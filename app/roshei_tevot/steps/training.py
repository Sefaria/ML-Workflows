import math
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer, TrainingArguments

from encoder_pretraining.steps.training.callbacks import SlackEvalCallback, ThroughputCallback
from encoder_pretraining.steps.training.checkpoints import GcsCheckpointCallback, restore_latest_checkpoint

from .data import (CrossCollator, ResampledDataset, SiameseCollator, attach_candidates, load_hebrew_names, load_lexicon,
                   load_split, sampling_weights)
from .evaluation import quick_top1
from .model import build_model, save_model

# Trainer's own step-based saving is pushed out of reach; GcsCheckpointCallback decides when to save.
NEVER_SAVE_STEPS = 10**9


class ExpansionTrainer(Trainer):
    """The model computes its own loss from the collated batch. Evaluation adds validation top-1 on rows with two or more candidates."""

    def __init__(self, *args, ranking_rows: Optional[list[dict]] = None, encode_settings: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranking_rows = ranking_rows or []
        self.encode_settings = encode_settings or {}
        self.best_top1: Optional[dict] = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.ranking_rows:
            device = next(self.model.parameters()).device.type
            started = time.monotonic()
            ranking = quick_top1(self.model, self.processing_class, self.ranking_rows, device, **self.encode_settings)
            extra = {f"{metric_key_prefix}_{name}": value for name, value in ranking.items()}
            print(f"step={self.state.global_step} validation top1={ranking['top1']:.4f} (frequency {ranking['frequency_top1']:.4f}) "
                  f"in {time.monotonic() - started:.0f}s")
            if self.best_top1 is None or ranking["top1"] > self.best_top1["top1"]:
                self.best_top1 = {"step": self.state.global_step, **ranking}
            self.log(extra)
            metrics.update(extra)
        return metrics


def run_training(
    base_model_path: str,
    output_dir: str,
    checkpoint_bucket: str,
    checkpoint_prefix: str,
    workflow_name: str,
    dataset_repo: str,
    architecture: str,
    pooling: str,
    within_weight: float,
    max_length: int,
    context_words: int,
    hard_negatives: int,
    min_candidates: int,
    key_alpha: float,
    synthetic_weight: float,
    samples_per_epoch: int,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    eval_steps: int,
    eval_max_rows: int,
    checkpoint_minutes: float,
    seed: int,
    dataloader_num_workers: int,
    max_train_rows: Optional[int] = None,
) -> dict:
    model, tokenizer = build_model(base_model_path, architecture, pooling, within_weight)
    lexicon = load_lexicon(dataset_repo)
    names = load_hebrew_names(dataset_repo)

    started = time.monotonic()
    train_rows, train_report = attach_candidates(load_split("train", dataset_repo, context_words, max_train_rows, names), lexicon, min_candidates, training=True)
    val_all = load_split("val", dataset_repo, context_words, None, names)
    val_rows, val_report = attach_candidates(val_all, lexicon, min_candidates, training=True)
    ranking_rows, ranking_report = attach_candidates(val_all, lexicon, min_candidates, training=False)
    val_rows, ranking_rows = val_rows[:eval_max_rows], ranking_rows[:eval_max_rows]
    print(f"Candidates attached in {time.monotonic() - started:.0f}s: train {train_report}, val {val_report}, ranking {ranking_report}")

    train_dataset = ResampledDataset(train_rows, samples_per_epoch, sampling_weights(train_rows, key_alpha, synthetic_weight), seed)
    eval_dataset = ResampledDataset(val_rows, len(val_rows))
    if architecture == "siamese":
        collator = SiameseCollator(tokenizer, max_length, context_words, hard_negatives=hard_negatives, seed=seed)
    else:
        collator = CrossCollator(tokenizer, max_length, context_words, hard_negatives=hard_negatives, seed=seed)

    steps_per_epoch = len(train_dataset) // batch_size
    max_steps = max(1, round(epochs * steps_per_epoch))

    checkpoint_dir = Path(output_dir) / "trainer"
    resume_path = restore_latest_checkpoint(checkpoint_bucket, checkpoint_prefix, str(checkpoint_dir))

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=math.ceil(warmup_ratio * max_steps),
        max_steps=max_steps,
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=NEVER_SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    encode_settings = {"max_length": max_length, "context_words": context_words}
    throughput = ThroughputCallback(batch_size * max_length)
    trainer = ExpansionTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        ranking_rows=ranking_rows,
        encode_settings=encode_settings,
        callbacks=[
            GcsCheckpointCallback(checkpoint_bucket, checkpoint_prefix, checkpoint_minutes * 60),
            throughput,
            SlackEvalCallback(workflow_name, max_steps / max(steps_per_epoch, 1), base_model_path,
                              metric_keys=("eval_loss", "eval_top1", "eval_frequency_top1")),
        ],
    )
    result = trainer.train(resume_from_checkpoint=resume_path)
    final_eval = trainer.evaluate()

    final_dir = Path(output_dir) / "final"
    save_model(trainer.model, tokenizer, str(final_dir), {
        **encode_settings, "base_model": base_model_path, "dataset_repo": dataset_repo, "min_candidates": min_candidates,
    })
    return {
        "plan": {"steps_per_epoch": steps_per_epoch, "max_steps": max_steps, "samples_per_epoch": len(train_dataset)},
        "architecture": architecture,
        "resumed_from_checkpoint": resume_path,
        "candidates": {"train": train_report, "val": val_report, "ranking": ranking_report},
        "train_metrics": result.metrics,
        "final_eval": final_eval,
        "best_validation_top1": trainer.best_top1,
        "mean_tokens_per_second": throughput.mean_tokens_per_second(),
        "final_model_dir": str(final_dir),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
