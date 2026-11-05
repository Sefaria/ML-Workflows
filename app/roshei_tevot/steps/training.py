import math
from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer, TrainingArguments

from encoder_pretraining.steps.training.callbacks import SlackEvalCallback, ThroughputCallback
from encoder_pretraining.steps.training.checkpoints import GcsCheckpointCallback, restore_latest_checkpoint

from .data import ExpansionCollator, ResampledDataset, load_split, sampling_weights
from .model import build_model, save_model

# Trainer's own step-based saving is pushed out of reach; GcsCheckpointCallback decides when to save.
NEVER_SAVE_STEPS = 10**9


class SiameseTrainer(Trainer):
    """The model computes its own loss from a query batch and a candidate pool; this only routes the batch to it."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs["queries"], inputs["candidates"], inputs["targets"])
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def run_training(
    base_model_path: str,
    output_dir: str,
    checkpoint_bucket: str,
    checkpoint_prefix: str,
    workflow_name: str,
    dataset_repo: str,
    pooling: str,
    max_length: int,
    context_words: int,
    hard_negatives: int,
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
    model, tokenizer = build_model(base_model_path, pooling)
    train_rows = load_split("train", dataset_repo, context_words, max_train_rows)
    val_rows = load_split("val", dataset_repo, context_words, eval_max_rows)
    print(f"Loaded {len(train_rows)} train and {len(val_rows)} validation rows")

    train_dataset = ResampledDataset(train_rows, samples_per_epoch, sampling_weights(train_rows, key_alpha, synthetic_weight), seed)
    eval_dataset = ResampledDataset(val_rows, len(val_rows))
    collator = ExpansionCollator(tokenizer, max_length, context_words, hard_negatives=hard_negatives, seed=seed)

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
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
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
        label_names=["targets"],
        disable_tqdm=True,
    )
    throughput = ThroughputCallback(batch_size * max_length)
    trainer = SiameseTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            GcsCheckpointCallback(checkpoint_bucket, checkpoint_prefix, checkpoint_minutes * 60),
            throughput,
            SlackEvalCallback(workflow_name, max_steps / max(steps_per_epoch, 1), base_model_path),
        ],
    )
    result = trainer.train(resume_from_checkpoint=resume_path)
    final_eval = trainer.evaluate()

    final_dir = Path(output_dir) / "final"
    save_model(trainer.model, tokenizer, str(final_dir), {
        "max_length": max_length, "context_words": context_words, "base_model": base_model_path, "dataset_repo": dataset_repo,
    })
    return {
        "plan": {"steps_per_epoch": steps_per_epoch, "max_steps": max_steps, "samples_per_epoch": len(train_dataset)},
        "resumed_from_checkpoint": resume_path,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_metrics": result.metrics,
        "final_eval": final_eval,
        "mean_tokens_per_second": throughput.mean_tokens_per_second(),
        "final_model_dir": str(final_dir),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
