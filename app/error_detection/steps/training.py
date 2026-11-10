import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments

from encoder_pretraining.steps.training.callbacks import SlackEvalCallback, ThroughputCallback
from encoder_pretraining.steps.training.checkpoints import GcsCheckpointCallback, restore_latest_checkpoint

from .data import ResampledDataset, TokenCollator, build_training_rows, chunk_rows, load_rows, sampling_weights
from .evaluation import token_metrics
from .model import WeightedTokenTrainer, build_model, save_model

# Trainer's own step-based saving is pushed out of reach; GcsCheckpointCallback decides when to save.
NEVER_SAVE_STEPS = 10**9


def run_training(
    base_model_path: str,
    output_dir: str,
    checkpoint_bucket: str,
    checkpoint_prefix: str,
    workflow_name: str,
    dataset_repo: str,
    max_length: int,
    paired_negatives: bool,
    negative_weight: float,
    synthetic_weight: float,
    pos_weight: float,
    samples_per_epoch: Optional[int],
    epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    eval_steps: int,
    checkpoint_minutes: float,
    seed: int,
    dataloader_num_workers: int,
    max_train_rows: Optional[int] = None,
) -> dict:
    model, tokenizer = build_model(base_model_path)

    started = time.monotonic()
    real = load_rows("real", "train", dataset_repo, max_train_rows)
    synthetic = load_rows("synthetic", "train", dataset_repo, max_train_rows) if synthetic_weight > 0 else []
    train_chunks = chunk_rows(tokenizer, build_training_rows(real, synthetic, paired_negatives), max_length)
    # Validation measures what the detector is for: real erroneous windows and their corrected copies, never synthetic text.
    validation = load_rows("real", "validation", dataset_repo)
    eval_chunks = chunk_rows(tokenizer, build_training_rows(validation, [], paired_negatives=True), max_length)
    data_report = {
        "train_chunks": dict(Counter(c["source"] for c in train_chunks)),
        "eval_chunks": dict(Counter(c["source"] for c in eval_chunks)),
        "train_rows": {"real": len(real), "synthetic": len(synthetic)},
    }
    print(f"Chunked in {time.monotonic() - started:.0f}s: {data_report}")

    size = samples_per_epoch or len(train_chunks)
    train_dataset = ResampledDataset(train_chunks, size, sampling_weights(train_chunks, synthetic_weight, negative_weight), seed)
    eval_dataset = ResampledDataset(eval_chunks, len(eval_chunks))
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
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
    throughput = ThroughputCallback(batch_size * max_length)
    trainer = WeightedTokenTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TokenCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        compute_metrics=token_metrics,
        pos_weight=pos_weight,
        callbacks=[
            GcsCheckpointCallback(checkpoint_bucket, checkpoint_prefix, checkpoint_minutes * 60),
            throughput,
            SlackEvalCallback(workflow_name, max_steps / steps_per_epoch, base_model_path,
                              metric_keys=("eval_loss", "eval_pr_auc", "eval_best_f1")),
        ],
    )
    result = trainer.train(resume_from_checkpoint=resume_path)
    last_eval = trainer.evaluate()
    restored = trainer.restore_best()
    best_eval = trainer.evaluate(metric_key_prefix="best") if restored else last_eval

    final_dir = Path(output_dir) / "final"
    save_model(trainer.model, tokenizer, str(final_dir), {
        "base_model": base_model_path, "dataset_repo": dataset_repo, "max_length": max_length, "pos_weight": pos_weight,
        "synthetic_weight": synthetic_weight, "paired_negatives": paired_negatives, "negative_weight": negative_weight,
        "validation_threshold": best_eval.get("best_threshold", best_eval.get("eval_threshold")),
        "best_step": trainer.best_step,
    })
    return {
        "plan": {"steps_per_epoch": steps_per_epoch, "max_steps": max_steps, "samples_per_epoch": len(train_dataset)},
        "data": data_report,
        "resumed_from_checkpoint": resume_path,
        "train_metrics": result.metrics,
        "last_eval": last_eval,
        "best_eval": best_eval,
        "best_step": trainer.best_step,
        "mean_tokens_per_second": throughput.mean_tokens_per_second(),
        "final_model_dir": str(final_dir),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
