import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from ..data import MaskingCollator, PackedTokenDataset, load_eval_windows
from ..data.dataset import read_manifest
from ..evaluation import StaticMaskedDataset
from .callbacks import SlackEvalCallback, ThroughputCallback
from .checkpoints import GcsCheckpointCallback, restore_latest_checkpoint
from .model import load_masked_lm, resolve_attention_implementation


# Trainer's own step-based saving is disabled by pushing it out of reach;
# GcsCheckpointCallback decides when to save.
NEVER_SAVE_STEPS = 10**9


def plan_training_steps(
    train_sequences: int,
    seq_len: int,
    global_batch_sequences: int,
    epochs: float,
    max_tokens: Optional[int],
) -> dict:
    steps_per_epoch = train_sequences // global_batch_sequences
    if steps_per_epoch < 1:
        raise ValueError("Train set is smaller than one global batch.")
    max_steps = max(1, round(epochs * steps_per_epoch))
    tokens_per_step = global_batch_sequences * seq_len
    if max_tokens is not None:
        max_steps = max(1, min(max_steps, max_tokens // tokens_per_step))
    return {
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "tokens_per_step": tokens_per_step,
        "planned_tokens": max_steps * tokens_per_step,
        "planned_epochs": max_steps / steps_per_epoch,
    }


def pretrain_masked_lm(
    corpus_dir: str,
    base_model_path: str,
    output_dir: str,
    checkpoint_bucket: str,
    checkpoint_prefix: str,
    mlm_probability: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    epochs: float,
    max_tokens: Optional[int],
    global_batch_sequences: int,
    per_device_batch_size: int,
    eval_steps: int,
    eval_max_sequences: int,
    checkpoint_minutes: float,
    attn_implementation: str,
    reference_compile: bool,
    seed: int,
    dataloader_num_workers: int,
    workflow_name: str,
) -> dict:
    if global_batch_sequences % per_device_batch_size:
        raise ValueError("global_batch_sequences must be divisible by per_device_batch_size.")

    manifest = read_manifest(corpus_dir)
    seq_len = manifest["seq_len"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    train_dataset = PackedTokenDataset(corpus_dir, "train", manifest)
    eval_windows = load_eval_windows(corpus_dir, manifest, eval_max_sequences)
    eval_dataset = StaticMaskedDataset(
        np.concatenate(list(eval_windows.values())),
        tokenizer,
        mlm_probability=mlm_probability,
        seed=seed,
    )

    plan = plan_training_steps(len(train_dataset), seq_len, global_batch_sequences, epochs, max_tokens)
    print(f"Training plan: {plan}")

    resolved_attention = resolve_attention_implementation(attn_implementation)
    print(f"Attention implementation: {resolved_attention}")
    model = load_masked_lm(base_model_path, resolved_attention, reference_compile)

    checkpoint_dir = Path(output_dir) / "trainer"
    resume_path = restore_latest_checkpoint(checkpoint_bucket, checkpoint_prefix, str(checkpoint_dir))

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=global_batch_sequences // per_device_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=math.ceil(warmup_ratio * plan["max_steps"]),
        max_steps=plan["max_steps"],
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
        disable_tqdm=True,
    )
    throughput = ThroughputCallback(plan["tokens_per_step"])
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MaskingCollator(tokenizer, mlm_probability),
        processing_class=tokenizer,
        callbacks=[
            GcsCheckpointCallback(checkpoint_bucket, checkpoint_prefix, checkpoint_minutes * 60),
            throughput,
            SlackEvalCallback(workflow_name, plan["planned_epochs"], base_model_path),
        ],
    )

    train_result = trainer.train(resume_from_checkpoint=resume_path)
    final_eval = trainer.evaluate()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    return {
        "plan": plan,
        "attn_implementation": resolved_attention,
        "resumed_from_checkpoint": resume_path,
        "train_metrics": train_result.metrics,
        "final_eval": final_eval,
        "mean_tokens_per_second": throughput.mean_tokens_per_second(),
        "final_model_dir": str(final_dir),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
