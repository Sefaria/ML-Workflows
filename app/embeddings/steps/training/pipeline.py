import math
from pathlib import Path
from typing import Callable, Optional

import torch
from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader

from .dataset import build_positive_pair_examples, read_jsonl, split_qrels_for_validation
from .validation import build_ir_validation_evaluator


def build_sentence_transformer_model(
    model_name_or_path: str,
    max_seq_length: int = 512,
    normalize_embeddings: bool = True,
) -> SentenceTransformer:
    transformer = models.Transformer(
        model_name_or_path,
        max_seq_length=max_seq_length,
    )
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    modules = [transformer, pooling]
    if normalize_embeddings:
        modules.append(models.Normalize())
    return SentenceTransformer(modules=modules)


def train_from_query_dataset_artifacts(
    documents_path: str,
    queries_path: str,
    qrels_path: str,
    base_model_path: str,
    output_dir: str,
    max_seq_length: int = 512,
    normalize_embeddings: bool = True,
    batch_size: int = 16,
    epochs: int = 1,
    warmup_steps: Optional[int] = None,
    learning_rate: float = 2e-5,
    max_examples: Optional[int] = None,
    use_amp: Optional[bool] = None,
    validation_fraction: float = 0.1,
    validation_seed: int = 0,
    evaluation_steps: Optional[int] = None,
    evaluation_callback: Optional[Callable[[dict[str, float], float, int], None]] = None,
) -> dict:
    documents = read_jsonl(documents_path)
    queries = read_jsonl(queries_path)
    qrels = read_jsonl(qrels_path)
    train_qrels, validation_qrels, split_report = split_qrels_for_validation(
        qrels=qrels,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        max_examples=max_examples,
    )
    examples, dataset_report = build_positive_pair_examples(
        documents=documents,
        queries=queries,
        qrels=train_qrels,
    )
    if not examples:
        raise ValueError("No positive query/document training pairs were built from the query dataset artifacts.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sentence_transformer_model(
        model_name_or_path=base_model_path,
        max_seq_length=max_seq_length,
        normalize_embeddings=normalize_embeddings,
    )
    model.to(device)

    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    evaluator, validation_report = build_ir_validation_evaluator(
        documents=documents,
        queries=queries,
        validation_qrels=validation_qrels,
        batch_size=batch_size,
        evaluation_callback=evaluation_callback,
    )
    resolved_warmup_steps = warmup_steps
    if resolved_warmup_steps is None:
        resolved_warmup_steps = max(1, int(steps_per_epoch * epochs * 0.1))
    resolved_evaluation_steps = evaluation_steps
    if evaluator is not None and resolved_evaluation_steps is None:
        resolved_evaluation_steps = steps_per_epoch
    if evaluator is not None and resolved_evaluation_steps <= 0:
        raise ValueError("evaluation_steps must be positive when validation is enabled.")
    if evaluator is None:
        resolved_evaluation_steps = 0
    resolved_use_amp = torch.cuda.is_available() if use_amp is None else use_amp

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=resolved_warmup_steps,
        optimizer_params={"lr": learning_rate},
        evaluation_steps=resolved_evaluation_steps,
        output_path=str(output_path),
        save_best_model=evaluator is not None,
        use_amp=resolved_use_amp,
        show_progress_bar=True,
    )

    return {
        "base_model_path": base_model_path,
        "output_dir": str(output_path),
        "device": device,
        "max_seq_length": max_seq_length,
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": resolved_warmup_steps,
        "evaluation_steps": resolved_evaluation_steps,
        "learning_rate": learning_rate,
        "use_amp": resolved_use_amp,
        "loss": "MultipleNegativesRankingLoss",
        "dataset": dataset_report,
        "validation_split": split_report,
        "validation": validation_report,
        "best_model": {
            "enabled": evaluator is not None,
            "selection_metric": validation_report["primary_metric"] if evaluator is not None else None,
            "selection_metric_key": validation_report["selection_metric_key"] if evaluator is not None else None,
            "output_dir": str(output_path),
        },
    }
