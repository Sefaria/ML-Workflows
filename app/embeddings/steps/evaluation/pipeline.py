from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.steps.patot.embedding import GeminiEmbedder, l2_normalize_vector


K_VALUES = [1, 3, 5, 10, 20, 50]
PRIMARY_METRIC = "ndcg@10"


@dataclass(frozen=True)
class EvaluationConfig:
    model_source: str
    sentence_transformer_model_path: Optional[str] = None
    sentence_transformer_batch_size: int = 32
    normalize_sentence_transformer_embeddings: bool = True
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-embedding-001"
    gemini_output_dimensionality: int = 1536
    gemini_query_task_type: str = "RETRIEVAL_QUERY"
    gemini_doc_task_type: str = "RETRIEVAL_DOCUMENT"
    gemini_similarity_metric: str = "dot"
    gemini_normalize_embeddings: bool = True
    gemini_cache_enabled: bool = True
    gemini_cache_path: str = "/cache/evaluation/gemini_embedding_cache.sqlite"
    gemini_max_workers: int = 4


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: str) -> dict:
    local_path = Path(path)
    if not local_path.exists():
        return {}
    return json.loads(local_path.read_text())


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def score_documents(query_vector: np.ndarray, document_matrix: np.ndarray, similarity_metric: str) -> np.ndarray:
    if similarity_metric == "dot":
        return document_matrix @ query_vector
    if similarity_metric == "cosine":
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(document_matrix, axis=1)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1.0
        return (document_matrix @ query_vector) / denom
    if similarity_metric == "euclidean":
        return -np.linalg.norm(document_matrix - query_vector, axis=1)
    raise ValueError(f"Unknown similarity metric: {similarity_metric}")


def ranked_documents(query_vector: np.ndarray, document_ids: list[str], document_matrix: np.ndarray, similarity_metric: str) -> list[tuple[str, float]]:
    scores = score_documents(query_vector, document_matrix, similarity_metric)
    order = np.argsort(-scores)
    return [(document_ids[index], float(scores[index])) for index in order]


def compute_ndcg(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_relevant_count = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    return dcg / idcg if idcg else 0.0


def compute_recall(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    return len(set(ranked_doc_ids[:k]) & relevant_doc_ids) / len(relevant_doc_ids)


def compute_mrr(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def compute_first_relevant_rank(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return float(rank)
    return float("inf")


def compute_query_metrics(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> dict[str, float]:
    metrics = {}
    for k in K_VALUES:
        metrics[f"ndcg@{k}"] = compute_ndcg(ranked_doc_ids, relevant_doc_ids, k)
    for k in K_VALUES:
        metrics[f"recall@{k}"] = compute_recall(ranked_doc_ids, relevant_doc_ids, k)
    for k in K_VALUES:
        metrics[f"mrr@{k}"] = compute_mrr(ranked_doc_ids, relevant_doc_ids, k)
    metrics["mean_first_relevant_rank"] = compute_first_relevant_rank(ranked_doc_ids, relevant_doc_ids)
    return metrics


def aggregate_metric(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("inf") if values else 0.0
    return sum(finite_values) / len(finite_values)


def summarize_query_rows(rows: list[dict]) -> dict:
    metric_names = [f"{prefix}@{k}" for prefix in ("ndcg", "recall", "mrr") for k in K_VALUES]
    metric_names.append("mean_first_relevant_rank")
    summary = {metric_name: aggregate_metric([row["metrics"][metric_name] for row in rows]) for metric_name in metric_names}
    summary["query_count"] = len(rows)
    return summary


def build_relevance_map(queries: list[dict], qrels: list[dict], documents: list[dict]) -> tuple[dict[str, set[str]], dict]:
    query_ids = {str(query["query_id"]) for query in queries}
    document_ids = {str(document["doc_id"]) for document in documents}
    relevant_doc_ids_by_query: dict[str, set[str]] = {}
    missing_queries = 0
    missing_documents = 0
    non_positive_qrels = 0
    for qrel in qrels:
        if int(qrel.get("relevance") or 0) <= 0:
            non_positive_qrels += 1
            continue
        query_id = str(qrel["query_id"])
        doc_id = str(qrel["doc_id"])
        if query_id not in query_ids:
            missing_queries += 1
            continue
        if doc_id not in document_ids:
            missing_documents += 1
            continue
        relevant_doc_ids_by_query.setdefault(query_id, set()).add(doc_id)
    return relevant_doc_ids_by_query, {
        "query_count": len(query_ids),
        "document_count": len(document_ids),
        "qrels_count": len(qrels),
        "judged_query_count": len(relevant_doc_ids_by_query),
        "missing_queries": missing_queries,
        "missing_documents": missing_documents,
        "non_positive_qrels": non_positive_qrels,
    }


def encode_with_sentence_transformer(documents: list[dict], queries: list[dict], config: EvaluationConfig) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict]:
    if not config.sentence_transformer_model_path:
        raise ValueError("sentence_transformer_model_path is required for SentenceTransformer evaluation.")
    model = SentenceTransformer(config.sentence_transformer_model_path)
    doc_ids = [str(document["doc_id"]) for document in documents]
    doc_texts = [str(document["text"]) for document in documents]
    query_texts = [str(query["text"]) for query in queries]
    query_ids = [str(query["query_id"]) for query in queries]
    document_matrix = model.encode(
        doc_texts,
        batch_size=config.sentence_transformer_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_sentence_transformer_embeddings,
        show_progress_bar=True,
    )
    query_matrix = model.encode(
        query_texts,
        batch_size=config.sentence_transformer_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_sentence_transformer_embeddings,
        show_progress_bar=True,
    )
    return doc_ids, document_matrix, dict(zip(query_ids, query_matrix)), {
        "backend": "sentence_transformer",
        "model_path": config.sentence_transformer_model_path,
        "batch_size": config.sentence_transformer_batch_size,
        "normalize_embeddings": config.normalize_sentence_transformer_embeddings,
        "similarity_metric": "cosine",
    }


def _embed_gemini_items(items: list[dict], text_key: str, id_key: str, task_type: str, config: EvaluationConfig) -> tuple[dict[str, list[float]], list[dict]]:
    if not config.gemini_api_key:
        raise ValueError("gemini_api_key is required for Gemini evaluation.")
    embedder = GeminiEmbedder(
        api_key=config.gemini_api_key,
        cache_enabled=config.gemini_cache_enabled,
        cache_path=config.gemini_cache_path,
    )
    vectors = {}
    failures = []

    def worker(item: dict) -> tuple[str, list[float]]:
        item_id = str(item[id_key])
        vector = embedder.embed_text(
            model=config.gemini_model,
            text=str(item[text_key]),
            output_dimensionality=config.gemini_output_dimensionality,
            task_type=task_type,
        )
        if config.gemini_normalize_embeddings:
            vector = l2_normalize_vector(vector)
        return item_id, vector

    with ThreadPoolExecutor(max_workers=max(1, config.gemini_max_workers)) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                item_id, vector = future.result()
                vectors[item_id] = vector
            except Exception as exc:
                failures.append(
                    {
                        "id": str(item.get(id_key)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return vectors, failures


def encode_with_gemini(documents: list[dict], queries: list[dict], config: EvaluationConfig) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict]:
    doc_vectors_by_id, doc_failures = _embed_gemini_items(documents, "text", "doc_id", config.gemini_doc_task_type, config)
    query_vectors_by_id, query_failures = _embed_gemini_items(queries, "text", "query_id", config.gemini_query_task_type, config)
    doc_ids = [str(document["doc_id"]) for document in documents if str(document["doc_id"]) in doc_vectors_by_id]
    if not doc_ids:
        raise ValueError("Gemini evaluation produced no document embeddings.")
    document_matrix = np.array([doc_vectors_by_id[doc_id] for doc_id in doc_ids], dtype=float)
    if config.gemini_normalize_embeddings:
        document_matrix = l2_normalize_matrix(document_matrix)
    query_vectors = {query_id: np.array(vector, dtype=float) for query_id, vector in query_vectors_by_id.items()}
    if config.gemini_normalize_embeddings:
        query_vectors = {
            query_id: vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector
            for query_id, vector in query_vectors.items()
        }
    return doc_ids, document_matrix, query_vectors, {
        "backend": "gemini",
        "model": config.gemini_model,
        "output_dimensionality": config.gemini_output_dimensionality,
        "query_task_type": config.gemini_query_task_type,
        "doc_task_type": config.gemini_doc_task_type,
        "similarity_metric": config.gemini_similarity_metric,
        "normalize_embeddings": config.gemini_normalize_embeddings,
        "cache_enabled": config.gemini_cache_enabled,
        "cache_path": config.gemini_cache_path,
        "max_workers": config.gemini_max_workers,
        "document_embedding_failures": doc_failures,
        "query_embedding_failures": query_failures,
    }


def evaluate_retrieval_dataset(
    documents_path: str,
    queries_path: str,
    qrels_path: str,
    metadata_path: Optional[str],
    output_dir: str,
    config: EvaluationConfig,
) -> dict:
    documents = read_jsonl(documents_path)
    queries = read_jsonl(queries_path)
    qrels = read_jsonl(qrels_path)
    metadata = read_json(metadata_path) if metadata_path else {}
    relevant_doc_ids_by_query, relevance_report = build_relevance_map(queries, qrels, documents)
    judged_queries = [query for query in queries if relevant_doc_ids_by_query.get(str(query["query_id"]))]
    if not judged_queries:
        raise ValueError("No judged queries found for evaluation.")

    if config.model_source == "gemini":
        doc_ids, document_matrix, query_vectors_by_id, encoder_report = encode_with_gemini(documents, judged_queries, config)
        similarity_metric = config.gemini_similarity_metric
    else:
        doc_ids, document_matrix, query_vectors_by_id, encoder_report = encode_with_sentence_transformer(documents, judged_queries, config)
        similarity_metric = "cosine"

    per_query_rows = []
    skipped_queries = []
    for query in judged_queries:
        query_id = str(query["query_id"])
        query_vector = query_vectors_by_id.get(query_id)
        if query_vector is None:
            skipped_queries.append({"query_id": query_id, "reason": "missing_query_embedding"})
            continue
        if query_vector.shape[0] != document_matrix.shape[1]:
            raise ValueError(
                f"Query/document dimension mismatch for {query_id}: query={query_vector.shape[0]} docs={document_matrix.shape[1]}"
            )
        relevant_doc_ids = relevant_doc_ids_by_query[query_id]
        ranked = ranked_documents(query_vector, doc_ids, document_matrix, similarity_metric)
        ranked_doc_ids = [doc_id for doc_id, _score in ranked]
        metrics = compute_query_metrics(ranked_doc_ids, relevant_doc_ids)
        per_query_rows.append(
            {
                "query_id": query_id,
                "query_text": query.get("text", ""),
                "query_type": query.get("type"),
                "query_lang": query.get("lang"),
                "positive_doc_ids": sorted(relevant_doc_ids),
                "top_10": [
                    {
                        "doc_id": doc_id,
                        "rank": rank,
                        "score": score,
                        "is_relevant": doc_id in relevant_doc_ids,
                    }
                    for rank, (doc_id, score) in enumerate(ranked[:10], start=1)
                ],
                "metrics": metrics,
            }
        )

    if not per_query_rows:
        raise ValueError("No query results were produced.")

    summary = summarize_query_rows(per_query_rows)
    summary_row = {
        "model_source": config.model_source,
        "backend": encoder_report["backend"],
        "similarity_metric": similarity_metric,
        "query_count": summary["query_count"],
        "ndcg@1": summary["ndcg@1"],
        "ndcg@3": summary["ndcg@3"],
        "ndcg@5": summary["ndcg@5"],
        "ndcg@10": summary["ndcg@10"],
        "ndcg@20": summary["ndcg@20"],
        "ndcg@50": summary["ndcg@50"],
        "recall@10": summary["recall@10"],
        "recall@50": summary["recall@50"],
        "mrr@10": summary["mrr@10"],
        "mean_first_relevant_rank": summary["mean_first_relevant_rank"],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path / "per_query_results.jsonl", per_query_rows)
    write_csv(output_path / "results_summary.csv", [summary_row])
    write_json(output_path / "results_summary.json", summary_row)
    report = {
        "status": "success",
        "primary_metric": PRIMARY_METRIC,
        "k_values": K_VALUES,
        "positive_only_qrels_caveat": "Unjudged query/document pairs are treated as implicit non-relevant.",
        "dataset": {
            "documents_count": len(documents),
            "queries_count": len(queries),
            "qrels_count": len(qrels),
            "metadata": metadata,
            **relevance_report,
        },
        "encoder": encoder_report,
        "summary": summary_row,
        "skipped_queries": skipped_queries,
    }
    write_json(output_path / "run_config.json", report)
    return report
