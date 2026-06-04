from typing import Callable, Optional

from sentence_transformers.evaluation import InformationRetrievalEvaluator, SentenceEvaluator


PRIMARY_RETRIEVAL_METRIC = "ndcg@10"
PRIMARY_SCORE_FUNCTION = "cosine"
SENTENCE_TRANSFORMERS_PRIMARY_METRIC = f"{PRIMARY_SCORE_FUNCTION}_{PRIMARY_RETRIEVAL_METRIC}"


def document_retrieval_role_counts(documents: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        metadata = document.get("metadata") or {}
        role = str(metadata.get("retrieval_role") or "unspecified")
        counts[role] = counts.get(role, 0) + 1
    return counts


class ReportingInformationRetrievalEvaluator(SentenceEvaluator):
    def __init__(
        self,
        evaluator: InformationRetrievalEvaluator,
        callback: Callable[[dict[str, float], float, int], None],
    ):
        super().__init__()
        self.evaluator = evaluator
        self.callback = callback
        self.primary_metric = evaluator.primary_metric
        self.greater_is_better = getattr(evaluator, "greater_is_better", True)

    def __call__(self, model, output_path: Optional[str] = None, epoch: float = -1, steps: int = -1, *args, **kwargs):
        metrics = self.evaluator(model, output_path=output_path, epoch=epoch, steps=steps, *args, **kwargs)
        self.primary_metric = self.evaluator.primary_metric
        self.callback(metrics, epoch, steps)
        return metrics

    def get_config_dict(self) -> dict:
        return self.evaluator.get_config_dict()


def build_ir_validation_evaluator(
    documents: list[dict],
    queries: list[dict],
    validation_qrels: list[dict],
    batch_size: int,
    name: str = "validation",
    evaluation_callback: Optional[Callable[[dict[str, float], float, int], None]] = None,
) -> tuple[Optional[InformationRetrievalEvaluator], dict]:
    document_text_by_id = {str(document["doc_id"]): str(document["text"]) for document in documents}
    query_text_by_id = {str(query["query_id"]): str(query["text"]) for query in queries}

    relevant_doc_ids_by_query: dict[str, set[str]] = {}
    missing_documents = 0
    missing_queries = 0

    for qrel in validation_qrels:
        if int(qrel.get("relevance") or 0) <= 0:
            continue

        query_id = str(qrel["query_id"])
        doc_id = str(qrel["doc_id"])

        if query_id not in query_text_by_id:
            missing_queries += 1
            continue
        if doc_id not in document_text_by_id:
            missing_documents += 1
            continue

        relevant_doc_ids_by_query.setdefault(query_id, set()).add(doc_id)

    validation_query_ids = sorted(relevant_doc_ids_by_query)
    validation_doc_ids = sorted({doc_id for doc_ids in relevant_doc_ids_by_query.values() for doc_id in doc_ids})

    report = {
        "evaluator": "InformationRetrievalEvaluator",
        "name": name,
        "query_count": len(validation_query_ids),
        "relevant_document_count": len(validation_doc_ids),
        "corpus_document_count": len(document_text_by_id),
        "retrieval_role_counts": document_retrieval_role_counts(documents),
        "qrels_count": len(validation_qrels),
        "missing_documents": missing_documents,
        "missing_queries": missing_queries,
        "primary_metric": PRIMARY_RETRIEVAL_METRIC,
        "primary_score_function": PRIMARY_SCORE_FUNCTION,
        "sentence_transformers_primary_metric": SENTENCE_TRANSFORMERS_PRIMARY_METRIC,
        "selection_metric_key": f"{name}_{SENTENCE_TRANSFORMERS_PRIMARY_METRIC}",
        "metrics": {
            "ndcg_at_k": [10],
            "mrr_at_k": [10],
            "precision_recall_at_k": [10, 50],
            "accuracy_at_k": [1, 3, 5, 10],
            "map_at_k": [100],
        },
    }

    if not validation_query_ids:
        return None, report

    evaluator = InformationRetrievalEvaluator(
        queries={query_id: query_text_by_id[query_id] for query_id in validation_query_ids},
        corpus=document_text_by_id,
        relevant_docs=relevant_doc_ids_by_query,
        ndcg_at_k=[10],
        mrr_at_k=[10],
        precision_recall_at_k=[10, 50],
        accuracy_at_k=[1, 3, 5, 10],
        map_at_k=[100],
        batch_size=batch_size,
        name=name,
        write_csv=True,
        main_score_function=PRIMARY_SCORE_FUNCTION,
    )
    evaluator.primary_metric = SENTENCE_TRANSFORMERS_PRIMARY_METRIC
    if evaluation_callback is not None:
        return ReportingInformationRetrievalEvaluator(evaluator, evaluation_callback), report
    return evaluator, report
