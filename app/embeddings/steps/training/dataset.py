import json
import random
from pathlib import Path
from typing import Optional

from sentence_transformers import InputExample


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_positive_pair_examples(
    documents: list[dict],
    queries: list[dict],
    qrels: list[dict],
    max_examples: Optional[int] = None,
) -> tuple[list[InputExample], dict]:
    document_text_by_id = {str(document["doc_id"]): str(document["text"]) for document in documents}
    query_text_by_id = {str(query["query_id"]): str(query["text"]) for query in queries}

    examples: list[InputExample] = []
    missing_documents = 0
    missing_queries = 0
    non_positive_qrels = 0

    for qrel in qrels:
        if max_examples is not None and len(examples) >= max_examples:
            break

        relevance = int(qrel.get("relevance") or 0)
        if relevance <= 0:
            non_positive_qrels += 1
            continue

        query_id = str(qrel["query_id"])
        doc_id = str(qrel["doc_id"])
        query_text = query_text_by_id.get(query_id)
        document_text = document_text_by_id.get(doc_id)

        if query_text is None:
            missing_queries += 1
            continue
        if document_text is None:
            missing_documents += 1
            continue

        examples.append(InputExample(texts=[query_text, document_text]))

    report = {
        "documents_count": len(documents),
        "queries_count": len(queries),
        "qrels_count": len(qrels),
        "positive_pair_count": len(examples),
        "max_examples": max_examples,
        "missing_documents": missing_documents,
        "missing_queries": missing_queries,
        "non_positive_qrels": non_positive_qrels,
    }
    return examples, report


def split_qrels_for_validation(
    qrels: list[dict],
    validation_fraction: float,
    validation_seed: int,
    max_examples: Optional[int] = None,
) -> tuple[list[dict], list[dict], dict]:
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be >= 0 and < 1.")

    positive_qrels = [qrel for qrel in qrels if int(qrel.get("relevance") or 0) > 0]
    if max_examples is not None:
        positive_qrels = positive_qrels[:max_examples]

    positive_doc_ids = sorted({str(qrel["doc_id"]) for qrel in positive_qrels})

    if not positive_qrels or validation_fraction == 0:
        report = {
            "strategy": "random_positive_doc_id_split",
            "validation_fraction": validation_fraction,
            "validation_seed": validation_seed,
            "positive_qrels_considered": len(positive_qrels),
            "positive_doc_ids_considered": len(positive_doc_ids),
            "validation_doc_ids_count": 0,
            "train_qrels_count": len(positive_qrels),
            "validation_qrels_count": 0,
        }
        return positive_qrels, [], report

    random.Random(validation_seed).shuffle(positive_doc_ids)
    validation_doc_count = max(1, int(len(positive_doc_ids) * validation_fraction))
    validation_doc_ids = set(positive_doc_ids[:validation_doc_count])
    validation_qrels = [qrel for qrel in positive_qrels if str(qrel["doc_id"]) in validation_doc_ids]
    train_qrels = [qrel for qrel in positive_qrels if str(qrel["doc_id"]) not in validation_doc_ids]
    if not train_qrels:
        train_qrels, validation_qrels = validation_qrels[:1], validation_qrels[1:]
        validation_doc_ids = {str(qrel["doc_id"]) for qrel in validation_qrels}

    report = {
        "strategy": "random_positive_doc_id_split",
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
        "positive_qrels_considered": len(positive_qrels),
        "positive_doc_ids_considered": len(positive_doc_ids),
        "validation_doc_ids_count": len(validation_doc_ids),
        "train_qrels_count": len(train_qrels),
        "validation_qrels_count": len(validation_qrels),
    }
    return train_qrels, validation_qrels, report
