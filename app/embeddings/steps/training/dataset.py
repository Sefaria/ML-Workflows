import json
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
