import json
from typing import Any

from .config import QueryGenerationConfig
from .types import ChunkedDocument


def build_keyword_rules(query_language_name: str) -> str:
    return (
        "- Keyword queries should be short and fragment-like.\n"
        "- Prefer 2-6 meaningful content words.\n"
        "- Avoid full-sentence phrasing unless necessary.\n"
        f"- Use natural {query_language_name} search terms rather than citation markers."
    )


def build_question_rules(query_language_name: str) -> str:
    return (
        "- Question queries should be natural user questions.\n"
        "- They should ask about the content or claim of the document, not its source.\n"
        f"- Write fluent {query_language_name} questions a real user might type into search."
    )


def build_sentence_rules(query_language_name: str) -> str:
    return (
        "- Sentence queries must not be phrased as questions.\n"
        "- Do not ask for information or use interrogative structure.\n"
        f"- Avoid question words such as who, what, when, where, why, how, and their {query_language_name} equivalents.\n"
        "- Sentence queries should read like declarative search inputs describing a topic, case, claim, law, argument, or scenario.\n"
        '- Prefer formulations like "discussion of...", "case of...", "law of...", "argument about...", "passage describing...".\n'
        "- If a sentence query could be rewritten as a natural question with only minor edits, rewrite it to be more clearly declarative."
    )


def build_type_specific_rules(query_type: str, query_language_name: str) -> str:
    if query_type == "keyword":
        return build_keyword_rules(query_language_name)
    if query_type == "question":
        return build_question_rules(query_language_name)
    if query_type == "sentence":
        return build_sentence_rules(query_language_name)
    raise ValueError(f"Unsupported query type: {query_type}")


def build_base_texts(metadata: dict[str, Any]) -> list[dict[str, str]]:
    base_text_mappings = metadata.get("baseTextMappings") or {}
    if not isinstance(base_text_mappings, dict):
        return []

    base_texts = []
    for source_segment_ref, base_text in base_text_mappings.items():
        if not isinstance(base_text, dict):
            continue
        base_texts.append(
            {
                "commentary_segment_ref": str(source_segment_ref),
                "base_ref": str(base_text.get("ref") or ""),
                "base_index_title": str(base_text.get("indexTitle") or ""),
                "base_text": str(base_text.get("text") or ""),
            }
        )
    return base_texts


def build_query_type_prompt(
    doc: ChunkedDocument,
    query_type: str,
    config: QueryGenerationConfig,
) -> str:
    query_lang = doc.lang
    query_language_name = config.query_language_names.get(query_lang, query_lang)
    type_specific_rules = build_type_specific_rules(query_type, query_language_name)
    base_texts = build_base_texts(doc.metadata)
    compact_doc = {
        "doc_id": doc.doc_id,
        "lang": query_lang,
        "commentary_text": doc.text,
        "base_texts": base_texts,
    }
    base_text_instruction = (
        "The document includes base_texts. Treat them only as background for understanding the commentary. "
        "The retrieval queries must target the commentary's added explanation, interpretation, ruling, distinction, "
        "question, answer, or other extra information beyond the base text."
        if base_texts
        else "No base text context is supplied. Generate queries from the supplied document text."
    )
    return f"""
You are creating an information-retrieval evaluation dataset for Jewish text search.

Create exactly {config.queries_per_type_per_doc} {query_type} retrieval queries for the supplied document.
This is NOT question answering. A query should map to useful documents.
{base_text_instruction}

Rules:
- Write every query in {query_language_name}.
- Query type must be: {query_type}.
- The supplied document is highly relevant to every query you create.
- Prefer realistic user search language over citation wording.
- Use the commentary_text as the basis for the queries.
- If base_texts are present, use them only as context. Do not write a query that is only about the base text.
- A valid query must specifically retrieve the commentary because of information the commentary adds beyond the base text.
- Do not infer or mention book names, authors, categories, titles, references, or other metadata.
- Do not generate queries about where the text comes from; generate queries only about what the text says.
- If the commentary does not add enough meaningful searchable information beyond the base text to create a good query, return the skip signal instead of weak queries.
{type_specific_rules}

Return only valid JSON. Do not use markdown fences.
Shape for usable examples:
{{
  "skip": false,
  "queries": [
    {{"text": "...", "reason": "why the commentary, not just the base text, is highly relevant"}}
  ]
}}

Shape for unusable examples:
{{
  "skip": true,
  "skip_reason": "why the commentary does not add enough searchable information beyond the base text",
  "queries": []
}}

Document:
{json.dumps(compact_doc, ensure_ascii=False)}
""".strip()


def build_system_prompt() -> str:
    return "You create careful retrieval evaluation datasets. Return only valid JSON."
