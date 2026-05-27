import json

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


def build_query_type_prompt(
    doc: ChunkedDocument,
    query_type: str,
    config: QueryGenerationConfig,
) -> str:
    query_lang = doc.lang
    query_language_name = config.query_language_names.get(query_lang, query_lang)
    type_specific_rules = build_type_specific_rules(query_type, query_language_name)
    compact_doc = {
        "doc_id": doc.doc_id,
        "lang": query_lang,
        "text": doc.text,
    }
    return f"""
You are creating an information-retrieval evaluation dataset for Jewish text search.

Create exactly {config.queries_per_type_per_doc} {query_type} retrieval queries for the supplied document.
This is NOT question answering. A query should map to useful documents.

Rules:
- Write every query in {query_language_name}.
- Query type must be: {query_type}.
- The supplied document is highly relevant to every query you create.
- Prefer realistic user search language over citation wording.
- Use only the document text itself as the basis for the queries.
- Do not infer or mention book names, authors, categories, titles, references, or other metadata.
- Do not generate queries about where the text comes from; generate queries only about what the text says.
{type_specific_rules}

Return only valid JSON. Do not use markdown fences.
Shape:
{{
  "queries": [
    {{"text": "...", "reason": "why this document is highly relevant"}}
  ]
}}

Document:
{json.dumps(compact_doc, ensure_ascii=False)}
""".strip()


def build_system_prompt() -> str:
    return "You create careful retrieval evaluation datasets. Return only valid JSON."
