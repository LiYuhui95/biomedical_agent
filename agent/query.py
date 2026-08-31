import json
import re

from agent.llm import LLMBackend


_PLACEHOLDER_PATTERN = re.compile(
    r"\b(compound|target|drug|gene|protein|disease)\s+[A-Z]\b"
)

_STOPWORDS = {
    "does", "is", "are", "the", "a", "an", "of", "in", "on",
    "affect", "affects", "improve", "improves", "increase",
    "decrease", "reduce", "enhance", "cause", "causes",
    "unknown", "result", "results", "effect", "impact",
    "play", "role", "and", "with", "to",
}


def _regex_fallback(question: str) -> str:
    cleaned = _PLACEHOLDER_PATTERN.sub("", question)
    cleaned = re.sub(r"[?.,]", "", cleaned)

    tokens = [
        tok for tok in cleaned.split()
        if tok.lower() not in _STOPWORDS
    ]

    return " AND ".join(tokens)

def _is_placeholder(term: str) -> bool:
    return bool(_PLACEHOLDER_PATTERN.search(term))

def extract_required_entities(
    question: str,
) -> list[str]:
    return re.findall(
        r"\b[A-Z][A-Z0-9-]{2,}\b",
        question,
    )

def ensure_required_entities(
    question: str,
    search_query: str,
) -> str:
    required_entities = extract_required_entities(
        question
    )

    missing_entities = [
        entity
        for entity in required_entities
        if entity.lower()
        not in search_query.lower()
    ]

    if not missing_entities:
        return search_query

    entity_prefix = " AND ".join(
        missing_entities
    )

    if not search_query:
        return entity_prefix

    return (
        f"{entity_prefix} AND "
        f"({search_query})"
    )

def rewrite_query(llm: LLMBackend, question: str) -> str:
    """
    Extract real biomedical terms from a question and combine
    them into a PubMed search query.

    Primary path: ask the LLM (JSON mode) to extract terms,
    dropping placeholder entities.
    Fallback path: deterministic regex extraction, used if the
    LLM call fails or returns something unusable.

    Returns "" if no real terms are found by either path.
    """

    prompt = f"""
You are a biomedical librarian.

Read the research question and extract a JSON list of the
real biomedical terms it contains (genes, proteins, drugs,
diseases, cell types, biological processes, etc.).

Drop placeholder or fictional entities (e.g. "compound X",
"target Z", "disease Y") but KEEP any real biomedical terms
that appear elsewhere in the same question.

Example:
Question: "Does compound X improve beta-cell survival?"
Answer: {{"terms": ["beta-cell survival"]}}

Example:
Question: "Does unknown target Z affect disease Y?"
Answer: {{"terms": []}}

Now extract terms for this question.
Return a JSON object of the form: {{"terms": ["term1", "term2"]}}

Research question:
{question}
"""

    used_fallback = False

    try:
        raw = llm.generate(prompt, json_mode=True)
        data = json.loads(raw)
        terms = data.get("terms", [])

        if not isinstance(terms, list):
            raise ValueError("terms is not a list")

        terms = [str(t).strip() for t in terms if str(t).strip()]

    except Exception:
        terms = []

    if not terms:
        used_fallback = True
        terms = _regex_fallback(question).split(" AND ")
        terms = [t for t in terms if t]

    terms = [t for t in terms if not _is_placeholder(t)]

    if not terms:
        search_query = ""
    else:
        search_query = " AND ".join(
            terms
        )

    search_query = ensure_required_entities(
        question=question,
        search_query=search_query,
    )

    print(
        f"DEBUG rewrite_query "
        f"terms={terms} "
        f"used_fallback={used_fallback} "
        f"final_query={search_query!r}"
    )

    return search_query

def refine_query(
    llm: LLMBackend,
    question: str,
    previous_query: str,
) -> str:
    prompt = f"""
You are refining a PubMed search query because the previous
query retrieved insufficient or mostly irrelevant evidence.

Research question:
{question}

Previous PubMed query:
{previous_query}

Generate a more precise PubMed query.

Requirements:
- Preserve every explicit gene, protein, drug, or target
  named in the research question.
- Keep the disease or biological condition.
- Add useful synonyms or PubMed field qualifiers when helpful.
- Do not invent entities.
- Return JSON only in this format:
  {{"query": "refined PubMed query"}}
"""

    try:
        raw = llm.generate(
            prompt,
            json_mode=True,
        )

        data = json.loads(raw)
        refined_query = str(
            data.get("query", "")
        ).strip()

    except Exception:
        refined_query = ""

    if not refined_query:
        refined_query = previous_query

    return ensure_required_entities(
        question=question,
        search_query=refined_query,
    )