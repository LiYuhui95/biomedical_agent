import json

from agent.llm import LLMBackend
from agent.state import ScientificEvidence


def extract_evidence(
    llm: LLMBackend,
    question: str,
    pmid: str,
    abstract: str,
) -> ScientificEvidence:

    prompt = f"""
You are a biomedical literature analyst.

Research question:
{question}

Paper PMID:
{pmid}

Paper abstract:
{abstract}

Extract evidence from this paper that is relevant
to the research question.

Return ONLY a valid JSON object.
Do not add markdown.
Do not add explanations outside JSON.

Set is_relevant=true only when this paper provides evidence
that helps answer the research question.

A paper is not relevant merely because it shares biomedical
keywords with the question.

Required schema:

{{
  "is_relevant": true,
  "claim": "Main finding or null",
  "evidence_type": "string",
  "model_system": "string or null",
  "intervention": "string or null",
  "outcome": "string or null",
  "supports_question": true,
  "limitations": ["string", "string"]
}}

Important:
- is_relevant indicates whether the paper helps answer the research question.
- supports_question indicates whether the evidence supports
  or argues against the hypothesis.
- A negative but relevant clinical trial should use:
  "is_relevant": true,
  "supports_question": false.
- If the paper is irrelevant, return:
  "is_relevant": false,
  "claim": null,
  "supports_question": null.
- "limitations" must always be a JSON array. Use [] when no limitation is available; never use null.
- If the research question names a specific target, gene, protein, drug, or intervention, the paper must discuss that
  entity to be relevant.
- A paper about the same disease but a different treatment or target is not relevant.
- For this question, a pulmonary fibrosis paper that does not discuss TNIK must be marked is_relevant=false.
- Do not invent information.
- Use null when the abstract does not provide the information.
- Distinguish animal, in vitro, observational, and clinical evidence.
- Clearly state important limitations.
"""

    raw = llm.generate(prompt)

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM did not return valid JSON.\n"
            f"Raw output:\n{raw}"
        ) from exc
    
    data.setdefault(
        "is_relevant",
        False,
    )

    if not data.get("evidence_type"):
        data["evidence_type"] = "unclear"

    limitations = data.get("limitations")

    if limitations is None:
        data["limitations"] = []
    elif isinstance(limitations, str):
        data["limitations"] = [limitations]
    elif not isinstance(limitations, list):
        data["limitations"] = []

    if not data["is_relevant"]:
        data["claim"] = None
        data["supports_question"] = None

    return ScientificEvidence(
        pmid=pmid,
        **data,
    )