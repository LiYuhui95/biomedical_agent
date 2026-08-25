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

Required schema:

{{
  "claim": "string",
  "evidence_type": "string",
  "model_system": "string or null",
  "intervention": "string or null",
  "outcome": "string or null",
  "supports_question": true,
  "limitations": ["string", "string"]
}}

Important:
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
    
    if not data.get("evidence_type"):
        data["evidence_type"] = "unclear"

    return ScientificEvidence(
        pmid=pmid,
        **data,
    )