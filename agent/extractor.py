import json

from agent.llm import LLMBackend
from agent.state import ScientificEvidence

def normalize_evidence_type(
    title: str,
    abstract: str,
    proposed_type: str,
) -> str:
    title_lower = title.lower()
    text_lower = (
        f"{title} {abstract}"
    ).lower()

    if "patent review" in title_lower:
        return "review"

    if "review" in title_lower:
        return "review"

    if "perspective" in title_lower:
        return "commentary"

    if "letter to editor" in title_lower:
        return "commentary"

    if (
        "randomized" in text_lower
        or "phase 2" in title_lower
        or "phase ii" in title_lower
    ):
        return "clinical_trial"

    if (
        "virtual screening" in title_lower
        or "molecular docking" in title_lower
    ):
        return "computational"

    valid_types = {
        "clinical_trial",
        "observational",
        "animal",
        "in_vitro",
        "computational",
        "review",
        "commentary",
        "unclear",
    }

    if proposed_type in valid_types:
        return proposed_type

    return "unclear"

def extract_evidence(
    llm: LLMBackend,
    question: str,
    pmid: str,
    title: str,
    abstract: str,
) -> ScientificEvidence:

    prompt = f"""
You are a biomedical literature analyst.

Research question:
{question}

Paper PMID:
{pmid}

Paper title:
{title}

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

"evidence_type" must be exactly one of:

- "clinical_trial": the current paper reports original
  interventional human trial results.
- "observational": the current paper reports original
  observational human data.
- "animal": the current paper reports in vivo animal evidence.
- "in_vitro": the current paper reports cellular or
  biochemical experimental evidence.
- "computational": the current paper primarily reports
  modeling, docking, or virtual screening.
- "review": the current paper summarizes other studies.
- "commentary": letter, editorial, or perspective.
- "unclear": insufficient information.

Do not label a review, perspective, patent review, or paper
that merely mentions a clinical trial as "clinical_trial".

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

    data["evidence_type"] = (
        normalize_evidence_type(
            title=title,
            abstract=abstract,
            proposed_type=data.get(
                "evidence_type",
                "unclear",
            ),
        )
    )

    return ScientificEvidence(
        pmid=pmid,
        **data,
    )