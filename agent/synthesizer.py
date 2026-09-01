import json

from agent.llm import LLMBackend
from agent.state import ScientificEvidence


def synthesize_answer(
    llm: LLMBackend,
    question: str,
    evidence: list[ScientificEvidence],
) -> str:

    if not evidence:
        return (
            "Insufficient evidence was retrieved "
            "to answer the question."
        )

    evidence_json = json.dumps(
        [
            item.model_dump()
            for item in evidence
        ],
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are a biomedical evidence synthesis assistant.

Answer the research question using ONLY the supplied
structured evidence.

Research question:
{question}

Evidence:
{evidence_json}

Instructions:
1. Give a direct answer to the research question.
2. Cite supporting statements using [PMID: number].
3. Compare evidence from different study types.
4. Distinguish preclinical evidence from clinical evidence.
5. Describe conflicting or negative evidence if present.
6. State important limitations and uncertainty.
7. Do not invent facts not contained in the evidence.
8. If the evidence is insufficient, explicitly say so.

- Distinguish evidence reported directly by the current paper
  from evidence merely discussed or cited by a review.
- Do not describe a review, perspective, or commentary as a
  clinical trial.
- Clinical evidence is generally more direct than preclinical
  evidence for evaluating therapeutic efficacy.
- Do not repeat the limitations section.
- Calibrate the conclusion to evidence strength. Use language
  such as "preliminary", "promising but unconfirmed", or
  "insufficient" when direct clinical evidence is limited.
  
Use this structure:

Conclusion:
<direct conclusion>

Evidence:
<synthesis with PMID citations>

Limitations:
<important limitations>

Return plain text only.
"""

    return llm.generate(prompt).strip()