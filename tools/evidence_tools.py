from agent.state import Evidence


MOCK_DATABASE = {
    "beta-cell": [
        Evidence(
            source_id="PMID001",
            claim=(
                "Compound X increased "
                "beta-cell survival under "
                "glucotoxic stress."
            ),
            evidence_type="in_vitro",
            strength=0.65
        ),
        Evidence(
            source_id="PMID002",
            claim=(
                "Compound X improved "
                "glucose tolerance in mice."
            ),
            evidence_type="animal",
            strength=0.75
        ),
    ]
}


def retrieve_evidence(
    question: str
) -> list[Evidence]:

    question_lower = question.lower()

    if "beta-cell" in question_lower:
        return MOCK_DATABASE["beta-cell"]

    return []

def evaluate_evidence(
    evidence: list[Evidence]
) -> float:

    if not evidence:
        return 0.0

    total = sum(
        item.strength
        for item in evidence
    )

    return total / len(evidence)