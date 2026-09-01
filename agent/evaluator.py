from agent.state import (
    AgentState,
    ScientificEvidence,
)


def is_usable_evidence(
    item: ScientificEvidence,
) -> bool:
    return (
        item.is_relevant
        and bool(item.claim)
    )


def get_usable_evidence(
    evidence: list[ScientificEvidence],
) -> list[ScientificEvidence]:
    return [
        item
        for item in evidence
        if is_usable_evidence(item)
    ]

from agent.state import (
    AgentState,
    ScientificEvidence,
)


def evaluate_evidence(
    evidence: list[ScientificEvidence],
    retrieved_count: int,
) -> dict[str, int | float]:
    """
    Deterministically evaluate extracted evidence.

    This function does not mutate any state.
    It can be reused by both the original agent
    and the LangGraph implementation.
    """

    usable_evidence_count = sum(
        item.is_relevant
        and bool(item.claim)
        for item in evidence
    )

    relevance_rate = (
        usable_evidence_count / len(evidence)
        if evidence
        else 0.0
    )

    return {
        "retrieved_count": retrieved_count,
        "usable_evidence_count": (
            usable_evidence_count
        ),
        "relevance_rate": relevance_rate,
    }

def evaluate_evidence_state(
    state: AgentState,
    retrieved_count: int,
) -> None:
    state.retrieved_count = retrieved_count

    if not state.evidence:
        state.usable_evidence_count = 0
        state.relevance_rate = 0.0
        return

    usable_evidence = get_usable_evidence(
        state.evidence
    )

    state.usable_evidence_count = len(
        usable_evidence
    )

    state.relevance_rate = (
        len(usable_evidence)
        / len(state.evidence)
    )