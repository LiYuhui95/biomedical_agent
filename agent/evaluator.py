from agent.state import AgentState


def evaluate_evidence_state(
    state: AgentState,
    retrieved_count: int,
) -> None:
    state.retrieved_count = retrieved_count

    if not state.evidence:
        state.usable_evidence_count = 0
        state.relevance_rate = 0.0
        return

    relevant_count = sum(
        item.is_relevant
        and bool(item.claim)
        for item in state.evidence
    )

    state.relevance_rate = (
        relevant_count / len(state.evidence)
    )

    state.usable_evidence_count = sum(
        item.is_relevant
        and bool(item.claim)
        for item in state.evidence
    )