from agent.state import  AgentState, WorkflowAction


def create_plan(
    state: AgentState,
) -> list[str]:
    return [
        "retrieve_evidence",
        "evaluate_evidence",
        "synthesize_answer",
    ]


def choose_next_action(
    state: AgentState,
    min_evidence: int = 3,
    min_relevance_rate: float = 0.4,
) -> WorkflowAction:

    if state.iteration >= state.max_iterations:
        if state.usable_evidence_count > 0:
            state.termination_reason = (
                "Maximum iterations reached; "
                "synthesizing available evidence."
            )
            return WorkflowAction.SYNTHESIZE

        state.termination_reason = (
            "Maximum iterations reached without "
            "usable evidence."
        )
        return WorkflowAction.STOP_INSUFFICIENT

    if state.retrieved_count == 0:
        return WorkflowAction.REFINE_QUERY

    if state.relevance_rate < min_relevance_rate:
        return WorkflowAction.REFINE_QUERY

    if state.usable_evidence_count < min_evidence:
        return WorkflowAction.FETCH_MORE

    return WorkflowAction.SYNTHESIZE