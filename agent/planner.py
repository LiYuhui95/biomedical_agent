from agent.state import  AgentState, WorkflowAction


def create_plan(
    state: AgentState,
) -> list[str]:
    return [
        "retrieve_evidence",
        "evaluate_evidence",
        "synthesize_answer",
    ]


def choose_workflow_action(
    *,
    iteration: int,
    max_iterations: int,
    retrieved_count: int,
    usable_evidence_count: int,
    relevance_rate: float,
    min_evidence: int = 3,
    min_relevance_rate: float = 0.4,
) -> tuple[WorkflowAction, str | None]:
    """
    Pure deterministic planner.

    Returns:
        (next action, termination reason)
    """

    if iteration >= max_iterations:
        if usable_evidence_count > 0:
            return (
                WorkflowAction.SYNTHESIZE,
                "Maximum iterations reached; "
                "synthesizing available evidence.",
            )

        return (
            WorkflowAction.STOP_INSUFFICIENT,
            "Maximum iterations reached without "
            "usable evidence.",
        )

    if retrieved_count == 0:
        return (
            WorkflowAction.REFINE_QUERY,
            None,
        )

    if relevance_rate < min_relevance_rate:
        return (
            WorkflowAction.REFINE_QUERY,
            None,
        )

    if usable_evidence_count < min_evidence:
        return (
            WorkflowAction.FETCH_MORE,
            None,
        )

    return (
        WorkflowAction.SYNTHESIZE,
        "Sufficient evidence found.",
    )

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