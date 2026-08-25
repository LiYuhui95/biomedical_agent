from agent.state import AgentState


def create_plan(
    state: AgentState
) -> list[str]:

    return [
        "retrieve_evidence",
        "evaluate_evidence",
        "synthesize_answer"
    ]