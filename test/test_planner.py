from agent.planner import choose_next_action
from agent.state import WorkflowAction, AgentState


def make_state(**updates) -> AgentState:
    state = AgentState(
        question="Is TNIK a therapeutic target for IPF?"
    )

    for name, value in updates.items():
        setattr(state, name, value)

    return state


def test_refines_query_when_nothing_retrieved():
    state = make_state(
        retrieved_count=0,
        iteration=1,
    )

    assert (
        choose_next_action(state)
        == WorkflowAction.REFINE_QUERY
    )


def test_refines_query_when_relevance_is_low():
    state = make_state(
        retrieved_count=10,
        relevance_rate=0.2,
        usable_evidence_count=2,
        iteration=1,
    )

    assert (
        choose_next_action(state)
        == WorkflowAction.REFINE_QUERY
    )


def test_fetches_more_when_evidence_is_insufficient():
    state = make_state(
        retrieved_count=10,
        relevance_rate=0.8,
        usable_evidence_count=2,
        iteration=1,
    )

    assert (
        choose_next_action(state)
        == WorkflowAction.FETCH_MORE
    )


def test_synthesizes_when_evidence_is_sufficient():
    state = make_state(
        retrieved_count=10,
        relevance_rate=0.8,
        usable_evidence_count=4,
        iteration=1,
    )

    assert (
        choose_next_action(state)
        == WorkflowAction.SYNTHESIZE
    )


def test_stops_after_budget_exhaustion():
    state = make_state(
        retrieved_count=0,
        usable_evidence_count=0,
        iteration=3,
        max_iterations=3,
    )

    assert (
        choose_next_action(state)
        == WorkflowAction.STOP_INSUFFICIENT
    )