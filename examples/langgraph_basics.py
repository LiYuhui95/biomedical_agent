from typing import TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)


class GraphState(TypedDict):
    question: str
    iteration: int
    usable_evidence_count: int
    decision: str
    final_answer: str


def retrieve_node(
    state: GraphState,
) -> dict:
    iteration = state["iteration"] + 1

    # Simulated retrieval:
    # first round finds 1 usable paper;
    # second round finds 3.
    usable_count = (
        1
        if iteration == 1
        else 3
    )

    print(
        f"Retrieve iteration {iteration}: "
        f"usable evidence={usable_count}"
    )

    return {
        "iteration": iteration,
        "usable_evidence_count": (
            usable_count
        ),
    }


def evaluate_node(
    state: GraphState,
) -> dict:
    if state["usable_evidence_count"] >= 3:
        decision = "synthesize"
    elif state["iteration"] >= 3:
        decision = "stop"
    else:
        decision = "fetch_more"

    print(
        f"Evaluation decision: {decision}"
    )

    return {
        "decision": decision
    }


def refine_node(
    state: GraphState,
) -> dict:
    print(
        "Preparing another retrieval round."
    )

    return {}


def synthesize_node(
    state: GraphState,
) -> dict:
    return {
        "final_answer": (
            "Sufficient evidence was found."
        )
    }


def stop_node(
    state: GraphState,
) -> dict:
    return {
        "final_answer": (
            "Insufficient evidence was found."
        )
    }


def route_after_evaluation(
    state: GraphState,
) -> str:
    return state["decision"]


builder = StateGraph(
    GraphState
)

builder.add_node(
    "retrieve",
    retrieve_node,
)

builder.add_node(
    "evaluate",
    evaluate_node,
)

builder.add_node(
    "refine",
    refine_node,
)

builder.add_node(
    "synthesize",
    synthesize_node,
)

builder.add_node(
    "stop",
    stop_node,
)

builder.add_edge(
    START,
    "retrieve",
)

builder.add_edge(
    "retrieve",
    "evaluate",
)

builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "fetch_more": "refine",
        "synthesize": "synthesize",
        "stop": "stop",
    },
)

builder.add_edge(
    "refine",
    "retrieve",
)

builder.add_edge(
    "synthesize",
    END,
)

builder.add_edge(
    "stop",
    END,
)

graph = builder.compile()

result = graph.invoke(
    {
        "question": (
            "Is TNIK a therapeutic target?"
        ),
        "iteration": 0,
        "usable_evidence_count": 0,
        "decision": "",
        "final_answer": "",
    },
    config={
        "recursion_limit": 10,
    },
)

print("\nFinal state:")
print(result)