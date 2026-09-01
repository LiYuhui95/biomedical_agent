import operator

from typing import Annotated, TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agent.llm import OllamaBackend
from agent.query import rewrite_query
from agent.state import (
    AgentAction,
    PaperRecord,
    ScientificEvidence,
    WorkflowAction,
)
from agent.extractor import extract_evidence
from agent.evaluator import evaluate_evidence
from agent.planner import choose_workflow_action
from agent.tools import execute_tool
from agent.synthesizer import synthesize_answer

from retrieval.ranker import SemanticRanker

import re

llm = OllamaBackend()
ranker = SemanticRanker()
_CITATION_PATTERN = re.compile(
    r"\[PMID:\s*(\d+)\]"
)

class BiomedicalGraphState(TypedDict):
    question: str
    search_query: str

    iteration: int
    max_iterations: int
    search_limit: int
    top_k: int

    pmids: list[str]
    papers: list[PaperRecord]

    ranked_papers: list[
        tuple[PaperRecord, float]
    ]

    evidence: list[ScientificEvidence]

    retrieved_count: int
    usable_evidence_count: int
    relevance_rate: float

    decision: str
    termination_reason: str | None

    final_answer: str

    citation_valid: bool
    invalid_citations: list[str]

    error: str | None

    trace: Annotated[
        list[str],
        operator.add,
    ]


def rewrite_query_node(
    state: BiomedicalGraphState,
) -> dict:
    query = rewrite_query(
        llm=llm,
        question=state["question"],
    )

    if not query:
        return {
            "search_query": "",
            "error": (
                "No valid biomedical query "
                "could be generated."
            ),
            "trace": [
                "Query rewrite failed."
            ],
        }

    return {
        "search_query": query,
        "trace": [
            f"Generated query: {query}"
        ],
    }

def route_after_rewrite(
    state: BiomedicalGraphState,
) -> str:
    if state["error"]:
        return "stop"

    if not state["search_query"]:
        return "stop"

    return "search"

def search_pubmed_node(
    state: BiomedicalGraphState,
) -> dict:
    action = AgentAction(
        action="search_pubmed",
        arguments={
            "query": state["search_query"],
            "retmax": 20,
        },
    )

    result = execute_tool(
        action
    )

    if not result.success:
        return {
            "pmids": [],
            "error": (
                result.error
                or "PubMed search failed."
            ),
            "trace": [
                "PubMed search failed."
            ],
        }

    pmids = result.data or []

    if not pmids:
        return {
            "pmids": [],
            "error": (
                "No PubMed results found."
            ),
            "trace": [
                "PubMed returned no results."
            ],
        }

    return {
        "pmids": pmids,
        "trace": [
            (
                "PubMed returned "
                f"{len(pmids)} PMIDs."
            )
        ],
    }

def finish_search_node(
    state: BiomedicalGraphState,
) -> dict:
    return {
        "final_answer": (
            f"Retrieved {len(state['pmids'])} "
            "PubMed IDs."
        ),
        "trace": [
            "Search demo completed."
        ],
    }


def stop_node(
    state: BiomedicalGraphState,
) -> dict:

    reason = (
        state["termination_reason"]
        or state["error"]
        or "Insufficient evidence."
    )

    return {
        "final_answer": (
            "Insufficient evidence was retrieved "
            "to answer the research question."
        ),
        "termination_reason": reason,
        "trace": [
            f"Graph stopped: {reason}"
        ],
    }

def route_after_search(
    state: BiomedicalGraphState,
) -> str:
    if state["error"]:
        return "stop"

    if not state["pmids"]:
        return "stop"

    return "fetch"

def fetch_papers_node(
    state: BiomedicalGraphState,
) -> dict:
    action = AgentAction(
        action="fetch_paper",
        arguments={
            "pmids": state["pmids"],
        },
    )

    result = execute_tool(action)

    if not result.success:
        return {
            "papers": [],
            "error": (
                result.error
                or "Paper fetching failed."
            ),
            "trace": [
                "Paper fetching failed."
            ],
        }

    papers = result.data or []

    if not papers:
        return {
            "papers": [],
            "error": (
                "No PaperRecord objects "
                "were produced."
            ),
            "trace": [
                "Fetch returned no papers."
            ],
        }

    return {
        "papers": papers,
        "error": None,
        "trace": [
            (
                f"Fetched {len(papers)} "
                "PaperRecord objects."
            )
        ],
    }

def route_after_fetch(
    state: BiomedicalGraphState,
) -> str:
    if state["error"]:
        return "stop"

    if not state["papers"]:
        return "stop"

    return "rank"

def rank_papers_node(
    state: BiomedicalGraphState,
) -> dict:
    ranked_papers = ranker.rank(
        question=state["question"],
        papers=state["papers"],
        top_k=state["top_k"],
    )

    if not ranked_papers:
        return {
            "ranked_papers": [],
            "error": (
                "Semantic ranking produced "
                "no results."
            ),
            "trace": [
                "Semantic ranking failed."
            ],
        }

    ranking_trace = [
        (
            f"Ranked PMID {paper.pmid} "
            f"with score {score:.4f}."
        )
        for paper, score in ranked_papers
    ]

    return {
        "ranked_papers": ranked_papers,
        "error": None,
        "trace": ranking_trace,
    }

def finish_retrieval_node(
    state: BiomedicalGraphState,
) -> dict:
    return {
        "final_answer": (
            f"Retrieved {len(state['papers'])} "
            "papers and ranked "
            f"{len(state['ranked_papers'])}."
        ),
        "trace": [
            "Retrieval demo completed."
        ],
    }

def extract_evidence_node(
    state: BiomedicalGraphState,
) -> dict:

    evidence = []

    for paper, _score in state["ranked_papers"]:

        if not paper.abstract:
            continue

        item = extract_evidence(
            llm=llm,
            question=state["question"],
            pmid=paper.pmid,
            title=paper.title,
            abstract=paper.abstract,
        )

        evidence.append(item)

    return {
        "evidence": evidence,
        "trace": [
            f"Extracted evidence from "
            f"{len(evidence)} papers"
        ],
    }

def evaluate_evidence_node(
    state: BiomedicalGraphState,
) -> dict:

    metrics = evaluate_evidence(
        evidence=state["evidence"],
        retrieved_count=len(state["papers"]),
    )

    return {
        **metrics,
        "trace": [
            "Evaluated evidence: "
            f"retrieved={metrics['retrieved_count']}, "
            f"usable={metrics['usable_evidence_count']}, "
            f"relevance_rate="
            f"{metrics['relevance_rate']:.2f}"
        ],
    }

def route_after_evaluation(
    state: BiomedicalGraphState,
) -> str:

    if state["iteration"] >= state["max_iterations"]:
        if state["usable_evidence_count"] > 0:
            return "synthesize"

        return "stop"

    if state["retrieved_count"] == 0:
        return "refine"

    if state["relevance_rate"] < 0.5:
        return "refine"

    if state["usable_evidence_count"] < 3:
        return "fetch_more"

    return "synthesize"

def planner_node(
    state: BiomedicalGraphState,
) -> dict:

    action, termination_reason = (
        choose_workflow_action(
            iteration=state["iteration"],
            max_iterations=state["max_iterations"],
            retrieved_count=state[
                "retrieved_count"
            ],
            usable_evidence_count=state[
                "usable_evidence_count"
            ],
            relevance_rate=state[
                "relevance_rate"
            ],
        )
    )

    return {
        "decision": action.value,
        "termination_reason": termination_reason,
        "trace": [
            f"Planner selected: {action.value}"
        ],
    }

def route_after_planner(
    state: BiomedicalGraphState,
) -> str:
    return state["decision"]

def refine_query_node(
    state: BiomedicalGraphState,
) -> dict:
    """
    Make the query more treatment-focused, then
    return to PubMed search.
    """

    current_query = state["search_query"]

    refinement = (
        "(inhibitor OR inhibition OR "
        "therapeutic OR treatment)"
    )

    if refinement in current_query:
        refined_query = current_query
    else:
        refined_query = (
            f"({current_query}) AND "
            f"{refinement}"
        )

    return {
        "search_query": refined_query,
        "iteration": state["iteration"] + 1,
        "error": None,
        "trace": [
            "Refined query: "
            f"{refined_query}"
        ],
    }

def fetch_more_node(
    state: BiomedicalGraphState,
) -> dict:
    """
    Increase retrieval and ranking budgets,
    then repeat search.
    """

    new_search_limit = min(
        state["search_limit"] + 10,
        50,
    )

    new_top_k = min(
        state["top_k"] + 5,
        new_search_limit,
    )

    return {
        "search_limit": new_search_limit,
        "top_k": new_top_k,
        "iteration": state["iteration"] + 1,
        "error": None,
        "trace": [
            "Expanded retrieval budget: "
            f"search_limit={new_search_limit}, "
            f"top_k={new_top_k}"
        ],
    }

def synthesize_node(
    state: BiomedicalGraphState,
) -> dict:

    usable_evidence = [
        item
        for item in state["evidence"]
        if item.is_relevant
        and bool(item.claim)
    ]

    if not usable_evidence:
        return {
            "final_answer": (
                "Insufficient usable evidence was "
                "retrieved to answer the question."
            ),
            "termination_reason": (
                "No usable evidence available "
                "for synthesis."
            ),
            "trace": [
                "Synthesis skipped: no usable evidence"
            ],
        }

    answer = synthesize_answer(
        llm=llm,
        question=state["question"],
        evidence=usable_evidence,
    )

    termination_reason = (
        state["termination_reason"]
        or "Sufficient evidence found."
    )

    return {
        "final_answer": answer,
        "termination_reason": termination_reason,
        "trace": [
            "Synthesized answer from "
            f"{len(usable_evidence)} evidence items"
        ],
    }

def validate_citations_node(
    state: BiomedicalGraphState,
) -> dict:

    cited_pmids = set(
        _CITATION_PATTERN.findall(
            state["final_answer"]
        )
    )

    valid_pmids = {
        item.pmid
        for item in state["evidence"]
        if item.is_relevant
        and bool(item.claim)
    }

    invalid_citations = sorted(
        cited_pmids - valid_pmids
    )

    citation_valid = (
        bool(cited_pmids)
        and not invalid_citations
    )

    if citation_valid:
        return {
            "citation_valid": True,
            "invalid_citations": [],
            "trace": [
                "Citation validation passed"
            ],
        }

    if invalid_citations:
        reason = (
            "The synthesized answer cited PMIDs "
            "not present in validated evidence: "
            + ", ".join(invalid_citations)
        )
    else:
        reason = (
            "The synthesized answer did not "
            "contain PMID citations."
        )

    return {
        "citation_valid": False,
        "invalid_citations": invalid_citations,
        "termination_reason": reason,
        "final_answer": (
            "The generated answer failed "
            "citation validation."
        ),
        "trace": [
            f"Citation validation failed: {reason}"
        ],
    }

builder = StateGraph(
    BiomedicalGraphState
)

# -------------------------------
# Register nodes
# -------------------------------

builder.add_node(
    "rewrite",
    rewrite_query_node,
)

builder.add_node(
    "search",
    search_pubmed_node,
)

builder.add_node(
    "fetch",
    fetch_papers_node,
)

builder.add_node(
    "rank",
    rank_papers_node,
)

builder.add_node(
    "extract",
    extract_evidence_node,
)

builder.add_node(
    "evaluate",
    evaluate_evidence_node,
)

builder.add_node(
    "planner",
    planner_node,
)

builder.add_node(
    "refine",
    refine_query_node,
)

builder.add_node(
    "fetch_more",
    fetch_more_node,
)

builder.add_node(
    "synthesize",
    synthesize_node,
)

builder.add_node(
    "validate_citations",
    validate_citations_node,
)

builder.add_node(
    "stop",
    stop_node,
)

# -------------------------------
# Main retrieval path
# -------------------------------

builder.add_edge(
    START,
    "rewrite",
)

builder.add_conditional_edges(
    "rewrite",
    route_after_rewrite,
    {
        "search": "search",
        "stop": "stop",
    },
)

builder.add_conditional_edges(
    "search",
    route_after_search,
    {
        "fetch": "fetch",
        "stop": "stop",
    },
)

builder.add_conditional_edges(
    "fetch",
    route_after_fetch,
    {
        "rank": "rank",
        "stop": "stop",
    },
)

builder.add_edge(
    "rank",
    "extract",
)

builder.add_edge(
    "extract",
    "evaluate",
)

builder.add_edge(
    "evaluate",
    "planner",
)

# -------------------------------
# Planner routing
# -------------------------------

builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        WorkflowAction.REFINE_QUERY.value: (
            "refine"
        ),
        WorkflowAction.FETCH_MORE.value: (
            "fetch_more"
        ),
        WorkflowAction.SYNTHESIZE.value: (
            "synthesize"
        ),
        WorkflowAction.STOP_INSUFFICIENT.value: (
            "stop"
        ),
    },
)

# -------------------------------
# Agent loops
# -------------------------------

builder.add_edge(
    "refine",
    "search",
)

builder.add_edge(
    "fetch_more",
    "search",
)

# -------------------------------
# Final answer path
# -------------------------------

builder.add_edge(
    "synthesize",
    "validate_citations",
)

builder.add_edge(
    "validate_citations",
    END,
)

builder.add_edge(
    "stop",
    END,
)

graph = builder.compile()


initial_state: BiomedicalGraphState = {
    "question": (
        "Is TNIK a promising therapeutic "
        "target for pulmonary fibrosis?"
    ),
    "search_query": "",

    "iteration": 0,
    "max_iterations": 3,
    "search_limit": 20,
    "top_k": 5,

    "pmids": [],
    "papers": [],
    "ranked_papers": [],
    "evidence": [],

    "retrieved_count": 0,
    "usable_evidence_count": 0,
    "relevance_rate": 0.0,

    "decision": "",
    "termination_reason": None,

    "final_answer": "",

    "citation_valid": False,
    "invalid_citations": [],

    "error": None,
    "trace": [],
}

result = graph.invoke(
    initial_state,
    config={
        "recursion_limit": 50,
    },
)

print("\nFinal decision:")
print(result["decision"])

print("\nIterations:")
print(result["iteration"])

print("\nRetrieved:")
print(result["retrieved_count"])

print("\nUsable evidence:")
print(result["usable_evidence_count"])

print("\nRelevance rate:")
print(
    f"{result['relevance_rate']:.2f}"
)

print("\nCitation valid:")
print(result["citation_valid"])

print("\nTermination reason:")
print(result["termination_reason"])

print("\nAnswer:")
print(result["final_answer"])

print("\nTrace:")
for step in result["trace"]:
    print(f"- {step}")