from agent.state import AgentState, AgentAction, WorkflowAction
from agent.evaluator import evaluate_evidence_state
from agent.planner import create_plan, choose_next_action
from agent.llm import LLMBackend, OllamaBackend
from agent.extractor import extract_evidence
from agent.query import rewrite_query, refine_query
from agent.tools import execute_tool
from retrieval.ranker import SemanticRanker
from agent.synthesizer import synthesize_answer

class BiomedicalAgent:
    def __init__(
        self,
        llm: LLMBackend | None = None,
        ranker = None,
    ):
        self.llm = llm or OllamaBackend()
        self.ranker = ranker if ranker is not None else SemanticRanker()

    def _run_retrieval_cycle(
        self,
        state: AgentState,
        search_query: str,
        top_k: int,
    ) -> bool:
        """
        Run one search-fetch-rank-extract-evaluate cycle.

        Returns True when the cycle completes successfully.
        Returns False when a tool fails or no papers are found.
        """

        # -----------------------------
        # Search
        # -----------------------------

        search_action = AgentAction(
            action="search_pubmed",
            arguments={
                "query": search_query,
                "retmax": 20,
            },
        )

        state.actions.append(search_action)

        search_result = execute_tool(
            search_action
        )

        state.tool_results.append(
            search_result
        )

        if not search_result.success:
            state.termination_reason = (
                "PubMed search failed."
            )
            state.final_answer = (
                "PubMed search failed: "
                f"{search_result.error}"
            )
            return False

        pmids = search_result.data

        if not pmids:
            state.papers = []
            state.evidence = []

            evaluate_evidence_state(
                state=state,
                retrieved_count=0,
            )

            return True

        # -----------------------------
        # Fetch
        # -----------------------------

        fetch_action = AgentAction(
            action="fetch_paper",
            arguments={
                "pmids": pmids,
            },
        )

        state.actions.append(fetch_action)

        fetch_result = execute_tool(
            fetch_action
        )

        state.tool_results.append(
            fetch_result
        )

        if not fetch_result.success:
            state.termination_reason = (
                "PubMed paper fetching failed."
            )
            state.final_answer = (
                "PubMed paper fetching failed: "
                f"{fetch_result.error}"
            )
            return False

        papers = fetch_result.data
        state.papers = papers

        # -----------------------------
        # Semantic ranking
        # -----------------------------

        ranked_papers = self.ranker.rank(
            question=state.question,
            papers=papers,
            top_k=top_k,
        )

        for paper, score in ranked_papers:
            state.reasoning_trace.append(
                f"Iteration {state.iteration}: "
                f"ranked PMID {paper.pmid} "
                f"with similarity {score:.4f}"
            )

        # -----------------------------
        # Evidence extraction
        # -----------------------------

        evidence = []

        for paper, _ in ranked_papers:
            if not paper.abstract:
                continue

            item = extract_evidence(
                llm=self.llm,
                question=state.question,
                pmid=paper.pmid,
                abstract=paper.abstract,
            )

            evidence.append(item)

        state.evidence = evidence

        # -----------------------------
        # Evidence evaluation
        # -----------------------------

        evaluate_evidence_state(
            state=state,
            retrieved_count=len(
                ranked_papers
            ),
        )

        return True

    def run(
        self,
        question: str,
    ) -> AgentState:
        state = AgentState(
            question=question
        )

        state.plan = create_plan(state)

        search_query = rewrite_query(
            self.llm,
            question,
        )

        if not search_query:
            state.termination_reason = (
                "Query rewrite failed."
            )
            state.final_answer = (
                "No valid biomedical search "
                "query could be generated."
            )
            return state

        state.completed_steps.append(
            "query_rewrite"
        )

        top_k = 5

        while (
            state.iteration
            < state.max_iterations
        ):
            state.iteration += 1

            state.reasoning_trace.append(
                f"Iteration {state.iteration}: "
                f"query={search_query!r}, "
                f"top_k={top_k}"
            )

            cycle_succeeded = (
                self._run_retrieval_cycle(
                    state=state,
                    search_query=search_query,
                    top_k=top_k,
                )
            )

            if not cycle_succeeded:
                return state

            state.next_action = (
                choose_next_action(state)
            )

            state.reasoning_trace.append(
                f"Iteration {state.iteration}: "
                f"decision="
                f"{state.next_action.value}"
            )

            # -------------------------
            # Synthesize
            # -------------------------

            if (
                state.next_action
                == WorkflowAction.SYNTHESIZE
            ):
                state.final_answer = (
                    synthesize_answer(
                        llm=self.llm,
                        question=state.question,
                        evidence=state.evidence,
                    )
                )

                state.completed_steps.append(
                    "synthesis"
                )

                state.termination_reason = (
                    state.termination_reason
                    or "Sufficient evidence found."
                )

                return state

            # -------------------------
            # Stop
            # -------------------------

            if (
                state.next_action
                == WorkflowAction.STOP_INSUFFICIENT
            ):
                state.termination_reason = (
                    state.termination_reason
                    or (
                        "Maximum iterations "
                        "reached without "
                        "sufficient evidence."
                    )
                )

                state.final_answer = (
                    "Insufficient evidence was "
                    "retrieved to answer the "
                    "research question."
                )

                return state

            # -------------------------
            # Fetch more
            # -------------------------

            if (
                state.next_action
                == WorkflowAction.FETCH_MORE
            ):
                if len(state.papers) > top_k:
                    top_k = min(
                        top_k + 5,
                        len(state.papers),
                    )

                    state.reasoning_trace.append(
                        f"Increasing top_k "
                        f"to {top_k}."
                    )

                else:
                    # PubMed did not return more
                    # candidates, so simply raising
                    # top_k would have no effect.
                    search_query = refine_query(
                        llm=self.llm,
                        question=state.question,
                        previous_query=search_query,
                    )

                    state.reasoning_trace.append(
                        "No additional candidates "
                        "available; refined query."
                    )

                continue

            # -------------------------
            # Refine query
            # -------------------------

            if (
                state.next_action
                == WorkflowAction.REFINE_QUERY
            ):
                search_query = refine_query(
                    llm=self.llm,
                    question=state.question,
                    previous_query=search_query,
                )

                state.reasoning_trace.append(
                    "Refined PubMed query."
                )

                continue

        # Defensive fallback: normally the planner
        # terminates during the final iteration.
        state.termination_reason = (
            "Agent loop ended after reaching "
            "the iteration limit."
        )

        state.final_answer = (
            "Insufficient evidence was retrieved "
            "to answer the research question."
        )

        return state