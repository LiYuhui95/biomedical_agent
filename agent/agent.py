from agent.state import AgentState, AgentAction, ToolResult
from agent.evaluator import evaluate_evidence_state
from agent.planner import create_plan, choose_next_action
from tools.pubmed import fetch_paper, search_pubmed
from agent.llm import LLMBackend, OllamaBackend
from agent.extractor import extract_evidence
from agent.query import rewrite_query
from agent.tools import execute_tool
from retrieval.ranker import SemanticRanker
from agent.synthesizer import synthesize_answer

class BiomedicalAgent:
    def __init__(self, llm: LLMBackend | None = None):
        self.llm = llm or OllamaBackend()
        self.ranker = SemanticRanker()

    def run(
        self,
        question: str
    ) -> AgentState:

        state = AgentState(
            question=question
        )

        # --------------------------------
        # 1. Planning
        # --------------------------------

        state.plan = create_plan(
            state
        )

        # --------------------------------
        # 2. Query rewrite
        # --------------------------------

        search_query = rewrite_query(
            self.llm,
            question
        )

        if not search_query:

            state.completed_steps.append(
                "query_rewrite"
            )

            state.final_answer = (
                "No valid biomedical "
                "search query could be generated."
            )

            return state

        state.completed_steps.append(
            "query_rewrite"
        )

        # --------------------------------
        # 3. Search Tool
        # --------------------------------

        search_action = AgentAction(
            action="search_pubmed",
            arguments={
                "query": search_query,
                "retmax": 20,
            },
        )

        state.actions.append(
            search_action
        )

        search_result = execute_tool(
            search_action
        )

        state.tool_results.append(
            search_result
        )

        if not search_result.success:

            state.final_answer = (
                "PubMed search failed: "
                f"{search_result.error}"
            )

            return state

        pmids = search_result.data

        if not pmids:

            state.final_answer = (
                "No PubMed results found."
            )

            return state

        # --------------------------------
        # 4. Fetch papers
        # --------------------------------

        fetch_action = AgentAction(
            action="fetch_paper",
            arguments={
                "pmids": pmids,
            },
        )

        state.actions.append(
            fetch_action
        )

        fetch_result = execute_tool(
            fetch_action
        )

        state.tool_results.append(
            fetch_result
        )

        if not fetch_result.success:

            state.final_answer = (
                "PubMed paper fetching failed: "
                f"{fetch_result.error}"
            )

            return state

        papers = fetch_result.data

        state.papers = papers

        state.completed_steps.append(
            "retrieval"
        )
        # --------------------------------
        # 4.5. Ranker
        # --------------------------------
        ranked_papers = self.ranker.rank(
            question=question,
            papers=papers,
            top_k=5,
        )
        for paper, score in ranked_papers:
            state.reasoning_trace.append(
                f"Ranked PMID {paper.pmid} "
                f"with similarity {score:.4f}"
            )
        # --------------------------------
        # 5. Evidence extraction
        # --------------------------------

        evidence = []

        for paper, _ in ranked_papers:

            if not paper.abstract:
                continue

            item = extract_evidence(
                llm=self.llm,
                question=question,
                pmid=paper.pmid,
                abstract=paper.abstract,
            )

            evidence.append(item)

        state.evidence = evidence

        state.completed_steps.append(
            "extraction"
        )
        # --------------------------------
        # 6. Evidence evaluation
        # --------------------------------

        evaluate_evidence_state(
            state=state,
            retrieved_count=len(ranked_papers),
        )

        state.next_action = choose_next_action(
            state
        )

        print(
            "Workflow decision:",
            state.next_action,
        )

        print(
            "Retrieved:",
            state.retrieved_count,
        )

        print(
            "Extracted evidence:",
            len(state.evidence),
        )

        print(
            "Usable evidence:",
            state.usable_evidence_count,
        )

        print(
            "Relevance rate:",
            state.relevance_rate,
        )

        # --------------------------------
        # 7. Temporary synthesis
        # --------------------------------

        state.final_answer = synthesize_answer(
            llm=self.llm,
            question=state.question,
            evidence=state.evidence,
        )

        state.completed_steps.append(
            "synthesis"
        )

        return state