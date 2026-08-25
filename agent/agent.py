from agent.state import AgentState
from agent.planner import create_plan
from tools.pubmed import fetch_pubmed_records, parse_pubmed_xml, search_pubmed
from agent.llm import LLMBackend, OllamaBackend
from agent.extractor import extract_evidence
from agent.query import rewrite_query

class BiomedicalAgent:
    def __init__(self, llm: LLMBackend | None = None):
        self.llm = llm or OllamaBackend()

    def run(
        self,
        question: str
    ) -> AgentState:

        state = AgentState(
            question=question
        )

        state.plan = create_plan(state)

        search_query = rewrite_query(self.llm, question)
        print(f"DEBUG search_query: {search_query!r}")
        if not search_query:
            state.papers = []
            state.completed_steps.append("retrieval")
            state.evidence = []
            state.final_answer = self._synthesize(state, 0.0)
            state.completed_steps.append("synthesis")
            return state
        
        pmids = search_pubmed(
            query=search_query,
            retmax=5,
        )

        xml = fetch_pubmed_records(
            pmids
        )

        papers = parse_pubmed_xml(
            xml
        )

        print(f"DEBUG papers found: {len(papers)}")
        for p in papers:
            print(f"DEBUG  pmid={p.pmid} has_abstract={bool(p.abstract)}")
            
        state.papers = papers
        state.completed_steps.append("retrieval")

        evidence = []

        for paper in papers:

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
        state.completed_steps.append("evaluation")

        if state.evidence:
            score = sum(
                1.0 if item.supports_question else 0.0
                for item in state.evidence
            ) / len(state.evidence)
        else:
            score = 0.0

        state.final_answer = self._synthesize(state, score)
        state.completed_steps.append("synthesis")

        return state

    def _synthesize(
        self,
        state: AgentState,
        score: float
    ) -> str:

        if not state.evidence:
            return (
                "Insufficient evidence "
                "was retrieved."
            )

        claims = " ".join(
            item.claim
            for item in state.evidence
        )

        return (
            f"Evidence summary: {claims} "
            f"Mean evidence score: "
            f"{score:.2f}."
        )