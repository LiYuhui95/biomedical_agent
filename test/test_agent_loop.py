import json

from agent.agent import BiomedicalAgent
from agent.state import ScientificEvidence


class FakeLLM:
    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> str:
        if json_mode:
            if "Previous PubMed query" in prompt:
                return json.dumps({
                    "query": (
                        'TNIK[Title/Abstract] AND '
                        '"pulmonary fibrosis"'
                        "[Title/Abstract]"
                    )
                })

            return json.dumps({
                "terms": [
                    "TNIK",
                    "pulmonary fibrosis",
                ]
            })

        return (
            "Conclusion:\n"
            "TNIK may be a promising target "
            "[PMID: 1].\n\n"
            "Evidence:\n"
            "Test evidence [PMID: 1].\n\n"
            "Limitations:\n"
            "This is a deterministic test."
        )


def make_evidence(
    count: int,
) -> list[ScientificEvidence]:
    return [
        ScientificEvidence(
            pmid=str(index),
            claim=f"Relevant claim {index}",
            evidence_type="preclinical",
            is_relevant=True,
            supports_question=True,
        )
        for index in range(1, count + 1)
    ]


def test_agent_loop_refines_fetches_and_synthesizes(
    monkeypatch,
):
    agent = BiomedicalAgent(
        llm=FakeLLM(),
        ranker=object(),
    )

    def fake_cycle(
        self,
        state,
        search_query,
        top_k,
    ):
        if state.iteration == 1:
            # No results → REFINE_QUERY
            state.retrieved_count = 0
            state.relevance_rate = 0.0
            state.usable_evidence_count = 0
            state.evidence = []

        elif state.iteration == 2:
            # Relevant, but too few → FETCH_MORE
            state.retrieved_count = 5
            state.relevance_rate = 0.8
            state.usable_evidence_count = 2
            state.evidence = make_evidence(2)

        else:
            # Sufficient evidence → SYNTHESIZE
            state.retrieved_count = 10
            state.relevance_rate = 0.8
            state.usable_evidence_count = 3
            state.evidence = make_evidence(3)

        return True

    monkeypatch.setattr(
        BiomedicalAgent,
        "_run_retrieval_cycle",
        fake_cycle,
    )

    result = agent.run(
        "Is TNIK a promising therapeutic "
        "target for pulmonary fibrosis?"
    )

    assert result.iteration == 3

    assert any(
        "decision=refine_query" in step
        for step in result.reasoning_trace
    )

    assert any(
        "decision=fetch_more" in step
        for step in result.reasoning_trace
    )

    assert any(
        "decision=synthesize" in step
        for step in result.reasoning_trace
    )

    assert result.usable_evidence_count == 3
    assert result.final_answer is not None
    assert "PMID: 1" in result.final_answer