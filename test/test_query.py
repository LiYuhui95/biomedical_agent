import json

from agent.query import refine_query


class FakeLLM:
    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps({
            "query": (
                '"pulmonary fibrosis"'
                "[Title/Abstract]"
            )
        })


def test_refine_query_preserves_required_entity():
    query = refine_query(
        llm=FakeLLM(),
        question=(
            "Is TNIK a promising therapeutic "
            "target for pulmonary fibrosis?"
        ),
        previous_query="pulmonary fibrosis",
    )

    assert "TNIK" in query
    assert "pulmonary fibrosis" in query