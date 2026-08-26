from agent.agent import BiomedicalAgent


def test_agent_run():

    agent = BiomedicalAgent()

    result = agent.run(
        "Is TNIK a promising therapeutic "
        "target for pulmonary fibrosis?"
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )

    assert result.question != ""

    assert len(
        result.papers
    ) > 0

    assert len(
        result.evidence
    ) > 0

    assert (
        "query_rewrite"
        in result.completed_steps
    )

    assert (
        "retrieval"
        in result.completed_steps
    )

    assert (
        "extraction"
        in result.completed_steps
    )