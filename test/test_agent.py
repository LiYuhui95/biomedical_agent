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

    print(result)

    for item in result.reasoning_trace:
        print(item)

    assert result.question != ""

    assert len(result.papers) > 0

    assert len(result.evidence) > 0

    assert len(result.evidence) <= 5

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

    assert result.final_answer is not None
    assert result.final_answer.strip()
    assert result.next_action is not None