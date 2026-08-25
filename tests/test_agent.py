from agent.agent import BiomedicalAgent


def test_beta_cell_question():

    agent = BiomedicalAgent()

    result = agent.run(
        "Does compound X improve "
        "beta-cell survival?"
    )

    assert len(
        result.evidence
    ) > 0

    assert (
        result.final_answer
        is not None
    )

    assert (
        "retrieval"
        in result.completed_steps
    )


def test_unknown_question():

    agent = BiomedicalAgent()

    result = agent.run(
        "Does unknown target Z "
        "affect disease Y?"
    )

    assert len(
        result.evidence
    ) == 0

    assert (
        "Insufficient evidence"
        in result.final_answer
    )