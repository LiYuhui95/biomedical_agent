from agent.agent import BiomedicalAgent


def main():

    agent = BiomedicalAgent()

    question = (
        "Does compound X improve "
        "beta-cell survival?"
    )

    result = agent.run(
        question
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()