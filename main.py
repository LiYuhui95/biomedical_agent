import argparse

from agent.agent import BiomedicalAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve, evaluate, and synthesize "
            "biomedical evidence from PubMed."
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help="Biomedical research question",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Display the agent reasoning trace",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = BiomedicalAgent()

    result = agent.run(
        args.question
    )

    print("\n" + "=" * 70)
    print("BIOMEDICAL EVIDENCE AGENT")
    print("=" * 70)

    print("\nQuestion:")
    print(result.question)

    print("\nAgent status:")
    print(f"Iterations: {result.iteration}")
    print(
        f"Final decision: "
        f"{result.next_action.value}"
    )
    print(
        f"Retrieved papers: "
        f"{result.retrieved_count}"
    )
    print(
        f"Usable evidence: "
        f"{result.usable_evidence_count}"
    )
    print(
        f"Relevance rate: "
        f"{result.relevance_rate:.2f}"
    )
    print(
        f"Citation valid: "
        f"{result.citation_valid}"
    )

    if result.invalid_citations:
        print(
            "Invalid citations: "
            + ", ".join(
                result.invalid_citations
            )
        )

    if result.termination_reason:
        print(
            "Termination reason: "
            f"{result.termination_reason}"
        )

    if args.trace:
        print("\nReasoning trace:")

        for step in result.reasoning_trace:
            print(f"- {step}")

    print("\nAnswer:")
    print(
        result.final_answer
        or "No answer was generated."
    )


if __name__ == "__main__":
    main()