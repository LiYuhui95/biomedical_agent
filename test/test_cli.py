from main import build_parser


def test_cli_parses_question():
    parser = build_parser()

    args = parser.parse_args([
        "Is TNIK a therapeutic target?"
    ])

    assert (
        args.question
        == "Is TNIK a therapeutic target?"
    )

    assert args.trace is False


def test_cli_enables_trace():
    parser = build_parser()

    args = parser.parse_args([
        "Is TNIK a therapeutic target?",
        "--trace",
    ])

    assert args.trace is True