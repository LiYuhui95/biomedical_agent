from agent.state import ScientificEvidence
from agent.validator import (
    extract_cited_pmids,
    find_invalid_citations,
)


def make_evidence(
    pmid: str,
) -> ScientificEvidence:
    return ScientificEvidence(
        pmid=pmid,
        claim="Relevant evidence",
        evidence_type="clinical",
        is_relevant=True,
        supports_question=True,
    )


def test_extract_cited_pmids():
    answer = (
        "Supported by [PMID: 123] "
        "and [PMID: 456]."
    )

    assert extract_cited_pmids(answer) == {
        "123",
        "456",
    }


def test_accepts_known_citations():
    evidence = [
        make_evidence("123"),
        make_evidence("456"),
    ]

    invalid = find_invalid_citations(
        answer=(
            "Supported by [PMID: 123] "
            "and [PMID: 456]."
        ),
        evidence=evidence,
    )

    assert invalid == set()


def test_rejects_unknown_citation():
    evidence = [
        make_evidence("123"),
    ]

    invalid = find_invalid_citations(
        answer=(
            "Supported by [PMID: 123] "
            "and [PMID: 999]."
        ),
        evidence=evidence,
    )

    assert invalid == {"999"}