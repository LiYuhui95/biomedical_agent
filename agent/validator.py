import re

from agent.state import ScientificEvidence


def extract_cited_pmids(
    answer: str,
) -> set[str]:
    return set(
        re.findall(
            r"PMID:\s*(\d+)",
            answer,
        )
    )


def find_invalid_citations(
    answer: str,
    evidence: list[ScientificEvidence],
) -> set[str]:
    cited_pmids = extract_cited_pmids(
        answer
    )

    allowed_pmids = {
        item.pmid
        for item in evidence
        if item.is_relevant
        and bool(item.claim)
    }

    return cited_pmids - allowed_pmids