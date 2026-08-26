from tools.pubmed import (
    search_pubmed,
    fetch_pubmed_records,
    parse_pubmed_xml,
)


pmids = search_pubmed(
    "TNIK fibrosis",
    retmax=5
)

print(
    "PMIDs:",
    pmids
)

xml = fetch_pubmed_records(
    pmids
)

papers = parse_pubmed_xml(
    xml
)

for paper in papers:

    print("=" * 60)

    print(
        "PMID:",
        paper.pmid
    )

    print(
        "Title:",
        paper.title
    )

    print(
        "Year:",
        paper.year
    )

    print(
        "Journal:",
        paper.journal
    )

    print(
        "Abstract:",
        (paper.abstract or "")[:500]
    )