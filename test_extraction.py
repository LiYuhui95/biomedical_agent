from agent.llm import OllamaBackend
from agent.extractor import extract_evidence

from tools.pubmed import (
    search_pubmed,
    fetch_pubmed_records,
    parse_pubmed_xml,
)


question = (
    "Is TNIK a promising therapeutic "
    "target for pulmonary fibrosis?"
)

print("Step 1: Searching PubMed...")

pmids = search_pubmed(
    "TNIK AND pulmonary fibrosis",
    retmax=3,
)

print("PMIDs:", pmids)

print("Step 2: Fetching papers...")

xml = fetch_pubmed_records(pmids)

print(
    "XML length:",
    len(xml)
)

print("Step 3: Parsing papers...")

papers = parse_pubmed_xml(xml)

print(
    "Number of papers:",
    len(papers)
)

for i, paper in enumerate(papers):

    print("=" * 70)
    print(
        f"Paper {i + 1}:",
        paper.pmid
    )
    print(
        "Title:",
        paper.title
    )
    print(
        "Abstract exists:",
        bool(paper.abstract)
    )

    if paper.abstract:
        print(
            "Abstract length:",
            len(paper.abstract)
        )
    else:
        print(
            "No abstract, skipping LLM."
        )
        continue

    print("Step 4: Calling local Qwen...")

    llm = OllamaBackend(
        model="qwen2.5:3b"
    )

    evidence = extract_evidence(
        llm=llm,
        question=question,
        pmid=paper.pmid,
        abstract=paper.abstract,
    )

    print("Scientific evidence:")
    print(
        evidence.model_dump_json(
            indent=2
        )
    )