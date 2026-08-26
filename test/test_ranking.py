from tools.pubmed import (
    search_pubmed,
    fetch_pubmed_records,
    parse_pubmed_xml,
)

from retrieval.ranker import SemanticRanker


question = (
    "Is TNIK a promising therapeutic "
    "target for pulmonary fibrosis?"
)

search_query = (
    "TNIK AND pulmonary fibrosis"
)

print("Searching PubMed...")

pmids = search_pubmed(
    search_query,
    retmax=20,
)

print(
    "PMIDs:",
    pmids
)

print("Fetching papers...")

xml = fetch_pubmed_records(
    pmids
)

papers = parse_pubmed_xml(
    xml
)

print(
    "Retrieved papers:",
    len(papers)
)

print("Loading embedding model...")

ranker = SemanticRanker()

print("Ranking papers...")

ranked = ranker.rank(
    question=question,
    papers=papers,
    top_k=5,
)

for i, (paper, score) in enumerate(
    ranked,
    start=1,
):

    print("=" * 70)

    print(
        f"Rank {i}"
    )

    print(
        f"Similarity: {score:.4f}"
    )

    print(
        f"PMID: {paper.pmid}"
    )

    print(
        f"Title: {paper.title}"
    )