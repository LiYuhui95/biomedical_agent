from tools.pubmed import (
    search_pubmed,
    fetch_paper,
)

from retrieval.ranker import SemanticRanker


def evaluate_at_k(
    ranked_pmids: list[str],
    relevant_pmids: set[str],
    k: int,
) -> dict[str, float]:
    
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if len(ranked_pmids) < k:
        raise ValueError(
            f"Cannot evaluate at k={k}: "
            f"only {len(ranked_pmids)} results"
    )
    
    top_k = ranked_pmids[:k]

    hits = sum(
        pmid in relevant_pmids
        for pmid in top_k
    )

    precision = hits / k

    recall = (
        hits / len(relevant_pmids)
        if relevant_pmids
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "hits": hits,
    }

def test_semantic_ranking():

    question = (
        "Is TNIK a promising therapeutic "
        "target for pulmonary fibrosis?"
    )

    search_query = (
        "TNIK AND pulmonary fibrosis"
    )

    # -----------------------------
    # 1. PubMed retrieval
    # -----------------------------

    pmids = search_pubmed(
        search_query,
        retmax=20,
    )

    papers = fetch_paper(
        pmids
    )

    print(
        f"\nRetrieved {len(papers)} papers"
    )

    # -----------------------------
    # 2. PubMed baseline
    # -----------------------------

    print("\n")
    print("=" * 80)
    print("PUBMED BASELINE")
    print("=" * 80)

    for i, paper in enumerate(
        papers,
        start=1,
    ):

        print(
            f"\nRank {i}"
        )

        print(
            f"PMID: {paper.pmid}"
        )

        print(
            f"Title: {paper.title}"
        )

    relevant_pmids = {
        "41475169",
        "40820280",
        "40461817",
        "38459338",
        "39422731",
        "40999821",
    }
    # -----------------------------
    # 3. Semantic ranking
    # -----------------------------

    ranker = SemanticRanker()

    ranked = ranker.rank(
        question=question,
        papers=papers,
        top_k=len(papers),
    )

    print("\n")
    print("=" * 80)
    print("SEMANTIC RANKING")
    print("=" * 80)
    # -----------------------------
    # 3.5  Retrieval evaluation
    # -----------------------------
    pubmed_pmids = [
        paper.pmid
        for paper in papers
    ]

    semantic_pmids = [
        paper.pmid
        for paper, score in ranked
    ]

    k = 5

    pubmed_metrics = evaluate_at_k(
        ranked_pmids=pubmed_pmids,
        relevant_pmids=relevant_pmids,
        k=k,
    )

    semantic_metrics = evaluate_at_k(
        ranked_pmids=semantic_pmids,
        relevant_pmids=relevant_pmids,
        k=k,
    )

    print("\n")
    print("=" * 80)
    print(f"RETRIEVAL EVALUATION @ {k}")
    print("=" * 80)

    print(
        "\nPubMed baseline:"
        f"\nHits: {pubmed_metrics['hits']}"
        f"\nPrecision@{k}: "
        f"{pubmed_metrics['precision']:.3f}"
        f"\nRecall@{k}: "
        f"{pubmed_metrics['recall']:.3f}"
    )

    print(
        "\nSemantic ranking:"
        f"\nHits: {semantic_metrics['hits']}"
        f"\nPrecision@{k}: "
        f"{semantic_metrics['precision']:.3f}"
        f"\nRecall@{k}: "
        f"{semantic_metrics['recall']:.3f}"
    )

    assert 0.0 <= pubmed_metrics["precision"] <= 1.0
    assert 0.0 <= pubmed_metrics["recall"] <= 1.0

    assert 0.0 <= semantic_metrics["precision"] <= 1.0
    assert 0.0 <= semantic_metrics["recall"] <= 1.0

    for i, (paper, score) in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"\nRank {i}"
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

    # -----------------------------
    # 4. Basic tests
    # -----------------------------

    assert len(papers) > 0

    assert len(ranked) == len(papers)
