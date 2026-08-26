import pytest

from agent.tools import (
    SearchPubMedArgs,
    FetchPaperArgs,
)


def test_valid_search_query():

    args = SearchPubMedArgs(
        query="metformin pancreatic beta cell"
    )

    assert (
        args.query
        == "metformin pancreatic beta cell"
    )


def test_invalid_short_query():

    with pytest.raises(ValueError):

        SearchPubMedArgs(
            query="ab"
        )


def test_valid_fetch_args():

    args = FetchPaperArgs(
        pmids=["12345678"]
    )

    assert args.pmids == [
        "12345678"
    ]


def test_invalid_pmid():

    with pytest.raises(ValueError):

        FetchPaperArgs(
            pmids=["abc123"]
        )


def test_empty_pmids():

    with pytest.raises(ValueError):

        FetchPaperArgs(
            pmids=[]
        )