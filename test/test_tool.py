from agent.state import AgentAction
from agent.tools import execute_tool


def test_search_pubmed_tool():

    action = AgentAction(
        action="search_pubmed",
        arguments={
            "query": (
                "TNIK AND pulmonary fibrosis"
            ),
            "retmax": 3,
        },
    )

    result = execute_tool(action)

    print(result)

    assert result.tool == "search_pubmed"
    assert result.success is True
    assert isinstance(
        result.data,
        list
    )


def test_fetch_paper_tool():

    action = AgentAction(
        action="fetch_paper",
        arguments={
            "pmids": [
                "41986665"
            ],
        },
    )

    result = execute_tool(action)

    print(result)

    assert result.tool == "fetch_paper"
    assert result.success is True