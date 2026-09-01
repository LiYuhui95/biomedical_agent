from langchain.tools import tool
from langchain_ollama import ChatOllama

from tools.pubmed import search_pubmed


@tool
def search_pubmed_tool(
    query: str,
    retmax: int = 5,
) -> list[str]:
    """
    Search PubMed for biomedical literature.

    Args:
        query:
            PubMed search query containing biomedical
            entities, diseases, drugs, or targets.
        retmax:
            Maximum number of PMIDs to return.

    Returns:
        A list of PubMed IDs.
    """
    return search_pubmed(
        query=query,
        retmax=retmax,
    )


model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

print("Tool name:")
print(search_pubmed_tool.name)

print("\nTool description:")
print(search_pubmed_tool.description)

print("\nTool argument schema:")
print(
    search_pubmed_tool.args_schema
    .model_json_schema()
)

direct_result = search_pubmed_tool.invoke({
    "query": (
        'TNIK AND '
        '"pulmonary fibrosis"'
    ),
    "retmax": 5,
})

print("\nDirect tool result:")
print(direct_result)

model_with_tools = model.bind_tools(
    [search_pubmed_tool]
)

response = model_with_tools.invoke(
    [
        (
            "system",
            "You are a biomedical literature assistant. "
            "Use the PubMed search tool when the user asks "
            "a biomedical evidence question.",
        ),
        (
            "human",
            (
                "Find PubMed papers about TNIK "
                "and pulmonary fibrosis."
            ),
        ),
    ]
)

print("Response type:")
print(type(response))

print("\nResponse content:")
print(response.content)

print("\nTool calls:")
print(response.tool_calls)


if response.tool_calls:
    tool_call = response.tool_calls[0]

    print("\nSelected tool:")
    print(tool_call["name"])

    print("\nGenerated arguments:")
    print(tool_call["args"])

    tool_result = (
        search_pubmed_tool.invoke(
            tool_call["args"]
        )
    )

    print("\nTool result:")
    print(tool_result)

else:
    print(
        "\nThe model did not generate "
        "a tool call."
    )