from langchain.agents import create_agent
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

    Use this tool when the user asks for papers or
    evidence about a biomedical research question.
    """
    return search_pubmed(
        query=query,
        retmax=retmax,
    )


model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[search_pubmed_tool],
    system_prompt=(
        "You are a biomedical literature assistant. "
        "Use PubMed when the user asks for biomedical "
        "papers or evidence. Do not invent PMIDs."
    ),
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Find PubMed papers about TNIK "
                "and pulmonary fibrosis."
            ),
        }
    ]
})

print("Result keys:")
print(result.keys())

print("\nMessages:")

for message in result["messages"]:
    message.pretty_print()