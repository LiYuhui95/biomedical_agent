from typing import Any

from .state import AgentAction, ToolResult
from tools.pubmed import search_pubmed, fetch_paper
from pydantic import BaseModel, Field, field_validator


def execute_tool(action: AgentAction) -> ToolResult:
    if action.action not in TOOLS:
        return ToolResult(
            tool=action.action,
            success=False,
            error=f"Unknown tool: {action.action}",
        )

    tool_config = TOOLS[action.action]

    try:
        args = tool_config["args_schema"](
            **action.arguments
        )

        data = tool_config["function"](
            **args.model_dump()
        )

        return ToolResult(
            tool=action.action,
            success=True,
            data=data,
        )

    except Exception as exc:
        return ToolResult(
            tool=action.action,
            success=False,
            error=str(exc),
        )

class SearchPubMedArgs(BaseModel):
    query: str = Field(min_length=3, description="The search query for PubMed.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("query cannot be empty")

        return value


class FetchPaperArgs(BaseModel):
    pmids: list[str] = Field(
        min_length=1,
        description="A list of PubMed PMIDs."
    )

    @field_validator("pmids")
    @classmethod
    def validate_pmids(
        cls,
        value: list[str]
    ) -> list[str]:

        for pmid in value:
            pmid = pmid.strip()

            if not pmid:
                raise ValueError(
                    "pmid cannot be empty"
                )

            if not pmid.isdigit():
                raise ValueError(
                    f"Invalid PMID: {pmid}"
                )

        return value

TOOLS = {
    "search_pubmed": {
        "function": search_pubmed,
        "args_schema": SearchPubMedArgs,
    },

    "fetch_paper": {
        "function": fetch_paper,
        "args_schema": FetchPaperArgs,
    },
}