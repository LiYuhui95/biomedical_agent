from typing import Any, List
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    pmid: str
    title: str
    abstract: str | None = None
    year: int | None = None
    journal: str | None = None
    authors: list[str] = []


class PaperRecord(BaseModel):
    pmid: str
    title: str
    abstract: str | None = None
    year: int | None = None
    journal: str | None = None
    authors: list[str] = Field(
        default_factory=list
    )


class ScientificEvidence(BaseModel):
    pmid: str
    claim: str
    evidence_type: str
    model_system: str | None = None
    intervention: str | None = None
    outcome: str | None = None
    supports_question: bool | None = None
    limitations: list[str] = Field(
        default_factory=list
    )


class AgentAction(BaseModel):
    """
    The next action that the LLM wants the agent to take.
    """

    action: str

    arguments: dict[str, Any] = Field(
        default_factory=dict
    )

class ToolResult(BaseModel):
    """
    The observation returned by a tool after an action is executed.
    """

    tool: str

    success: bool

    data: Any = None

    error: str | None = None


class AgentState(BaseModel):
    """
    Shared working memory of the agent.
    """

    question: str

    plan: list[str] = Field(
        default_factory=list
    )

    papers: list[PaperRecord] = Field(
        default_factory=list
    )

    evidence: list[ScientificEvidence] = Field(
        default_factory=list
    )

    completed_steps: list[str] = Field(
        default_factory=list
    )

    tool_results: list[ToolResult] = Field(
        default_factory=list
    )

    actions: list[AgentAction] = Field(
    default_factory=list
    )
    
    reasoning_trace: list[str] = Field(
        default_factory=list
    )

    final_answer: str | None = None
