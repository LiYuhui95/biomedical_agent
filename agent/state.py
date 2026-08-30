from typing import Any, List
from pydantic import BaseModel, Field
from enum import Enum

class AgentAction(BaseModel):
    action: str
    arguments: dict[str, Any]

class WorkflowAction(str, Enum):
    SEARCH = "search"
    FETCH_MORE = "fetch_more"
    REFINE_QUERY = "refine_query"
    SYNTHESIZE = "synthesize"
    STOP_INSUFFICIENT = "stop_insufficient"

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
    claim: str | None = None
    evidence_type: str = 'unclear'
    is_relevant: bool = False
    model_system: str | None = None
    intervention: str | None = None
    outcome: str | None = None
    supports_question: bool | None = None
    limitations: list[str] = Field(
        default_factory=list
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

    iteration: int = 0
    max_iterations: int = 3

    retrieved_count: int = 0
    usable_evidence_count: int = 0
    relevance_rate: float = 0.0

    next_action: WorkflowAction = WorkflowAction.SEARCH
    termination_reason: str | None = None

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
