from typing import List
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


class AgentState(BaseModel):
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

    final_answer: str | None = None