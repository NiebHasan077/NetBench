"""Pydantic models for the benchmark generator pipeline.

Question shape stays back-compatible with v4 (existing fields kept; new fields
are additive). See PLAN.md §5 for the full final schema.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DifficultyT = Literal["easy", "medium", "hard"]
QuestionTypeT = Literal[
    "concept",
    "reasoning",
    "calculation",
    "comparison",
    "diagnosis",
    "scenario",
    "task",
]


class Passage(BaseModel):
    paper_id: str
    passage_id: str
    text: str

    model_config = ConfigDict(extra="ignore")


class EvidenceAnchor(BaseModel):
    passage_id: str
    claim: str

    model_config = ConfigDict(extra="ignore")


class PaperCard(BaseModel):
    paper_id: str
    main_problem: str
    system_context: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    important_formulas: list[str] = Field(default_factory=list)
    important_numbers: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    possible_question_topics: list[str] = Field(default_factory=list)
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list)
    cluster_id: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


class Evidence(BaseModel):
    paper_id: str
    passage_id: str
    quote: str

    model_config = ConfigDict(extra="ignore")


class Question(BaseModel):
    id: str
    benchmark_version: str = "v5.0"
    category: str
    difficulty: DifficultyT
    question_type: QuestionTypeT
    question: str
    reference_answer: str
    source_papers: list[str]
    evidence: list[Evidence]
    keywords: list[str] = Field(default_factory=list)
    requires_calculation: bool = False
    synthetic_scenario: bool = False
    inspired_by_papers: list[str] = Field(default_factory=list)
    generator_model: str = ""
    critic_model: str = ""
    prompt_version: str = ""
    seed: int = 0
    # Provenance enrichment (stage 09; additive, optional). See stage_09_enrich.py.
    calibration: Optional[dict] = None  # {ladder:{weak,mid,strong}, difficulty_original}
    validation: Optional[dict] = None   # {checks_passed:[...], issues:[{check,reason}]}

    model_config = ConfigDict(extra="ignore")

    @field_validator("source_papers")
    @classmethod
    def at_least_one_paper(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("source_papers must be non-empty")
        return v
