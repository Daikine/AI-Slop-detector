"""Pydantic models for the main analyze API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Input: raw HTML (or email HTML body) to analyze."""

    html: str | None = None
    raw_email: str | None = None


class ParsedData(BaseModel):
    text: str = ""
    images: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class AnalyzerResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    features: dict = Field(default_factory=dict)


class DecisionResult(BaseModel):
    verdict: str
    score: float = Field(..., ge=0.0, le=1.0)
    analyzer_count: int
    individual_scores: list[float]
    reasons: list[str]


class AnalyzeResponse(BaseModel):
    text_analysis: AnalyzerResult
    image_analysis: dict
    decision: DecisionResult
    raw_email: str | None = None 
