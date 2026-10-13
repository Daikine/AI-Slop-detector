"""Combine only available text and vision evidence into the existing verdict scale."""
from __future__ import annotations

import math
from models import AnalyzerResult, DecisionResult

THRESHOLDS = (
    (0.2, "DEFINITELY_HUMAN"),
    (0.4, "PROBABLY_HUMAN"),
    (0.6, "MIXED"),
    (0.8, "PROBABLY_AI"),
    (1.01, "DEFINITELY_AI"),
)


def _valid_score(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def make_decision(
    text_result: AnalyzerResult,
    image_result: dict,
    *,
    text_available: bool = True,
) -> DecisionResult:
    """Average valid signals; unavailable modalities must never count as zero."""
    results: list[tuple[str, float, str]] = []
    reasons: list[str] = []

    text_score = _valid_score(text_result.score)
    if text_available and text_score is not None:
        results.append(("text", text_score, text_result.reason))
    else:
        reasons.append("text: skipped (no usable text)")

    image_status = image_result.get("status")
    if image_status is None:  # Compatibility with older image analyzer responses.
        image_status = "ok" if image_result.get("images") else "unavailable"
    image_score = _valid_score(image_result.get("score"))
    if image_status in {"ok", "partial"} and image_score is not None:
        results.append(("images", image_score, str(image_result.get("reason") or "vision analysis")))
    else:
        reasons.append("images: skipped (" + str(image_result.get("reason") or "no reliable vision result") + ")")

    if not results:
        return DecisionResult(
            verdict="MIXED", score=0.5, analyzer_count=0,
            individual_scores=[],
            reasons=reasons + ["insufficient evidence: no valid analysis was available"],
        )

    average = sum(score for _, score, _ in results) / len(results)
    verdict = next(label for bound, label in THRESHOLDS if average < bound)
    for name, score, detail in results:
        reasons.append(
            f"{name}: {'AI-like signals' if score >= 0.6 else 'weak or mixed AI signals'} "
            f"(score={score:.3f}); {detail}"
        )
    reasons.append("scores are indicators, not verified authorship probabilities")
    return DecisionResult(
        verdict=verdict,
        score=round(average, 3),
        analyzer_count=len(results),
        individual_scores=[round(score, 4) for _, score, _ in results],
        reasons=reasons,
    )
