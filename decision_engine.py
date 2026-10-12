"""Combine analyzer scores into a final explainable verdict."""
from __future__ import annotations

from models import AnalyzerResult, DecisionResult

# Average score thresholds: 0 = human, 1 = AI
THRESHOLDS = [
    (0.2, "DEFINITELY_HUMAN"),
    (0.4, "PROBABLY_HUMAN"),
    (0.6, "MIXED"),
    (0.8, "PROBABLY_AI"),
    (1.1, "DEFINITELY_AI"),
]


def make_decision(
    text_result: AnalyzerResult,
    image_result: dict,
) -> DecisionResult:
    """Merge text + image analysis into one verdict."""
    results: list[dict] = [
        {
            "analyzer_type": "text",
            "score": text_result.score,
            "reason": text_result.reason,
        }
    ]

    image_score = float(image_result.get("score", 0.0))
    image_reason = image_result.get("reason") or "image analysis"
    results.append(
        {
            "analyzer_type": "images",
            "score": image_score,
            "reason": image_reason,
        }
    )

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    verdict = next(v for th, v in THRESHOLDS if avg < th)

    reasons = []
    for r in results:
        score = r["score"]
        atype = r["analyzer_type"]
        if score > 0.5:
            reasons.append(f"{atype}: AI-like signals detected (score={score})")
        else:
            reasons.append(f"{atype}: no strong AI signals (score={score})")
        detail = r.get("reason")
        if detail:
            reasons.append(f"  → {detail}")

    return DecisionResult(
        verdict=verdict,
        score=round(avg, 3),
        analyzer_count=len(results),
        individual_scores=[round(s, 4) for s in scores],
        reasons=reasons,
    )
