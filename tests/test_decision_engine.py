from decision_engine import make_decision
from models import AnalyzerResult


def test_make_decision_averages_scores():
    text = AnalyzerResult(score=0.2, reason="human-like", features={})
    images = {"score": 0.8, "reason": "qr found"}
    decision = make_decision(text, images)
    assert decision.score == 0.5
    assert decision.verdict == "MIXED"
    assert decision.analyzer_count == 2


def test_definitely_human():
    text = AnalyzerResult(score=0.05, reason="clean", features={})
    images = {"score": 0.0, "reason": "no images"}
    decision = make_decision(text, images)
    assert decision.verdict == "DEFINITELY_HUMAN"
