from text_analyzer import analyze_text


def test_heuristic_fallback_on_phishing_words():
    result = analyze_text("URGENT: verify your password and account immediately")
    assert result.score > 0
    assert "heuristic" in result.reason or result.score >= 0.5
