from text_analyzer import analyze_text


def test_heuristic_fallback_on_phishing_words():
    result = analyze_text("URGENT: verify your password and account immediately")
    assert result.score > 0
    assert "heuristic" in result.reason or result.score >= 0.5


def test_semantic_profile_is_embedded_in_text_analysis():
    text = (
        "We are excited to announce a comprehensive solution for your needs. "
        "Furthermore, our innovative platform offers a seamless experience."
    )
    result = analyze_text(text, subject="Product update")

    semantic = result.features["semantic_analysis"]
    assert semantic["method"] == "offline-rule-based-v1"
    assert 0.0 <= semantic["score"] <= 1.0
    assert semantic["signals"]
    assert any(segment["type"] == "subject" for segment in semantic["segments"])
    assert any(segment["type"] == "paragraph" for segment in semantic["segments"])


def test_semantic_profile_detects_formulaic_phrasing():
    result = analyze_text(
        "Please be advised that we would like to inform you about our comprehensive "
        "solution. We appreciate your patience while we provide a seamless experience."
    )

    semantic = result.features["semantic_analysis"]
    assert semantic["score"] >= 0.25
    assert any("formulaic_language" in signal for signal in semantic["signals"])


def test_semantic_profile_calculates_subject_body_relevance():
    result = analyze_text(
        "The product update includes a fix for the login issue and the latest test results.",
        subject="Product update and login fix",
    )

    relevance = result.features["semantic_analysis"]["subject_body_relevance"]
    assert relevance is not None
    assert 0.0 <= relevance <= 1.0
    assert relevance > 0.0


def test_semantic_profile_marks_short_text_as_insufficient_context():
    result = analyze_text("Hi, thanks!")
    semantic = result.features["semantic_analysis"]

    assert semantic["context_sufficient"] is False
    assert semantic["label"] == "insufficient_context"


def test_semantic_segments_include_valid_spans_and_signature():
    text = (
        "We are excited to announce a comprehensive solution for your business needs.\n\n"
        "Best regards,\nExample Team"
    )
    result = analyze_text(text)
    segments = result.features["semantic_analysis"]["segments"]

    signature = next(segment for segment in segments if segment["type"] == "signature")
    start, end = signature["span"]
    assert text[start:end] == signature["content"]
    assert any(segment["type"] == "paragraph" for segment in segments)


def test_main_passes_parsed_email_subject_into_text_profile(monkeypatch):
    from unittest.mock import AsyncMock

    import main as main_module
    from fastapi.testclient import TestClient

    parsed_email = {
        "text": (
            "We are excited to announce a comprehensive solution for your business needs. "
            "Furthermore, our innovative platform offers a seamless experience."
        ),
        "headers": {"subject": "Business platform update"},
        "images": [],
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return parsed_email

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, json):
            assert json["raw_email"] == "ZHVtbXk="
            return FakeResponse()

    monkeypatch.setattr(main_module, "PARSER_URL", "http://parser.test/parse")
    monkeypatch.setattr(main_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        main_module,
        "analyze_images",
        AsyncMock(return_value={"score": 0.0, "reason": "test", "images": []}),
    )

    response = TestClient(main_module.app).post(
        "/analyze", json={"raw_email": "ZHVtbXk="}
    )
    assert response.status_code == 200
    semantic = response.json()["text_analysis"]["features"]["semantic_analysis"]
    assert any(
        segment["type"] == "subject" and segment["content"] == "Business platform update"
        for segment in semantic["segments"]
    )


def test_semantic_rules_do_not_lower_a_trained_model_score():
    from models import AnalyzerResult
    from text_analyzer import TextPredictor

    predictor = TextPredictor()
    base = AnalyzerResult(
        score=0.8,
        reason="existing model result",
        features={"backend": "baseline", "model_version": "test"},
    )
    result = predictor._add_semantic_signals(
        base,
        "I sent the report yesterday and can review the attached file tomorrow.",
        subject="Report follow-up",
    )

    assert result.score >= base.score
    assert result.features["model_score"] == base.score
