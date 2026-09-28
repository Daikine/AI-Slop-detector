from app.main import analyze_image


def test_empty_image_is_safe():
    score, features, reason = analyze_image(b"")
    assert score == 0
    assert features["qr_codes"] == []
    assert "no image" in reason
