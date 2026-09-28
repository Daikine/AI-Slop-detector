from app.main import analyze_html


def test_detects_tailwind_and_interactive_markup():
    score, features, reason = analyze_html('<form><div class="text-red-500 p-4"><script>x()</script></div></form>')
    assert score > 0
    assert features["tailwind_class_count"] == 2
    assert "active" in reason
