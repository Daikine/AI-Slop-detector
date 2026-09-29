from parser import parse_html


def test_parse_extracts_text_and_links():
    html = '<html><body><p>Hello <a href="https://example.com/x">world</a></p></body></html>'
    parsed = parse_html(html)
    assert "Hello" in parsed.text
    assert "world" in parsed.text
    assert "https://example.com/x" in parsed.links


def test_parse_extracts_images():
    html = '<img src="https://cdn.example.com/a.png"><img src="data:image/png;base64,aaa">'
    parsed = parse_html(html)
    assert len(parsed.images) == 2
    assert parsed.images[0].startswith("https://")


def test_skips_script_content():
    html = "<p>Visible</p><script>secret()</script>"
    parsed = parse_html(html)
    assert "Visible" in parsed.text
    assert "secret" not in parsed.text
