import base64
import hashlib
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pytest
from fastapi.testclient import TestClient

from mime_walker import parse_email
from main import app


def simple_email() -> bytes:
    return (b"From: alice@example.com\r\nTo: bob@x.ru\r\n"
            b"Subject: Hi\r\n\r\nHello world\r\n")


def alternative_email() -> bytes:
    outer = MIMEMultipart("alternative")
    outer["Subject"] = "Sale"
    outer.attach(MIMEText("Fallback text", "plain"))
    outer.attach(MIMEText("<html><body><b>Big Sale</b></body></html>", "html"))
    return outer.as_bytes()


def related_email() -> bytes:
    outer = MIMEMultipart("related")
    outer["From"] = "shop@news.example.com"
    outer["Subject"] = "Sale!"
    outer.attach(MIMEText("Fallback", "plain"))
    outer.attach(MIMEText('<html><body><img src="cid:logo123"></body></html>', "html"))
    img = MIMEImage(b"\x89PNG fake", _subtype="png")
    img.add_header("Content-ID", "<logo123>")
    outer.attach(img)
    return outer.as_bytes()


def attachment_email() -> bytes:
    outer = MIMEMultipart("mixed")
    outer["Subject"] = "Report"
    outer.attach(MIMEText("See attached", "plain"))
    pdf = MIMEApplication(b"%PDF-fake", _subtype="pdf")
    pdf.add_header("Content-Disposition", "attachment", filename="report.pdf")
    outer.attach(pdf)
    return outer.as_bytes()


def links_email() -> bytes:
    html = ('<html><head><base href="https://cdn.news.example/"></head><body>'
            '<a href="/l/123">Shop now</a>'
            '<a href="javascript:void(0)">click</a>'
            '<img src="https://track.example/pixel.gif">'
            '<img src="/img/hero.png"></body></html>')
    outer = MIMEMultipart("alternative")
    outer.attach(MIMEText("x", "plain"))
    outer.attach(MIMEText(html, "html"))
    return outer.as_bytes()


# ---------- тесты ----------

def test_plain_only():
    p = parse_email(simple_email())
    assert "Hello world" in p.text
    assert p.html is None
    assert p.parse_metadata.text_source == "mime_plain"


def test_alternative_prefers_html():
    p = parse_email(alternative_email())
    assert "<b>Big Sale</b>" in p.html
    assert p.parse_metadata.text_source == "mime_plain"  # авторский plain сохранён


def test_cid_resolution():
    p = parse_email(related_email())
    inline = [i for i in p.images if i.cid == "logo123"]
    assert len(inline) == 1
    assert inline[0].sha256 == hashlib.sha256(b"\x89PNG fake").hexdigest()
    assert inline[0].source == "html+mime"
    assert not any("not resolved" in w for w in p.parse_metadata.warnings)


def test_attachment_metadata():
    p = parse_email(attachment_email())
    a = p.attachments[0]
    assert a.filename == "report.pdf"
    assert a.mime_type == "application/pdf"
    assert a.sha256 == hashlib.sha256(b"%PDF-fake").hexdigest()


def test_links_with_base_href():
    p = parse_email(links_email())
    urls = [l.url for l in p.links]
    assert "https://cdn.news.example/l/123" in urls          # relative resolved
    assert any(l.scheme == "javascript" for l in p.links)     # сохранены как фича
    img_urls = [i.url for i in p.images if i.url]
    assert "https://cdn.news.example/img/hero.png" in img_urls
    assert "https://track.example/pixel.gif" in img_urls


def test_truncated_multipart_yields_warning():
    p = parse_email(alternative_email()[:-20])   # откусили закрывающий boundary
    assert p.parse_metadata.warnings             # дефект из msg.defects


def test_empty_body():
    p = parse_email(b"From: a@b.c\r\nSubject: empty\r\n\r\n")
    assert p.text is None and p.html is None
    assert p.parse_metadata.warnings


def test_api_invalid_base64():
    client = TestClient(app)
    r = client.post("/parse", json={"raw_email": "!!!not-base64!!!"})
    assert r.status_code == 400


def test_api_happy_path():
    client = TestClient(app)
    r = client.post("/parse", json={
        "raw_email": base64.b64encode(related_email()).decode()})
    assert r.status_code == 200
    data = r.json()
    assert data["headers"]["from"] == "shop@news.example.com"
    assert any(i["cid"] == "logo123" for i in data["images"])
    assert data["parse_metadata"]["parse_time_ms"] is not None