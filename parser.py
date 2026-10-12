"""HTML / email body parser: extract plain text, image refs, and links."""
from __future__ import annotations

import re
from html.parser import HTMLParser

from models import ParsedData

_HREF_RE = re.compile(r'href=["\']?(https?://[^"\'\s>]+)', re.I)
_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+', re.I)
_WS_RE = re.compile(r"\s+")


class _HTMLExtractor(HTMLParser):
    """Collect visible text, img src values, and href links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[str] = []
        self.links: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {k.lower(): (v or "") for k, v in attrs}
        if tag in {"script", "style"}:
            self._skip = True
            return
        if tag == "img":
            src = values.get("src", "").strip()
            if src:
                self.images.append(src)
        if tag == "a":
            href = values.get("href", "").strip()
            if href.startswith(("http://", "https://")):
                self.links.append(href)
        if tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data:
            self.parts.append(data)


def parse_html(html: str) -> ParsedData:
    """Parse HTML into plain text, image sources, and links."""
    extractor = _HTMLExtractor()
    try:
        extractor.feed(html or "")
    except Exception:
        pass

    text = _WS_RE.sub(" ", "".join(extractor.parts)).strip()
    images = list(dict.fromkeys(extractor.images))
    links = list(dict.fromkeys(extractor.links + _HREF_RE.findall(html or "")))

    # Bare URLs in leftover text (e.g. plain-text email pasted as HTML)
    if text:
        links = list(dict.fromkeys(links + _URL_RE.findall(text)))

    return ParsedData(text=text, images=images, links=links)
