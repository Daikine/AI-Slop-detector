import re
from typing import List, Tuple
from urllib.parse import urljoin, urlparse, unquote

from bs4 import BeautifulSoup

from models import LinkInfo, ImageInfo

try:
    import lxml  # noqa: F401
    _PARSER = "lxml"
except ImportError:
    _PARSER = "html.parser"

CID_RE = re.compile(r"^cid:\s*(.+?)\s*$", re.IGNORECASE)
CSS_URL_RE = re.compile(r"""url\(\s*['"]?([^'")]+)['"]?\s*\)""", re.IGNORECASE)
DANGEROUS_SCHEMES = {"javascript", "vbscript", "data", "blob"}


def extract_html_artifacts(html: str) -> Tuple[List[ImageInfo], List[LinkInfo]]:
    soup = BeautifulSoup(html, _PARSER)

    # <base href> меняет резолв всех относительных ссылок
    base_tag = soup.find("base", href=True)
    base_url = base_tag["href"].strip() if base_tag else None

    def _resolve(url: str) -> str:
        url = url.strip()
        return urljoin(base_url, url) if base_url else url

    links: List[LinkInfo] = []
    images: List[ImageInfo] = []
    pos = 0

    # --- <a href> ---
    for a in soup.find_all("a", href=True):
        raw_href = a["href"].strip()
        if not raw_href:
            continue
        m = CID_RE.match(raw_href)
        context = a.get_text(" ", strip=True)[:200] or None
        if m:
            links.append(LinkInfo(url=raw_href, context=context,
                                  position=pos, scheme="cid", is_cid=True))
        else:
            url = _resolve(raw_href)
            links.append(LinkInfo(url=url, context=context, position=pos,
                                  scheme=urlparse(url).scheme.lower() or None))
        pos += 1

    # --- <img> (CID-вариант разрулит cid_resolver) ---
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src:
            continue
        m = CID_RE.match(src)
        if m:
            images.append(ImageInfo(cid=unquote(m.group(1)),
                                    context=img.get("alt") or None, source="html"))
        else:
            url = _resolve(src)
            if urlparse(url).scheme.lower() not in DANGEROUS_SCHEMES:
                images.append(ImageInfo(url=url, context=img.get("alt") or None,
                                        source="html"))

    # --- CSS background из <style> ---
    for style in soup.find_all("style"):
        css = style.string or style.get_text()
        for m in CSS_URL_RE.finditer(css):
            target = m.group(1).strip()
            if CID_RE.match(target):
                images.append(ImageInfo(cid=unquote(target[4:].strip()), source="html"))
            else:
                url = _resolve(target)
                if urlparse(url).scheme.lower() not in DANGEROUS_SCHEMES:
                    links.append(LinkInfo(url=url, context="css_background",
                                          position=pos, scheme=urlparse(url).scheme.lower()))
                    pos += 1

    # --- атрибут background="" (старый Outlook любит) ---
    for tag in soup.find_all(attrs={"background": True}):
        target = tag["background"].strip()
        if target and not CID_RE.match(target):
            url = _resolve(target)
            links.append(LinkInfo(url=url, context="bg_attr", position=pos,
                                  scheme=urlparse(url).scheme.lower()))
            pos += 1

    return images, links