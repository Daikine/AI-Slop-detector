"""Minimal email/.eml parser for ML data loading (stdlib only)."""
from __future__ import annotations

import email
import re
from email import policy

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class EmailParser:
    def parse(self, raw: str | bytes):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        msg = email.message_from_string(raw, policy=policy.default)
        text_body, html_body = "", ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition", "")).lower()
                if "attachment" in disp:
                    continue
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                try:
                    decoded = payload.decode(charset, errors="ignore")
                except LookupError:
                    decoded = payload.decode("utf-8", errors="ignore")
                if ctype == "text/plain":
                    text_body += decoded
                elif ctype == "text/html":
                    html_body += decoded
        else:
            payload = msg.get_payload(decode=True) or b""
            charset = msg.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="ignore")
            except LookupError:
                decoded = payload.decode("utf-8", errors="ignore")
            if msg.get_content_type() == "text/html":
                html_body = decoded
            else:
                text_body = decoded

        class _Parsed:
            pass

        out = _Parsed()
        out.text_body = text_body
        out.html_body = html_body
        return out

    @staticmethod
    def strip_html(html: str) -> str:
        if not html:
            return ""
        return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()
