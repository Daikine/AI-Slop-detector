import email
import logging
from email import policy
from email.message import EmailMessage

from decoding import decode_bytes
from extractors.attachment_extractor import collect_attachment
from extractors.image_extractor import collect_mime_image
from extractors.header_extractor import extract_headers
from extractors.html_extractor import extract_html_artifacts
from extractors.text_extractor import html_to_text
from cid_resolver import resolve_images
from models import ParsedEmail, ParseMetadata

logger = logging.getLogger(__name__)

MAX_RAW_SIZE = 25 * 1024 * 1024   # защита от DoS гигантским письмом
MAX_NESTING_DEPTH = 5             # защита от бесконечной вложенности rfc822


class EmailTooLarge(Exception):
    pass


class _Collector:
    """Аккумулятор всего найденного при обходе дерева."""
    def __init__(self, include_inline_content: bool, max_inline_bytes: int):
        self.include_inline_content = include_inline_content
        self.max_inline_bytes = max_inline_bytes
        self.warnings: list[str] = []
        self.html_parts: list[str] = []
        self.plain_parts: list[str] = []
        self.mime_images: list = []
        self.attachments: list = []
        self.multipart_types: set[str] = set()
        self.max_depth = 0


def parse_email(raw: bytes, *, include_inline_content=False,
                max_inline_bytes=2_000_000) -> ParsedEmail:
    """Точка входа: bytes сырого письма -> ParsedEmail."""
    if len(raw) > MAX_RAW_SIZE:
        raise EmailTooLarge(f"email too large: {len(raw)} bytes")

    msg = email.message_from_bytes(raw, policy=policy.default)
    c = _Collector(include_inline_content, max_inline_bytes)

    # stdlib сам нашёл дефекты (битый boundary и т.п.) — забираем как warnings
    for part in msg.walk():
        for defect in part.defects:
            c.warnings.append(f"{type(defect).__name__} @ {part.get_content_type()}")

    _walk(msg, c, depth=0, in_related=False)

    # canonical HTML: последний найденный (в multipart/alternative части
    # идут по возрастанию «качества»: plain, потом html)
    html = c.html_parts[-1] if c.html_parts else None

    headers = extract_headers(msg)

    html_images, links = [], []
    if html:
        html_images, links = extract_html_artifacts(html)

    images, cid_warnings = resolve_images(html_images, c.mime_images)
    c.warnings.extend(cid_warnings)

    if c.plain_parts:
        text, text_source = c.plain_parts[0], "mime_plain"
    elif html:
        text, text_source = html_to_text(html), "derived_from_html"
    else:
        text, text_source = None, None

    return ParsedEmail(
        headers=headers,
        html=html,
        text=text,
        images=images,
        attachments=c.attachments,
        links=links,
        parse_metadata=ParseMetadata(
            text_source=text_source,
            multipart_type="+".join(sorted(c.multipart_types)) or None,
            nesting_depth=c.max_depth,
            warnings=c.warnings,
        ),
    )

def _walk(msg: EmailMessage, c: _Collector, depth: int, in_related: bool) -> None:
    c.max_depth = max(c.max_depth, depth)
    ctype = msg.get_content_type()               # никогда не None, дефолт text/plain
    disposition = msg.get_content_disposition()  # None | 'inline' | 'attachment'

    # --- контейнеры ---
    if ctype.startswith("multipart/"):
        subtype = ctype.split("/", 1)[1]
        c.multipart_types.add(subtype)
        parts = list(msg.iter_parts())
        if not parts:
            c.warnings.append(f"empty multipart/{subtype}")
        for part in parts:
            _walk(part, c, depth + 1,
                  in_related=in_related or subtype == "related")
        return

    if ctype == "message/rfc822":
        _walk_rfc822(msg, c, depth, in_related)
        return

    # --- листовые части; порядок проверок важен! ---
    # 1) явное вложение (в т.ч. ПРИЛОЖЕННАЯ картинка — не путать с инлайн)
    if disposition == "attachment":
        c.attachments.append(collect_attachment(msg, ctype, c.warnings))
        return

    # 2) инлайновая картинка (обычно с Content-ID)
    if ctype.startswith("image/"):
        c.mime_images.append(collect_mime_image(
            msg, c.warnings, c.include_inline_content, c.max_inline_bytes))
        return

    # 3) прочее с filename (inline-приложенный pdf и т.п.)
    if msg.get_filename():
        c.attachments.append(collect_attachment(msg, ctype, c.warnings))
        return

    # 4) тело письма
    if ctype.startswith("text/"):
        text, charset, warns = _get_text_part(msg)
        c.warnings.extend(warns)
        if text is None:
            return
        if ctype == "text/html":
            c.html_parts.append(text)
        elif ctype == "text/plain":
            c.plain_parts.append(text)
        else:
            c.warnings.append(f"ignored text part: {ctype}")   # text/calendar и пр.
        return

    c.warnings.append(f"skipped leaf part: {ctype}")

def _walk_rfc822(msg, c, depth, in_related) -> None:
    if depth >= MAX_NESTING_DEPTH:
        c.warnings.append("max message/rfc822 nesting depth exceeded")
        return
    try:
        inner = msg.get_content()   # при policy.default вернёт EmailMessage
    except Exception as e:
        inner = None
        c.warnings.append(f"rfc822 unwrap failed: {e.__class__.__name__}")
    if not isinstance(inner, EmailMessage):
        payload = msg.get_payload()
        inner = payload[0] if isinstance(payload, list) and payload else None
    if inner is None:
        c.warnings.append("empty message/rfc822 part")
        return
    _walk(inner, c, depth + 1, in_related)

def _get_text_part(part: EmailMessage):
    warnings: list[str] = []
    declared = part.get_content_charset()

    try:
        content = part.get_content()      # CTE + charset одним вызовом
        if isinstance(content, str):
            if not content.strip():
                return None, declared, ["empty text part"]
            return content, declared, warnings
        warnings.append(f"get_content() -> {type(content).__name__} instead of str")
    except LookupError:
        warnings.append(f"unknown charset: {declared}")
    except UnicodeDecodeError:
        warnings.append(f"declared charset failed: {declared}")
    except Exception as e:                # битый base64 и прочее
        warnings.append(f"decode failed: {e.__class__.__name__}")

    # fallback: сырой payload + наш каскад из decode_bytes
    payload = part.get_payload(decode=True)
    if not payload:
        warnings.append("no decodable payload in text part")
        return None, declared, warnings
    text, used, extra = decode_bytes(payload, declared)
    warnings.extend(extra)
    return text, used, warnings