import base64
import hashlib
from email.message import EmailMessage

from models import ImageInfo
from extractors.attachment_extractor import get_bytes   # общий декодер байтов


def collect_mime_image(msg: EmailMessage, warnings: list,
                       include_content: bool, max_inline_bytes: int) -> ImageInfo:
    raw_cid = msg.get("Content-ID")
    # MIME: Content-ID: <logo123@host>  |  HTML: cid:logo123
    cid = raw_cid.strip().strip("<>") if raw_cid else None

    payload, warns = get_bytes(msg)
    warnings.extend(warns)

    img = ImageInfo(
        cid=cid,
        content_type=msg.get_content_type(),
        size_bytes=len(payload) if payload else 0,
        sha256=hashlib.sha256(payload).hexdigest() if payload else None,
        source="mime",
    )
    # base64 в ответ — только по флагу и только для маленьких,
    # иначе response раздувается до десятков мегабайт
    if include_content and payload and len(payload) <= max_inline_bytes:
        img.content_b64 = base64.b64encode(payload).decode("ascii")
    return img