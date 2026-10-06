import base64
import hashlib
import re
from email.message import EmailMessage

from models import AttachmentInfo


def collect_attachment(msg: EmailMessage, mime_type: str, warnings: list) -> AttachmentInfo:
    payload, warns = get_bytes(msg)
    warnings.extend(warns)
    return AttachmentInfo(
        filename=msg.get_filename(),   # RFC2231/RFC2047 имена разрулены policy.default
        mime_type=mime_type,
        size=len(payload) if payload else 0,
        sha256=hashlib.sha256(payload).hexdigest() if payload else None,
        inline=msg.get_content_disposition() == "inline",
    )


def get_bytes(msg: EmailMessage):
    """Декодирование бинарной части с fallback на «битый» base64."""
    warnings = []
    payload = None
    try:
        payload = msg.get_payload(decode=True)
    except Exception as e:
        warnings.append(f"CTE decode failed: {e.__class__.__name__}")

    if payload is None:
        raw = msg.get_payload()
        if isinstance(raw, str) and raw:
            try:
                payload = robust_b64decode(raw)
            except Exception:
                warnings.append("binary payload unrecoverable")
    return payload, warnings


def robust_b64decode(raw: str) -> bytes:
    """Вытаскивает base64 даже из обрезанного/замусоренного payload."""
    cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
    cleaned = cleaned.rstrip("=")              # пересобираем padding заново
    cleaned += "=" * (-len(cleaned) % 4)
    return base64.b64decode(cleaned)