import re
from typing import Optional
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr

from models import HeaderInfo, AuthResults

# spf=pass smtp.mailfrom=x.com; dkim=fail header.d=y.com; dmarc=none
AUTH_METHOD_RE = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([a-z0-9\-]+)", re.IGNORECASE)


def extract_headers(msg: EmailMessage) -> HeaderInfo:
    from_raw = _clean(msg.get("From"))
    from_addr, from_domain = _parse_address(from_raw)

    return HeaderInfo(**{
        "from": from_addr,
        "from_domain": from_domain,
        "subject": _clean(msg.get("Subject")),
        "reply_to": _clean(msg.get("Reply-To")),
        "return_path": _clean(msg.get("Return-Path")),
        "x_mailer": _clean(msg.get("X-Mailer")),
        "date": _clean(msg.get("Date")),
        "message_id": _clean(msg.get("Message-ID")),
        "list_unsubscribe": _clean(msg.get("List-Unsubscribe")),
        "authentication": _parse_auth_results(msg),
    })


def _clean(value) -> Optional[str]:
    """str(header) + ручная добивка RFC2047, если policy не справился."""
    if value is None:
        return None
    s = str(value).strip()
    if "=?" in s:   # остались нераскодированные encoded-words
        try:
            s = str(make_header(decode_header(s)))
        except Exception:
            pass
    return s or None


def _parse_address(raw: Optional[str]):
    if not raw:
        return None, None
    _name, addr = parseaddr(raw)          # 'Иван <ivan@x.ru>' -> 'ivan@x.ru'
    domain = addr.rsplit("@", 1)[-1].lower() if "@" in addr else None
    return (addr or raw), domain


def _parse_auth_results(msg: EmailMessage) -> Optional[AuthResults]:
    values = msg.get_all("Authentication-Results")   # заголовков может быть несколько
    if not values:
        return None
    found = {m.lower(): r.lower()
             for m, r in AUTH_METHOD_RE.findall("; ".join(str(v) for v in values))}
    if not found:
        return None
    return AuthResults(spf=found.get("spf"),
                       dkim=found.get("dkim"),
                       dmarc=found.get("dmarc"))