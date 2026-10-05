import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Реальные alias'ы, встречающиеся в дикой природе
CHARSET_ALIASES = {
    "cp1251": "windows-1251",
    "1251": "windows-1251",
    "windows1251": "windows-1251",
    "utf8": "utf-8",
    "utf_8": "utf-8",
    "ks_c_5601-1987": "cp949",      # корейский из Outlook
    "gb2312": "gb18030",            # superset, безопаснее
    "gbk": "gb18030",
    "unknown-8bit": None,
    "x-unknown": None,
    "user-defined": None,
    "default": None,
    "ansi_x3.110-1983": "cp1252",   # бред, который шлёт Outlook
}

# 'РџСЂРёРІРµС‚' — utf-8, прочитанный как cp1251: пары [Р или С]+кириллица
MOJIBAKE_RE = re.compile(r"(?:[РС][\u0400-\u04ff]){4,}")
FIXABLE = {"windows-1251", "cp1251", "koi8-r", "koi8-u", "cp1252",
           "latin-1", "iso-8859-1"}


def normalize_charset(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().strip('"').lower()
    return CHARSET_ALIASES.get(n, n)


def decode_bytes(data: bytes, declared_charset: Optional[str]) -> Tuple[str, str, list]:
    """Декодирует байты. НИКОГДА не бросает исключений.
    Возвращает (text, charset_used, warnings)."""
    warnings: list[str] = []
    charset = normalize_charset(declared_charset)

    # 1) декларированный charset из Content-Type
    if charset:
        try:
            return _finish(data.decode(charset), charset), charset, warnings
        except (UnicodeDecodeError, LookupError) as e:
            warnings.append(f"declared charset '{charset}' failed: {e.__class__.__name__}")

    # 2) строгий utf-8 — частый случай «charset забыли». Валидация мультибайтом
    #    достаточно строгая, ложных срабатываний почти нет
    try:
        return _finish(data.decode("utf-8"), "utf-8"), "utf-8 (detected)", warnings
    except UnicodeDecodeError:
        pass

    # 3) статистическое определение (charset-normalizer — замена chardet)
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best is not None:
            enc = best.encoding or "utf-8"
            warnings.append(f"charset detected: {enc}")
            return _finish(str(best), enc), f"{enc} (detected)", warnings
    except Exception:
        pass

    # 4) latin-1 маппит байт-в-символ и не падает никогда — последний рубеж
    warnings.append("all decoders failed, using latin-1 fallback")
    return _finish(data.decode("latin-1"), "latin-1"), "latin-1 (fallback)", warnings


def _finish(text: str, charset: str) -> str:
    text = text.lstrip("\ufeff")          # BOM
    return unwrap_mojibake(text, charset)


def unwrap_mojibake(text: str, charset_used: Optional[str]) -> str:
    """Best-effort починка двойного кодирования."""
    cs = normalize_charset(charset_used)
    if not cs or cs not in FIXABLE:
        return text
    if not MOJIBAKE_RE.search(text):
        return text
    src = "cp1252" if cs in {"latin-1", "iso-8859-1"} else cs
    try:
        return text.encode(src).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text