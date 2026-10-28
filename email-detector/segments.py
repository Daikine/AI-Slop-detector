"""MIME-разбор реального .eml, без вложений; ровно четыре зоны на выходе."""
import email
import re
from email import policy
from html.parser import HTMLParser

KINDS = ('subject', 'greeting', 'body', 'signature')
GREETING = re.compile(r'^(?:hi\b|hey\b|hello\b|dear\b|good (?:morning|evening|afternoon)\b|'
                      r'здравств\w*|добр\w*|привет\w*|хай\b|уважаем\w*|коллеги\b|всем привет)', re.I)
SIGNOFF = re.compile(r'^(?:--\s*$|с уважением\b|с наилучшими\b|всего доброго\b|'
                     r'best regards\b|kind regards\b|regards\b|sincerely\b|cheers\b)', re.I)
CONTACT = re.compile(r'(?:[\w.+-]+@[\w.-]+\.\w+|\+\d[\d ()-]{6,}|(?:тел|phone|tel)\s*[:.])', re.I)


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'style', 'script', 'head'}:
            self.hidden += 1
        if tag in {'p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'style', 'script', 'head'}:
            self.hidden = max(self.hidden - 1, 0)
        if tag in {'p', 'div', 'li', 'tr'}:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def parse_eml(raw: bytes | str) -> tuple[str, str]:
    if isinstance(raw, str):
        raw = raw.encode('utf-8', 'replace')
    msg = email.message_from_bytes(raw, policy=policy.default)
    subject = str(msg.get('Subject') or '').strip()
    plain, html = [], []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == 'attachment':
            continue
        if part.get_content_type() not in {'text/plain', 'text/html'}:
            continue
        try:
            value = part.get_content()
        except (LookupError, UnicodeError, ValueError):
            value = (part.get_payload(decode=True) or b'').decode('utf-8', 'replace')
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        (plain if part.get_content_type() == 'text/plain' else html).append(value)
    if plain:
        body = '\n'.join(plain)
    else:
        parser = _HTMLText()
        parser.feed('\n'.join(html))
        body = ''.join(parser.parts)
    body = re.sub(r'\n{3,}', '\n\n', body.replace('\r\n', '\n')).strip()
    return subject, body


def split_email(subject: str, body: str) -> dict[str, str]:
    lines = [line.rstrip() for line in body.replace('\r\n', '\n').split('\n')]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    greeting = signature = ''
    if lines and len(lines[0]) <= 120 and GREETING.match(lines[0].strip()):
        greeting = lines.pop(0).strip()
        while lines and not lines[0].strip():
            lines.pop(0)

    # Подпись может идти без пустой строки, после стандартного sign-off.
    for i in range(max(0, len(lines) - 7), len(lines)):
        if SIGNOFF.match(lines[i].strip()):
            signature = '\n'.join(lines[i:]).strip()
            lines = lines[:i]
            break
    if not signature and len(lines) >= 3:
        blocks = re.split(r'\n\s*\n', '\n'.join(lines))
        last = blocks[-1].strip()
        if len(blocks) > 1 and len(last.splitlines()) <= 4 and CONTACT.search(last):
            signature = last
            lines = '\n\n'.join(blocks[:-1]).split('\n')
    return {'subject': subject.strip(), 'greeting': greeting,
            'body': '\n'.join(lines).strip(), 'signature': signature}
