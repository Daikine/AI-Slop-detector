import re
from bs4 import BeautifulSoup

try:
    import lxml  # noqa: F401
    _PARSER = "lxml"
except ImportError:
    _PARSER = "html.parser"

DROP = ["script", "style", "noscript", "template", "head", "title"]
BLOCK = ["p", "div", "table", "tr", "li", "ul", "ol", "h1", "h2", "h3",
         "h4", "h5", "h6", "section", "article", "blockquote", "pre"]

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(DROP):
        tag.decompose()

    # <br> -> перенос строки, иначе весь текст склеится в одну строку
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # разделитель "\n" между ВСЕМИ текстовыми узлами:
    # get_text() без него склеивает <td>цена</td><td>100</td> -> "цена100"
    text = soup.get_text("\n")

    text = re.sub(r"[ \t\xa0]+", " ", text)              # пробелы + &nbsp;
    text = re.sub(r"\n[ \t]*\n[ \t\n]*", "\n\n", text)   # схлопываем пустые строки
    return text.strip()