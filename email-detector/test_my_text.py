r"""Проверить произвольный текст существующей моделью AI Email Detector.

Примеры на Windows (из папки проекта):
    .venv\Scripts\python.exe test_my_text.py my_text.txt
    .venv\Scripts\python.exe test_my_text.py ai_example.txt
    .venv\Scripts\python.exe test_my_text.py

Ничего не переобучает, не изменяет .env и не печатает API-ключ.
Если подключён API, текст может быть передан настроенному провайдеру.
"""
from __future__ import annotations

import argparse
from email.message import EmailMessage
from pathlib import Path

from analyze import RU, score_email
from segments import KINDS


def get_text(filename: str | None) -> str:
    if filename:
        return Path(filename).read_text(encoding="utf-8-sig").strip()

    print("Вставь текст сообщения. Заверши отдельной строкой END и нажми Enter.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверить свой TXT-текст обученным детектором писем"
    )
    parser.add_argument("file", nargs="?", help="UTF-8 TXT-файл; если не задан, ввод вручную")
    args = parser.parse_args()

    text = get_text(args.file)
    if not text:
        parser.error("Текст пустой")

    # Создаём корректное письмо .eml только в оперативной памяти. Тема намеренно
    # отсутствует, чтобы сравнивать авторство именно введённого текста.
    email = EmailMessage()
    email.set_content(text, charset="utf-8")
    result = score_email(email.as_bytes())

    print("\n--- РЕЗУЛЬТАТ ---")
    print(f"Объём: {len(text.split())} слов")
    print(f"Режим: {result.get('mode', 'нет')}; эмбеддинги: {result.get('embedding_backend', 'нет')}")
    print(f"{'Сегмент':<17}{'TF-IDF':>9}{'EMBED':>9}{'LLM':>9}{'SCORE':>9}{'ПОРОГ':>9}  Метка")
    for kind in KINDS:
        item = result['segments'][kind]
        if not item['present']:
            print(f"{RU[kind]:<17}(отсутствует)")
            continue
        fmt = lambda key: '—' if item.get(key) is None else f"{item[key]:.3f}"
        print(f"{RU[kind]:<17}{fmt('tfidf'):>9}{fmt('embedding'):>9}"
              f"{fmt('semantic_llm'):>9}{fmt('score'):>9}{fmt('threshold'):>9}  {item['label']}")
    print(f"Дерево решений: {result.get('tree_probability', '—')}")
    print(f"Итоговый скор: {result.get('final_score', '—')}")
    print(f"ВЕРДИКТ: {result['verdict']}")
    for warning in result.get('warnings', []):
        print("ВНИМАНИЕ:", warning)
    print("Важно: это экспериментальный детектор писем, а не доказательство авторства.")
    if len(text.split()) < 50:
        print("Мало слов: на коротких SMS классификация особенно ненадёжна.")


if __name__ == "__main__":
    main()
