"""Быстрый санити-чек: прогони 2 текста через обученную модель.

Usage:
    python ML/training/predict.py "Уважаемый клиент, срочно подтвердите данные"
    python ML/training/predict.py                # использует дефолтные примеры
"""
from __future__ import annotations

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "ML/models/text_detector"
DEFAULTS = [
    "Уважаемый клиент! Ваш аккаунт будет заблокирован в течение 24 часов. "
    "Настоящим сообщаем о необходимости подтверждения данных.",
    "Привет, скинь отчёт за прошлый месяц, я не могу найти его в почте :)",
]

def main(texts: list[str]) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    for t in texts:
        enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[0]
        label = "AI" if p[1] > 0.5 else "HUMAN"
        print(f"[{label} p={p[1]:.3f}] {t[:70]}")

if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULTS)