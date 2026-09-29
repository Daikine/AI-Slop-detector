"""Download public AI-text datasets and merge into training CSVs.

Sources:
  - Hello-SimpleAI/HC3 (Human ChatGPT Comparison Corpus) — English Q&A
  - Local synthetic seed (always available offline)

Usage:
  python ML/data/download_datasets.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from ML.preprocess import TextPreprocessor

DATA = Path(__file__).resolve().parent
PROCESSED = DATA / "processed"
SEEDS = DATA / "seeds"
OUT = PROCESSED / "external_hc3.csv"
MAX_PER_CLASS = 800
SEED = 42

random.seed(SEED)


def download_hc3() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install datasets package: pip install datasets\n" + str(exc)
        ) from exc

    print("[i] downloading Hello-SimpleAI/HC3 (all subset)…")
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    rows: list[dict] = []
    seen: set[str] = set()
    preprocessor = TextPreprocessor()
    human, ai = 0, 0
    indices = list(range(len(ds)))
    random.shuffle(indices)
    for i in indices:
        row = ds[i]
        # HC3: human_answers / chatgpt_answers are lists of strings
        if human < MAX_PER_CLASS:
            for ans in row.get("human_answers") or []:
                text = (ans or "").strip()
                normalized = preprocessor.clean_text(text)
                if len(text) > 40 and normalized not in seen:
                    seen.add(normalized)
                    rows.append(
                        {
                            "text": text[:4000],
                            "label": 0,
                            "source": "human",
                            "ai_assisted": 0,
                        }
                    )
                    human += 1
                    break
        if ai < MAX_PER_CLASS:
            for ans in row.get("chatgpt_answers") or []:
                text = (ans or "").strip()
                normalized = preprocessor.clean_text(text)
                if len(text) > 40 and normalized not in seen:
                    seen.add(normalized)
                    rows.append(
                        {
                            "text": text[:4000],
                            "label": 1,
                            "source": "ai",
                            "ai_assisted": 1,
                        }
                    )
                    ai += 1
                    break
        if human >= MAX_PER_CLASS and ai >= MAX_PER_CLASS:
            break
    print(f"[i] HC3 rows: human={human} ai={ai}")
    return rows


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SEEDS.mkdir(parents=True, exist_ok=True)
    try:
        rows = download_hc3()
    except Exception as exc:
        print(f"[warn] HC3 download failed ({exc}); continuing without external data")
        rows = []

    if not rows:
        print("[!] no external rows written")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["text", "label", "source", "ai_assisted"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
