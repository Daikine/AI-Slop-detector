"""Train and persist a reproducible TF-IDF + logistic regression baseline."""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ML.preprocess import TextPreprocessor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ML" / "data"
RAW = DATA / "raw"
SYNTH = DATA / "processed" / "synthetic_full.csv"
EXTERNAL = DATA / "processed" / "external_hc3.csv"
PROCESSED = DATA / "processed"
MODELS = ROOT / "ML" / "models"
TARGET = "ai_assisted"
SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
C = 1.0
MAX_ITER = 1000


def load_eml() -> pd.DataFrame:
    """Load optional .eml files, using parent folder spam/ham as spam label."""
    from ML.email_parser import EmailParser

    parser = EmailParser()
    rows = []
    for path in RAW.rglob("*"):
        if not path.is_file():
            continue
        parent = path.parent.name.lower()
        label = 1 if "spam" in parent else 0 if "ham" in parent else None
        if label is None:
            continue
        try:
            parsed = parser.parse(path.read_text(encoding="utf-8", errors="ignore"))
            text = parsed.text_body or parser.strip_html(parsed.html_body)
        except Exception:
            continue
        if text.strip():
            rows.append({"text": text, "label": label, "source": "email", TARGET: 0})
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def train(df: pd.DataFrame):
    pre = TextPreprocessor(language="english")
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    df["clean"] = df["text"].map(pre.clean_text)
    df = df[df["clean"].str.len() > 0].drop_duplicates("clean").reset_index(drop=True)
    if df[TARGET].nunique() < 2:
        raise ValueError(f"Target '{TARGET}' must contain both classes")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df[TARGET], test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    model = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight="balanced", random_state=SEED)
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)
    model.fit(X_tr_v, y_tr)
    y_pred = model.predict(X_te_v)
    y_proba = model.predict_proba(X_te_v)[:, 1]
    model.fit(vec.transform(df["clean"]), df[TARGET])
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, y_proba)),
        "rows": int(len(df)),
        "test_rows": int(len(y_te)),
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "combined_clean.parquet", index=False)
    pd.DataFrame({"text": X_te, "label": y_te}).to_parquet(PROCESSED / "baseline_test.parquet", index=False)
    (MODELS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_te, y_pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    return model, vec


def save(model, vec) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    with (MODELS / "baseline_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (MODELS / "baseline_vectorizer.pkl").open("wb") as f:
        pickle.dump(vec, f)
    print(f"Saved -> {MODELS}")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    parts = [load_eml(), load_csv(SYNTH), load_csv(EXTERNAL)]
    df = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    if df.empty:
        raise SystemExit("No data. Run ML/data/generate_synthetic.py first.")
    print(f"Total rows: {len(df)} | AI-assisted: {int(df[TARGET].sum())}")
    save(*train(df))
