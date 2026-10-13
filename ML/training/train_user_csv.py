"""Experimental text baseline training for current GitHub TextPredictor.

Usage from repository ROOT (PowerShell):
    python ML/training/train_user_csv.py --csv ML/data/raw/balanced_ai_human_dataset.csv --allow-confounded-data

The source dataset is NOT a valid independent benchmark of AI authorship.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ML.preprocess import TextPreprocessor  # noqa: E402


def strip_synthetic_headers(text: str) -> str:
    """Strip leading RFC822-like header block only when clearly present.

    Dataset AI examples are built as a fake full email, whereas almost all human
    examples contain only the body. The output is an email BODY classifier.
    """
    raw = str(text).lstrip("\ufeff\r\n")
    if not re.match(r"(?i)^return-path\s*:", raw):
        return raw
    parts = re.split(r"\r?\n\s*\r?\n", raw, maxsplit=1)
    if len(parts) == 2 and re.search(r"(?mi)^Subject\s*:", parts[0]):
        return parts[1]
    return raw  # keep suspicious malformed emails rather than silently remove content


def inspect_dataset(df: pd.DataFrame) -> dict:
    ai = df[df.is_ai == 1]
    human = df[df.is_ai == 0]
    header = df.text.str.contains(r"(?mi)^Return-Path\s*:", regex=True)
    source_cross = pd.crosstab(df.source.fillna("unspecified"), df.is_ai)
    inseparable = bool((source_cross.gt(0).sum(axis=1) == 1).all())
    stats = {
        "rows": len(df),
        "human_rows": len(human), "ai_rows": len(ai),
        "human_median_raw_chars": float(human.text.str.len().median()),
        "ai_median_raw_chars": float(ai.text.str.len().median()),
        "human_header_rate": float(header[df.is_ai == 0].mean()),
        "ai_header_rate": float(header[df.is_ai == 1].mean()),
        "header_only_rule_accuracy": float(np.mean(header.astype(int).to_numpy() == df.is_ai.to_numpy())),
        "sources_exclusive_to_one_label": inseparable,
    }
    stats["strong_confound"] = bool(
        stats["header_only_rule_accuracy"] >= 0.9
        or stats["ai_median_raw_chars"] >= 2 * max(stats["human_median_raw_chars"], 1)
        or inseparable
    )
    return stats


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"text", "is_ai"}.issubset(df.columns):
        raise ValueError("CSV must have columns text,is_ai[,source]")
    df = df.dropna(subset=["text", "is_ai"]).copy()
    df["is_ai"] = pd.to_numeric(df.is_ai, errors="raise").astype(int)
    if set(df.is_ai.unique()) != {0, 1}:
        raise ValueError("is_ai must have BOTH 0=human and 1=AI")
    if "source" not in df:
        df["source"] = "unspecified"
    df["text"] = df.text.astype(str)
    df["body"] = df.text.map(strip_synthetic_headers)
    pre = TextPreprocessor()
    df["clean"] = df.body.map(pre.clean_text)
    df = df[df.clean.str.len() >= 15].copy()
    df["key"] = df.clean.str.casefold().str.replace(r"\s+", " ", regex=True)
    contradictory = df.groupby("key").is_ai.nunique().gt(1).sum()
    if contradictory:
        raise ValueError(f"{contradictory} cleaned duplicate texts have contradictory labels")
    return df.drop_duplicates(subset=["key"]).reset_index(drop=True)


def train(args: argparse.Namespace) -> dict:
    raw = pd.read_csv(args.csv)
    if "source" not in raw:
        raw["source"] = "unspecified"
    if not {"text", "is_ai"}.issubset(raw):
        raise ValueError("CSV needs text and is_ai columns")
    diagnostics = inspect_dataset(raw)
    print("=== RAW DATA DIAGNOSTICS (not a quality benchmark) ===")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if diagnostics["strong_confound"] and not args.allow_confounded_data:
        raise ValueError(
            "Dataset has strong class/source/format confounds. "
            "For DEMO training only, rerun with --allow-confounded-data. "
            "For a trustworthy detector collect matched human and AI email bodies from independent sources."
        )

    df = load_data(args.csv)
    if args.max_rows_per_class:
        df = pd.concat([
            g.sample(n=min(len(g), args.max_rows_per_class), random_state=42)
            for _, g in df.groupby("is_ai")
        ], ignore_index=True)
    if min(df.is_ai.value_counts()) < 3:
        raise ValueError("Need at least 3 distinct examples per class")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.is_ai)
    assert not set(train_df.key) & set(test_df.key), "Normalized text leakage detected"
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=35_000,
                                 min_df=2, sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=35_000,
                                 min_df=2, sublinear_tf=True)),
    ])
    X_train = vectorizer.fit_transform(train_df.clean)
    X_test = vectorizer.transform(test_df.clean)
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    model.fit(X_train, train_df.is_ai)
    pos_idx = list(model.classes_).index(1)
    scores = model.predict_proba(X_test)[:, pos_idx]
    labels = (scores >= 0.5).astype(int)
    metrics = {
        "status": "EXPERIMENTAL_UNVALIDATED_AUTHORSHIP",
        "warning": "Train/test share source families. Class was confounded with header presence and length. Do not quote this as real-world AI detection accuracy.",
        "train_count": len(train_df), "test_count": len(test_df),
        "metrics_on_biased_holdout": {
            "balanced_accuracy": float(balanced_accuracy_score(test_df.is_ai, labels)),
            "roc_auc": float(roc_auc_score(test_df.is_ai, scores)),
            "confusion_matrix_0_human_1_ai": confusion_matrix(test_df.is_ai, labels, labels=[0,1]).tolist(),
            "classification_report": classification_report(test_df.is_ai, labels, output_dict=True, zero_division=0),
        },
        "raw_diagnostics": diagnostics,
        "processed_body_median_human_chars": float(df[df.is_ai == 0].clean.str.len().median()),
        "processed_body_median_ai_chars": float(df[df.is_ai == 1].clean.str.len().median()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # Matches existing GitHub TextPredictor which reads two pickle files.
    with (args.out / "baseline_vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.out / "baseline_model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    (args.out / "training_report.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved compatible files:", args.out / "baseline_model.pkl", args.out / "baseline_vectorizer.pkl", sep="\n")
    print("Biased holdout balanced_accuracy:", round(metrics["metrics_on_biased_holdout"]["balanced_accuracy"],4))
    print("WARNING: this metric does NOT show real AI authorship detection accuracy.")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train current GitHub word+char TF-IDF model (DEMO only)")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "ML" / "models")
    p.add_argument("--max-rows-per-class", type=int, default=None, help="quick smoke mode, e.g. 300")
    p.add_argument("--allow-confounded-data", action="store_true", help="allow experimental model training")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
