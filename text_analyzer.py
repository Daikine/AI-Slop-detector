"""Local text AI-slop analyzer.

Priority:
  1) DistilBERT weights in MODEL_DIR (if present)
  2) Baseline TF-IDF + LogisticRegression pickles
  3) Keyword heuristics
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

from models import AnalyzerResult

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ML"))

from preprocess import TextPreprocessor  # noqa: E402

MODEL_DIR = os.getenv("MODEL_DIR", str(ROOT / "ML" / "models" / "text_detector"))
BASELINE_MODEL = Path(
    os.getenv("BASELINE_MODEL", str(ROOT / "ML" / "models" / "baseline_model.pkl"))
)
BASELINE_VEC = Path(
    os.getenv(
        "BASELINE_VEC",
        str(ROOT / "ML" / "models" / "baseline_vectorizer.pkl"),
    )
)


class TextPredictor:
    def __init__(self, model_dir: str = MODEL_DIR, max_length: int = 256) -> None:
        self.max_length = max_length
        self.pre = TextPreprocessor()
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None
        self.baseline_model = None
        self.baseline_vec = None
        self.device = "cpu"
        self.model_version = os.getenv("MODEL_VERSION", self.model_dir.name)

        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_dir)
                )
                self.model.to(self.device).eval()
            except Exception as exc:
                print(f"[text] transformer load failed: {exc}", flush=True)

        if self.model is None and BASELINE_MODEL.exists() and BASELINE_VEC.exists():
            try:
                with open(BASELINE_MODEL, "rb") as f:
                    self.baseline_model = pickle.load(f)
                with open(BASELINE_VEC, "rb") as f:
                    self.baseline_vec = pickle.load(f)
                self.model_version = "baseline-tfidf-logreg"
            except Exception as exc:
                print(f"[text] baseline load failed: {exc}", flush=True)

    def predict(self, text: str) -> AnalyzerResult:
        clean = self.pre.clean_text(text or "")
        if self.model is not None and self.tokenizer is not None:
            return self._transformer_predict(clean)
        if self.baseline_model is not None and self.baseline_vec is not None:
            return self._baseline_predict(clean)
        return self._fallback_predict(clean)

    def _transformer_predict(self, clean: str) -> AnalyzerResult:
        import torch

        enc = self.tokenizer(
            clean, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]
        score = round(float(probs[1]), 4)
        return AnalyzerResult(
            score=score,
            reason=self._reason(score),
            features={
                "model_version": self.model_version,
                "clean_length": len(clean),
                "device": self.device,
                "backend": "transformer",
            },
        )

    def _baseline_predict(self, clean: str) -> AnalyzerResult:
        X = self.baseline_vec.transform([clean])
        proba = self.baseline_model.predict_proba(X)[0]
        score = round(float(proba[1]), 4)
        return AnalyzerResult(
            score=score,
            reason=self._reason(score),
            features={
                "model_version": self.model_version,
                "clean_length": len(clean),
                "device": "cpu",
                "backend": "baseline",
            },
        )

    @staticmethod
    def _reason(score: float) -> str:
        if score >= 0.8:
            return "strong AI-generated writing patterns"
        if score >= 0.6:
            return "probable AI assistance in writing style"
        if score >= 0.4:
            return "mixed signals: style between human and AI"
        return "no strong AI-writing patterns"

    @staticmethod
    def _fallback_predict(text: str) -> AnalyzerResult:
        indicators = (
            "urgent", "immediately", "verify", "password", "account", "confirm",
            "срочно", "немедленно", "подтвердите", "парол", "аккаунт", "заблокирован",
            "настоящим сообщаем", "уважаемый клиент", "hereby inform",
        )
        matches = sum(1 for indicator in indicators if indicator in text)
        score = round(min(matches / 4.0, 1.0), 4)
        return AnalyzerResult(
            score=score,
            reason="heuristic fallback: no trained model mounted",
            features={
                "model_version": "heuristic-fallback",
                "clean_length": len(text),
                "device": "cpu",
                "backend": "heuristic",
                "model_available": False,
            },
        )


_predictor: TextPredictor | None = None


def _get_predictor() -> TextPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TextPredictor()
    return _predictor


def analyze_text(text: str) -> AnalyzerResult:
    """Analyze plain text for AI-assisted writing signals."""
    return _get_predictor().predict(text)
