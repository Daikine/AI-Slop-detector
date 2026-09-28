"""Model wrapper: email text -> P(ai_assisted). Uses the same preprocessing as training."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ML"))

from preprocess import TextPreprocessor


class TextPredictor:
    def __init__(self, model_dir: str, max_length: int = 256) -> None:
        self.max_length = max_length  # = cfg.max_length при обучении
        self.pre = TextPreprocessor()
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device).eval()
        self.model_version = os.getenv("MODEL_VERSION", Path(model_dir).name)

    def predict(self, text: str) -> tuple[float, dict, str]:
        clean = self.pre.clean_text(text or "")
        if self.model is None or self.tokenizer is None:
            return self._fallback_predict(clean)
        enc = self.tokenizer(
            clean, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]
        score = round(float(probs[1]), 4)  # P(ai_assisted), label=1 при обучении
        features = {
            "model_version": self.model_version,
            "clean_length": len(clean),
            "device": self.device,
        }
        if score >= 0.8:
            reason = "strong AI-generated writing patterns"
        elif score >= 0.6:
            reason = "probable AI assistance in writing style"
        elif score >= 0.4:
            reason = "mixed signals: style between human and AI"
        else:
            reason = "no strong AI-writing patterns"
        return score, features, reason

    @staticmethod
    def _fallback_predict(text: str) -> tuple[float, dict, str]:
        """Development fallback used when a trained model is not mounted."""
        indicators = (
            "urgent", "immediately", "verify", "password", "account", "confirm",
            "срочно", "немедленно", "подтвердите", "парол", "аккаунт", "заблокирован",
        )
        matches = sum(1 for indicator in indicators if indicator in text)
        score = round(min(matches / 4.0, 1.0), 4)
        return score, {
            "model_version": "heuristic-fallback",
            "clean_length": len(text),
            "device": "cpu",
            "model_available": False,
        }, "heuristic fallback: trained text model is not mounted"
