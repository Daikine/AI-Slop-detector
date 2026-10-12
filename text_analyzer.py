"""Local AI-writing detector with lightweight, explainable semantic signals.

Inference priority:
  1) DistilBERT weights in MODEL_DIR (if present)
  2) Baseline TF-IDF + LogisticRegression pickles
  3) Legacy keyword fallback

Semantic profiling runs *inside* TextPredictor. It uses deterministic, offline
linguistic/content cues; it adds explainability without requiring a downloaded
model, another service, or additional runtime dependencies.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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

# Keep semantic analysis bounded for very large MIME messages. Model scoring and
# the existing response contract remain unchanged; only the new metadata is capped.
_MAX_SEMANTIC_CHARS = 100_000
_MAX_SEMANTIC_SEGMENTS = 50
_MAX_SEGMENT_PREVIEW = 500
_MIN_SEMANTIC_WORDS = 8

_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?…](?=\s|$)|$)", re.DOTALL)
_PARAGRAPH_RE = re.compile(r"\S(?:.*?\S)?(?=(?:\r?\n[ \t]*){2,}|\Z)", re.DOTALL)
_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{2,}|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)(?!\w)")

# Phrase groups describe formulaic or generic wording, not proof of AI authorship.
# Russian and English are both supported because the project data contains both.
_FORMULAIC_PHRASES = (
    "i hope this email finds you well",
    "i hope this message finds you well",
    "we are excited to announce",
    "please be advised",
    "we would like to inform you",
    "thank you for your understanding",
    "at your earliest convenience",
    "do not hesitate to contact us",
    "in today's fast-paced world",
    "unlock the full potential",
    "tailored to meet your needs",
    "comprehensive solution",
    "seamless experience",
    "cutting-edge",
    "state-of-the-art",
    "it is important to note",
    "we appreciate your patience",
    "настоящим сообщаем",
    "настоящим уведомляем",
    "обращаем ваше внимание",
    "рады сообщить",
    "мы рады сообщить",
    "благодарим за понимание",
    "для вашего удобства",
    "с целью повышения",
    "комплексное решение",
    "индивидуальный подход",
    "широкий спектр услуг",
    "уникальная возможность",
    "в современном мире",
    "позвольте представить",
    "не стесняйтесь обращаться",
    "в случае возникновения вопросов",
    "для получения дополнительной информации",
    "в целях обеспечения",
)

_FORMAL_TRANSITIONS = (
    "furthermore", "moreover", "additionally", "consequently", "in conclusion",
    "it is worth noting", "следует отметить", "таким образом", "кроме того",
    "более того", "в заключение", "в дополнение",
)

_GENERIC_TERMS = (
    "solution", "approach", "opportunity", "benefits", "experience", "platform",
    "innovative", "customer needs", "various services", "value proposition",
    "решение", "возможность", "преимущества", "опыт", "платформа",
    "инновационный", "потребности клиентов", "широкий спектр", "ценность",
)

_SIGNATURE_RE = re.compile(
    r"\b(?:best regards|kind regards|warm regards|regards|sincerely|"
    r"yours truly|respectfully|с уважением|с наилучшими пожеланиями|"
    r"с благодарностью)\b",
    re.IGNORECASE,
)

_SUBJECT_STOPWORDS = {
    # English
    "the", "and", "for", "with", "from", "your", "you", "our", "this", "that",
    "are", "was", "were", "have", "has", "will", "please", "about", "into",
    # Russian
    "для", "как", "что", "это", "при", "или", "вам", "ваш", "ваша", "ваши",
    "мы", "вы", "они", "его", "ее", "её", "над", "под", "про", "при",
}


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]

def _add_semantic_signals(
    self,
    base_result: AnalyzerResult,
    raw_text: str,
    subject: str | None,
) -> AnalyzerResult:
    """Fuse semantic cues into the existing score. Supports Neural API or local rules."""
    
    
    use_neural_api = os.getenv("USE_NEURAL_API", "false").lower() == "true"
    
    if use_neural_api:
        semantic = get_neural_semantic_score(raw_text, subject or "")
    else:
        semantic = _build_semantic_profile(raw_text, subject)

    base_score = float(base_result.score)
    backend = str(base_result.features.get("backend", "unknown"))

    
    if "api_error" in str(semantic.get("signals", [])) or "api_key_missing" in str(semantic.get("signals", [])):
        semantic_weight = 0.0
    else:
        semantic_weight = 0.25 if backend in {"transformer", "baseline"} else 1.0
        
    semantic_score = float(semantic.get("score", 0.0))
    
    if backend in {"transformer", "baseline"} and semantic_weight > 0:
        
        final_score = base_score + semantic_weight * semantic_score * (1.0 - base_score)
        fusion_strategy = "model_plus_semantic_evidence"
    else:
        
        final_score = max(base_score, semantic_score)
        fusion_strategy = "maximum_of_legacy_fallback_and_semantic_profile"
        
    final_score = round(min(max(final_score, 0.0), 1.0), 4)

    features = dict(base_result.features)
    features.update({
        "model_score": round(base_score, 4),
        "semantic_analysis": {
            **semantic,
            "fusion_weight": semantic_weight,
            "fusion_strategy": fusion_strategy,
        },
    })

    reason = (
        "heuristic fallback: no trained model mounted"
        if backend == "heuristic"
        else self._reason(final_score)
    )
    
    signals = semantic.get("signals", [])
    if signals:
        signal_categories = list(dict.fromkeys(
            signal.split(":", 1)[0].replace("_", " ")
            for signal in signals
        ))
        reason += f"; semantic signals: {', '.join(signal_categories[:3])}"
    elif not semantic.get("context_sufficient", True):
        reason += "; semantic analysis has insufficient text context"
    else:
        reason += "; no strong semantic style signals"

    return AnalyzerResult(score=final_score, reason=reason, features=features)
  
def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _split_body_segments(body: str) -> tuple[list[tuple[str, int, int]], bool]:
    """Return paragraph-like blocks with spans and report whether the view was capped."""
    blocks: list[tuple[str, int, int]] = []
    segments_truncated = False
    for match in _PARAGRAPH_RE.finditer(body):
        start, end = match.span()
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start += leading
        end -= trailing
        content = body[start:end]
        if content.strip():
            blocks.append((content.strip(), start, end))
        if len(blocks) > _MAX_SEMANTIC_SEGMENTS:
            segments_truncated = True
            blocks = blocks[:_MAX_SEMANTIC_SEGMENTS]
            break

    # HTML-to-text conversion often produces single newlines rather than blank
    # lines. If there is one large block, retain sentence spans instead.
    if len(blocks) == 1 and not segments_truncated:
        block, start, _end = blocks[0]
        sentence_matches = list(_SENTENCE_RE.finditer(block))
        non_empty = [m for m in sentence_matches if m.group(0).strip()]
        if len(non_empty) > 1:
            split_blocks: list[tuple[str, int, int]] = []
            for sentence in non_empty:
                content = sentence.group(0).strip()
                left_trim = len(sentence.group(0)) - len(sentence.group(0).lstrip())
                sentence_start = start + sentence.start() + left_trim
                split_blocks.append((content, sentence_start, sentence_start + len(content)))
            if len(split_blocks) > _MAX_SEMANTIC_SEGMENTS:
                segments_truncated = True
            blocks = split_blocks[:_MAX_SEMANTIC_SEGMENTS]

    return blocks, segments_truncated


def _segment_semantics(text: str) -> dict[str, Any]:
    """Score one segment using conservative, explainable style/content cues."""
    words = _tokens(text)
    if len(words) < _MIN_SEMANTIC_WORDS:
        return {
            "score": 0.0,
            "signals": [],
            "reason": "insufficient context for semantic style signals",
            "context_sufficient": False,
        }

    lower = text.casefold()
    score = 0.0
    signals: list[str] = []

    matched_phrases = [phrase for phrase in _FORMULAIC_PHRASES if phrase in lower]
    if matched_phrases:
        score += min(0.26 + 0.12 * (len(matched_phrases) - 1), 0.50)
        signals.append("formulaic_language: " + "; ".join(matched_phrases[:3]))

    transition_hits = [phrase for phrase in _FORMAL_TRANSITIONS if phrase in lower]
    if transition_hits:
        score += min(0.04 * len(transition_hits), 0.12)
        signals.append("formal_transitions: " + "; ".join(transition_hits[:3]))

    generic_count = sum(lower.count(term) for term in _GENERIC_TERMS)
    concrete_count = len(_NUMBER_RE.findall(text))
    if generic_count >= 2 and concrete_count < 2:
        score += 0.08
        signals.append("generic_wording: several broad claims with few concrete details")

    sentences = _split_sentences(text)
    sentence_lengths = [len(_tokens(sentence)) for sentence in sentences if _tokens(sentence)]
    if len(sentence_lengths) >= 4:
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((length - mean_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        coefficient_of_variation = math.sqrt(variance) / mean_length if mean_length else 1.0
        if mean_length >= 8 and coefficient_of_variation <= 0.22:
            score += 0.08
            signals.append("uniform_sentence_structure: sentence lengths are unusually consistent")

    sentence_keys = [re.sub(r"\W+", " ", sentence.casefold()).strip() for sentence in sentences]
    sentence_keys = [sentence for sentence in sentence_keys if _word_count(sentence) >= 5]
    if sentence_keys:
        counts = Counter(sentence_keys)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        if repeated:
            score += min(0.15 + 0.04 * (repeated - 1), 0.23)
            signals.append("repeated_content: one or more sentences are duplicated")
        elif len(sentence_keys) <= 40:
            # Look for near-identical sentences, a possible sign of templated variants.
            token_sets = [set(_tokens(sentence)) for sentence in sentence_keys]
            near_duplicate = False
            for i, left in enumerate(token_sets):
                for right in token_sets[i + 1:]:
                    if len(left) >= 5 and len(right) >= 5:
                        union = left | right
                        if union and len(left & right) / len(union) >= 0.82:
                            near_duplicate = True
                            break
                if near_duplicate:
                    break
            if near_duplicate:
                score += 0.12
                signals.append("near_repeated_content: sentences use highly similar wording")

    # Several concrete identifiers can indicate message-specific wording. This is
    # deliberately only a small adjustment; AI-generated text can contain details.
    if concrete_count >= 3 and not matched_phrases:
        score = max(0.0, score - 0.04)

    score = round(min(max(score, 0.0), 0.95), 4)
    if not signals:
        reason = "no strong formulaic or repetitive semantic patterns detected"
    else:
        reason = "; ".join(signal.split(":", 1)[0].replace("_", " ") for signal in signals)
    return {
        "score": score,
        "signals": signals,
        "reason": reason,
        "context_sufficient": True,
    }


def _semantic_label(score: float, context_sufficient: bool) -> str:
    if not context_sufficient:
        return "insufficient_context"
    if score >= 0.60:
        return "ai_like"
    if score >= 0.35:
        return "mixed"
    return "no_strong_ai_signals"


def _build_semantic_profile(text: str, subject: str | None) -> dict[str, Any]:
    """Create the explainable semantic section added to the existing result."""
    raw_text = text or ""
    body = raw_text[:_MAX_SEMANTIC_CHARS]
    input_truncated = len(raw_text) > len(body)
    segments: list[dict[str, Any]] = []
    scored_segments: list[tuple[float, int]] = []
    all_signals: list[str] = []

    def add_segment(kind: str, content: str, span: list[int]) -> None:
        profile = _segment_semantics(content)
        sufficient = bool(profile["context_sufficient"])
        score = float(profile["score"])
        if sufficient:
            scored_segments.append((score, min(_word_count(content), 80)))
        for signal in profile["signals"]:
            if signal not in all_signals:
                all_signals.append(signal)

        preview = content[:_MAX_SEGMENT_PREVIEW]
        segments.append({
            "type": kind,
            "content": preview,
            "span": span,
            "generation_probability": round(score, 4),
            "label": _semantic_label(score, sufficient),
            "reason": profile["reason"],
            "signals": profile["signals"],
            "truncated": len(content) > len(preview),
        })

    clean_subject = (subject or "").strip()
    if clean_subject:
        add_segment("subject", clean_subject, [0, len(clean_subject)])

    body_blocks, segments_truncated = _split_body_segments(body)
    for index, (content, start, end) in enumerate(body_blocks):
        kind = "signature" if index == len(body_blocks) - 1 and _SIGNATURE_RE.search(content) else "paragraph"
        add_segment(kind, content, [start, end])

    if scored_segments:
        weights = [max(math.sqrt(word_count), 1.0) for _score, word_count in scored_segments]
        weighted_mean = sum(score * weight for (score, _count), weight in zip(scored_segments, weights)) / sum(weights)
        highest = max(score for score, _count in scored_segments)
        semantic_score = 0.75 * weighted_mean + 0.25 * highest
        context_sufficient = True
    else:
        semantic_score = 0.0
        context_sufficient = False

    subject_body_relevance: float | None = None
    if clean_subject and body.strip():
        subject_terms = {
            token for token in _tokens(clean_subject)
            if len(token) >= 3 and token not in _SUBJECT_STOPWORDS
        }
        body_terms = {
            token for token in _tokens(body)
            if len(token) >= 3 and token not in _SUBJECT_STOPWORDS
        }
        if subject_terms:
            subject_body_relevance = round(len(subject_terms & body_terms) / len(subject_terms), 4)

    semantic_score = round(min(max(semantic_score, 0.0), 0.95), 4)
    return {
        "method": "offline-rule-based-v1",
        "score": semantic_score,
        "label": _semantic_label(semantic_score, context_sufficient),
        "context_sufficient": context_sufficient,
        "signals": all_signals[:12],
        "subject_body_relevance": subject_body_relevance,
        "segments": segments,
        "segments_truncated": segments_truncated,
        "input_truncated": input_truncated,
    }


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

    def predict(self, text: str, subject: str | None = None) -> AnalyzerResult:
        raw_text = text or ""
        clean = self.pre.clean_text(raw_text)
        if self.model is not None and self.tokenizer is not None:
            base_result = self._transformer_predict(clean)
        elif self.baseline_model is not None and self.baseline_vec is not None:
            base_result = self._baseline_predict(clean)
        else:
            base_result = self._fallback_predict(clean)

        return self._add_semantic_signals(base_result, raw_text, subject)

    def _add_semantic_signals(
        self,
        base_result: AnalyzerResult,
        raw_text: str,
        subject: str | None,
    ) -> AnalyzerResult:
        """Fuse semantic cues into the existing score without lowering legacy scores."""
        semantic = _build_semantic_profile(raw_text, subject)
        base_score = float(base_result.score)
        backend = str(base_result.features.get("backend", "unknown"))

        # Keep model predictions stable; when no trained model is available, semantic
        # rules have more influence because the legacy fallback is intentionally simple.
        semantic_weight = 0.20 if backend in {"transformer", "baseline"} else 1.0
        semantic_score = float(semantic["score"])
        if backend in {"transformer", "baseline"}:
            # Treat semantic rules as a secondary signal for an existing trained model.
            final_score = base_score + semantic_weight * semantic_score * (1.0 - base_score)
            fusion_strategy = "model_plus_semantic_evidence"
        else:
            # The legacy fallback is intentionally simple; use the stronger of its
            # signal and the new semantic profile, never lowering an old fallback score.
            final_score = max(base_score, semantic_score)
            fusion_strategy = "maximum_of_legacy_fallback_and_semantic_profile"
        final_score = round(min(max(final_score, 0.0), 1.0), 4)

        features = dict(base_result.features)
        features.update({
            "model_score": round(base_score, 4),
            "semantic_analysis": {
                **semantic,
                "fusion_weight": semantic_weight,
                "fusion_strategy": fusion_strategy,
            },
        })

        reason = (
            "heuristic fallback: no trained model mounted"
            if backend == "heuristic"
            else self._reason(final_score)
        )
        if semantic["signals"]:
            signal_categories = list(dict.fromkeys(
                signal.split(":", 1)[0].replace("_", " ")
                for signal in semantic["signals"]
            ))
            reason += "; semantic signals: " + ", ".join(signal_categories[:3])
        elif not semantic["context_sufficient"]:
            reason += "; semantic analysis has insufficient text context"
        else:
            reason += "; no strong semantic style signals"

        return AnalyzerResult(score=final_score, reason=reason, features=features)

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


def analyze_text(text: str, subject: str | None = None) -> AnalyzerResult:
    """Analyze an email body and optional subject using the existing text pipeline."""
    return _get_predictor().predict(text, subject=subject)
