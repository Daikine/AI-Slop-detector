"""Image analyzer service: HTTP API + external vision model (API key)."""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

_PLACEHOLDER_KEYS = {
    "",
    "your_key",
    "changeme",
    "replace_me",
    "sk-xxx",
    "sk-your-key",
}


def _resolve_api_key() -> str:
    raw = (os.getenv("AI_API_KEY") or "").strip()
    if raw.lower() in {k.lower() for k in _PLACEHOLDER_KEYS}:
        return ""
    return raw


AI_API_KEY = _resolve_api_key()
AI_API_URL = os.getenv(
    "AI_API_URL",
    "https://api.openai.com/v1/chat/completions",
)
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

app = FastAPI(title="Image Analyzer", version="0.2.0")


class ImageRequest(BaseModel):
    images: list[str] = Field(default_factory=list)


class ImageResponse(BaseModel):
    score: float
    reason: str
    images: list[dict[str, Any]]


def _decode_image(ref: str) -> bytes | None:
    """Decode data-URI / raw base64; return None for remote URLs."""
    if not ref:
        return None
    if ref.startswith("data:"):
        try:
            return base64.b64decode(ref.split(",", 1)[-1], validate=False)
        except Exception:
            return None
    if "://" not in ref and len(ref) > 64:
        try:
            return base64.b64decode(ref, validate=False)
        except Exception:
            return None
    return None


def _local_analyze(image_data: bytes | None, ref: str) -> dict[str, Any]:
    """Heuristic fallback when AI_API_KEY is not set or vision API fails."""
    features: dict[str, Any] = {"ref_preview": ref[:80], "qr_codes": []}
    if not image_data:
        if ref.startswith(("http://", "https://")):
            return {
                "score": 0.1,
                "reason": "remote image URL only; local decode skipped",
                "features": features,
            }
        return {"score": 0.0, "reason": "no image payload", "features": features}

    features["bytes"] = len(image_data)
    score = 0.0
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_data))
        features.update(
            {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
        )
        if not image.info:
            score += 0.10
    except Exception as exc:
        features["decode_error"] = str(exc)
        return {"score": 0.0, "reason": "image could not be decoded", "features": features}

    try:
        import cv2
        import numpy as np

        matrix = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if matrix is not None:
            detector = cv2.QRCodeDetector()
            ok, decoded, _, _ = detector.detectAndDecodeMulti(matrix)
            if ok:
                features["qr_codes"] = [v for v in decoded if v]
            else:
                value, _, _ = detector.detectAndDecode(matrix)
                if value:
                    features["qr_codes"] = [value]
    except ImportError:
        features["qr_detection"] = "opencv unavailable"
    except Exception as exc:
        features["qr_detection_error"] = str(exc)

    if features["qr_codes"]:
        score += 0.25
    reason = (
        "QR code detected"
        if features["qr_codes"]
        else "no QR code; AI-image score is heuristic"
    )
    return {"score": round(min(score, 1.0), 3), "reason": reason, "features": features}


def _local_batch(images: list[str]) -> ImageResponse:
    per_image = []
    for ref in images:
        payload = _decode_image(ref)
        per_image.append(_local_analyze(payload, ref))
    scores = [item["score"] for item in per_image]
    avg = sum(scores) / len(scores) if scores else 0.0
    reasons = [item["reason"] for item in per_image]
    return ImageResponse(
        score=round(avg, 3),
        reason="; ".join(dict.fromkeys(reasons)),
        images=per_image,
    )


async def _api_analyze(images: list[str]) -> dict[str, Any]:
    """Call an OpenAI-compatible vision API and map the reply to a score."""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You analyze email images for AI-generated / GenAI phishing artifacts. "
                "Reply with ONLY a JSON object: "
                '{"score": <float 0..1>, "reason": "<short explanation>"} '
                "where score is P(AI-generated or AI-assisted phishing imagery)."
            ),
        }
    ]
    for ref in images[:8]:
        if ref.startswith(("http://", "https://", "data:")):
            content.append({"type": "image_url", "image_url": {"url": ref}})
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ref}"},
                }
            )

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200,
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(AI_API_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(
                f"vision API error: {response.status_code} {response.text[:300]}"
            )
        data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"bad vision API response: {exc}") from exc

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {
            "score": 0.5,
            "reason": f"unparseable model reply: {text[:200]}",
            "images": [{"raw": text[:500]}],
        }
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "score": 0.5,
            "reason": f"invalid JSON from model: {text[:200]}",
            "images": [{"raw": text[:500]}],
        }

    score = float(parsed.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 3),
        "reason": str(parsed.get("reason", "vision model assessment")),
        "images": [{"model": AI_MODEL, "api": AI_API_URL}],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_configured": bool(AI_API_KEY),
    }


@app.post("/analyze", response_model=ImageResponse)
async def analyze_images(request: ImageRequest):
    if not request.images:
        return ImageResponse(score=0.0, reason="no images provided", images=[])

    if AI_API_KEY:
        try:
            result = await _api_analyze(request.images)
            return ImageResponse(
                score=result["score"],
                reason=result["reason"],
                images=result.get("images", []),
            )
        except Exception as exc:
            # Invalid key / network / quota — degrade gracefully
            fallback = _local_batch(request.images)
            fallback.reason = f"vision API failed ({exc}); {fallback.reason}"
            return fallback

    return _local_batch(request.images)
