"""Main application: parse → text analysis → image analysis → decision."""
from __future__ import annotations

import logging
import os

import httpx
from fastapi import FastAPI

from decision_engine import make_decision
from image_client import analyze_images
from models import AnalyzeRequest, AnalyzeResponse
from parser import parse_html
from text_analyzer import analyze_text

logger = logging.getLogger("orchestrator")

PARSER_URL = os.getenv("PARSER_URL")

app = FastAPI(
    title="AI-Slop-detector",
    version="0.2.0",
    description="Simplified two-app architecture: local text analysis + remote image analysis.",
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


def _collect_image_refs(parsed: dict) -> list[str]:
    """Parser отдаёт [{url, cid, content_b64}]; image-analyzer ждёт list[str]."""
    refs = []
    for img in parsed.get("images", []):
        if img.get("url"):
            refs.append(img["url"])
        elif img.get("content_b64"):
            refs.append(img["content_b64"])   # голый base64 — image-analyzer умеет
    return refs


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    parsed = None
    if request.raw_email and PARSER_URL:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(PARSER_URL, json={
                    "raw_email": request.raw_email,
                    "include_inline_content": True,      # чтобы CID-картинки доехали байтами
                    "max_inline_image_bytes": 500_000,
                })
                r.raise_for_status()
                parsed = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            parsed = None
            logger.warning("parser-service unavailable: %s", exc)

    if parsed:
        text = parsed.get("text") or ""
        image_refs = _collect_image_refs(parsed)
    else:
        # legacy-путь: старый локальный парсер, как работало раньше
        legacy = parse_html(request.html or "")
        text = legacy.text
        image_refs = [
            i["url"] if isinstance(i, dict) else getattr(i, "url", i)
            for i in legacy.images
        ]

    # 1. Analyze text locally
    text_result = analyze_text(text)

    # 2. Send images to image-analyzer over HTTP
    try:
        image_result = await analyze_images(image_refs)
    except Exception as exc:
        # Soft-fail: still return text + decision with empty image signal
        image_result = {
            "score": 0.0,
            "reason": f"image-analyzer unavailable: {exc}",
            "images": [],
        }

    # 3. Final decision
    decision = make_decision(text_result=text_result, image_result=image_result)

    return AnalyzeResponse(
        text_analysis=text_result,
        image_analysis=image_result,
        decision=decision,
    )