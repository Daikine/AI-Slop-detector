"""Orchestrator: MIME/HTML parsing -> text -> vision -> decision."""
from __future__ import annotations

import base64
import html
import logging
import os

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from decision_engine import make_decision
from image_client import analyze_images
from models import AnalyzeRequest, AnalyzeResponse, AnalyzerResult
from parser import parse_html
from text_analyzer import analyze_text

logger = logging.getLogger("orchestrator")
PARSER_URL = os.getenv("PARSER_URL")
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

app = FastAPI(
    title="AI-Slop-detector", version="0.3.0",
    description="Email text and real image uploads with explainable AI-content indicators.",
)


@app.get("/health")
async def health():
    return {"status": "healthy", "note": "check image-analyzer:8001/health for vision API status"}


def _collect_image_refs(parsed: dict) -> list[str]:
    """Prefer the actual inline MIME bytes over remote links/CID identifiers."""
    refs: list[str] = []
    for img in parsed.get("images", []):
        if not isinstance(img, dict):
            continue
        content = img.get("content_b64")
        url = img.get("url")
        mime = str(img.get("content_type") or "image/jpeg").lower()
        if content and mime in ALLOWED_IMAGE_MIMES:
            refs.append(f"data:{mime};base64,{content}")
        elif isinstance(url, str) and url.startswith(("http://", "https://", "data:image/")):
            refs.append(url)
    return list(dict.fromkeys(refs))[:8]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    parsed = None
    subject = None

    if request.raw_email:
        if not PARSER_URL:
            raise HTTPException(status_code=503, detail="PARSER_URL is not configured")
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(PARSER_URL, json={
                    "raw_email": request.raw_email,
                    "include_inline_content": True,
                    "max_inline_image_bytes": 2_000_000,
                })
                response.raise_for_status()
                parsed = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("parser-service failed: %s", exc)
            raise HTTPException(status_code=502, detail="MIME parser unavailable or email invalid") from exc

    if parsed is not None:
        text = parsed.get("text") or ""
        subject = (parsed.get("headers") or {}).get("subject")
        image_refs = _collect_image_refs(parsed)
    else:
        legacy = parse_html(request.html or "")
        text = legacy.text
        image_refs = [
            item if isinstance(item, str) else item.get("url", "")
            for item in legacy.images
        ]

    image_refs = list(dict.fromkeys(image_refs + request.images))[:8]
    has_text = bool(text.strip() or (subject and subject.strip()))
    text_result = (
        analyze_text(text or subject or "", subject=subject)
        if has_text
        else AnalyzerResult(score=0.0, reason="no text provided", features={"status": "no_text"})
    )

    try:
        image_result = await analyze_images(image_refs)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("image-analyzer failed: %s", exc)
        image_result = {
            "score": 0.0, "status": "error",
            "reason": "image-analyzer service unavailable", "images": [],
        }

    decision = make_decision(text_result, image_result, text_available=has_text)
    return AnalyzeResponse(
        text_analysis=text_result,
        image_analysis=image_result,
        decision=decision,
    )


@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_uploaded_image(
    file: UploadFile = File(..., description="JPEG, PNG, WebP or GIF image"),
    text: str = Form("", description="Optional email/body text"),
):
    """Real photo upload from Swagger, Postman, curl or a frontend."""
    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=415, detail="Supported: JPG, PNG, WebP, GIF")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be <= 6 MiB")
    ref = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return await analyze(AnalyzeRequest(html=html.escape(text), images=[ref]))
