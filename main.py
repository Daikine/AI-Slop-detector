"""Main application: parse → text analysis → image analysis → decision."""
from __future__ import annotations

from fastapi import FastAPI

from decision_engine import make_decision
from image_client import analyze_images
from models import AnalyzeRequest, AnalyzeResponse
from parser import parse_html
from text_analyzer import analyze_text

app = FastAPI(
    title="AI-Slop-detector",
    version="0.2.0",
    description="Simplified two-app architecture: local text analysis + remote image analysis.",
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    # 1. Parse HTML
    parsed = parse_html(request.html)

    # 2. Analyze text locally
    text_result = analyze_text(parsed.text)

    # 3. Send images to image-analyzer over HTTP
    try:
        image_result = await analyze_images(parsed.images)
    except Exception as exc:
        # Soft-fail: still return text + decision with empty image signal
        image_result = {
            "score": 0.0,
            "reason": f"image-analyzer unavailable: {exc}",
            "images": [],
        }

    # 4. Final decision
    decision = make_decision(text_result=text_result, image_result=image_result)

    return AnalyzeResponse(
        text_analysis=text_result,
        image_analysis=image_result,
        decision=decision,
    )
