"""HTTP transport from the main app to image-analyzer."""
from __future__ import annotations

import os
import httpx

IMAGE_ANALYZER_URL = os.getenv("IMAGE_ANALYZER_URL", "http://localhost:8001/analyze")


async def analyze_images(images: list[str]) -> dict:
    if not images:
        return {"score": 0.0, "status": "no_images", "reason": "no images provided", "images": []}
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(IMAGE_ANALYZER_URL, json={"images": images[:8]})
        response.raise_for_status()
        return response.json()
