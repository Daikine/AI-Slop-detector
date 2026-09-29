"""HTTP client for the separate image-analyzer service."""
from __future__ import annotations

import os

import httpx

IMAGE_ANALYZER_URL = os.getenv(
    "IMAGE_ANALYZER_URL",
    "http://localhost:8001/analyze",
)


async def analyze_images(images: list[str]) -> dict:
    """Send image refs (URLs or data-URIs) to image-analyzer over HTTP."""
    if not images:
        return {
            "score": 0.0,
            "reason": "no images in message",
            "images": [],
        }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            IMAGE_ANALYZER_URL,
            json={"images": images},
        )
        response.raise_for_status()
        return response.json()
