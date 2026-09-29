"""Smoke e2e against running docker stack (app + image-analyzer)."""
from __future__ import annotations

import httpx

APP = "http://localhost:8000"


def main() -> None:
    html = """
    <html><body>
      <p>Please confirm your account password immediately.</p>
      <img src="https://example.com/banner.png">
      <a href="http://fake-bank.example/login">Click here</a>
    </body></html>
    """
    with httpx.Client(timeout=60.0) as client:
        health = client.get(f"{APP}/health")
        health.raise_for_status()
        print("health:", health.json())

        resp = client.post(f"{APP}/analyze", json={"html": html})
        resp.raise_for_status()
        data = resp.json()
        print("decision:", data["decision"])
        print("text score:", data["text_analysis"]["score"])
        print("image score:", data["image_analysis"].get("score"))


if __name__ == "__main__":
    main()
