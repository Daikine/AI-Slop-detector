"""Text AI-slop analyzer: emails.raw -> analysis.text (DistilBERT)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from predictor import TextPredictor  # noqa: E402  (лежит рядом)

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
MODEL_DIR = os.getenv("MODEL_DIR", str(ROOT / "ML/models/text_detector"))


async def main() -> None:
    predictor = TextPredictor(MODEL_DIR)
    consumer = AIOKafkaConsumer(
        "emails.raw", bootstrap_servers=KAFKA, group_id="text-analyzer"
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA, value_serializer=lambda v: json.dumps(v).encode()
    )
    await consumer.start()
    await producer.start()
    print(f"Text analyzer: emails.raw -> analysis.text (model: {MODEL_DIR})", flush=True)
    try:
        async for msg in consumer:
            try:
                data = json.loads(msg.value.decode())
                text = data.get("text") or data.get("text_part") or ""
                score, features, reason = predictor.predict(text)
                await producer.send_and_wait(
                    "analysis.text",
                    {
                        "task_id": data["task_id"],
                        "analyzer_type": "text",
                        "score": score,
                        "features": features,
                        "reason": reason,
                        "analyzed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as exc:
                print(f"[text] skipped malformed message: {exc}", flush=True)
    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
