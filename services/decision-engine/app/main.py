"""Decision Engine: analysis.aggregated -> verdicts.final + Postgres."""
import asyncio
import json
import os
from datetime import datetime, timezone

import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dev:dev@localhost:5435/slop")
INPUT_TOPIC = "analysis.aggregated"
OUTPUT_TOPIC = "verdicts.final"
DLQ_TOPIC = "dead-letter-queue"

# Пороги для вердикта (средний скор: 0 = человек, 1 = AI)
THRESHOLDS = [
    (0.2, "DEFINITELY_HUMAN"),
    (0.4, "PROBABLY_HUMAN"),
    (0.6, "MIXED"),
    (0.8, "PROBABLY_AI"),
    (1.1, "DEFINITELY_AI"),
]
def make_reasons(results: list) -> list:
    reasons = []
    for r in results:
        score = r.get("score", 0)
        atype = r.get("analyzer_type", "?")
        if score > 0.5:
            reasons.append(f"{atype}: обнаружены AI-подобные признаки (score={score})")
        else:
            reasons.append(f"{atype}: признаков AI-генерации нет (score={score})")
    return reasons

def make_verdict(task_id: str, results: list) -> dict:
    scores = [r.get("score", 0.0) for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0

    verdict = next(v for th, v in THRESHOLDS if avg < th)

    return {
        "task_id": task_id,
        "verdict": verdict,
        "score": round(avg, 3),
        "analyzer_count": len(results),
        "individual_scores": scores,
        "reasons": make_reasons(results),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


async def save_to_pg(pool, v: dict):
    """Сохраняем вердикт в Postgres (схема: email_id, overall_risk, ai_assistance_score...)."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO verdicts
                    (email_id, overall_risk, ai_assistance_score,
                     classification, cluster_id, full_verdict, decided_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                """,
                v["task_id"],          # email_id
                v["verdict"],          # overall_risk
                v["score"],            # ai_assistance_score
                v["verdict"],          # classification
                None,                  # cluster_id — пока не используем
                json.dumps(v),         # full_verdict (JSONB)
            )
        print(f"[decision] task={v['task_id']} saved to Postgres", flush=True)
    except Exception as e:
        print(f"[decision] PG save failed for {v['task_id']}: {e}", flush=True)

async def main():
    consumer = AIOKafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="decision-engine",
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await consumer.start()
    await producer.start()

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    print(f"Decision Engine: {INPUT_TOPIC} -> {OUTPUT_TOPIC} + Postgres", flush=True)

    try:
        async for msg in consumer:
            try:
                data = json.loads(msg.value.decode("utf-8"))
            except Exception as e:
                await producer.send_and_wait(DLQ_TOPIC, {
                    "topic": msg.topic, "error": str(e),
                    "raw": msg.value.decode("utf-8", errors="replace"),
                })
                print(f"[decision] bad message -> DLQ: {e}", flush=True)
                continue

            task_id = data.get("task_id")
            results = data.get("results", [])
            if not task_id or not results:
                print(f"[decision] skip malformed aggregated task: {data}", flush=True)
                continue

            verdict = make_verdict(task_id, results)

            await producer.send_and_wait(OUTPUT_TOPIC, verdict)
            await save_to_pg(pool, verdict)
            print(f"[decision] task={task_id} verdict={verdict['verdict']} "
                  f"score={verdict['score']} -> {OUTPUT_TOPIC}", flush=True)
    finally:
        await pool.close()
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())