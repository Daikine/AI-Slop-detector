"""HTML structure analyzer: emails.raw -> analysis.html."""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")


class StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: dict[str, int] = {}
        self.max_depth = self.depth = 0
        self.inline_styles = 0
        self.external_resources = 0
        self.hidden_elements = 0
        self.forms = 0
        self.scripts = 0
        self.tailwind_classes = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.tags[tag] = self.tags.get(tag, 0) + 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        values = {k.lower(): (v or "") for k, v in attrs}
        if "style" in values:
            self.inline_styles += 1
        if tag in {"img", "link", "script", "iframe"}:
            self.external_resources += 1
        if tag == "form":
            self.forms += 1
        if tag == "script":
            self.scripts += 1
        if "hidden" in values or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", values.get("style", ""), re.I):
            self.hidden_elements += 1
        classes = values.get("class", "").split()
        self.tailwind_classes += sum(1 for c in classes if re.match(r"(?:sm:|md:|lg:)?(?:text|bg|p|m|flex|grid|rounded|font)-", c))

    def handle_endtag(self, tag: str) -> None:
        self.depth = max(0, self.depth - 1)


def analyze_html(html: str) -> tuple[float, dict, str]:
    parser = StructureParser()
    try:
        parser.feed(html or "")
    except Exception:
        pass
    total = sum(parser.tags.values())
    signals = {
        "tag_count": total, "max_dom_depth": parser.max_depth,
        "inline_style_count": parser.inline_styles,
        "external_resource_count": parser.external_resources,
        "hidden_element_count": parser.hidden_elements,
        "form_count": parser.forms, "script_count": parser.scripts,
        "tailwind_class_count": parser.tailwind_classes,
    }
    points = 0.0
    reasons = []
    if parser.tailwind_classes:
        points += .30; reasons.append("Tailwind-like utility classes in email HTML")
    if parser.scripts or parser.forms:
        points += .25; reasons.append("active or interactive HTML elements")
    if parser.max_depth > 12:
        points += .15; reasons.append("unusually deep DOM")
    if parser.inline_styles > 40 or total > 180:
        points += .15; reasons.append("unusually complex inline layout")
    if parser.hidden_elements > 3:
        points += .10; reasons.append("multiple hidden elements")
    if not reasons:
        reasons.append("no strong anomalous HTML structure signals")
    return round(min(points, 1.0), 3), signals, "; ".join(reasons)


async def main() -> None:
    consumer = AIOKafkaConsumer("emails.raw", bootstrap_servers=KAFKA, group_id="html-analyzer")
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA, value_serializer=lambda v: json.dumps(v).encode())
    await consumer.start(); await producer.start()
    print("HTML analyzer: emails.raw -> analysis.html", flush=True)
    try:
        async for msg in consumer:
            try:
                data = json.loads(msg.value.decode())
                score, features, reason = analyze_html(data.get("html") or data.get("html_part") or "")
                await producer.send_and_wait("analysis.html", {"task_id": data["task_id"], "analyzer_type": "html", "score": score, "features": features, "reason": reason, "analyzed_at": datetime.now(timezone.utc).isoformat()})
            except Exception as exc:
                print(f"[html] skipped malformed message: {exc}", flush=True)
    finally:
        await consumer.stop(); await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
