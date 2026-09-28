cat > README.md <<'READMEEOF'
# AI-Slop-detector

Multi-signal analysis system for detecting AI-generated and AI-assisted email spam.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Makefile Commands](#makefile-commands)
- [Project Structure](#project-structure)
- [Reliability Features](#reliability-features)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

AI-Slop-detector is a multi-signal analysis system designed to detect AI-generated
and AI-assisted email spam.

Instead of relying solely on traditional text classification, the system analyzes
a message across multiple independent signals:

- email text
- HTML structure
- embedded images
- links and metadata

Each signal is scored independently, then all scores are combined into a single
explainable verdict that indicates the likelihood of AI-assisted origin.

The system is built as an additional analytical layer on top of an existing
anti-spam pipeline: it does not just output a class, but explains **which elements**
of the message look AI-generated and **how confident** the system is.

### Research Foundation

The detection methodology is based on recent academic research:

- *Do Spammers Dream of Electric Sheep?* — IMC 2025
- *Machine Learning and Watermarking for Accurate Detection of AI-Generated Phishing Emails*
- *SpearBot*
- *Phish-Master*

## Key Features

### Multi-Signal Analysis

The system combines several independent analyzers instead of relying on a single
classifier. Each analyzer runs as an independent service and publishes its result
to its own Kafka topic:

| Analyzer | Topic | What it detects |
|---|---|---|
| **Text Analyzer** | `analysis.text` | LLM-specific writing patterns, style inconsistencies |
| **HTML Analyzer** | `analysis.html` | AI-generated boilerplate, unusual DOM structures |
| **Image Analyzer** | `analysis.images` | GenAI artifacts, OCR/content mismatches |
| **Link/Metadata Analyzer** | `analysis.links-meta` | Anchor/URL mismatch, suspicious domains |

### Explainable AI (XAI)

The Decision Engine produces a structured, explainable verdict for every message.
All individual analyzer scores are preserved, so an analyst can see **why** a
message was flagged.

**Example verdict stored in PostgreSQL:**

~~~json
{
  "task_id": "6237d5f5-5b1e-4ce8-ac14-05b20ad86dec",
  "verdict": "PROBABLY_HUMAN",
  "score": 0.35,
  "analyzer_count": 4,
  "individual_scores": [0.332, 0.581, 0.19, 0.297],
  "decided_at": "2026-09-21T14:41:45.104644+00:00"
}
~~~

**Verdict scale:**

| Average score | Verdict |
|---|---|
| 0.0 – 0.2 | `DEFINITELY_HUMAN` |
| 0.2 – 0.4 | `PROBABLY_HUMAN` |
| 0.4 – 0.6 | `MIXED` |
| 0.6 – 0.8 | `PROBABLY_AI` |
| 0.8 – 1.0 | `DEFINITELY_AI` |

## Architecture

~~~mermaid
graph TD
    Client["CLIENT: POST /api/v1/analyze"]
    Gateway["API GATEWAY (FastAPI, task_id)"]
    MockAnalyzer["MOCK TEXT ANALYZER (placeholder for ML)"]
    E2E["E2E TEST HARNESS (e2e_test.py)"]
    Aggregator["AGGREGATOR (4-of-N, dedup, timeout, DLQ)"]
    Decision["DECISION ENGINE (verdicts + explainability)"]
    PG["PostgreSQL verdicts"]
    Output["OUTPUT: verdicts.final"]

    Client --> Gateway
    Gateway -->|emails.raw| MockAnalyzer
    MockAnalyzer -->|analysis.text| Aggregator
    E2E -->|analysis.html / images / links-meta| Aggregator
    Aggregator -->|analysis.aggregated| Decision
    Decision -->|verdicts.final| Output
    Decision --> PG
~~~

**Data flow:**

~~~
POST /analyze -> api-gateway -> emails.raw -> text analyzer --+
                                                              |   +------------+
   e2e harness -> analysis.html ------------------------------+-->| AGGREGATOR |
              ->  analysis.images -----------------------------+   +-----+------+
              ->  analysis.links-meta                          |         |
                                                               |         v  analysis.aggregated
                                                       (text from mock)
                                                                         +-----------------+
                                                               |         | DECISION ENGINE |
                                                               |         +---+---------+---+
                                                      dead-letter-queue <--+         |
                                                                                v
                                                       verdicts.final + PostgreSQL
~~~

### Service Communication

All services communicate exclusively through Kafka topics — no direct calls.
Message schemas are fixed contracts defined in `shared/models/`, which allows
the ML team to plug their real analyzer in place of the mock **without any
changes to backend code**.

## Tech Stack

| Category | Technology |
|---|---|
| Message Broker | Apache Kafka 3.9 |
| Primary Database | PostgreSQL 15 |
| Cache / Task State | Redis 7 |
| API Framework | FastAPI |
| Language | Python 3.13 |
| Containerization | Docker Compose |
| Kafka Clients | aiokafka |

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Git
- Make (optional but recommended)

### 1. Clone the repository

~~~bash
git clone https://github.com/kosavka223/AI-Slop-detector.git
cd AI-Slop-detector
~~~

### 2. Set up the Python environment

~~~bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pydantic aiokafka redis asyncpg httpx
~~~

### 3. Start the full stack

~~~bash
make up-all
~~~

This builds and starts **7 containers**: Kafka, PostgreSQL, Redis,
api-gateway, aggregator, decision-engine and mock-analyzer.

**Wait ~40 seconds after startup — services need time to connect to Kafka.**

### 4. Verify

~~~bash
make status
curl http://localhost:8000/health
~~~

### 5. Run an end-to-end test

~~~bash
python tools/e2e_test.py
~~~

Then check the verdict in PostgreSQL:

~~~bash
docker exec -it slop-postgres psql -U dev -d slop -P pager=off -c \
  "SELECT email_id, overall_risk, ai_assistance_score, decided_at FROM verdicts ORDER BY decided_at DESC LIMIT 5;"
~~~

## API Reference

Interactive Swagger UI is available at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze` | Submit text for analysis. Returns `task_id` (async processing) |
| `GET` | `/api/v1/status/{task_id}` | Get task status and verdict when ready |
| `GET` | `/health` | Service health check |

**Example:**

~~~bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Some text to analyze..."}'
~~~

## Makefile Commands

| Command | Description |
|---|---|
| `make up-all` | Start the full stack (infra + all services) |
| `make up` | Start infrastructure only (Kafka, PostgreSQL, Redis) |
| `make status` | Show container status |
| `make logs` | Tail logs of all backend services |
| `make psql` | Open PostgreSQL console |
| `make down` | Stop everything |

## Project Structure

~~~
AI-Slop-detector/
├── docker-compose.yml          # Full system: 7 containers
├── Dockerfile.service          # Shared image for all services
├── Makefile                    # make up-all / status / logs / psql / down
├── init.sql                    # Database schema (verdicts table)
├── services/                   # Backend microservices
│   ├── api-gateway/            # HTTP entry point, Kafka producer, Redis task store
│   ├── aggregator/             # 4-of-N result collection, dedup, timeouts, DLQ
│   └── decision-engine/        # Verdict logic + PostgreSQL persistence
├── shared/                     # Contracts shared across teams
│   ├── models/                 # Topic names, message schemas (pydantic)
│   └── proto/                  # gRPC contract stub for the ML team
├── tools/
│   ├── mock_analyzer.py        # Mock text analyzer (placeholder for ML service)
│   └── e2e_test.py             # Automated end-to-end pipeline test
└── ML/                         # ML team workspace
~~~

## Reliability Features

| Concern | Solution |
|---|---|
| Analyzer never responds | Aggregation timeout (30 s) → partial result or DLQ |
| Malformed message in a topic | Parsed safely, sent to Dead Letter Queue, service keeps running |
| Duplicate analyzer results | Deduplication in the aggregator task buffer |
| Kafka temporarily unavailable | restart: on-failure — Docker restarts the service, messages preserved in topics |
| Client blocking on slow analysis | Async pattern: task_id + polling /status endpoint |

## Roadmap

### Phase 1 — Core Backend (DONE)

- Multi-service event-driven pipeline on Kafka
- API Gateway with async task processing
- Aggregator with deduplication, timeouts and DLQ
- Decision Engine with explainable verdicts persisted to PostgreSQL
- Full Dockerization, automated e2e testing

### Phase 2 — ML Integration

- Replace mock analyzer with a real ML service (same Kafka contract)
- Implement analyzers per gRPC contract (shared/proto/analyzer.proto)
- Model quality evaluation: precision, recall, F1-score, ROC-AUC

### Phase 3 — Production Hardening

- Kafka healthchecks + ordered startup
- Horizontal scaling of analyzers (Kafka partitions)
- Metrics and observability

## License

This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for the complete license text.

## Support

- Issues: https://github.com/kosavka223/AI-Slop-detector/issues
- Contact: m.gavrilenko@g.nsu.ru
READMEEOF
