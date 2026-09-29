# AI-Slop-detector

Multi-signal analysis for detecting AI-generated / AI-assisted email spam.

## Architecture

Two applications, no Kafka / gRPC / microservice mesh:

```text
Клиент
  │
  ▼
main.py  (:8000)
  ├── parser
  ├── text analyzer
  ├── decision engine
  └── HTTP → image-analyzer (:8001) → vision API (AI_API_KEY)
```

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env   # optional: set AI_API_KEY for real vision API
make up                # builds & starts app + image-analyzer
curl http://localhost:8000/health
```

### Local (without Docker)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r image-analyzer/requirements.txt

# terminal 1
cd image-analyzer && uvicorn main:app --port 8001

# terminal 2
export IMAGE_ANALYZER_URL=http://localhost:8001/analyze
uvicorn main:app --port 8000
```

## API

Swagger: http://localhost:8000/docs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Sync analysis of HTML → text + images + verdict |
| `GET` | `/health` | Health check |

**Example:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"html": "<p>Verify your password immediately.</p><img src=\"https://example.com/x.png\">"}'
```

**Verdict scale (average of analyzer scores):**

| Score | Verdict |
|---|---|
| 0.0 – 0.2 | `DEFINITELY_HUMAN` |
| 0.2 – 0.4 | `PROBABLY_HUMAN` |
| 0.4 – 0.6 | `MIXED` |
| 0.6 – 0.8 | `PROBABLY_AI` |
| 0.8 – 1.0 | `DEFINITELY_AI` |

## Project structure

```text
├── main.py                 # FastAPI entry: POST /analyze
├── parser.py               # HTML → text + images + links
├── text_analyzer.py        # Local text scoring
├── decision_engine.py      # Combine scores → verdict
├── image_client.py         # HTTP client → image-analyzer
├── models.py               # Pydantic models
├── requirements.txt
├── Dockerfile
├── docker-compose.yml      # app + image-analyzer
├── image-analyzer/
│   ├── main.py             # FastAPI + vision API / heuristics
│   ├── requirements.txt
│   └── Dockerfile
└── ML/                     # Training data, scripts and local baseline model artifacts
```

## Train local text model

The checked-in seed data generates reproducible synthetic samples and trains the local TF-IDF + LogisticRegression baseline used by the API. From the project root:

```bash
python3 ML/data/generate_synthetic.py
python3 ML/training/train_baseline.py
```

The training script writes model artifacts to `ML/models/` and evaluation metrics to `ML/models/baseline_metrics.json`; model binary files are intentionally excluded from Git. `make up` mounts those local artifacts into the app container. The built-in dataset is small and synthetic, so its reported metrics are not an estimate of real-world performance. To add optional HC3 samples, install Hugging Face `datasets` and run `python3 ML/data/download_datasets.py` before training.

## Makefile

| Command | Description |
|---|---|
| `make up` | Build & start both containers |
| `make status` | `docker compose ps` |
| `make logs` | Tail app + image-analyzer logs |
| `make down` | Stop stack |

## E2E smoke test

With the stack running:

```bash
python tools/e2e_test.py
```

## License

GNU GPL v3.0 — see [LICENSE](LICENSE).
