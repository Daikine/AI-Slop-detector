# AI-Slop-detector

Multi-signal analysis for detecting AI-generated / AI-assisted email content.

## Architecture

The project keeps semantic text analysis inside the existing local `TextPredictor`—there is no separate semantic service and no extra container to deploy.

```text
Client
  │
  ▼
main.py (:8000)
  ├── parser service (:8002) — MIME, headers, body, images, links
  ├── TextPredictor — trained model / legacy fallback + semantic signals
  ├── decision engine — combines text and image scores
  └── HTTP → image-analyzer (:8001) — vision API or local fallback
```

### Semantic signals

The text pipeline preserves its current priority: local transformer weights, then TF-IDF + LogisticRegression artifacts, then the legacy keyword fallback. After that existing score is produced, an offline semantic profile adds explainable content/style evidence to the same `text_analysis` result:

- formulaic or boilerplate phrases in Russian and English;
- generic wording with few concrete details;
- repeated or near-repeated sentences;
- unusually uniform sentence lengths;
- subject-to-body term overlap as a separate relevance feature;
- per-segment metadata for the subject, body paragraphs/sentences and closing signature, including spans, labels, scores and reasons.

When a trained text model is available, semantic evidence only supplements its score and cannot lower it. When it is not available, the higher of the legacy fallback and the semantic profile is used. All semantic rules run locally with the Python standard library; there is no model download, external API request or new runtime dependency required for this feature.

The semantic score is a heuristic signal, not a calibrated probability or proof that a person did or did not use AI. Corporate templates, newsletters and automatic notifications can naturally contain formulaic language, so results should be interpreted alongside other signals and reviewed before high-impact decisions.

## Quick start

### Docker (recommended)

```bash
cp .env.example .env   # optional: set AI_API_KEY for the external vision API
make up                # builds and starts app + parser + image-analyzer
curl http://localhost:8000/health
```

The stack can run without `AI_API_KEY`; the image analyzer then falls back to its local logic. A trained text model is optional as well.

### Local (without Docker)

Run these in separate terminals from the repository root:

```bash
# Terminal 1 — MIME parser
python -m pip install -r services/parser/requirements.txt
cd services/parser && uvicorn main:app --host 127.0.0.1 --port 8002
```

```bash
# Terminal 2 — image analyzer
python -m pip install -r image-analyzer/requirements.txt
cd image-analyzer && uvicorn main:app --host 127.0.0.1 --port 8001
```

```bash
# Terminal 3 — main application
python -m pip install -r requirements.txt
# Run from the project root. Set these URLs in the shell before starting:
# Linux/macOS:
export PARSER_URL=http://127.0.0.1:8002/parse
export IMAGE_ANALYZER_URL=http://127.0.0.1:8001/analyze
uvicorn main:app --host 127.0.0.1 --port 8000
```

On PowerShell, use `$env:PARSER_URL="http://127.0.0.1:8002/parse"` and `$env:IMAGE_ANALYZER_URL="http://127.0.0.1:8001/analyze"` instead of `export`.

## API

Swagger: http://localhost:8000/docs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Analyze HTML or a raw MIME email, including text and images |
| `GET` | `/health` | Main application health check |
| `GET` | `http://localhost:8002/health` | MIME parser health check |

**Example:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"html": "<p>Verify your password immediately.</p><img src=\"https://example.com/x.png\">"}'
```

The semantic profile is included in `text_analysis.features.semantic_analysis`; the existing top-level response shape remains unchanged. For each returned segment, `span` is a character range into that segment's source (`subject` or message body), `generation_probability` is the rule-based semantic score, and `signals` / `reason` explain which patterns contributed.

**Verdict scale** (average of the text and image scores):

| Score | Verdict |
|---|---|
| 0.0 – 0.2 | `DEFINITELY_HUMAN` |
| 0.2 – 0.4 | `PROBABLY_HUMAN` |
| 0.4 – 0.6 | `MIXED` |
| 0.6 – 0.8 | `PROBABLY_AI` |
| 0.8 – 1.0 | `DEFINITELY_AI` |

The names are inherited from the existing API and describe detector scores, not verified authorship.

## Project structure

```text
├── main.py                 # FastAPI entry: POST /analyze
├── parser.py               # Legacy HTML → text + images + links fallback
├── text_analyzer.py        # Existing text model plus integrated semantic profile
├── decision_engine.py      # Combine scores
├── image_client.py         # HTTP client → image-analyzer
├── models.py               # Main API Pydantic models
├── services/parser/        # MIME parser service and MIME metadata extractors
├── image-analyzer/         # Vision analysis / local fallback
├── tests/                  # Unit and MIME parser tests
└── ML/                     # Training data, scripts and optional local model artifacts
```

## Train the local text model

The checked-in seed data generates synthetic samples and trains the local TF-IDF + LogisticRegression baseline. From the project root:

```bash
python ML/data/generate_synthetic.py
python ML/training/train_baseline.py
```

The training script writes model artifacts to `ML/models/`; binary files are intentionally excluded from Git. Docker Compose mounts the local artifacts into the app container. The dataset is synthetic and small, so its metrics are not an estimate of real-world performance.

## Tests

Run the regression suite from the project root:

```bash
python -m pytest -q
```

## Makefile

| Command | Description |
|---|---|
| `make up` | Build and start all three services |
| `make status` | `docker compose ps` |
| `make logs` | Tail application and analyzer logs |
| `make down` | Stop the stack |

## License

GNU GPL v3.0 — see [LICENSE](LICENSE).
