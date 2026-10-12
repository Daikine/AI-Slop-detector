# Architecture Overview

The application retains the existing service boundaries: the MIME parser extracts an email, the main application performs local text scoring and combines signals, and the image analyzer handles image analysis. **Semantic text profiling is part of `text_analyzer.py` and is not a separate service.**

```text
Client
  │
  ▼
main.py (FastAPI :8000)
  ├── parser service (:8002)
  │     └── headers, MIME body, links, images, attachments
  ├── TextPredictor (in-process)
  │     ├── DistilBERT (optional local weights)
  │     ├── TF-IDF + LogisticRegression (optional local artifacts)
  │     ├── legacy keyword fallback
  │     └── semantic profile (offline, explainable rules)
  ├── decision engine
  └── HTTP → image-analyzer (:8001)
                    └── configured vision API or local fallback
```

## Data flow

`POST /analyze` → MIME parse (when `raw_email` is provided and the parser is available) or legacy HTML parsing → text model/fallback + integrated semantic signals → image analysis → decision → JSON response.

The parser supplies the email subject and text body to the text pipeline. Text analysis adds an explainable `features.semantic_analysis` object without changing the existing top-level response model. Segment entries may represent the subject, body paragraphs/sentences, and closing signature; each has a source span, semantic style score, label, and reasons.

## Semantic profile

The profile is fully offline and uses only the Python standard library. Signals include formulaic phrases, generic wording, repeated/near-repeated sentences, sentence-length regularity, and subject/body term overlap. Subject/body overlap is reported as a relevance feature and does not by itself imply AI authorship.

A trained text model remains the primary predictor and the semantic profile acts as supplemental evidence. If no trained model is mounted, the existing fallback remains in place and the semantic score can supply a stronger signal. These rules are heuristic indicators rather than a calibrated model; legitimate marketing or corporate emails can use similar language.

## Environment

| Variable | Where | Purpose |
|---|---|---|
| `PARSER_URL` | app | MIME parser endpoint (`http://parser:8002/parse` in Docker) |
| `IMAGE_ANALYZER_URL` | app | Image analyzer endpoint (`http://image-analyzer:8001/analyze` in Docker) |
| `BASELINE_MODEL` | app | Optional TF-IDF/LogisticRegression model pickle |
| `BASELINE_VEC` | app | Optional TF-IDF vectorizer pickle |
| `MODEL_DIR` | app | Optional local transformer weights |
| `AI_API_KEY` | image-analyzer | Optional key for a vision API |
| `AI_API_URL` | image-analyzer | OpenAI-compatible chat completions endpoint |
| `AI_MODEL` | image-analyzer | Vision model name |

## Docker

```bash
docker compose up -d --build
```

The Compose stack has three services: `app` (:8000), `image-analyzer` (:8001), and `parser` (:8002). Semantic analysis adds no service or model-download requirement.
