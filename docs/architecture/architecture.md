# Architecture Overview

Упрощённая архитектура без Kafka, gRPC и отдельных микросервисов.

Система состоит из **двух приложений**:

1. **Основное приложение** (`main.py`) — парсинг HTML, анализ текста,
   принятие решения, HTTP-вызов image-analyzer.
2. **Image analyzer** — отдельный HTTP-сервис; по API-ключу обращается
   к внешней нейросети (или использует локальные эвристики, если ключа нет).

```text
Клиент
  │
  ▼
main.py  (FastAPI :8000)
  ├── parser
  ├── text analyzer
  ├── decision engine
  └── HTTP → image-analyzer (:8001)
                    │
                    ▼
             Нейросеть по AI_API_KEY
```

## Data flow

`POST /analyze` → parse HTML → analyze text locally → HTTP images →
decision → JSON response (синхронно).

## Environment

| Variable | Where | Purpose |
|---|---|---|
| `IMAGE_ANALYZER_URL` | app | URL of image-analyzer (`http://image-analyzer:8001/analyze` in Docker) |
| `AI_API_KEY` | image-analyzer | Key for external vision API |
| `AI_API_URL` | image-analyzer | OpenAI-compatible chat completions endpoint |
| `AI_MODEL` | image-analyzer | Model name (default `gpt-4o-mini`) |
| `MODEL_DIR` | app | Optional DistilBERT weights for text analysis |

## Docker

```bash
docker compose up -d --build
```

Два контейнера: `slop-app` (:8000) и `slop-image-analyzer` (:8001).
