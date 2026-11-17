# Подготовка Qwen-версии к коммиту

## Конфигурация

1. Скопируй `.env.example` в `.env`.
2. Заполни `API_KEY` настоящим ключом, который выдала команда NSU.
3. Оставь `API_URL` и `MODEL` из шаблона, если команда не выдала другие значения.
4. Не добавляй `.env` в Git. В репозитории хранится только `.env.example` с безопасным placeholder.

## Архитектура

- Эмбеддинги: Qwen3-Embedding-0.6B локально через LM Studio.
- Чатовая оценка и необязательная сегментация: Qwen Chat через OpenAI-compatible API.
- Клиент: официальный Python SDK `openai`.
- Ручной ввод: всё письмо одной строкой, затем Enter.
- Стандартная Qwen v2-модель не требует TF-IDF и не создаёт фиктивный `tfidf.pkl`.
- `tree.pkl`, `centroids.json`, `thresholds.json` и `test_metrics.json` — обученные артефакты; SQLite-кэш не нужен для воспроизведения модели.

## Перед коммитом

- Убедись, что `.env` не отображается в `git status`.
- Не добавляй `.venv/`, `.venv-local/`, `__pycache__/`, `*.sqlite3*`.
- Если коммитишь уже обученную гибридную модель, добавь `model_hybrid_qwen/tree.pkl`, `centroids.json`, `thresholds.json`, `test_metrics.json`.
- Не добавляй `model_hybrid_qwen/tfidf.pkl`: в Qwen-only режиме этот файл не нужен.
- Запусти `pytest -q` и проверь `git diff --cached --check` перед коммитом.

Пример сообщения коммита: `feat: add Qwen hybrid email detection`.
