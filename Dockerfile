FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py parser.py text_analyzer.py decision_engine.py image_client.py models.py ./
COPY ML/preprocess.py ./ML/preprocess.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
