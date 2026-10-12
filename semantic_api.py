import os
import json
import re
import httpx
from typing import Dict, Any

def get_neural_semantic_score(text: str, subject: str = "") -> Dict[str, Any]:
    api_key = os.getenv("NEURAL_API_KEY")
    api_url = os.getenv("NEURAL_API_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("NEURAL_MODEL", "gpt-4o-mini")
    
    if not api_key:
        return {
            "score": 0.0, 
            "signals": ["api_key_missing"], 
            "reason": "NEURAL_API_KEY не задан в .env",
            "method": "neural-api-fallback"
        }

   
    truncated_text = text[:3000] 
    
    prompt = f"""Ты - детектор AI-текста (AI Slop Detector). Проанализируй текст на вероятность того, что он сгенерирован ИИ.
Тема письма: {subject}
Текст: {truncated_text}

Верни ТОЛЬКО валидный JSON без markdown-оберток (без ```json) в следующем формате:
{{
  "score": 0.85,
  "signals": ["шаблонные фразы", "отсутствие конкретики", "водные переходы"],
  "reason": "Текст содержит много общих фраз и стандартные переходы, характерные для LLM."
}}
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"} 
    }
    
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            
            content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content, flags=re.IGNORECASE)
            
            result = json.loads(content)
            return {
                "score": float(result.get("score", 0.0)),
                "signals": result.get("signals", []),
                "reason": result.get("reason", "Неизвестная причина"),
                "method": "neural-api-v1"
            }
    except Exception as e:
        return {
            "score": 0.0, 
            "signals": [f"api_error: {str(e)}"], 
            "reason": "Ошибка вызова API нейросети",
            "method": "neural-api-error"
        }
