"""Настройки. Переменные окружения имеют приоритет над локальным .env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for raw_line in (ROOT / '.env').read_text(encoding='utf-8').splitlines() if (ROOT / '.env').is_file() else []:
    line = raw_line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.split('=', 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"\'')

API_KEY = os.getenv('API_KEY', '').strip()
if API_KEY.lower() in {'', 'sk-...', 'sk-xxx', 'your_api_key_here', 'changeme'}:
    API_KEY = ''
API_URL = os.getenv('API_URL', 'https://deepcode.ci.nsu.ru/api/v1').rstrip('/')
EMBED_API_URL = os.getenv('EMBED_API_URL', '').rstrip('/')
EMBED_API_KEY = os.getenv('EMBED_API_KEY', '').strip()
MODEL = os.getenv('MODEL', 'Qwen3.8-Flash-Next')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
SEM_BATCH = max(1, int(os.getenv('SEM_BATCH', '8')))
SEM_CONC = max(1, min(32, int(os.getenv('SEM_CONC', '8'))))
EMB_BATCH = max(1, int(os.getenv('EMB_BATCH', '32')))
EMB_CONC = max(1, min(32, int(os.getenv('EMB_CONC', '6'))))
API_TIMEOUT = max(5., float(os.getenv('API_TIMEOUT', '35')))
API_RETRIES = max(0, min(5, int(os.getenv('API_RETRIES', '2'))))
MODEL_DIR = ROOT / os.getenv('DETECTOR_MODEL_DIR', 'model')
DATA_DIR = ROOT / 'data'
