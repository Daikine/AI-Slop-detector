"""Local Qwen embeddings plus semantic LLM configured in .env."""
import asyncio
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'qwen3-embedding-local'
MODEL_KEY = 'text-embedding-qwen3-embedding-0.6b'
LOCAL_MODEL_DIR = ROOT / 'model_local_qwen_llm'


def configure():
    import config  # Load the user's chat API key, URL and model from .env first.
    os.environ.update(EMBED_API_URL='http://127.0.0.1:1234/v1', EMBED_API_KEY='lm-studio',
                      EMBED_MODEL=MODEL_ID, SEM_CONC='2',
                      EMB_BATCH='8', EMB_CONC='1', API_TIMEOUT='120', API_RETRIES='1',
                      DETECTOR_MODEL_DIR=LOCAL_MODEL_DIR.name)
    importlib.reload(config)


def ensure_semantic():
    from config import API_KEY, MODEL
    from api import semantic_many
    if not API_KEY:
        raise RuntimeError('Set API_KEY in .env for the semantic LLM.')
    values = asyncio.run(semantic_many([('body', 'Please send the revised meeting agenda tomorrow.')],
                                      verbose=True))
    if values[0] is None:
        raise RuntimeError('LLM did not return a valid score. Check API_KEY, API_URL and MODEL in .env.')
    print(f'OK: semantic LLM {MODEL}; score={values[0]:.3f}.', flush=True)


def ensure_embedding():
    import httpx
    lms = shutil.which('lms')
    if not lms:
        candidate = Path.home() / '.lmstudio/bin/lms.exe'
        lms = str(candidate) if candidate.is_file() else None
    if not lms:
        raise RuntimeError('LM Studio CLI not found. Install/launch LM Studio first.')
    base = os.environ['EMBED_API_URL']
    with httpx.Client(timeout=120, trust_env=False) as client:
        try:
            response = client.get(base + '/models')
            response.raise_for_status()
        except httpx.TransportError:
            subprocess.run([lms, 'server', 'start', '--port', '1234', '--bind', '127.0.0.1'], check=True)
        loaded = subprocess.run([lms, 'ps', '--json'], capture_output=True, text=True, check=True)
        if not any(m.get('identifier') == MODEL_ID for m in json.loads(loaded.stdout)):
            subprocess.run([lms, 'load', MODEL_KEY, '--identifier', MODEL_ID,
                            '--gpu', 'max', '--context-length', '4096', '-y'], check=True)
        response = client.post(base + '/embeddings', json={
            'model': MODEL_ID, 'input': ['This is a local embedding connection test.']})
        response.raise_for_status()
        vector = response.json()['data'][0]['embedding']
        if len(vector) != 1024:
            raise RuntimeError(f'Expected 1024 dimensions, received {len(vector)}')
        print(f'OK: local Qwen embeddings, {len(vector)} dimensions.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'train', 'text', 'analyze', 'server'])
    args, rest = parser.parse_known_args()
    os.chdir(ROOT)
    configure()
    ensure_embedding()
    if args.command in {'check', 'train'}:
        ensure_semantic()
    if args.command == 'check':
        return
    if args.command == 'train':
        if '--offline' in rest or '--no-llm' in rest:
            parser.error('Local Qwen training requires embeddings; omit --offline/--no-llm.')
        if '--no-semantic' in rest:
            parser.error('This profile requires semantic LLM calibration.')
        import sqlite3
        LOCAL_MODEL_DIR.mkdir(exist_ok=True)
        source = ROOT / 'model_local_qwen/cache.sqlite3'
        target = LOCAL_MODEL_DIR / 'cache.sqlite3'
        if source.exists() and not target.exists():
            with sqlite3.connect(source) as old, sqlite3.connect(target) as new:
                old.backup(new)
        sys.argv = ['train.py', *rest]
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
        saved = json.loads((LOCAL_MODEL_DIR / 'centroids.json').read_text(encoding='utf-8'))
        if saved['backend'] != 'api' or saved['model'] != MODEL_ID:
            raise RuntimeError('Training did not produce Qwen embedding centroids.')
        thresholds = json.loads((LOCAL_MODEL_DIR / 'thresholds.json').read_text(encoding='utf-8'))
        if not thresholds['engine'].get('with_llm'):
            raise RuntimeError('LLM calibration failed; with_llm thresholds are missing.')
    else:
        path = LOCAL_MODEL_DIR / 'centroids.json'
        if not path.exists():
            raise RuntimeError('Train first: local_qwen.py train')
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['backend'] != 'api' or saved['model'] != MODEL_ID:
            raise RuntimeError('Local model is incompatible. Run local_qwen.py train again.')
        if args.command == 'server':
            import uvicorn
            uvicorn.run('server:app', host='127.0.0.1', port=8000)
        else:
            script = 'test_my_text.py' if args.command == 'text' else 'analyze.py'
            sys.argv = [script, *rest]
            runpy.run_path(str(ROOT / script), run_name='__main__')


if __name__ == '__main__':
    main()
