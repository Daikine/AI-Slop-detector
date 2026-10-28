"""Анализ .eml: 4 зоны -> API-сигналы+TF-IDF -> дерево -> decision engine."""
import argparse
import asyncio
import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from api import SignalCache, api_signals
from config import API_KEY, EMBED_API_KEY, MODEL_DIR
from scoring import (embedding_score, features, final_score, full_semantic_coverage,
                     scores_for_email)
from segments import KINDS, parse_eml, split_email

RU = {'subject': 'тема', 'greeting': 'приветствие',
      'body': 'тело', 'signature': 'подпись'}


@lru_cache(maxsize=1)
def load_model():
    names = ('tfidf.pkl', 'tree.pkl', 'thresholds.json', 'centroids.json')
    missing = [name for name in names if not (MODEL_DIR / name).exists()]
    if missing:
        raise RuntimeError('Не найдены модели: ' + ', '.join(missing) + '. Сначала: python train.py')
    # Никогда не загружать чужой/непроверенный pickle: Python pickle может выполнять код.
    with (MODEL_DIR / 'tfidf.pkl').open('rb') as f:
        tfidf = pickle.load(f)
    with (MODEL_DIR / 'tree.pkl').open('rb') as f:
        tree = pickle.load(f)
    thresholds = json.loads((MODEL_DIR / 'thresholds.json').read_text(encoding='utf-8'))
    centers = json.loads((MODEL_DIR / 'centroids.json').read_text(encoding='utf-8'))
    return tfidf, tree, thresholds, centers


def score_email(raw: bytes | str):
    tfidf, tree, thresholds, centroids = load_model()
    subject, body = parse_eml(raw)
    segs = split_email(subject, body)
    present = [(k, segs[k]) for k in KINDS if segs[k].strip()]
    if not present:
        return {'subject': subject, 'verdict': 'UNCERTAIN', 'reason': 'empty_message',
                'segments': {k: {'present': False, 'score': None} for k in KINDS},
                'final_score': None, 'flagged_segments': []}
    texts = [text for _, text in present]
    predictions = tfidf['clf'].predict_proba(tfidf['vec'].transform(texts))[:, 1]
    tf_scores = dict(zip((k for k, _ in present), map(float, predictions)))

    wants_embedding = centroids['backend'] == 'api'
    # API вызовы запускаются параллельно. HTTP-сервер вызывает score_email в threadpool.
    if API_KEY or EMBED_API_KEY:
        with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
            embed_api, sem_api = asyncio.run(api_signals([segs], want_embedding=wants_embedding,
                                                          want_semantic=bool(thresholds['engine'].get('with_llm')),
                                                          cache=cache))
    else:
        embed_api, sem_api = {}, {}
    em_scores, sem_scores = {}, {}
    for kind, text in present:
        c = centroids['centroids']
        em_scores[kind] = embedding_score(embed_api.get((kind, text)),
                                          c.get(f'{kind}_1'), c.get(f'{kind}_0'))
        sem_scores[kind] = sem_api.get((kind, text))
    scores = scores_for_email(segs, tf_scores, em_scores, sem_scores)
    model_features = np.asarray([features(scores)])
    tree_p = float(tree.predict_proba(model_features)[0][1])

    engine = thresholds['engine']
    mode = 'with_llm' if engine.get('with_llm') and full_semantic_coverage(scores) else 'no_llm'
    final = final_score(scores, tree_p, mode)
    cuts = engine[mode]
    warnings = []
    # Независимый fallback после отказа LLM используется только в обученном no_llm-режиме.
    if centroids['backend'] == 'api' and any(
        scores[k]['embedding'] is None and
        centroids['centroids'].get(f'{k}_1') is not None and
        centroids['centroids'].get(f'{k}_0') is not None for k, _ in present
    ):
        warnings.append('API эмбеддингов недоступен: распределение признаков изменилось. Ручная проверка.')
    if API_KEY and mode == 'no_llm':
        warnings.append('Семантика API недоступна или не калибрована: использован no_llm режим.')
    verdict = ('HUMAN' if final < cuts['human_cut'] else
               'AI' if final >= cuts['ai_cut'] else 'UNCERTAIN')
    if warnings and centroids['backend'] == 'api' and any(s['embedding'] is None for s in scores.values() if s['present']):
        verdict = 'UNCERTAIN'

    out = {}
    flagged = []
    for kind in KINDS:
        item = scores[kind]
        if not item['present']:
            out[kind] = {'present': False, 'score': None}
            continue
        seg_mode = mode if mode == 'no_llm' else (
            'with_llm' if item['semantic_llm'] is not None else 'no_llm')
        threshold = thresholds['segments'][kind].get(seg_mode)
        value = item[seg_mode]
        label = ('AI' if threshold is not None and value >= threshold else
                 'HUMAN' if threshold is not None else 'UNDETERMINED')
        if label == 'AI':
            flagged.append(kind)
        out[kind] = {'present': True, 'text_preview': segs[kind][:300],
                     'tfidf': item['tfidf'], 'embedding': item['embedding'],
                     'semantic_llm': item['semantic_llm'], 'score': value,
                     'threshold': threshold, 'label': label}

    certain_labels = [v['label'] for v in out.values() if v.get('label') in {'AI', 'HUMAN'}]
    if 'AI' in certain_labels and 'HUMAN' in certain_labels and verdict != 'UNCERTAIN':
        verdict = 'UNCERTAIN'
        warnings.append('У сегментов конфликтующие метки: возможно смешанное авторство.')

    return {'subject': subject, 'segments': out, 'mode': mode,
            'tree_probability': round(tree_p, 5), 'final_score': round(float(final), 5),
            'engine_thresholds': cuts, 'verdict': verdict,
            'flagged_segments': flagged, 'warnings': warnings,
            'api_key_set': bool(API_KEY), 'embedding_backend': centroids['backend'],
            'note': 'Scores/пороги не являются достоверной вероятностью авторства ИИ.'}


def fmt(value):
    return f'{value:.3f}' if value is not None else '   — '


def main():
    parser = argparse.ArgumentParser(description='Детектор AI-авторства .eml')
    parser.add_argument('file', help='путь к письму .eml или - для stdin')
    parser.add_argument('--json', action='store_true', help='машиночитаемый JSON')
    args = parser.parse_args()
    raw = sys.stdin.buffer.read() if args.file == '-' else Path(args.file).read_bytes()
    result = score_email(raw)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f'Тема: {result["subject"] or "(нет)"}')
    print(f'Режим: {result.get("mode", "нет")}; embeddings: {result.get("embedding_backend", "нет")}')
    print(f'{"СЕГМЕНТ":<16} {"TF-IDF":>9} {"EMBED":>9} {"LLM":>9} {"SCORE":>9} {"ПОРОГ":>9} МЕТКА')
    for k in KINDS:
        v = result['segments'][k]
        if not v['present']:
            print(f'{RU[k]:<16} (зона отсутствует)')
            continue
        print(f'{RU[k]:<16} {fmt(v["tfidf"]):>9} {fmt(v["embedding"]):>9} '
              f'{fmt(v["semantic_llm"]):>9} {fmt(v["score"]):>9} '
              f'{fmt(v["threshold"]):>9} {v["label"]}')
    print(f'Decision Tree: {fmt(result.get("tree_probability"))}')
    print(f'Final score: {fmt(result.get("final_score"))}')
    print(f'ВЕРДИКТ: {result["verdict"]}')
    for warning in result.get('warnings', []):
        print('[!] ' + warning)
    print('Оценка вероятностная и не доказывает реальное авторство.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        sys.exit(f'Ошибка: {exc}')
