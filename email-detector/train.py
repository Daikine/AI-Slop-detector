"""4-stage fit: base TF-IDF/centroids -> stack decision tree -> calibration -> held-out test.

Не используем тест при выборе порогов. LLM на обучающем base вообще не нужен;
на больших датасетах вызовы API ограничены выборками в каждом этапе.
"""
import argparse
import asyncio
import csv
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, export_text

from api import SignalCache, api_signals, embedding_many
from config import API_KEY, EMBED_API_KEY, DATA_DIR, MODEL_DIR
from scoring import (embedding_score, feature_names, features, final_score,
                     full_semantic_coverage, scores_for_email)
from segments import KINDS, parse_eml, split_email


def load_data(csv_path=None):
    if csv_path:
        path = Path(csv_path)
    else:
        path = DATA_DIR / 'dataset.csv'
    rows = []
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            col = next((key for key in ('is_ai', 'ai_assisted', 'label') if key in fields), None)
            if col is None or not fields.intersection({'text', 'body'}):
                raise ValueError('CSV должен содержать text (или body) и is_ai (или ai_assisted / label)')
            if col == 'label' and 'source' in fields:
                print('[!] label может означать СПАМ, не ИИ! Проверь корректность разметки.')
            if col == 'ai_assisted' and 'label' in fields:
                print('[i] выбран ai_assisted как метка ИИ; поле label игнорируется.')
            for line in reader:
                raw_label = str(line.get(col, '')).strip().lower()
                text = str(line.get('text') or line.get('body') or '').strip()
                if not text or raw_label not in {'0', '1', 'true', 'false', 'ai', 'human', 'yes', 'no'}:
                    continue
                label = int(raw_label in {'1', 'true', 'ai', 'yes'})
                rows.append((str(line.get('subject') or '').strip(), text, label))
        print(f'[data] {path}: {len(rows)} строк')
    else:
        if csv_path:
            raise FileNotFoundError(f'Нет файла: {path}')
        for folder, label in (('human', 0), ('ai', 1)):
            for file in sorted((DATA_DIR / folder).rglob('*.eml')):
                subject, body = parse_eml(file.read_bytes())
                if body:
                    rows.append((subject, body, label))
        if not rows:
            raise RuntimeError('Нет размеченных писем. Укажи --csv data/demo_synthetic.csv для '
                               'ДЕМОНСТРАЦИИ или добавь data/dataset.csv / data/human/*.eml + data/ai/*.eml')
    # Убираем точные и очевидные нормализованные дубликаты ДО разделения.
    unique = {}
    contradictory = 0
    for subject, body, label in rows:
        fingerprint = re.sub(r'\s+', ' ', subject + '\n' + body).strip().casefold()
        if fingerprint in unique:
            if unique[fingerprint][2] != label:
                contradictory += 1
                unique.pop(fingerprint)
            continue
        unique[fingerprint] = (subject, body, label)
    rows = list(unique.values())
    if contradictory:
        print(f'[!] найдено {contradictory} противоречивых дубликатов; исключены')
    labels = Counter(x[2] for x in rows)
    print(f'[data] после дедупликации: {len(rows)} (human={labels[0]}, ai={labels[1]})')
    if min(labels[0], labels[1]) < 20:
        raise RuntimeError('Для обучающей/калибровочной/тестовой частей нужно >=20 уникальных '
                           'размеченных писем каждого класса; лучше сотни.')
    return rows


def split_rows(rows, seed, stack_limit, calib_limit, test_limit):
    rng = np.random.default_rng(seed)
    split = {'base': [], 'stack': [], 'calib': [], 'test': []}
    for label in (0, 1):
        idx = np.array([i for i, row in enumerate(rows) if row[2] == label])
        rng.shuffle(idx)
        n = len(idx)
        n_stack = min(stack_limit, max(4, int(round(n * .20))))
        n_calib = min(calib_limit, max(4, int(round(n * .10))))
        n_test = min(test_limit, max(4, int(round(n * .10))))
        if n_stack + n_calib + n_test >= n - 5:
            raise RuntimeError('Слишком мало писем для независимых разделов. Нужно >=20 каждого класса.')
        cursor = 0
        for key, count in (('stack', n_stack), ('calib', n_calib), ('test', n_test)):
            split[key].extend(rows[i] for i in idx[cursor:cursor + count])
            cursor += count
        split['base'].extend(rows[i] for i in idx[cursor:])
    for rows_sub in split.values():
        rng.shuffle(rows_sub)
    return split


def fit_tfidf(rows):
    texts, labels = [], []
    for subject, body, label in rows:
        for text in split_email(subject, body).values():
            if text.strip():
                texts.append(text)
                labels.append(label)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                          max_features=30000, token_pattern=r'(?u)\b\w+\b')
    x = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=350, class_weight='balanced', solver='liblinear', random_state=42)
    clf.fit(x, labels)
    print(f'[tfidf] {len(texts)} зон, {x.shape[1]} признаков')
    return {'vec': vec, 'clf': clf}


def normalize_center(vectors):
    avg = np.asarray(vectors, dtype=float).mean(axis=0)
    norm = np.linalg.norm(avg)
    return (avg / norm).tolist() if norm > 0 else None


def fit_centroids(rows, cache, per_class):
    if not (API_KEY or EMBED_API_KEY):
        return {'backend': 'none', 'centroids': {}, 'model': None}
    from config import EMBED_MODEL
    # API availability probe; ложный hash fallback ЗАПРЕЩЁН.
    probe = asyncio.run(embedding_many(['embedding probe'], cache=cache))
    if not probe[0]:
        print('[embedding] /embeddings недоступен: signal=None, не подменяем хэшированием.')
        return {'backend': 'none', 'centroids': {}, 'model': None}
    rng = np.random.default_rng(42)
    subset = []
    for label in (0, 1):
        candidates = [row for row in rows if row[2] == label]
        chosen = rng.choice(len(candidates), size=min(per_class, len(candidates)), replace=False)
        subset.extend(candidates[i] for i in chosen)
    segs = [split_email(subject, body) for subject, body, _ in subset]
    emb, _ = asyncio.run(api_signals(segs, want_semantic=False, cache=cache, verbose=True))
    grouped = {}
    for row, zones in zip(subset, segs):
        for kind in KINDS:
            value = emb.get((kind, zones[kind]))
            if value is not None:
                grouped.setdefault((kind, row[2]), []).append(value)
    centroids = {}
    for (kind, label), values in grouped.items():
        if len(values) >= 3:
            center = normalize_center(values)
            if center is not None:
                centroids[f'{kind}_{label}'] = center
    print(f'[embedding] API-векторы: {len(emb)} уникальных зон; {len(centroids)} центроидов')
    return {'backend': 'api' if centroids else 'none', 'centroids': centroids,
            'model': EMBED_MODEL if centroids else None}


def score_rows(rows, tfidf, centroids, cache, use_sem):
    segs = [split_email(subject, body) for subject, body, _ in rows]
    pairs = [(i, kind, zones[kind]) for i, zones in enumerate(segs)
             for kind in KINDS if zones[kind].strip()]
    texts = [text for _, _, text in pairs]
    predictions = (tfidf['clf'].predict_proba(tfidf['vec'].transform(texts))[:, 1]
                   if texts else np.array([]))
    tf_by_row = [{} for _ in rows]
    for (i, kind, _), prob in zip(pairs, predictions):
        tf_by_row[i][kind] = float(prob)
    wants_emb = centroids['backend'] == 'api'
    if wants_emb or use_sem:
        em_api, sem_api = asyncio.run(api_signals(segs, want_embedding=wants_emb,
                                                 want_semantic=use_sem, cache=cache, verbose=True))
    else:
        em_api, sem_api = {}, {}
    results = []
    for i, (zones, row) in enumerate(zip(segs, rows)):
        em_scores, sem_scores = {}, {}
        for kind in KINDS:
            if not zones[kind].strip():
                continue
            cen = centroids['centroids']
            em_scores[kind] = embedding_score(em_api.get((kind, zones[kind])),
                                              cen.get(f'{kind}_1'), cen.get(f'{kind}_0'))
            sem_scores[kind] = sem_api.get((kind, zones[kind]))
        scores = scores_for_email(zones, tf_by_row[i], em_scores, sem_scores)
        results.append({'scores': scores, 'label': row[2], 'features': features(scores)})
    return results


def best_cut(labels, scores):
    """O(N log N): реальные границы между отсортированными score, максимум Youden J."""
    pairs = sorted(zip(scores, labels))
    humans = sum(label == 0 for _, label in pairs)
    ais = len(pairs) - humans
    if not humans or not ais:
        return None, None
    distinct = sorted(set(score for score, _ in pairs))
    candidates = ([max(0., distinct[0] - 1e-6)] +
                  [(a + b) / 2 for a, b in zip(distinct[:-1], distinct[1:])] +
                  [min(1., distinct[-1] + 1e-6)])
    values = []
    for cut in candidates:
        tp = sum(s >= cut and y == 1 for s, y in pairs) / ais
        fp = sum(s >= cut and y == 0 for s, y in pairs) / humans
        values.append((tp - fp, cut))
    best = max(x for x, _ in values)
    threshold = float(np.median([cut for x, cut in values if abs(x - best) < 1e-10]))
    return round(threshold, 5), round(best, 4)


def calibrate_segments(rows):
    thresholds = {}
    for kind in KINDS:
        thresholds[kind] = {}
        for mode in ('no_llm', 'with_llm'):
            subset = [(r['label'], r['scores'][kind][mode]) for r in rows
                      if r['scores'][kind]['present'] and
                      (mode == 'no_llm' or r['scores'][kind]['semantic_llm'] is not None)]
            count = Counter(y for y, _ in subset)
            if min(count[0], count[1]) >= 3:
                threshold, youden = best_cut([y for y, _ in subset], [s for _, s in subset])
                thresholds[kind][mode] = threshold
                thresholds[kind][mode + '_n'] = len(subset)
                thresholds[kind][mode + '_youden'] = youden
            elif mode == 'no_llm':
                # При отсутствии достаточных меток нельзя придумывать границу.
                thresholds[kind][mode] = None
    return thresholds


def engine_thresholds(calib, tree, mode):
    eligible = [r for r in calib if mode == 'no_llm' or full_semantic_coverage(r['scores'])]
    if min(Counter(r['label'] for r in eligible).values(), default=0) < 3 or len({r['label'] for r in eligible}) < 2:
        return None
    tree_p = tree.predict_proba(np.array([r['features'] for r in eligible]))[:, 1]
    labels = [r['label'] for r in eligible]
    finals = [final_score(r['scores'], p, mode) for r, p in zip(eligible, tree_p)]
    split, j = best_cut(labels, finals)
    human = float(np.quantile([s for s, y in zip(finals, labels) if y == 0], .90))
    ai = float(np.quantile([s for s, y in zip(finals, labels) if y == 1], .10))
    if human < ai:
        # Не делаем ВСЮ разделимую область неопределённой: берём вокруг
        # оптимума Youden узкий диапазон из 15% межквантильного зазора.
        gap = min(.15, max(.025, .15 * (ai - human)))
        lower, upper = max(human, split - gap), min(ai, split + gap)
        if lower < upper:
            human, ai = lower, upper
        else:
            human, ai = max(0., split - .025), min(1., split + .025)
    else:
        # На плохо разделимых данных оставляем narrow UNCERTAIN около Youden.
        human, ai = max(0., split - .025), min(1., split + .025)
    return {'human_cut': round(human, 5), 'ai_cut': round(ai, 5),
            'youden_cut': split, 'youden_j': j, 'n_calibration': len(eligible)}


def verdict(score, thresholds):
    return ('HUMAN' if score < thresholds['human_cut'] else
            'AI' if score >= thresholds['ai_cut'] else 'UNCERTAIN')


def evaluate(rows, tree, engine):
    report = {}
    for mode in ('no_llm', 'with_llm'):
        cut = engine.get(mode)
        if not cut:
            continue
        eligible = [r for r in rows if mode == 'no_llm' or full_semantic_coverage(r['scores'])]
        if not eligible or len({r['label'] for r in eligible}) < 2:
            continue
        probs = tree.predict_proba(np.asarray([r['features'] for r in eligible]))[:, 1]
        finals = [final_score(r['scores'], p, mode) for r, p in zip(eligible, probs)]
        y = [r['label'] for r in eligible]
        preds = [int(s >= cut['youden_cut']) for s in finals]
        decisions = [verdict(s, cut) for s in finals]
        report[mode] = {
            'n_test': len(y),
            'roc_auc': round(float(roc_auc_score(y, finals)), 4),
            'balanced_accuracy_at_youden': round(float(balanced_accuracy_score(y, preds)), 4),
            'precision_at_youden': round(float(precision_score(y, preds, zero_division=0)), 4),
            'recall_at_youden': round(float(recall_score(y, preds, zero_division=0)), 4),
            'uncertain_fraction': round(decisions.count('UNCERTAIN') / len(decisions), 4),
            'human_wrongly_flagged_ai': sum(z == 'AI' and label == 0 for z, label in zip(decisions, y)),
            'ai_wrongly_flagged_human': sum(z == 'HUMAN' and label == 1 for z, label in zip(decisions, y)),
            'confusion_at_youden': confusion_matrix(y, preds, labels=[0, 1]).tolist(),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='явный путь к CSV; иначе data/dataset.csv или .eml')
    parser.add_argument('--offline', '--no-llm', action='store_true', help='не отправлять запросы во внешние API')
    parser.add_argument('--no-semantic', action='store_true', help='API эмбеддинги оставить, LLM семантику выключить')
    parser.add_argument('--centroid-per-class', type=int, default=120)
    parser.add_argument('--stack-per-class', type=int, default=150)
    parser.add_argument('--calib-per-class', type=int, default=120)
    parser.add_argument('--test-per-class', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    t0 = time.perf_counter()
    rows = load_data(args.csv)
    parts = split_rows(rows, args.seed, args.stack_per_class, args.calib_per_class, args.test_per_class)
    print('[split]', {k: dict(Counter(row[2] for row in v)) for k, v in parts.items()})
    tfidf = fit_tfidf(parts['base'])
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    use_api = bool(API_KEY or EMBED_API_KEY) and not args.offline
    use_sem = bool(API_KEY) and not args.offline and not args.no_semantic
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        centroids = fit_centroids(parts['base'], cache, max(3, args.centroid_per_class)) if use_api else {
            'backend': 'none', 'centroids': {}, 'model': None}
        print('[stack] Decision Tree: отдельные письма, не base train')
        stack = score_rows(parts['stack'], tfidf, centroids, cache, use_sem=False)
        x_stack = np.asarray([r['features'] for r in stack])
        y_stack = np.asarray([r['label'] for r in stack])
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(3, len(stack) // 60),
                                      class_weight='balanced', random_state=args.seed)
        tree.fit(x_stack, y_stack)
        print(export_text(tree, feature_names=feature_names()))
        print('[calib] пороги на третьем независимом наборе')
        calib = score_rows(parts['calib'], tfidf, centroids, cache, use_sem=use_sem)
        segment_cuts = calibrate_segments(calib)
        engine = {'no_llm': engine_thresholds(calib, tree, 'no_llm')}
        if use_sem:
            engine['with_llm'] = engine_thresholds(calib, tree, 'with_llm')
        print('[thresholds] зоны:', json.dumps(segment_cuts, ensure_ascii=False))
        print('[thresholds] engine:', engine)
        print('[test] НОВЫЕ данные: не использовались для обучения/порогов')
        test = score_rows(parts['test'], tfidf, centroids, cache, use_sem=use_sem)
    report = evaluate(test, tree, engine)
    print('[metrics] test:', json.dumps(report, indent=2, ensure_ascii=False))
    thresholds = {'segments': segment_cuts, 'engine': engine,
                  'embedding_backend': centroids['backend'],
                  'class_mapping': {'0': 'HUMAN', '1': 'AI'}}
    for name, artifact in (('tfidf.pkl', tfidf), ('tree.pkl', tree)):
        with (MODEL_DIR / name).open('wb') as fh:
            pickle.dump(artifact, fh)
    for name, obj in (('centroids.json', centroids), ('thresholds.json', thresholds),
                      ('test_metrics.json', report)):
        (MODEL_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] {time.perf_counter() - t0:.1f} s; модели и отчёт в {MODEL_DIR}')
    print('[!] Метрики синтетического датасета не доказывают качество на реальных письмах.')


if __name__ == '__main__':
    main()
