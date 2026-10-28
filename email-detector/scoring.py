"""Единая математика для train и analyze: никаких разных формул."""
import math

from segments import KINDS


def mean_present(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def embedding_score(vec, ai_center, human_center):
    """Псевдо-скор из нормализованных косинусов; НЕ калиброванная вероятность."""
    if vec is None or ai_center is None or human_center is None:
        return None
    if len(vec) != len(ai_center) or len(vec) != len(human_center):
        return None
    similarity_ai = sum(a * b for a, b in zip(vec, ai_center))
    similarity_human = sum(a * b for a, b in zip(vec, human_center))
    d = max(-50., min(50., 8. * (similarity_ai - similarity_human)))
    return 1 / (1 + math.exp(-d))


def scores_for_email(segs, tf_scores, emb_scores, sem_scores):
    result = {}
    for kind in KINDS:
        present = bool(segs[kind].strip())
        tf = tf_scores.get(kind) if present else None
        em = emb_scores.get(kind) if present else None
        sm = sem_scores.get(kind) if present else None
        result[kind] = {'present': present, 'tfidf': tf, 'embedding': em,
                        'semantic_llm': sm, 'no_llm': mean_present((tf, em)),
                        'with_llm': mean_present((tf, em, sm))}
    return result


def features(scores):
    no_llm = [scores[k]['no_llm'] for k in KINDS]
    tfidf = [scores[k]['tfidf'] for k in KINDS]
    flags = [float(scores[k]['present']) for k in KINDS]
    return ([x if x is not None else 0. for x in no_llm] + flags +
            [mean_present(no_llm) or 0., mean_present(tfidf) or 0.])


def mean_zones(scores, mode):
    return mean_present([scores[k][mode] for k in KINDS])


def full_semantic_coverage(scores):
    return all(not scores[k]['present'] or scores[k]['semantic_llm'] is not None for k in KINDS)


def final_score(scores, tree_prob, mode):
    avg = mean_zones(scores, mode)
    if avg is None:
        return None
    return avg if tree_prob is None else (avg + float(tree_prob)) / 2


def feature_names():
    return ([f'score_{k}' for k in KINDS] + [f'has_{k}' for k in KINDS] +
            ['mean_score', 'mean_tf'])
