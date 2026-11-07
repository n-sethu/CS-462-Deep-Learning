"""CPU-only calculations on cached scores. Ties are broken by source image ID."""
import numpy as np

def ranked(errors, scores, ids, coverage):
    errors, scores, ids = map(np.asarray, (errors, scores, ids))
    if not (0 < coverage <= 1) or len(errors) == 0:
        raise ValueError('Coverage must be in (0, 1] with nonempty data')
    if errors.shape != scores.shape or ids.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Expected aligned finite scores')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Source IDs must be unique within a condition')
    order = np.lexsort((ids, scores))
    kept = int(np.ceil(coverage * len(errors)))
    accepted = np.zeros(len(errors), dtype=bool)
    accepted[order[:kept]] = True
    curve = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return dict(accepted=accepted, kept=kept, errors=int(errors[accepted].sum()), risk=float(curve[kept-1]), aurc=float(curve.mean()), curve=curve)

def choose_examples(errors, ids, accept_a, accept_b):
    """Deterministic examples at current coverage; absent categories are disclosed."""
    masks = [
        ('Novelty success', (errors == 1) & accept_a & ~accept_b),
        ('Novelty regression', (errors == 1) & ~accept_a & accept_b),
        ('Both miss an error', (errors == 1) & accept_a & accept_b),
        ('Both accept correctly', (errors == 0) & accept_a & accept_b),
    ]
    used=set();result=[]
    for name,mask in masks:
        candidates=sorted(np.flatnonzero(mask),key=lambda j:int(ids[j]))
        chosen=next((int(j) for j in candidates if int(j) not in used),None)
        if chosen is not None:used.add(chosen)
        result.append((name,chosen))
    return result
