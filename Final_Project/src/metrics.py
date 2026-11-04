"""Selective classification: risk increases with score; ties use original image ID."""
import numpy as np
from sklearn.metrics import roc_auc_score


def risk_curve(errors, scores, ids):
    errors, scores, ids = map(np.asarray, (errors, scores, ids))
    if errors.ndim != 1 or errors.shape != scores.shape or ids.shape != errors.shape or not len(errors):
        raise ValueError('Expected nonempty aligned vectors')
    if not np.isfinite(scores).all() or not np.isin(errors, [0, 1]).all():
        raise ValueError('Invalid risks or binary errors')
    order = np.lexsort((ids, scores))
    return np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)


def metrics(errors, scores, ids):
    curve = risk_curve(errors, scores, ids)
    return dict(aurc=float(curve.mean()),
                auroc=float(roc_auc_score(errors, scores)) if len(np.unique(errors)) == 2 else None,
                risk80=float(curve[int(np.ceil(.8 * len(curve))) - 1]),
                risk90=float(curve[int(np.ceil(.9 * len(curve))) - 1]))
