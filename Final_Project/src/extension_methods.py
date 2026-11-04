"""Geometry and depth extraction for the separately registered H7/H8 studies."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader
from .validation import distances, output_features, sha


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def write_npz(path, **arrays):
    """Exclusive artifact creation: existing outputs are never overwritten."""
    with Path(path).open('xb') as handle:
        np.savez(handle, **arrays)


def fit_geometry(train_features, train_labels, centroids=None, mahalanobis=False):
    x = np.asarray(train_features)
    y = np.asarray(train_labels)
    if x.ndim != 3 or x.shape[1] != 3 or len(x) != len(y) or set(np.unique(y)) != set(range(10)):
        raise ValueError('Expected aligned CNN-training features for three members and ten classes')
    if centroids is None:
        centroids = np.stack([x[y == k].mean(0) for k in range(10)], axis=1)
    if centroids.shape != (3, 10, x.shape[-1]):
        raise ValueError('Invalid centroid shape')
    result = dict(centroids=centroids)
    if mahalanobis:
        covariance, cholesky, shrinkage = [], [], []
        for m in range(3):
            residuals = x[:, m].astype(np.float64) - centroids[m, y].astype(np.float64)
            fitted = LedoitWolf(assume_centered=True, store_precision=False).fit(residuals)
            covariance.append(fitted.covariance_)
            cholesky.append(scipy.linalg.cholesky(fitted.covariance_, lower=True, check_finite=True))
            shrinkage.append(fitted.shrinkage_)
        result.update(covariance=np.stack(covariance), cholesky=np.stack(cholesky), shrinkage=np.asarray(shrinkage))
    return result


def raw_distances(features, predictions, geometry):
    x, predicted = np.asarray(features), np.asarray(predictions)
    if x.ndim != 3 or predicted.shape != (len(x),) or not np.isin(predicted, np.arange(10)).all():
        raise ValueError('Expected features and predicted classes, without labels')
    if 'cholesky' not in geometry:
        return distances(x, predicted, geometry['centroids'])
    result = np.empty((len(x), 3), dtype=np.float64)
    for m in range(3):
        delta = x[:, m].astype(np.float64) - geometry['centroids'][m, predicted].astype(np.float64)
        whitened = scipy.linalg.solve_triangular(geometry['cholesky'][m], delta.T, lower=True, check_finite=True)
        result[:, m] = np.linalg.norm(whitened, axis=0)
    return result


def fit_normalization(detector_features, detector_logits, geometry):
    d = raw_distances(detector_features, output_features(detector_logits)['prediction'], geometry)
    return dict(geometry, mean=d.mean(0), std=d.std(0))


def make_features(ids, logits, features, geometry):
    """Evaluation labels are deliberately excluded from this API."""
    out = output_features(logits)
    d = raw_distances(features, out['prediction'], geometry)
    z = (d - geometry['mean']) / (geometry['std'] + 1e-8)
    table = pd.DataFrame(dict(id=ids, **out, mean_novelty=z.mean(1), novelty_variation=z.std(1), max_novelty=z.max(1)))
    for m in range(3):
        table[f'model_{m+1}_novelty'] = z[:, m]
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite novelty features')
    return table


def add_targets(table, labels):
    result = table.copy()
    result['label'] = np.asarray(labels)
    result['error'] = (result.prediction.to_numpy() != np.asarray(labels)).astype(int)
    return result


class StageCapture:
    """Temporary hooks leave the original model and checkpoint interface intact."""
    def __init__(self, model):
        self.values = {}
        self.handles = []
        for stage in ('stage1', 'stage2', 'stage4'):
            layer = getattr(model.network, stage.replace('stage', 'layer'))
            def capture(_module, _args, output, key=stage):
                self.values[key] = output.mean(dim=(-2, -1))
            self.handles.append(layer.register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()


def extract_stages(dataset, ids, models, device, original, path, register_sha):
    """One joint pass for early stages, with original logits/features as a checksum."""
    import json
    path = Path(path)
    meta = path.with_suffix('.json')
    if path.exists():
        metadata = json.loads(meta.read_text())
        if metadata['registration_sha256'] != register_sha or metadata['sha256'] != sha(path):
            raise ValueError('Depth cache checksum or registration mismatch')
        cached = load_npz(path)
        np.testing.assert_array_equal(cached['id'], ids)
        return cached
    np.testing.assert_array_equal(original['id'], ids)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    hooks = [StageCapture(model) for model in models]
    values = {stage: [] for stage in ('stage1', 'stage2')}
    max_logits_difference = max_features_difference = 0.
    offset = 0
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device)
                outputs = [model(images, return_features=True) for model in models]
                logits = torch.stack([out[0] for out in outputs], 1).cpu().numpy()
                late = torch.stack([hook.values['stage4'] for hook in hooks], 1).cpu().numpy()
                end = offset + len(images)
                np.testing.assert_array_equal(labels.numpy(), original['label'][offset:end])
                np.testing.assert_allclose(logits, original['logits'][offset:end], rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(late, original['features'][offset:end], rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(output_features(logits)['prediction'], output_features(original['logits'][offset:end])['prediction'])
                max_logits_difference = max(max_logits_difference, float(np.abs(logits-original['logits'][offset:end]).max()))
                max_features_difference = max(max_features_difference, float(np.abs(late-original['features'][offset:end]).max()))
                for stage in values:
                    values[stage].append(torch.stack([h.values[stage] for h in hooks], 1).cpu().numpy())
                offset = end
    finally:
        for hook in hooks:
            hook.close()
    if offset != len(ids):
        raise ValueError('Incomplete depth extraction')
    result = dict(id=np.asarray(ids), **{k: np.concatenate(v) for k, v in values.items()})
    for stage, dim in [('stage1', 64), ('stage2', 128)]:
        if result[stage].shape != (len(ids), 3, dim) or not np.isfinite(result[stage]).all():
            raise ValueError('Invalid stage shape or features')
    write_npz(path, **result)
    with meta.open('x') as handle:
        json.dump(dict(registration_sha256=register_sha, sha256=sha(path), wall_seconds=time.perf_counter()-start,
                       max_logits_difference=max_logits_difference, max_stage4_difference=max_features_difference), handle, indent=2)
    return result
