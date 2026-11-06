"""Reproducible pre-test pipeline: python -m src.validation --device cuda."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
import pandas as pd
import scipy.optimize
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .data import MEAN, STD, split_indices
from .metrics import metrics
from .models import CIFARResNet18
from .train import seed_everything

OUTPUT = ['confidence', 'entropy', 'margin', 'mutual_information']
FEATURES = {key: OUTPUT + ['mean_novelty', 'novelty_variation', 'max_novelty'][:n]
            for key, n in zip('ABCD', range(4))}


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def probabilities(logits, temperature=1.):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def output_features(logits):
    """Label-free: shape N x M x K. Average probabilities, never logits."""
    p = probabilities(logits)
    mean = p.mean(1)
    entropy = lambda q: -(q * np.log(np.maximum(q, 1e-300))).sum(-1)
    top = np.sort(mean, axis=1)
    predictions = p.argmax(-1)
    return dict(confidence=top[:, -1], entropy=entropy(mean), margin=top[:, -1] - top[:, -2],
                mutual_information=entropy(mean) - entropy(p).mean(1),
                prediction=mean.argmax(1), unanimous=(predictions == predictions[:, :1]).all(1))


def distances(features, predictions, centroids):
    """No labels accepted: each network uses its own predicted-class centroid."""
    return np.linalg.norm(features - centroids[:, predictions, :].transpose(1, 0, 2), axis=-1)


def extract(config, root, device):
    dataset = datasets.CIFAR10(config['data_dir'], train=True, download=False,
                              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    if not Path(config['split_path']).exists():
        raise FileNotFoundError('Use the saved training split; extraction cannot create a replacement')
    splits = split_indices(dataset.targets, config['split_path'], config['split_seed'])
    checkpoints = [Path(config['output_dir']) / f'seed_{s}/best.pt' for s in (1, 2, 3)]
    provenance = dict(checkpoints={str(p): sha(p) for p in checkpoints}, split_sha256=sha(config['split_path']),
                      preprocessing=dict(mean=MEAN, std=STD), torch=torch.__version__, numpy=np.__version__)
    models = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        saved = checkpoint['config']
        assert saved['split_seed'] == config['split_seed']
        assert saved['normalization_mean'] == list(MEAN) and saved['normalization_std'] == list(STD)
        model = CIFARResNet18().to(device).eval()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        models.append(model)
    assert len(set(provenance['checkpoints'].values())) == 3, 'Identical checkpoint files'
    # Check actual weights, since serialized metadata alone can differ.
    for i in range(3):
        for j in range(i):
            assert any(not torch.equal(a, b) for a, b in zip(models[i].parameters(), models[j].parameters()))
    manifest = root / 'cache_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError('Cache provenance changed; choose a new output directory')
    manifest.write_text(json.dumps(provenance, indent=2))
    caches = {}
    for name, ids in splits.items():
        path = root / f'{name}.npz'
        if path.exists():
            with np.load(path) as saved:
                caches[name] = {k: saved[k] for k in saved.files}
            np.testing.assert_array_equal(caches[name]['id'], ids)
            np.testing.assert_array_equal(caches[name]['label'], np.asarray(dataset.targets)[ids])
            continue
        loader = DataLoader(Subset(dataset, ids.tolist()), batch_size=128, shuffle=False, num_workers=2)
        logits, features, labels = [], [], []
        with torch.inference_mode():
            for batch, target in loader:
                outputs = [model(batch.to(device), return_features=True) for model in models]
                logits.append(torch.stack([x[0] for x in outputs], 1).cpu().numpy())
                features.append(torch.stack([x[1] for x in outputs], 1).cpu().numpy())
                labels.append(target.numpy())
        cache = dict(id=ids, label=np.concatenate(labels), logits=np.concatenate(logits), features=np.concatenate(features))
        assert cache['features'].shape == (len(ids), 3, 512)
        assert np.isfinite(cache['features']).all() and np.isfinite(cache['logits']).all()
        cache['probabilities'] = probabilities(cache['logits']).astype(np.float32)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, **cache)
        temporary.replace(path)
        caches[name] = cache
        print(f'Cached {name}: {len(ids)} images', flush=True)
    return caches


def fit_mlp(x, y, xv, yv, ids, seed, epochs):
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)
    xv = torch.tensor(xv, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best, state, patience, best_epoch = float('inf'), None, 0, 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x), generator=generator).split(256):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = model(xv).sigmoid().numpy().ravel()
        value = metrics(yv, score, ids)['aurc']
        if value < best:
            best, state, patience, best_epoch = value, copy.deepcopy(model.state_dict()), 0, epoch + 1
        else:
            patience += 1
        if patience >= 20:
            break
    model.load_state_dict(state)
    with torch.no_grad():
        score = model(xv).sigmoid().numpy().ravel()
    return state, score, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ensemble.json')
    parser.add_argument('--output', default='results/validation')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    seed_everything(42)
    torch.set_num_threads(2)
    config = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    caches = extract(config, root, args.device)
    train, detector, val = [caches[k] for k in ('train', 'detector_train', 'val')]
    centroids = np.stack([train['features'][train['label'] == k].mean(0) for k in range(10)], axis=1)
    predicted = {k: output_features(v['logits']) for k, v in caches.items()}
    raw = {k: distances(v['features'], predicted[k]['prediction'], centroids) for k, v in caches.items()}
    mean, std = raw['detector_train'].mean(0), raw['detector_train'].std(0)
    np.savez(root / 'novelty_stats.npz', centroids=centroids, mean=mean, std=std)
    tables = {}
    for key, cache in caches.items():
        standardized = (raw[key] - mean) / (std + 1e-8)
        table = pd.DataFrame(dict(id=cache['id'], label=cache['label'], **predicted[key],
                                  mean_novelty=standardized.mean(1), novelty_variation=standardized.std(1),
                                  max_novelty=standardized.max(1)))
        table['error'] = (table.prediction != table.label).astype(int)
        for m in range(3):
            table[f'model_{m+1}_prediction'] = cache['logits'][:, m].argmax(1)
            table[f'model_{m+1}_novelty'] = standardized[:, m]
        table.to_csv(root / f'{key}_features.csv', index=False)
        tables[key] = table
    vt, dt = tables['val'], tables['detector_train']
    predictions = val['logits'].argmax(2)
    disagreements = {f'{i+1}_{j+1}': int((predictions[:, i] != predictions[:, j]).sum()) for i in range(3) for j in range(i)}
    assert all(count > 0 for count in disagreements.values()), 'Members make identical validation predictions'
    verification = dict(model_accuracy=[float((predictions[:, m] == val['label']).mean()) for m in range(3)],
                        ensemble_accuracy=float(1 - vt.error.mean()), pairwise_disagreements=disagreements,
                        unanimous_fraction=float(vt.unanimous.mean()))
    (root / 'ensemble_verification.json').write_text(json.dumps(verification, indent=2))
    y, yv, ids = dt.error.to_numpy(), vt.error.to_numpy(), vt.id.to_numpy()
    rows, scores = [], {}
    def record(name, score):
        scores[name] = score
        rows.append(dict(detector=name, **metrics(yv, score, ids)))
    for name, sign in zip(OUTPUT, [-1, 1, -1, 1]):
        record(name, sign * vt[name].to_numpy())
    def nll(log_t):
        p = probabilities(val['logits'], np.exp(log_t)).mean(1)
        return -np.log(np.maximum(p[np.arange(len(yv)), val['label']], 1e-300)).mean()
    fitted = scipy.optimize.minimize_scalar(nll, bounds=(-4, 4), method='bounded')
    if not fitted.success:
        raise RuntimeError('Temperature fitting failed')
    temperature = float(np.exp(fitted.x))
    calibrated = probabilities(val['logits'], temperature).mean(1)
    record('temperature_confidence', -calibrated.max(1))
    selections = {}
    for key, columns in FEATURES.items():
        assert not {'label', 'error', 'id'} & set(columns)
        scaler = StandardScaler().fit(dt[columns])
        x, xv = scaler.transform(dt[columns]), scaler.transform(vt[columns])
        lr = LogisticRegression(C=1., max_iter=2000, random_state=42).fit(x, y)
        joblib.dump(dict(model=lr, scaler=scaler, features=columns), root / f'{key}_lr.joblib')
        record(f'{key}_lr', lr.predict_proba(xv)[:, 1])
        candidates = []
        for seed in (11, 22, 33):
            state, score, epoch = fit_mlp(x, y, xv, yv, ids, seed, args.epochs)
            torch.save(dict(state_dict=state, features=columns, mean=scaler.mean_, scale=scaler.scale_, seed=seed, epoch=epoch), root / f'{key}_mlp_{seed}.pt')
            record(f'{key}_mlp_{seed}', score)
            candidates.append((rows[-1]['aurc'], seed))
        selected = min(candidates)[1]
        selections[key] = selected
        record(f'{key}_mlp_selected', scores[f'{key}_mlp_{selected}'])
        print(f'{key}: validation-selected MLP seed {selected}', flush=True)
    pd.DataFrame(rows).to_csv(root / 'validation_metrics.csv', index=False)
    pd.DataFrame(dict(id=ids, error=yv, **scores)).to_csv(root / 'validation_scores.csv', index=False)
    vt[['mean_novelty', 'novelty_variation', 'max_novelty', 'mutual_information']].corr().to_csv(root / 'feature_correlations.csv')
    freeze = dict(temperature=temperature, temperature_argmax_changes=int((calibrated.argmax(1) != vt.prediction).sum()),
                  selected_mlp_seeds=selections, epochs=args.epochs, learning_rate=.001, batch_size=256,
                  patience=20, lr_C=1., feature_sets=FEATURES, test_evaluated=False,
                  checkpoint_selection_caveat='Saved CNNs selected by minimum validation loss, not maximum accuracy.',
                  novelty_reference='Centroids: CNN train; population mean/std: detector train; novelty variation ddof=0.',
                  tie_break='Ascending risk, then original CIFAR training image ID',
                  files={p.name: sha(p) for p in root.glob('*') if p.suffix in ('.pt', '.joblib', '.npz')})
    (root / 'frozen_protocol.json').write_text(json.dumps(freeze, indent=2))
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
