"""Frozen evaluation only: no detector fitting and no test-dependent selection."""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .data import MEAN, STD
from .metrics import metrics, risk_curve
from .models import CIFARResNet18
from .train import seed_everything
from .validation import FEATURES, OUTPUT, distances, output_features, probabilities, sha

CORRUPTIONS = ('gaussian_noise', 'motion_blur', 'jpeg_compression')
SEVERITIES = (1, 3, 5)
SEEDS = (11, 22, 33)
MIN_ERRORS = 20  # Predeclared caution threshold, not a claim of adequate power above it.


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_artifacts(validation):
    protocol = json.loads((validation / 'frozen_protocol.json').read_text())
    manifest = json.loads((validation / 'cache_manifest.json').read_text())
    for name, digest in protocol['files'].items():
        if sha(validation / name) != digest:
            raise ValueError(f'Frozen artifact changed: {name}')
    for name, digest in manifest['checkpoints'].items():
        if sha(name) != digest:
            raise ValueError(f'CNN checkpoint changed: {name}')
    return protocol, manifest


def freeze(validation, root, resamples):
    protocol, manifest = verify_artifacts(validation)
    scores = pd.read_csv(validation / 'validation_scores.csv')
    # Score-only validation quantiles; resulting test coverage is allowed to differ.
    thresholds = {name: {str(c): float(np.sort(scores[name])[int(np.ceil(c * len(scores))) - 1])
                         for c in (.8, .9)} for name in scores.columns if name not in ('id', 'error')}
    sealed = dict(validation_protocol_sha256=sha(validation / 'frozen_protocol.json'),
                  validation_manifest_sha256=sha(validation / 'cache_manifest.json'),
                  thresholds=thresholds, selected_mlp_seeds=protocol['selected_mlp_seeds'],
                  primary_family='mlp_selected', primary_comparison='C_minus_B',
                  bootstrap_seed=20261103, bootstrap_resamples=resamples, confidence_level=.95,
                  corruptions=CORRUPTIONS, severities=SEVERITIES, min_subset_errors_for_auroc=MIN_ERRORS,
                  permutation_seed=20261104, permutations=100,
                  source_sha256={str(p): sha(p) for p in [Path(__file__), Path('src/metrics.py'), Path('src/validation.py'), Path('src/models.py')]})
    # Normalize tuples to JSON lists before comparison.
    sealed = json.loads(json.dumps(sealed))
    path = root / 'sealed_protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != sealed:
            raise ValueError('Final protocol changed: refuse to rerun with different settings')
    else:
        with path.open('x') as handle:
            json.dump(sealed, handle, indent=2)
    return protocol, manifest, sealed


class FrozenDetectors:
    def __init__(self, validation, protocol):
        self.lr, self.mlp = {}, {}
        self.selected = protocol['selected_mlp_seeds']
        for key, columns in FEATURES.items():
            artifact = joblib.load(validation / f'{key}_lr.joblib')
            if artifact['features'] != columns:
                raise ValueError('LR feature order changed')
            self.lr[key] = artifact
            for seed in SEEDS:
                saved = torch.load(validation / f'{key}_mlp_{seed}.pt', map_location='cpu', weights_only=False)
                if saved['features'] != columns or saved['seed'] != seed:
                    raise ValueError('MLP feature order/seed changed')
                model = nn.Sequential(nn.Linear(len(columns), 32), nn.ReLU(), nn.Linear(32, 1)).eval()
                model.load_state_dict(saved['state_dict'])
                self.mlp[key, seed] = (model, saved['mean'], saved['scale'])

    def predict_one(self, table, key='C', family='mlp_selected'):
        columns = FEATURES[key]
        if family == 'lr':
            art = self.lr[key]
            return art['model'].predict_proba(art['scaler'].transform(table[columns]))[:, 1]
        model, mean, scale = self.mlp[key, self.selected[key]]
        with torch.inference_mode():
            return model(torch.tensor((table[columns].to_numpy()-mean)/scale, dtype=torch.float32)).sigmoid().numpy().ravel()

    def predict(self, table):
        scores = {name: sign * table[name].to_numpy() for name, sign in zip(OUTPUT, (-1, 1, -1, 1))}
        for key, columns in FEATURES.items():
            artifact = self.lr[key]
            scores[f'{key}_lr'] = artifact['model'].predict_proba(artifact['scaler'].transform(table[columns]))[:, 1]
            for seed in SEEDS:
                model, mean, scale = self.mlp[key, seed]
                x = (table[columns].to_numpy() - mean) / scale
                with torch.inference_mode():
                    scores[f'{key}_mlp_{seed}'] = model(torch.tensor(x, dtype=torch.float32)).sigmoid().numpy().ravel()
            scores[f'{key}_mlp_selected'] = scores[f'{key}_mlp_{self.selected[key]}']
        return scores


def feature_table(cache, stats):
    features = output_features(cache['logits'])
    d = distances(cache['features'], features['prediction'], stats['centroids'])
    z = (d - stats['mean']) / (stats['std'] + 1e-8)
    table = pd.DataFrame(dict(id=cache['id'], **features, mean_novelty=z.mean(1),
                              novelty_variation=z.std(1), max_novelty=z.max(1)))
    for m in range(3):
        table[f'model_{m+1}_prediction'] = cache['logits'][:, m].argmax(1)
        table[f'model_{m+1}_novelty'] = z[:, m]
    return table  # Labels cannot enter feature construction.


class CorruptedImages(Dataset):
    def __init__(self, path, severity, labels, transform):
        self.images = np.load(path, mmap_mode='r')[(severity - 1) * 10000:severity * 10000]
        if self.images.shape != (10000, 32, 32, 3):
            raise ValueError('Invalid CIFAR-10-C image layout')
        self.labels, self.transform = labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index])


def load_models(manifest, device):
    models = []
    for path in manifest['checkpoints']:
        model = CIFARResNet18().to(device).eval()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        models.append(model)
    return models


def infer(dataset, models, device, path, provenance):
    metadata = path.with_suffix('.json')
    if path.exists():
        if not metadata.exists() or json.loads(metadata.read_text())['provenance'] != provenance:
            raise ValueError('Inference cache provenance mismatch')
        saved_meta = json.loads(metadata.read_text())
        if sha(path) != saved_meta['sha256']:
            raise ValueError('Inference cache checksum mismatch')
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}
    loader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    logits, features, labels = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device)
            outputs = [model(images, return_features=True) for model in models]
            logits.append(torch.stack([x[0] for x in outputs], 1).cpu().numpy())
            features.append(torch.stack([x[1] for x in outputs], 1).cpu().numpy())
            labels.append(target.numpy())
    cache = dict(id=np.arange(len(dataset)), label=np.concatenate(labels), logits=np.concatenate(logits), features=np.concatenate(features))
    cache['probabilities'] = probabilities(cache['logits']).astype(np.float32)
    assert cache['features'].shape == (len(dataset), 3, 512)
    assert np.isfinite(cache['features']).all() and np.isfinite(cache['logits']).all()
    elapsed = time.perf_counter() - start
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, **cache)
    temp.replace(path)
    atomic_json(metadata, dict(provenance=provenance, sha256=sha(path), inference_wall_seconds=elapsed))
    return cache


def paired_bootstrap(errors, left, right, ids, resamples=2000, seed=20261103, groups=None):
    """Left-minus-right AURC. Pooled rows sharing a source are sampled together."""
    errors, left, right, ids = map(np.asarray, (errors, left, right, ids))
    if not (errors.shape == left.shape == right.shape == ids.shape):
        raise ValueError('Unaligned bootstrap arrays')
    rng = np.random.default_rng(seed)
    if groups is None:
        units = None
    else:
        groups = np.asarray(groups)
        if groups.shape != errors.shape:
            raise ValueError('Unaligned group IDs')
        unique = np.unique(groups)
        units = [np.flatnonzero(groups == g) for g in unique]
        lengths = {len(u) for u in units}
        if len(lengths) != 1:
            raise ValueError('Expected complete repeated conditions per source image')
        units = np.stack(units)
    values = np.empty(resamples)
    for i in range(resamples):
        n = len(errors) if units is None else len(units)
        draw = rng.integers(0, n, n)
        idx = draw if units is None else units[draw].ravel()
        values[i] = risk_curve(errors[idx], left[idx], ids[idx]).mean() - risk_curve(errors[idx], right[idx], ids[idx]).mean()
    point = float(risk_curve(errors, left, ids).mean() - risk_curve(errors, right, ids).mean())
    low, high = np.quantile(values, [.025, .975])
    return dict(delta_aurc=point, ci_low=float(low), ci_high=float(high), resamples=resamples,
                supports_improvement=bool(point < 0 and high < 0))


def evaluate_condition(name, cache, detectors, stats, protocol, sealed, root):
    table = feature_table(cache, stats)
    scores = detectors.predict(table)
    calibrated = probabilities(cache['logits'], protocol['temperature']).mean(1)
    scores['temperature_confidence'] = -calibrated.max(1)
    table['label'] = cache['label']
    table['error'] = (table.prediction != table.label).astype(int)
    errors, ids = table.error.to_numpy(), table.id.to_numpy()
    output = root / name
    output.mkdir(exist_ok=True)
    table.to_csv(output / 'features.csv', index=False)
    pd.DataFrame(dict(id=ids, error=errors, **scores)).to_csv(output / 'scores.csv', index=False)
    rows, threshold_rows = [], []
    masks = dict(all=np.ones(len(table), dtype=bool), unanimous=table.unanimous.to_numpy(),
                 confident_unanimous=(table.unanimous & (table.confidence >= .9)).to_numpy())
    for subset, mask in masks.items():
        n, wrong = int(mask.sum()), int(errors[mask].sum())
        for detector, score in scores.items():
            if n:
                values = metrics(errors[mask], score[mask], ids[mask])
                if min(wrong, n - wrong) < MIN_ERRORS:
                    values['auroc'] = None
            else:
                values = dict(aurc=None, auroc=None, risk80=None, risk90=None)
            rows.append(dict(condition=name, subset=subset, detector=detector, count=n, errors=wrong,
                             correct=n - wrong, few_errors=min(wrong, n - wrong) < MIN_ERRORS, **values))
            for coverage in (.8, .9):
                threshold = sealed['thresholds'][detector][str(coverage)]
                accepted = mask & (score <= threshold)
                threshold_rows.append(dict(subset=subset, detector=detector, validation_coverage=coverage,
                                           threshold=threshold, accepted=int(accepted.sum()),
                                           actual_coverage=float(accepted.sum()/n) if n else None,
                                           accepted_errors=int(errors[accepted].sum()),
                                           risk=float(errors[accepted].mean()) if accepted.any() else None))
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(output / 'threshold_metrics.csv', index=False)
    atomic_json(output / 'ensemble.json', dict(accuracy=float(1-errors.mean()),
                model_accuracy=[float((cache['logits'][:, m].argmax(1) == cache['label']).mean()) for m in range(3)],
                calibrated_argmax_changes=int((calibrated.argmax(1) != table.prediction).sum())))
    print(f'{name}: accuracy={1-errors.mean():.4f}; errors={errors.sum()}', flush=True)
    return table, scores, rows


def gallery(name, images, table, scores, sealed, root):
    selected = []
    for family in ('lr', 'mlp_selected'):
        b, c = scores[f'B_{family}'], scores[f'C_{family}']
        tb, tc = [sealed['thresholds'][f'{k}_{family}']['0.9'] for k in 'BC']
        for subset, mask in [('all_errors', table.error.eq(1).to_numpy()),
                             ('confident_unanimous_errors', (table.error.eq(1) & table.unanimous & table.confidence.ge(.9)).to_numpy())]:
            for direction, criterion in [('C_catches_B_misses', (c > tc) & (b <= tb)),
                                         ('B_catches_C_misses', (b > tb) & (c <= tc))]:
                candidates = np.flatnonzero(mask & criterion)
                # First six by original source ID: no visual cherry-picking.
                chosen = candidates[np.argsort(table.id.to_numpy()[candidates])][:6]
                for idx in chosen:
                    selected.append(dict(family=family, subset=subset, direction=direction, **table.iloc[idx].to_dict(),
                                         B_score=float(b[idx]), C_score=float(c[idx]), B_threshold=tb, C_threshold=tc,
                                         total_candidates=len(candidates)))
                if not len(chosen):
                    continue
                fig, axes = plt.subplots(1, len(chosen), figsize=(3*len(chosen), 3), squeeze=False)
                for ax, idx in zip(axes.ravel(), chosen):
                    ax.imshow(images[idx])
                    row = table.iloc[idx]
                    ax.set_title(f'ID {int(row.id)} true={int(row.label)} pred={int(row.prediction)}\nc={row.confidence:.3f} B={b[idx]:.3f} C={c[idx]:.3f}', fontsize=8)
                    ax.axis('off')
                fig.suptitle(f'{name}: {family}, {subset}, {direction}', fontsize=9)
                fig.tight_layout()
                fig.savefig(root / name / f'{family}_{subset}_{direction}.png', dpi=140)
                plt.close(fig)
    pd.DataFrame(selected).to_csv(root / name / 'gallery_examples.csv', index=False)


def benchmark(models, dataset, device, detectors, stats, root):
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    rows = []
    for batch_size in (1, 128):
        batch = torch.stack([dataset[i][0] for i in range(batch_size)]).to(device)
        def forward(count, joint):
            return [model(batch, return_features=joint) for model in models[:count]]
        with torch.inference_mode():
            output = forward(3, True)
            cache = dict(id=np.arange(batch_size), logits=torch.stack([x[0] for x in output], 1).cpu().numpy(),
                         features=torch.stack([x[1] for x in output], 1).cpu().numpy())
            table = feature_table(cache, stats)
            def end_to_end(single=False):
                outputs = forward(3, True)
                c = dict(id=np.arange(batch_size), logits=torch.stack([x[0] for x in outputs], 1).cpu().numpy(),
                         features=torch.stack([x[1] for x in outputs], 1).cpu().numpy())
                t = feature_table(c, stats)
                return detectors.predict_one(t) if single else detectors.predict(t)
            operations = dict(single_network=lambda: forward(1, False), ensemble_logits=lambda: forward(3, False),
                              ensemble_logits_and_features=lambda: forward(3, True),
                              novelty_and_output_features=lambda: feature_table(cache, stats),
                              all_16_detectors=lambda: detectors.predict(table),
                              ensemble_features_all_detectors=end_to_end,
                              ensemble_features_C_mlp=lambda: end_to_end(single=True))
            # Single deployed detector measured separately, rather than charging all sensitivity runs.
            for family in ('lr', 'mlp_selected'):
                def predict_one(family=family):
                    columns = FEATURES['C']
                    if family == 'lr':
                        art = detectors.lr['C']
                        return art['model'].predict_proba(art['scaler'].transform(table[columns]))
                    model, mean, scale = detectors.mlp['C', detectors.selected['C']]
                    return model(torch.tensor((table[columns].to_numpy()-mean)/scale, dtype=torch.float32)).sigmoid()
                operations[f'C_{family}_detector'] = predict_one
            for name, operation in operations.items():
                for _ in range(5):
                    operation()
                samples = []
                for _ in range(30):
                    sync()
                    start = time.perf_counter()
                    operation()
                    sync()
                    samples.append((time.perf_counter()-start)*1000)
                rows.append(dict(operation=name, batch_size=batch_size, median_ms=float(np.median(samples)),
                                 p10_ms=float(np.quantile(samples,.1)), p90_ms=float(np.quantile(samples,.9)),
                                 median_ms_per_image=float(np.median(samples)/batch_size)))
    pd.DataFrame(rows).to_csv(root / 'latency.csv', index=False)
    atomic_json(root / 'environment.json', dict(torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu', cpu_threads=torch.get_num_threads(),
                timing='5 warmups, 30 repeats; CUDA synchronization; preprocessing and disk I/O excluded'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', default='results/validation')
    parser.add_argument('--output', default='results/final')
    parser.add_argument('--data', default='data')
    parser.add_argument('--corruptions', default='data/CIFAR-10-C')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--stage', choices=['clean', 'shift', 'all'], default='all')
    args = parser.parse_args()
    if args.resamples < 1000:
        raise ValueError('Final evaluation requires at least 1000 bootstrap resamples')
    seed_everything(42)
    torch.set_num_threads(2)
    validation, root = Path(args.validation), Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    protocol, manifest, sealed = freeze(validation, root, args.resamples)
    models = load_models(manifest, args.device)
    detectors = FrozenDetectors(validation, protocol)
    with np.load(validation / 'novelty_stats.npz') as saved:
        stats = {key: saved[key] for key in saved.files}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # This is the first access to test images, after the immutable seal above.
    clean = datasets.CIFAR10(args.data, train=False, download=False, transform=transform)
    provenance = dict(seal=sha(root / 'sealed_protocol.json'), checkpoints=manifest['checkpoints'], preprocessing=manifest['preprocessing'])
    all_rows, shift_data = [], []
    if args.stage in ('clean', 'all'):
        p = dict(provenance, dataset='official CIFAR-10 test', image_sha256=__import__('hashlib').sha256(clean.data.tobytes()).hexdigest())
        cache = infer(clean, models, args.device, root / 'clean.npz', p)
        table, scores, rows = evaluate_condition('clean', cache, detectors, stats, protocol, sealed, root)
        gallery('clean', clean.data, table, scores, sealed, root)
        all_rows += rows
        contrasts = []
        for family in ('lr', 'mlp_selected'):
            for left, right in [('B', 'A'), ('C', 'B'), ('D', 'C')]:
                result = paired_bootstrap(table.error, scores[f'{left}_{family}'], scores[f'{right}_{family}'], table.id,
                                          args.resamples, sealed['bootstrap_seed'])
                contrasts.append(dict(family=family, comparison=f'{left}_minus_{right}', **result))
        for key in 'ABCD':
            contrasts.append(dict(family=key, comparison='MLP_minus_LR', **paired_bootstrap(table.error,
                scores[f'{key}_mlp_selected'], scores[f'{key}_lr'], table.id, args.resamples, sealed['bootstrap_seed'])))
        pd.DataFrame(contrasts).to_csv(root / 'clean_bootstrap.csv', index=False)
        sensitivity = []
        for key in 'ABCD':
            for metric in ('aurc', 'auroc'):
                values = [metrics(table.error, scores[f'{key}_mlp_{seed}'], table.id)[metric] for seed in SEEDS]
                sensitivity.append(dict(configuration=key, metric=metric, mean=float(np.mean(values)), std=float(np.std(values)), values=str(values)))
        pd.DataFrame(sensitivity).to_csv(root / 'seed_sensitivity.csv', index=False)
        permutation = []
        for family in ('lr', 'mlp_selected'):
            rng = np.random.default_rng(sealed['permutation_seed'])
            base = metrics(table.error, scores[f'C_{family}'], table.id)['aurc']
            for repeat in range(sealed['permutations']):
                shuffled = table.copy()
                shuffled['novelty_variation'] = rng.permutation(shuffled.novelty_variation.to_numpy())
                score = detectors.predict(shuffled)[f'C_{family}']
                permutation.append(dict(family=family, repeat=repeat, delta_aurc=metrics(table.error, score, table.id)['aurc']-base))
        pd.DataFrame(permutation).to_csv(root / 'permutation_diagnostic.csv', index=False)
        benchmark(models, clean, args.device, detectors, stats, root)
    if args.stage in ('shift', 'all'):
        directory = Path(args.corruptions)
        labels_file = directory / 'labels.npy'
        labels = np.load(labels_file)
        if labels.shape not in ((10000,), (50000,)):
            raise ValueError('Unexpected corruption label layout')
        for start in range(0, len(labels), 10000):
            np.testing.assert_array_equal(labels[start:start+10000], clean.targets)
        shift_contrasts = []
        for corruption in CORRUPTIONS:
            path = directory / f'{corruption}.npy'
            source_hash = sha(path)
            for severity in SEVERITIES:
                name = f'{corruption}_{severity}'
                dataset = CorruptedImages(path, severity, np.asarray(clean.targets), transform)
                p = dict(provenance, corruption=corruption, severity=severity, source_sha256=source_hash, labels_sha256=sha(labels_file))
                cache = infer(dataset, models, args.device, root / f'{name}.npz', p)
                table, scores, rows = evaluate_condition(name, cache, detectors, stats, protocol, sealed, root)
                gallery(name, dataset.images, table, scores, sealed, root)
                all_rows += rows
                shift_data.append((table, scores))
                for family in ('lr', 'mlp_selected'):
                    result = paired_bootstrap(table.error, scores[f'C_{family}'], scores[f'B_{family}'], table.id, args.resamples, sealed['bootstrap_seed'])
                    shift_contrasts.append(dict(condition=name, family=family, comparison='C_minus_B', **result))
        errors = np.concatenate([t.error for t, _ in shift_data])
        groups = np.concatenate([t.id for t, _ in shift_data])
        ids = np.arange(len(errors))
        for family in ('lr', 'mlp_selected'):
            left = np.concatenate([s[f'C_{family}'] for _, s in shift_data])
            right = np.concatenate([s[f'B_{family}'] for _, s in shift_data])
            result = paired_bootstrap(errors, left, right, ids, args.resamples, sealed['bootstrap_seed'], groups=groups)
            shift_contrasts.append(dict(condition='pooled_source_clustered', family=family, comparison='C_minus_B', **result))
        pd.DataFrame(shift_contrasts).to_csv(root / 'shift_bootstrap.csv', index=False)
    # Keep independently completed stages when running clean and shift separately.
    frames = [pd.read_csv(p) for p in sorted(root.glob('*/metrics.csv'))]
    pd.concat(frames, ignore_index=True).to_csv(root / 'all_metrics.csv', index=False)
    verify_artifacts(validation)
    atomic_json(root / f'{args.stage}_complete.json', dict(sealed_protocol_sha256=sha(root / 'sealed_protocol.json'),
                conditions=[p.parent.name for p in sorted(root.glob('*/metrics.csv'))], frozen_artifacts_verified=True))
    print('Frozen evaluation stage complete.', flush=True)


if __name__ == '__main__':
    main()
