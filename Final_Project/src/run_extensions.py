"""New-files-only H7/H8 runner. Each study registers, prepares, fits, freezes, evaluates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import MEAN, STD
from .extension_methods import load_npz, write_npz, fit_geometry, fit_normalization, make_features, add_targets, extract_stages
from .final_evaluation import FrozenDetectors, load_models, CorruptedImages, verify_artifacts
from .metrics import metrics, risk_curve
from .train import seed_everything
from .validation import FEATURES, fit_mlp, sha

DEFAULT_ROOT = '/common/users/ns1788/beyond-ensemble-agreement/training/extensions_h7_h8_v1'
VALIDATION = Path('results/validation')
FINAL = Path('results/final')
CONFIG = Path('configs/extensions_h7_h8.json')
SEEDS = (11, 22, 33)


def json_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)


def csv_new(path, frame):
    with Path(path).open('x') as handle:
        frame.to_csv(handle, index=False)


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def verify_hashes(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError(f'Artifact or source changed: {path}')


def audit_originals(root):
    snapshot = json.loads((root / 'original_files.json').read_text())
    verify_hashes(snapshot)
    return len(snapshot)


def register(root, study):
    path = root / study
    path.mkdir(exist_ok=True)
    if (path / 'registration.json').exists():
        check_registration(path)
        return
    protocol, manifest = verify_artifacts(VALIDATION)
    config = json.loads(CONFIG.read_text())
    sources = [CONFIG, Path('src/extension_methods.py'), Path(__file__), Path('src/extension_report.py')]
    sources += [Path(p) for p in json.loads((FINAL / 'sealed_protocol.json').read_text())['source_sha256']]
    sources += [Path('src/data.py'), Path('src/train.py')]
    recorded = dict(study=study, registered_utc=datetime.now(timezone.utc).isoformat(), exploratory=True,
                    original_test_results_already_known=True, protocol=config,
                    originals_snapshot_sha256=sha(root / 'original_files.json'),
                    original_protocol_sha256=sha(VALIDATION / 'frozen_protocol.json'),
                    original_final_seal_sha256=sha(FINAL / 'sealed_protocol.json'),
                    source_sha256={str(p.absolute()): sha(p) for p in sources},
                    original_selected_mlp_seeds=protocol['selected_mlp_seeds'])
    json_new(path / 'registration.json', recorded)
    json_new(path / 'environment.json', {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']})
    print(f'{study}: protocol registered before preparation.', flush=True)


def check_registration(path):
    registered = json.loads((path / 'registration.json').read_text())
    verify_hashes(registered['source_sha256'])
    if sha(VALIDATION / 'frozen_protocol.json') != registered['original_protocol_sha256'] or sha(FINAL / 'sealed_protocol.json') != registered['original_final_seal_sha256']:
        raise ValueError('Original frozen protocol changed')
    return registered


def variants(study):
    return ['mahalanobis'] if study == 'h7' else ['stage1', 'stage2']


def original_cache(condition):
    source = VALIDATION if condition in ('train', 'detector_train', 'val') else FINAL
    path = source / f'{condition}.npz'
    if source == FINAL:
        metadata = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != metadata['sha256']:
            raise ValueError(f'Original evaluation cache changed: {condition}')
    return load_npz(path)


def depth_dataset(condition, original):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if condition in ('train', 'detector_train', 'val'):
        dataset = datasets.CIFAR10('data', train=True, download=False, transform=transform)
        return Subset(dataset, original['id'].tolist())
    if condition == 'clean':
        return datasets.CIFAR10('data', train=False, download=False, transform=transform)
    corruption, severity = condition.rsplit('_', 1)
    return CorruptedImages(Path('data/CIFAR-10-C') / f'{corruption}.npy', int(severity), original['label'], transform)


def stage_features(root, condition, original, models, device):
    cache_dir = root / 'h8' / 'depth_cache'
    cache_dir.mkdir(exist_ok=True)
    return extract_stages(depth_dataset(condition, original), original['id'], models, device, original,
                          cache_dir / f'{condition}.npz', sha(root / 'h8/registration.json'))


def prepare(root, study, device):
    path = root / study
    check_registration(path)
    if (path / 'prepared.json').exists():
        verify_hashes(json.loads((path / 'prepared.json').read_text())['files'])
        print(f'{study}: prepared artifacts verified and reused.', flush=True)
        return
    if study == 'h8' and not (root / 'h7/evaluated.json').exists():
        raise ValueError('Finish H7 before preparing H8')
    _, manifest = verify_artifacts(VALIDATION)
    models = load_models(manifest, device) if study == 'h8' else None
    original = {s: original_cache(s) for s in ('train', 'detector_train', 'val')}
    if study == 'h8':
        stage_caches = {s: stage_features(root, s, cache, models, device) for s, cache in original.items()}
    files = []
    for variant in variants(study):
        dest = path / variant
        dest.mkdir(exist_ok=True)
        training = original['train']['features'] if study == 'h7' else stage_caches['train'][variant]
        detector = original['detector_train']['features'] if study == 'h7' else stage_caches['detector_train'][variant]
        centroids = load_npz(VALIDATION / 'novelty_stats.npz')['centroids'] if study == 'h7' else None
        geometry = fit_geometry(training, original['train']['label'], centroids, mahalanobis=study == 'h7')
        geometry = fit_normalization(detector, original['detector_train']['logits'], geometry)
        write_npz(dest / 'geometry.npz', **geometry)
        files.append(dest / 'geometry.npz')
        for split in ('detector_train', 'val'):
            cache = original[split]
            x = cache['features'] if study == 'h7' else stage_caches[split][variant]
            table = add_targets(make_features(cache['id'], cache['logits'], x, geometry), cache['label'])
            csv_new(dest / f'{split}_features.csv', table)
            files.append(dest / f'{split}_features.csv')
        print(f'{study}/{variant}: geometry and development features prepared.', flush=True)
    json_new(path / 'prepared.json', dict(registration_sha256=sha(path / 'registration.json'), files={str(p): sha(p) for p in files}))


class ExtensionBank:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lr, self.mlp = {}, {}
        self.selection = json.loads((self.directory / 'selection.json').read_text())['selected_mlp_seeds']
        for key in 'BCD':
            art = joblib.load(self.directory / f'{key}_lr.joblib')
            if art['features'] != FEATURES[key]:
                raise ValueError('Detector feature columns changed')
            self.lr[key] = art
            for seed in SEEDS:
                saved = torch.load(self.directory / f'{key}_mlp_{seed}.pt', map_location='cpu', weights_only=False)
                if saved['features'] != FEATURES[key] or saved['seed'] != seed:
                    raise ValueError('Saved MLP configuration changed')
                model = nn.Sequential(nn.Linear(len(saved['features']), 32), nn.ReLU(), nn.Linear(32, 1)).eval()
                model.load_state_dict(saved['state_dict'])
                self.mlp[key, seed] = (model, saved['mean'], saved['scale'])

    def predict(self, table):
        result = {}
        for key in 'BCD':
            cols = FEATURES[key]
            if {'label', 'error', 'id'} & set(cols):
                raise ValueError('Label/ID leakage into detector')
            art = self.lr[key]
            result[f'{key}_lr'] = art['model'].predict_proba(art['scaler'].transform(table[cols]))[:, 1]
            for seed in SEEDS:
                model, mean, scale = self.mlp[key, seed]
                x = (table[cols].to_numpy() - mean) / scale
                with torch.inference_mode():
                    result[f'{key}_mlp_{seed}'] = model(torch.tensor(x, dtype=torch.float32)).sigmoid().numpy().ravel()
            result[f'{key}_mlp_selected'] = result[f'{key}_mlp_{self.selection[key]}']
        return result


def fit(root, study):
    path = root / study
    check_registration(path)
    verify_hashes(json.loads((path / 'prepared.json').read_text())['files'])
    if (path / 'frozen.json').exists():
        raise ValueError('Study is frozen; detector fitting is disabled')
    for variant in variants(study):
        dest = path / variant
        if (dest / 'selection.json').exists():
            print(f'{variant}: detector selection exists; freeze verifies its artifacts.', flush=True)
            continue
        dt, vt = [read_csv(dest / f'{s}_features.csv') for s in ('detector_train', 'val')]
        y, yv, ids = dt.error.to_numpy(), vt.error.to_numpy(), vt.id.to_numpy()
        rows, predictions, selected = [], {}, {}
        for key in 'BCD':
            columns = FEATURES[key]
            scaler = StandardScaler().fit(dt[columns])
            x, xv = scaler.transform(dt[columns]), scaler.transform(vt[columns])
            lr = LogisticRegression(C=1., max_iter=2000, random_state=42).fit(x, y)
            with (dest / f'{key}_lr.joblib').open('xb') as handle:
                joblib.dump(dict(model=lr, scaler=scaler, features=columns), handle)
            score = lr.predict_proba(xv)[:, 1]
            predictions[f'{key}_lr'] = score
            rows.append(dict(detector=f'{key}_lr', **metrics(yv, score, ids)))
            candidates = []
            for seed in SEEDS:
                state, score, epoch = fit_mlp(x, y, xv, yv, ids, seed, 200)
                with (dest / f'{key}_mlp_{seed}.pt').open('xb') as handle:
                    torch.save(dict(state_dict=state, features=columns, mean=scaler.mean_, scale=scaler.scale_, seed=seed, epoch=epoch), handle)
                name = f'{key}_mlp_{seed}'
                predictions[name] = score
                row = dict(detector=name, **metrics(yv, score, ids))
                rows.append(row)
                candidates.append((row['aurc'], seed))
            selected[key] = min(candidates)[1]
            predictions[f'{key}_mlp_selected'] = predictions[f'{key}_mlp_{selected[key]}']
            rows.append(dict(detector=f'{key}_mlp_selected', **metrics(yv, predictions[f'{key}_mlp_selected'], ids)))
            print(f'{study}/{variant}/{key}: selected seed {selected[key]} on validation.', flush=True)
        csv_new(dest / 'validation_metrics.csv', pd.DataFrame(rows))
        csv_new(dest / 'validation_scores.csv', pd.DataFrame(dict(id=ids, error=yv, **predictions)))
        thresholds = {name: {str(c): float(np.sort(score)[int(np.ceil(c*len(score)))-1]) for c in (.8, .9)} for name, score in predictions.items()}
        artifacts = {str(p): sha(p) for p in dest.iterdir() if p.is_file()}
        json_new(dest / 'selection.json', dict(selected_mlp_seeds=selected, thresholds=thresholds, files=artifacts))


def freeze(root, study):
    path = root / study
    registered = check_registration(path)
    verify_hashes(json.loads((path / 'prepared.json').read_text())['files'])
    files = {}
    for variant in variants(study):
        dest = path / variant
        selection = json.loads((dest / 'selection.json').read_text())
        verify_hashes(selection['files'])
        bank = ExtensionBank(dest)
        vt = read_csv(dest / 'val_features.csv')
        expected = read_csv(dest / 'validation_scores.csv')
        for name, values in bank.predict(vt).items():
            np.testing.assert_allclose(values, expected[name], atol=2e-7, rtol=2e-6)
        files.update(selection['files'])
        files[str(dest / 'selection.json')] = sha(dest / 'selection.json')
    value = dict(registration_sha256=sha(path / 'registration.json'), frozen_utc=datetime.now(timezone.utc).isoformat(),
                 files=files, original_files_verified=audit_originals(root), exploratory=True)
    json_new(path / 'frozen.json', value)
    print(f'{study}: detector artifacts, preprocessing, seeds and thresholds frozen.', flush=True)


def verify_freeze(path):
    check_registration(path)
    frozen = json.loads((path / 'frozen.json').read_text())
    if frozen['registration_sha256'] != sha(path / 'registration.json'):
        raise ValueError('Registration changed after freeze')
    verify_hashes(frozen['files'])
    return frozen


def condition_metrics(condition, variant, table, scores, thresholds):
    e, ids = table.error.to_numpy(), table.id.to_numpy()
    masks = dict(all=np.ones(len(table), dtype=bool), unanimous=table.unanimous.to_numpy(),
                 confident_unanimous=(table.unanimous & table.confidence.ge(.9)).to_numpy())
    rows, decisions = [], []
    for subset, mask in masks.items():
        n, wrong = int(mask.sum()), int(e[mask].sum())
        for name, values in scores.items():
            score = np.asarray(values)
            m = metrics(e[mask], score[mask], ids[mask]) if n else dict(aurc=None, auroc=None, risk80=None, risk90=None)
            if min(wrong, n-wrong) < 20:
                m['auroc'] = None
            rows.append(dict(condition=condition, variant=variant, subset=subset, detector=name, count=n, errors=wrong,
                             correct=n-wrong, few_errors=min(wrong,n-wrong)<20, **m))
            if name in thresholds:
                for coverage in (.8,.9):
                    threshold=thresholds[name][str(coverage)]
                    accepted=mask & (score <= threshold)
                    decisions.append(dict(condition=condition, variant=variant, subset=subset, detector=name, validation_coverage=coverage,
                                          threshold=threshold, coverage=float(accepted.sum()/n) if n else None,
                                          accepted=int(accepted.sum()), accepted_errors=int(e[accepted].sum()),
                                          risk=float(e[accepted].mean()) if accepted.any() else None))
    return rows, decisions


def bootstrap_contrasts(errors, scores, ids, contrasts, resamples=2000, seed=20261103, groups=None):
    """Same samples for every score/contrast; linear contrasts include difference-in-differences."""
    errors, ids = np.asarray(errors), np.asarray(ids)
    if errors.ndim != 1 or ids.shape != errors.shape or not len(errors):
        raise ValueError('Unaligned nonempty errors and IDs required')
    names = sorted({name for terms in contrasts.values() for name in terms})
    arrays = {name: np.asarray(scores[name]) for name in names}
    for name, score in arrays.items():
        if score.shape != errors.shape or not np.isfinite(score).all():
            raise ValueError(f'Unaligned or nonfinite scores: {name}')
    units = None
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != errors.shape:
            raise ValueError('Unaligned cluster IDs')
        order = np.argsort(groups, kind='stable')
        _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
        if len(np.unique(counts)) != 1:
            raise ValueError('Incomplete repeated conditions per source image')
        units = order.reshape(len(starts), int(counts[0]))
    point_scores = {name: float(risk_curve(errors, arrays[name], ids).mean()) for name in names}
    values = np.empty((resamples, len(contrasts)))
    rng = np.random.default_rng(seed)
    for repeat in range(resamples):
        n = len(errors) if units is None else len(units)
        draw = rng.integers(n, size=n)
        idx = draw if units is None else units[draw].ravel()
        sample_scores = {name: float(risk_curve(errors[idx], arrays[name][idx], ids[idx]).mean()) for name in names}
        values[repeat] = [sum(weight*sample_scores[name] for name, weight in terms.items()) for terms in contrasts.values()]
    result=[]
    for i, (contrast, terms) in enumerate(contrasts.items()):
        low, high = np.quantile(values[:,i],[.025,.975])
        point = sum(weight*point_scores[name] for name, weight in terms.items())
        result.append(dict(comparison=contrast, delta_aurc=point, ci_low=float(low), ci_high=float(high), resamples=resamples,
                           negative_interval=bool(high<0), interval_includes_zero=bool(low<=0<=high)))
    return result


def comparisons(study, family):
    contrasts={}
    for v in variants(study):
        def add(name, terms):
            contrasts[f'{family}:{name}']={f'{variant}:{key}_{family}':weight for variant,key,weight in terms}
        for key in 'BC':
            add(f'{v}_{key}_minus_original_{key}',[(v,key,1),('original',key,-1)])
        add(f'{v}_C_minus_B',[(v,'C',1),(v,'B',-1)])
        add(f'{v}_B_minus_A',[(v,'B',1),('original','A',-1)])
        add(f'{v}_D_minus_C',[(v,'D',1),(v,'C',-1)])
        if study=='h7':
            add('geometry_change_in_C_minus_B',[(v,'C',1),(v,'B',-1),('original','C',-1),('original','B',1)])
    return contrasts


def sensitivity(condition, variant, table, scores):
    result=[]
    for key in 'BCD':
        for metric in ('aurc','auroc'):
            values=[metrics(table.error, scores[f'{key}_mlp_{seed}'], table.id)[metric] for seed in SEEDS]
            result.append(dict(condition=condition, variant=variant, configuration=key, metric=metric,
                               mean=float(np.mean(values)), std=float(np.std(values)), values=json.dumps(values)))
    differences=[]
    for seed in SEEDS:
        b=metrics(table.error,scores[f'B_mlp_{seed}'],table.id)['aurc']
        c=metrics(table.error,scores[f'C_mlp_{seed}'],table.id)['aurc']
        differences.append(dict(condition=condition,variant=variant,seed=seed,B_aurc=b,C_aurc=c,delta_aurc=c-b))
    return result,differences


def corruption_response(root):
    path=root/'h8'
    variants_all=['original','stage1','stage2']
    result=[]
    clean={v:read_csv(FINAL/'clean/features.csv' if v=='original' else path/v/'clean/features.csv') for v in variants_all}
    config=json.loads(CONFIG.read_text())
    for condition in config['evaluation']['conditions'][1:]:
        for v in variants_all:
            corrupted=read_csv(FINAL/condition/'features.csv' if v=='original' else path/v/condition/'features.csv')
            base=clean[v]
            np.testing.assert_array_equal(base.id,corrupted.id)
            groups=dict(all=np.ones(len(base),bool),remains_correct=(base.error.eq(0)&corrupted.error.eq(0)).to_numpy(),
                        becomes_wrong=(base.error.eq(0)&corrupted.error.eq(1)).to_numpy(),
                        remains_wrong=(base.error.eq(1)&corrupted.error.eq(1)).to_numpy(),
                        becomes_correct=(base.error.eq(1)&corrupted.error.eq(0)).to_numpy())
            for group,mask in groups.items():
                delta=(corrupted.mean_novelty-base.mean_novelty).to_numpy()[mask]
                result.append(dict(condition=condition,stage='stage4' if v=='original' else v,transition=group,count=int(mask.sum()),
                                   mean_delta=float(delta.mean()) if len(delta) else None,median_delta=float(np.median(delta)) if len(delta) else None,
                                   p25=float(np.quantile(delta,.25)) if len(delta) else None,p75=float(np.quantile(delta,.75)) if len(delta) else None))
    csv_new(path/'corruption_response.csv',pd.DataFrame(result))


def evaluate(root, study, device):
    path=root/study
    if (path/'evaluated.json').exists():
        verify_freeze(path)
        verify_hashes(json.loads((path/'evaluated.json').read_text())['files'])
        print(f'{study}: completed evaluation verified and reused.',flush=True)
        return
    verify_freeze(path)
    protocol,manifest=verify_artifacts(VALIDATION)
    original_thresholds=json.loads((FINAL/'sealed_protocol.json').read_text())['thresholds']
    models=load_models(manifest,device) if study=='h8' else None
    banks={v:ExtensionBank(path/v) for v in variants(study)}
    all_rows,threshold_rows,seed_rows,seed_differences=[],[],[],[]
    pooled_e,pooled_groups,pooled_scores=[],[],{}
    all_ci=[]
    contrasts={}
    for family in ('lr','mlp_selected'):
        contrasts.update(comparisons(study,family))
    for condition in json.loads(CONFIG.read_text())['evaluation']['conditions']:
        cache=original_cache(condition)
        original_table=read_csv(FINAL/condition/'features.csv')
        orig_scores=read_csv(FINAL/condition/'scores.csv')
        np.testing.assert_array_equal(cache['id'],orig_scores.id)
        np.testing.assert_array_equal(cache['label'],original_table.label)
        original_e=(output_features_cached(cache)!=cache['label']).astype(int)
        np.testing.assert_array_equal(original_e,orig_scores.error)
        rows,decisions=condition_metrics(condition,'original',original_table,{k:orig_scores[k].to_numpy() for k in orig_scores if k not in ('id','error')},original_thresholds)
        all_rows+=rows
        threshold_rows+=decisions
        feature_cache=stage_features(root,condition,cache,models,device) if study=='h8' else None
        combined={f'original:{key}_{family}':orig_scores[f'{key}_{family}'].to_numpy() for key in 'ABCD' for family in ('lr','mlp_selected')}
        for variant,bank in banks.items():
            dest=path/variant/condition
            dest.mkdir(exist_ok=True)
            geometry=load_npz(path/variant/'geometry.npz')
            x=cache['features'] if study=='h7' else feature_cache[variant]
            table=add_targets(make_features(cache['id'],cache['logits'],x,geometry),cache['label'])
            np.testing.assert_array_equal(table.prediction,original_table.prediction)
            np.testing.assert_array_equal(table.error,original_e)
            scores=bank.predict(table)
            scores.update({name:orig_scores[name].to_numpy() for name in orig_scores if name.startswith('A_') or name in ('confidence','entropy','margin','mutual_information','temperature_confidence')})
            csv_new(dest/'features.csv',table)
            csv_new(dest/'scores.csv',pd.DataFrame(dict(id=cache['id'],error=original_e,**scores)))
            thresholds=dict(original_thresholds,**json.loads((path/variant/'selection.json').read_text())['thresholds'])
            rows,decisions=condition_metrics(condition,variant,table,scores,thresholds)
            csv_new(dest/'metrics.csv',pd.DataFrame(rows))
            csv_new(dest/'threshold_metrics.csv',pd.DataFrame(decisions))
            all_rows+=rows
            threshold_rows+=decisions
            rows,differences=sensitivity(condition,variant,table,scores)
            seed_rows+=rows
            seed_differences+=differences
            combined.update({f'{variant}:{key}_{family}':scores[f'{key}_{family}'] for key in 'BCD' for family in ('lr','mlp_selected')})
        print(f'{study}/{condition}: frozen scores computed; paired bootstrap starting.',flush=True)
        current=bootstrap_contrasts(original_e,combined,cache['id'],contrasts)
        all_ci += [dict(condition=condition,**row) for row in current]
        if condition!='clean':
            pooled_e.append(original_e)
            pooled_groups.append(cache['id'])
            for name,values in combined.items():
                pooled_scores.setdefault(name,[]).append(values)
    errors=np.concatenate(pooled_e)
    print(f'{study}: all ten conditions done; source-clustered pooled bootstrap starting.',flush=True)
    pooled=bootstrap_contrasts(errors,{k:np.concatenate(v) for k,v in pooled_scores.items()},np.arange(len(errors)),contrasts,groups=np.concatenate(pooled_groups))
    all_ci += [dict(condition='pooled_source_clustered',**row) for row in pooled]
    csv_new(path/'metrics.csv',pd.DataFrame(all_rows))
    csv_new(path/'threshold_metrics.csv',pd.DataFrame(threshold_rows))
    csv_new(path/'bootstrap.csv',pd.DataFrame(all_ci))
    csv_new(path/'seed_sensitivity.csv',pd.DataFrame(seed_rows))
    csv_new(path/'paired_seed_sensitivity.csv',pd.DataFrame(seed_differences))
    if study=='h8':
        corruption_response(root)
    verify_freeze(path)
    count=audit_originals(root)
    files={str(p):sha(p) for p in path.rglob('*') if p.is_file() and p.name not in ('evaluated.json',)}
    json_new(path/'evaluated.json',dict(frozen_sha256=sha(path/'frozen.json'),conditions=10,original_files_unchanged=count,
                                      completed_utc=datetime.now(timezone.utc).isoformat(),files=files,exploratory=True))
    print(f'{study}: evaluated; all {count} original files unchanged.',flush=True)


def output_features_cached(cache):
    from .validation import output_features
    return output_features(cache['logits'])['prediction']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','prepare','fit','freeze','evaluate','report','audit'])
    parser.add_argument('--study',choices=['h7','h8'],required=True)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    root=Path(args.root).absolute()
    for protected in (VALIDATION.resolve(),FINAL.resolve()):
        if root.resolve()==protected or root.resolve().is_relative_to(protected):
            raise ValueError('Extension outputs cannot be placed inside original experiment directories')
    if not (root/'original_files.json').exists():
        raise ValueError('A pre-implementation original-files snapshot is required')
    seed_everything(42)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.command=='register':register(root,args.study)
        elif args.command=='prepare':prepare(root,args.study,args.device)
        elif args.command=='fit':fit(root,args.study)
        elif args.command=='freeze':freeze(root,args.study)
        elif args.command=='evaluate':evaluate(root,args.study,args.device)
        elif args.command=='report':
            from .extension_report import report
            report(root,args.study)
        else:
            print(f'Unchanged originals: {audit_originals(root)}',flush=True)


if __name__=='__main__':
    main()
