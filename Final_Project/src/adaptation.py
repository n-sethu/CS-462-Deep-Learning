"""Separate Gaussian-noise adaptation: prepare, fit/freeze, evaluate, report."""
import argparse
import copy
import json
import platform
from importlib.metadata import distributions
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .data import MEAN, STD, split_indices
from .extension_methods import add_targets, load_npz, make_features
from .final_evaluation import infer, load_models, verify_artifacts
from .metrics import metrics, risk_curve
from .train import seed_everything
from .validation import FEATURES, sha

VALIDATION = Path('results/validation')
FINAL = Path('results/final')


def save_json(path, obj):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise ValueError(f'Existing artifact differs: {path}')
        return
    with path.open('x') as f:
        json.dump(obj, f, indent=2)


def hashes(paths):
    return {str(p): sha(p) for p in paths}


def verify(files):
    for p, digest in files.items():
        if sha(p) != digest:
            raise ValueError(f'Artifact changed: {p}')


def noise_image(image, severity, source_id, split_code, config):
    """Official CIFAR-C formula; local per-source RNG, invariant to worker order."""
    sigma = config['gaussian_sigma'][str(severity)]
    rng = np.random.default_rng(np.random.SeedSequence(
        [config['corruption_seed'], split_code, int(source_id), severity]))
    image = np.asarray(image, dtype=np.float64) / 255.
    return (255 * np.clip(image + rng.normal(scale=sigma, size=image.shape), 0, 1)).astype(np.uint8)


class DevelopmentNoise(Dataset):
    def __init__(self, dataset, ids, severity, split_code, config):
        self.dataset, self.ids = dataset, np.asarray(ids)
        self.severity, self.split_code, self.config = severity, split_code, config
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        source = int(self.ids[index])
        image = noise_image(self.dataset.data[source], self.severity, source, self.split_code, self.config)
        return self.transform(Image.fromarray(image)), int(self.dataset.targets[source])


def register(root, config_path):
    config = json.loads(config_path.read_text())
    expected = ['clean'] + [f'gaussian_noise_{s}' for s in (1, 3, 5)]
    evaluation = ['clean'] + [f'{c}_{s}' for c in ('gaussian_noise', 'motion_blur', 'jpeg_compression') for s in (1, 3, 5)]
    if config['development_conditions'] != expected or config['evaluation_conditions'] != evaluation:
        raise ValueError('Only the registered noise-only development design is supported')
    if config['mlp_seeds'] != [11, 22, 33] or config['epochs'] < 1 or config['patience'] < 1 or config['bootstrap_resamples'] < 1000:
        raise ValueError('Invalid seed, training, or bootstrap protocol')
    original, manifest = verify_artifacts(VALIDATION)
    ensemble = json.loads(Path('configs/ensemble.json').read_text())
    if not Path(ensemble['split_path']).exists():
        raise FileNotFoundError('Existing saved split required')
    sources = [config_path, Path(__file__), Path('src/metrics.py'), Path('src/validation.py'),
               Path('src/final_evaluation.py'), Path('src/extension_methods.py'), Path('src/data.py'),
               Path('src/models.py'), Path('src/train.py'), Path('configs/ensemble.json'),
               Path(ensemble['split_path']), VALIDATION/'frozen_protocol.json',
               VALIDATION/'cache_manifest.json', FINAL/'sealed_protocol.json']
    record = dict(config=config, sources=hashes(sources), original_files={str(VALIDATION/k): v for k,v in original['files'].items()},
                  checkpoints=manifest['checkpoints'])
    save_json(root/'registration.json', record)
    save_json(root/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, packages={d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}))
    return config, ensemble, manifest


def check(root):
    record = json.loads((root/'registration.json').read_text())
    for name in ('sources', 'original_files', 'checkpoints'):
        verify(record[name])
    return record['config']


def prepare(root, config_path, device):
    config, ensemble, manifest = register(root, config_path)
    if (root/'prepared.json').exists():
        verify(json.loads((root/'prepared.json').read_text())['files'])
        return
    dataset = datasets.CIFAR10(ensemble['data_dir'], train=True, download=False)
    splits = split_indices(dataset.targets, ensemble['split_path'], ensemble['split_seed'])
    assert not set(splits['detector_train']) & set(splits['val'])
    models = load_models(manifest, device)
    stats = load_npz(VALIDATION/'novelty_stats.npz')
    files = []
    for split_code, split in enumerate(('detector_train','val'), 1):
        folder = root/'development'/split
        folder.mkdir(parents=True, exist_ok=True)
        ids = splits[split]
        clean = load_npz(VALIDATION/f'{split}.npz')
        np.testing.assert_array_equal(clean['id'], ids)
        np.testing.assert_array_equal(clean['label'], np.asarray(dataset.targets)[ids])
        for condition in config['development_conditions']:
            if condition == 'clean':
                cache = clean
            else:
                severity = int(condition.rsplit('_',1)[1])
                cache_path = folder/f'{condition}.npz'
                provenance = dict(registration_sha256=sha(root/'registration.json'), split=split, condition=condition)
                cache = infer(DevelopmentNoise(dataset, ids, severity, split_code, config), models, device, cache_path, provenance)
                np.testing.assert_array_equal(cache['id'], np.arange(len(ids)))
                cache = dict(cache, id=ids)
                files += [cache_path, cache_path.with_suffix('.json')]
            np.testing.assert_array_equal(cache['label'], clean['label'])
            table = add_targets(make_features(ids, cache['logits'], cache['features'], stats), cache['label'])
            table['condition'] = condition
            path = folder/f'{condition}.csv'
            if path.exists():
                pd.testing.assert_frame_equal(pd.read_csv(path), table, check_dtype=False, rtol=1e-6, atol=1e-8)
            else:
                table.to_csv(path, index=False)
            files.append(path)
            print(f'Prepared {split}/{condition}: {len(table)}', flush=True)
    save_json(root/'prepared.json', dict(files=hashes(files)))


def development(root, split, config):
    frames = [pd.read_csv(root/'development'/split/f'{c}.csv') for c in config['development_conditions']]
    for frame in frames:
        if not frame.id.is_unique:
            raise ValueError('Duplicate source IDs within condition')
        np.testing.assert_array_equal(frame.id, frames[0].id)
        np.testing.assert_array_equal(frame.label, frames[0].label)
        np.testing.assert_array_equal(frame.error, frame.prediction.ne(frame.label).astype(int))
    return pd.concat(frames, ignore_index=True)


def validation_objective(table, score, conditions):
    return float(np.mean([metrics(table.loc[table.condition.eq(c),'error'],
        np.asarray(score)[table.condition.eq(c)], table.loc[table.condition.eq(c),'id'])['aurc'] for c in conditions]))


def fit_mixture_mlp(x, y, xv, vt, seed, config):
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,1))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    x = torch.tensor(x,dtype=torch.float32); target=torch.tensor(y[:,None],dtype=torch.float32)
    xv = torch.tensor(xv,dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed)
    best=float('inf'); state=None; waiting=0; best_epoch=0; steps=0
    for epoch in range(config['epochs']):
        model.train()
        for idx in torch.randperm(len(x),generator=generator).split(config['batch_size']):
            optimizer.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(x[idx]),target[idx]).backward()
            optimizer.step(); steps+=1
        model.eval()
        with torch.inference_mode(): score=model(xv).sigmoid().numpy().ravel()
        objective=validation_objective(vt,score,config['development_conditions'])
        if objective < best:
            best=objective; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; waiting=0
        else: waiting+=1
        if waiting >= config['patience']: break
    model.load_state_dict(state)
    with torch.inference_mode(): score=model(xv).sigmoid().numpy().ravel()
    return state, score, dict(best_epoch=best_epoch, epochs_run=epoch+1, optimizer_steps=steps,
                             checkpoint_optimizer_steps=best_epoch*int(np.ceil(len(x)/config['batch_size'])), validation_objective=best)


def fit(root):
    config=check(root)
    if (root/'frozen.json').exists():
        verify(json.loads((root/'frozen.json').read_text())['files'])
        print('Adapted detectors already frozen; verified and reused.',flush=True)
        return
    prepared=json.loads((root/'prepared.json').read_text()); verify(prepared['files'])
    dt,vt=[development(root,s,config) for s in ('detector_train','val')]
    if set(dt.id)&set(vt.id): raise ValueError('Development source leakage')
    path=root/'detectors'; path.mkdir(exist_ok=True)
    selected={}; thresholds={}; scores={}; training=[]
    for key in 'AB':
        columns=FEATURES[key]
        assert not {'label','error','id','condition'}&set(columns)
        scaler=StandardScaler().fit(dt[columns])
        x,xv=scaler.transform(dt[columns]),scaler.transform(vt[columns])
        lr=LogisticRegression(C=config['lr_C'], max_iter=config['lr_max_iter'],random_state=42).fit(x,dt.error.to_numpy())
        joblib.dump(dict(model=lr,scaler=scaler,features=columns),path/f'{key}_lr.joblib')
        scores[f'{key}_lr']=lr.predict_proba(xv)[:,1]
        candidates=[]
        for seed in config['mlp_seeds']:
            state,score,details=fit_mixture_mlp(x,dt.error.to_numpy(),xv,vt,seed,config)
            torch.save(dict(state_dict=state,features=columns,mean=scaler.mean_,scale=scaler.scale_,seed=seed,**details),path/f'{key}_mlp_{seed}.pt')
            scores[f'{key}_mlp_{seed}']=score
            candidates.append((details['validation_objective'],seed))
            training.append(dict(feature_set=key,seed=seed,**details))
        selected[key]=min(candidates)[1]
        scores[f'{key}_mlp_selected']=scores[f'{key}_mlp_{selected[key]}']
        print(f'{key}: selected seed {selected[key]} by mean condition validation AURC',flush=True)
    clean=vt.condition.eq('clean').to_numpy()
    for name,score in scores.items():
        thresholds[name]={str(c):float(np.sort(score[clean])[int(np.ceil(c*clean.sum()))-1]) for c in config['coverage']}
    rows=[]
    for name,score in scores.items():
        for condition in config['development_conditions']:
            mask=vt.condition.eq(condition)
            rows.append(dict(detector=name,condition=condition,**metrics(vt.error[mask],score[mask],vt.id[mask])))
    pd.DataFrame(rows).to_csv(root/'validation_metrics.csv',index=False)
    pd.DataFrame(dict(id=vt.id,condition=vt.condition,error=vt.error,**scores)).to_csv(root/'validation_scores.csv',index=False)
    pd.DataFrame(training).to_csv(root/'training_history.csv',index=False)
    files=list(path.glob('*.pt'))+list(path.glob('*.joblib'))+[root/'validation_metrics.csv',root/'validation_scores.csv',root/'training_history.csv',root/'prepared.json']
    save_json(root/'frozen.json',dict(selected_mlp_seeds=selected,thresholds=thresholds,files=hashes(files),
        prepared_files=prepared['files'],registration_sha256=sha(root/'registration.json')))


class AdaptedBank:
    def __init__(self, root):
        self.sealed=json.loads((root/'frozen.json').read_text())
        verify(self.sealed['files']); verify(self.sealed['prepared_files'])
        if sha(root/'registration.json') != self.sealed['registration_sha256']: raise ValueError('Registration changed')
        self.root=root

    def predict(self, table):
        scores={}
        for key in 'AB':
            lr=joblib.load(self.root/'detectors'/f'{key}_lr.joblib')
            assert lr['features']==FEATURES[key]
            scores[f'{key}_lr']=lr['model'].predict_proba(lr['scaler'].transform(table[FEATURES[key]]))[:,1]
            for seed in (11,22,33):
                saved=torch.load(self.root/'detectors'/f'{key}_mlp_{seed}.pt',map_location='cpu',weights_only=False)
                assert saved['features']==FEATURES[key] and saved['seed']==seed
                model=nn.Sequential(nn.Linear(len(FEATURES[key]),32),nn.ReLU(),nn.Linear(32,1)).eval()
                model.load_state_dict(saved['state_dict'])
                x=(table[FEATURES[key]].to_numpy()-saved['mean'])/saved['scale']
                with torch.inference_mode(): scores[f'{key}_mlp_{seed}']=model(torch.tensor(x,dtype=torch.float32)).sigmoid().numpy().ravel()
            scores[f'{key}_mlp_selected']=scores[f"{key}_mlp_{self.sealed['selected_mlp_seeds'][key]}"]
        return scores


def contrast_statistics(errors,scores,ids,coverage):
    """All four fixed systems share the same examples and stable risk/ID order."""
    values={}; accepted={}
    for name,score in scores.items():
        order=np.lexsort((ids,score)); ordered=errors[order]
        values[name]=float((np.cumsum(ordered)/np.arange(1,len(errors)+1)).mean())
        accepted[name]={str(c):int(ordered[:int(np.ceil(c*len(errors)))] .sum()) for c in coverage}
    out={}
    for name,terms in {'original_B_minus_A':{'original_B':1,'original_A':-1},
                       'adapted_B_minus_A':{'adapted_B':1,'adapted_A':-1},
                       'Gamma':{'adapted_B':1,'adapted_A':-1,'original_B':-1,'original_A':1}}.items():
        out[name]=sum(w*values[k] for k,w in terms.items())
    for regime in ('original','adapted'):
        for c in coverage:
            out[f'{regime}_additional_errors_caught_{c}']=accepted[f'{regime}_A'][str(c)]-accepted[f'{regime}_B'][str(c)]
    return out


def bootstrap(errors,scores,ids,config):
    errors,ids=np.asarray(errors),np.asarray(ids)
    if not len(errors) or any(np.asarray(v).shape!=errors.shape for v in scores.values()): raise ValueError('Unaligned scores')
    point=contrast_statistics(errors,scores,ids,config['coverage'])
    rng=np.random.default_rng(config['bootstrap_seed'])
    draws={k:[] for k in point}
    for _ in range(config['bootstrap_resamples']):
        idx=rng.integers(len(errors),size=len(errors))
        sample=contrast_statistics(errors[idx],{k:np.asarray(v)[idx] for k,v in scores.items()},ids[idx],config['coverage'])
        for k,v in sample.items(): draws[k].append(v)
    return [dict(comparison=k,estimate=v,ci_low=float(np.quantile(draws[k],.025)),ci_high=float(np.quantile(draws[k],.975)),
                 resamples=config['bootstrap_resamples'],metric='accepted_error_count' if 'caught' in k else 'aurc') for k,v in point.items()]


def evaluate(root):
    config=check(root); bank=AdaptedBank(root)
    if (root/'evaluated.json').exists():
        verify(json.loads((root/'evaluated.json').read_text())['files']); return
    original_thresholds=json.loads((FINAL/'sealed_protocol.json').read_text())['thresholds']
    # Record immutable evaluation inputs only after adapted detector selection is sealed.
    inputs=[FINAL/c/f for c in config['evaluation_conditions'] for f in ('features.csv','scores.csv')]
    save_json(root/'evaluation_inputs.json',hashes(inputs))
    rows=[]; thresholds=[]; uncertainty=[]; sensitivity=[]; files=[]
    for condition in config['evaluation_conditions']:
        table=pd.read_csv(FINAL/condition/'features.csv').sort_values('id').reset_index(drop=True)
        original=pd.read_csv(FINAL/condition/'scores.csv').sort_values('id').reset_index(drop=True)
        if not table.id.is_unique: raise ValueError('Duplicate evaluation source IDs')
        np.testing.assert_array_equal(table.id,original.id); np.testing.assert_array_equal(table.error,original.error)
        np.testing.assert_array_equal(table.error,table.prediction.ne(table.label).astype(int))
        adapted=bank.predict(table)
        scores={f'original_{k}_{f}':original[f'{k}_{f}'].to_numpy() for k in 'AB' for f in ('lr','mlp_selected')}
        scores.update({f'adapted_{k}':v for k,v in adapted.items()})
        folder=root/'evaluation'/condition; folder.mkdir(parents=True,exist_ok=True)
        score_path=folder/'scores.csv'; pd.DataFrame(dict(id=table.id,error=table.error,**scores)).to_csv(score_path,index=False); files.append(score_path)
        masks=dict(all=np.ones(len(table),bool),unanimous=table.unanimous.to_numpy(bool),
                   confident_unanimous=(table.unanimous & table.confidence.ge(.9)).to_numpy())
        for subset,mask in masks.items():
            n=int(mask.sum()); wrong=int(table.error[mask].sum())
            for name,score in scores.items():
                if '_mlp_' in name and not name.endswith('selected'): continue
                result=metrics(table.error[mask],score[mask],table.id[mask]) if n else dict(aurc=None,auroc=None,risk80=None,risk90=None)
                if min(wrong,n-wrong)<20: result['auroc']=None
                order=np.lexsort((table.id.to_numpy()[mask],score[mask]))
                ordered_errors=table.error.to_numpy()[mask][order]
                counts={}
                for coverage in config['coverage']:
                    kept=int(np.ceil(coverage*n))
                    suffix=str(int(100*coverage))
                    counts[f'accepted{suffix}']=kept
                    counts[f'accepted_errors{suffix}']=int(ordered_errors[:kept].sum())
                rows.append(dict(condition=condition,subset=subset,detector=name,count=n,errors=wrong,**counts,**result))
                regime,detector=name.split('_',1)
                reference=original_thresholds if regime=='original' else bank.sealed['thresholds']
                for c in config['coverage']:
                    threshold=reference[detector][str(c)]; accept=mask&(score<=threshold)
                    thresholds.append(dict(condition=condition,subset=subset,detector=name,validation_coverage=c,threshold=threshold,
                        actual_coverage=float(accept.sum()/n) if n else None,accepted=int(accept.sum()),accepted_errors=int(table.error[accept].sum()),
                        risk=float(table.error[accept].mean()) if accept.any() else None))
        for family in ('lr','mlp_selected'):
            four={f'{regime}_{k}':scores[f'{regime}_{k}_{family}'] for regime in ('original','adapted') for k in 'AB'}
            uncertainty.extend(dict(condition=condition,family=family,**r) for r in bootstrap(table.error.to_numpy(),four,table.id.to_numpy(),config))
        for key in 'AB':
            for metric in ('aurc','auroc'):
                vals=[metrics(table.error,adapted[f'{key}_mlp_{seed}'],table.id)[metric] for seed in config['mlp_seeds']]
                sensitivity.append(dict(condition=condition,feature_set=key,metric=metric,mean=float(np.mean(vals)),std=float(np.std(vals)),values=json.dumps(vals)))
        print(f'Evaluated {condition}',flush=True)
    for name,data in [('metrics.csv',rows),('threshold_metrics.csv',thresholds),('bootstrap.csv',uncertainty),('seed_sensitivity.csv',sensitivity)]:
        p=root/name; pd.DataFrame(data).to_csv(p,index=False); files.append(p)
    files.append(root/'evaluation_inputs.json')
    save_json(root/'evaluated.json',dict(files=hashes(files),frozen_sha256=sha(root/'frozen.json')))
    report(root)


def report(root):
    check(root); AdaptedBank(root)
    complete=json.loads((root/'evaluated.json').read_text()); verify(complete['files'])
    if sha(root/'frozen.json')!=complete['frozen_sha256']: raise ValueError('Frozen detectors changed')
    verify(json.loads((root/'evaluation_inputs.json').read_text()))
    b=pd.read_csv(root/'bootstrap.csv')
    lines=['# Exploratory Gaussian-noise detector adaptation','',
        'CNNs and novelty references were fixed. A/B detectors were fitted on clean plus Gaussian noise 1/3/5 development images. Blur/JPEG were excluded from adaptation development. All test conditions had already been inspected; this is exploratory, conditional on one ensemble.','',
        'Negative AURC differences favor novelty. Recovery requires adapted B−A below zero with its interval excluding zero. Negative Gamma alone establishes no recovery. Intervals are paired 95% percentile intervals from 2,000 image resamples; multiple comparisons are unadjusted. No pooled analysis was performed.','',
        '| Condition | Family | Original B−A [CI] | Adapted B−A [CI] | Gamma [CI] |','| --- | --- | --- | --- | --- |']
    for (condition,family),g in b.groupby(['condition','family'],sort=False):
        cells=[]
        for name in ('original_B_minus_A','adapted_B_minus_A','Gamma'):
            r=g[g.comparison.eq(name)].iloc[0]; cells.append(f'{r.estimate:+.6f} [{r.ci_low:+.6f}, {r.ci_high:+.6f}]')
        lines.append('| '+' | '.join([condition,family]+cells)+' |')
    lines+=['','See metrics.csv for clean/shift and agreement-subset AUROC/AURC and exact-coverage risk; bootstrap.csv for matched-coverage additional-error counts and uncertainty; threshold_metrics.csv for actual coverage and accepted risk at clean-validation thresholds; seed_sensitivity.csv for all three adapted MLP seeds. Counts greater than zero favor adapted-feature B over A within the named training regime.','',
        'Training uses equal-weight mean per-condition validation AURC for checkpoint/seed selection. Clean-only original selections used clean validation AURC. Mixture epochs contain four times as many rows; training_history.csv records actual epochs and optimizer steps. Hence effects concern the full adaptation regime, not exposure isolated from training compute or model-selection changes.','',
        'Improvement after adaptation is consistent with a transfer limitation but does not establish a causal mechanism. A/B feature scalers are fitted on the adaptation mixture; novelty standardization remains the original clean reference. Failure to recover benefit does not establish the absence of useful representation information. Original H1 results are unchanged.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','fit','evaluate','report','all'],default='all')
    parser.add_argument('--config',type=Path,default=Path('configs/adaptation.json'))
    parser.add_argument('--output',type=Path,default=Path('results/adaptation'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    # Prevent accidental writes to original artifacts, including via output aliases.
    out=args.output.resolve()
    protected=[VALIDATION.resolve(),FINAL.resolve(),Path('results/exploratory').resolve()]
    if any(out==p or p in out.parents or out in p.parents for p in protected): raise ValueError('Output overlaps protected results')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if args.stage in ('prepare','all'): prepare(args.output,args.config,args.device)
        if args.stage in ('fit','all'): fit(args.output)
        if args.stage in ('evaluate','all'): evaluate(args.output)
        if args.stage=='report': report(args.output)


if __name__=='__main__':
    main()
