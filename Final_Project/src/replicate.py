"""Isolated ten-ensemble replication using a frozen source snapshot and compact retention."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2)


def initialize(repo,root):
    repo,root=repo.resolve(),root.resolve()
    if root.exists():
        record=json.loads((root/'registration.json').read_text())
        for p,h in record['snapshot_hashes'].items():
            if digest(root/p)!=h: raise ValueError(f'Snapshot changed: {p}')
        return
    root.mkdir(parents=True)
    snapshot=root/'snapshot'; snapshot.mkdir()
    shutil.copytree(repo/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(repo/'configs',snapshot/'configs')
    shutil.copytree(repo/'splits',snapshot/'splits')
    config=json.loads((snapshot/'configs/replication.json').read_text())
    seeds=[s for row in config['member_seeds'] for s in row]
    if len(seeds)!=30 or len(set(seeds))!=30 or set(seeds)&{1,2,3}: raise ValueError('Require 30 distinct new CNN seeds')
    if not (repo/'data/CIFAR-10-C/gaussian_noise.npy').exists(): raise FileNotFoundError('Existing CIFAR-10-C required')
    snapshot_hashes={str(p.relative_to(root)):digest(p) for p in snapshot.rglob('*') if p.is_file()}
    write(root/'registration.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),config=config,
        snapshot_hashes=snapshot_hashes,original_repo=str(repo),data=str((repo/'data').resolve()),
        original_results=str((repo/'results/final').resolve()),split_sha256=digest(snapshot/'splits/cifar10_seed42.npz')))
    for index,seeds in enumerate(config['member_seeds'],1):
        work=root/f'ensemble_{index:02d}'; work.mkdir()
        (work/'src').symlink_to(snapshot/'src',target_is_directory=True)
        (work/'data').symlink_to((repo/'data').resolve(),target_is_directory=True)
        (work/'splits').symlink_to(snapshot/'splits',target_is_directory=True)
        shutil.copytree(snapshot/'configs',work/'configs')
        ensemble=json.loads((work/'configs/ensemble.json').read_text())
        ensemble.update(output_dir=str(work/'training'),data_dir=str((repo/'data').resolve()),split_path=str(snapshot/'splits/cifar10_seed42.npz'))
        (work/'configs/ensemble.json').write_text(json.dumps(ensemble,indent=2))
        write(work/'identity.json',dict(ensemble=index,member_seeds=seeds,member_slots=[1,2,3],registration_sha256=digest(root/'registration.json')))
    (root/'logs').mkdir()
    print(f'Registered 10 ensembles at {root}',flush=True)


def invoke(work,stage):
    log=work/f'{stage}.log'
    with log.open('a') as f:
        subprocess.run([sys.executable,'-u','-B','-m','src.replicate','internal',stage],cwd=work,stdout=f,stderr=subprocess.STDOUT,check=True)


def completed(work,stage):
    return (work/f'{stage}_complete.json').exists()


def worker(root,index):
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError('A Slurm GPU allocation is required')
    root=root.resolve(); record=json.loads((root/'registration.json').read_text())
    for p,h in record['snapshot_hashes'].items():
        if digest(root/p)!=h: raise ValueError(f'Snapshot changed: {p}')
    work=root/f'ensemble_{index:02d}'
    if not work.exists(): raise ValueError('Unknown ensemble index')
    if (work/'compact_complete.json').exists():
        manifest=json.loads((work/'compact_complete.json').read_text())
        for p,h in manifest['retained'].items():
            if digest(work/p)!=h: raise ValueError(f'Retained artifact changed: {p}')
        return
    for stage in ['train','baseline','extensions','exploratory','adaptation','member_metrics','analysis','compact']:
        if not completed(work,stage): invoke(work,stage)
        print(f'ensemble {index:02d}: {stage} complete',flush=True)


def suppress_plots():
    from matplotlib.figure import Figure
    Figure.savefig=lambda *a,**kw: None


def train(work):
    import torch
    from . import train as t
    seeds=json.loads((work/'identity.json').read_text())['member_seeds']
    t.plot_history=lambda *a,**kw:None
    for slot,seed in enumerate(seeds,1):
        destination=work/'training'/f'seed_{slot}'
        if (destination/'trained.json').exists(): continue
        actual=work/'training'/f'seed_{seed}'
        if actual.exists() or destination.exists(): raise ValueError(f'Interrupted CNN run retained for diagnosis: {seed}')
        sys.argv=['train','--config','configs/ensemble.json','--seed',str(seed),'--device','cuda']
        t.main()
        # Remove optimizer state before any downstream protocol/checksum is created.
        checkpoint=torch.load(actual/'best.pt',map_location='cpu',weights_only=False)
        checkpoint.pop('optimizer_state_dict',None); checkpoint.pop('scheduler_state_dict',None)
        torch.save(checkpoint,actual/'best.pt')
        write(actual/'trained.json',dict(seed=seed,slot=slot,checkpoint_sha256=digest(actual/'best.pt')))
        actual.rename(destination)


def baseline(work):
    from . import validation as v, final_evaluation as f
    (work/'results/validation').mkdir(parents=True,exist_ok=True)
    if not (work/'results/validation/frozen_protocol.json').exists():
        sys.argv=['validation','--config','configs/ensemble.json','--device','cuda']; v.main()
    f.gallery=lambda *a,**kw:None
    for stage in ('clean','shift'):
        if (work/f'results/final/{stage}_complete.json').exists(): continue
        sys.argv=['final_evaluation','--stage',stage,'--device','cuda']; f.main()


def extensions(work):
    from . import run_extensions as e
    root=work/'results/extensions'; root.mkdir(exist_ok=True)
    if not (root/'original_files.json').exists():
        originals=[p for folder in ('results/validation','results/final','training') for p in (work/folder).rglob('*') if p.is_file()]
        write(root/'original_files.json',{str(p):digest(p) for p in originals})
    for study in ('h7','h8'):
        if (root/study/'evaluated.json').exists():
            e.verify_hashes(json.loads((root/study/'evaluated.json').read_text())['files']); continue
        e.register(root,study)
        e.prepare(root,study,'cuda')
        if not (root/study/'frozen.json').exists(): e.fit(root,study); e.freeze(root,study)
        e.evaluate(root,study,'cuda')
        e.audit_originals(root)


def exploratory(work):
    from . import exploratory_analysis as e
    e.ROOT=work; e.FINAL=work/'results/final'; e.OUT=work/'results/exploratory'
    e.main()


def adaptation(work):
    from . import adaptation as a
    root=work/'results/adaptation'; root.mkdir(exist_ok=True)
    a.prepare(root,Path('configs/adaptation.json'),'cuda')
    a.fit(root); a.evaluate(root)


def member_metrics(work):
    import numpy as np
    import pandas as pd
    from .metrics import metrics
    from .validation import output_features
    rows=[]
    for p in sorted((work/'results/final').glob('*.npz')):
        with np.load(p) as cache:
            for member in range(3):
                features=output_features(cache['logits'][:,member:member+1])
                error=(features['prediction']!=cache['label']).astype(int)
                for name,sign in [('confidence',-1),('entropy',1),('margin',-1)]:
                    rows.append(dict(condition=p.stem,member=member+1,accuracy=float(1-error.mean()),errors=int(error.sum()),
                        score=name,**metrics(error,sign*features[name],cache['id'])))
    pd.DataFrame(rows).to_csv(work/'results/single_member_metrics.csv',index=False)


def analysis(work):
    """All ablations/subsets per condition; pooled tests cluster source images."""
    import numpy as np
    import pandas as pd
    from .run_extensions import bootstrap_contrasts
    contrasts={}
    for family in ('lr','mlp_selected'):
        for left,right in [('B','A'),('C','B'),('D','C')]:
            contrasts[f'{family}:{left}_minus_{right}']={f'{left}_{family}':1,f'{right}_{family}':-1}
    for key in 'ABCD':
        contrasts[f'{key}:MLP_minus_LR']={f'{key}_mlp_selected':1,f'{key}_lr':-1}
    intervals=[]; counts=[]; sensitivity=[]; pooled=[]; pooled_scores={}
    conditions=json.loads(Path('configs/adaptation.json').read_text())['evaluation_conditions']
    for condition in conditions:
        table=pd.read_csv(work/f'results/final/{condition}/features.csv')
        scores=pd.read_csv(work/f'results/final/{condition}/scores.csv')
        np.testing.assert_array_equal(table.id,scores.id)
        np.testing.assert_array_equal(table.error,scores.error)
        masks=dict(all=np.ones(len(table),bool),unanimous=table.unanimous.to_numpy(bool),
            confident_unanimous=(table.unanimous&table.confidence.ge(.9)).to_numpy())
        for subset,mask in masks.items():
            if not mask.any(): continue
            current={k:scores[k].to_numpy()[mask] for k in scores if k not in ('id','error')}
            rows=bootstrap_contrasts(table.error.to_numpy()[mask],current,table.id.to_numpy()[mask],contrasts)
            intervals.extend(dict(condition=condition,subset=subset,**r) for r in rows)
            for family in ('lr','mlp_selected'):
                for c in (.8,.9):
                    accepted={}
                    for key in 'ABCD':
                        order=np.lexsort((table.id.to_numpy()[mask],current[f'{key}_{family}']))
                        accepted[key]=int(table.error.to_numpy()[mask][order[:int(np.ceil(c*mask.sum()))]].sum())
                    counts.append(dict(condition=condition,subset=subset,family=family,coverage=c,
                        **{f'accepted_errors_{k}':v for k,v in accepted.items()},additional_B_vs_A=accepted['A']-accepted['B'],additional_C_vs_B=accepted['B']-accepted['C']))
        for family in ('lr','mlp_selected'):
            # Reuse four-system paired-count implementation to quantify A/B and B/C.
            from .adaptation import bootstrap
            cfg=json.loads(Path('configs/adaptation.json').read_text())
            four=dict(original_A=scores[f'A_{family}'].to_numpy(),original_B=scores[f'B_{family}'].to_numpy(),
                      adapted_A=scores[f'B_{family}'].to_numpy(),adapted_B=scores[f'C_{family}'].to_numpy())
            for r in bootstrap(table.error.to_numpy(),four,table.id.to_numpy(),cfg):
                if r['metric']=='accepted_error_count':
                    r['comparison']=r['comparison'].replace('original_','B_vs_A_').replace('adapted_','C_vs_B_')
                    intervals.append(dict(condition=condition,subset='all',comparison=f"{family}:{r['comparison']}",delta_aurc=None,**{k:v for k,v in r.items() if k!='comparison'}))
        for seed in (11,22,33):
            from .metrics import metrics
            b=metrics(table.error,scores[f'B_mlp_{seed}'],table.id)['aurc']
            c=metrics(table.error,scores[f'C_mlp_{seed}'],table.id)['aurc']
            sensitivity.append(dict(condition=condition,seed=seed,B_aurc=b,C_aurc=c,C_minus_B=c-b))
        if condition!='clean':
            pooled.append(table)
            for k in scores:
                if k not in ('id','error'): pooled_scores.setdefault(k,[]).append(scores[k].to_numpy())
    all_images=pd.concat(pooled,ignore_index=True)
    rows=bootstrap_contrasts(all_images.error.to_numpy(),{k:np.concatenate(v) for k,v in pooled_scores.items()},np.arange(len(all_images)),contrasts,groups=all_images.id.to_numpy())
    intervals.extend(dict(condition='pooled_source_clustered',subset='all',**r) for r in rows)
    pd.DataFrame(intervals).to_csv(work/'results/replication_bootstrap.csv',index=False)
    pd.DataFrame(counts).to_csv(work/'results/matched_coverage.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(work/'results/paired_seed_sensitivity.csv',index=False)


def compact(work):
    """Prune only this newly generated workspace after all dependent stages succeed."""
    import numpy as np
    for stage in ('train','baseline','extensions','exploratory','adaptation','member_metrics','analysis'):
        if not completed(work,stage): raise ValueError(f'Cannot compact before {stage}')
    from .final_evaluation import verify_artifacts
    from .adaptation import AdaptedBank
    verify_artifacts(work/'results/validation'); AdaptedBank(work/'results/adaptation')
    for study in ('h7','h8'):
        from .run_extensions import verify_hashes
        verify_hashes(json.loads((work/f'results/extensions/{study}/evaluated.json').read_text())['files'])
    removed=[]
    # Keep raw logits/labels/IDs for all baseline conditions and development splits;
    # discard huge raw representations only after every downstream consumer finished.
    for folder in ('results/validation','results/final'):
        for path in (work/folder).glob('*.npz'):
            if path.name=='novelty_stats.npz': continue
            with np.load(path) as z:
                arrays={k:z[k] for k in ('id','label','logits') if k in z}
            if not arrays: continue
            target=path.with_name(path.stem+'_outputs.npz')
            np.savez_compressed(target,**arrays)
            removed.append(dict(path=str(path.relative_to(work)),sha256=digest(path),bytes=path.stat().st_size,reason='representation cache; compact logits retained'))
            path.unlink()
    # Early raw features and adaptation development raw inference are regenerable.
    for folder in ('results/extensions/h8/depth_cache','results/adaptation/development'):
        for p in (work/folder).rglob('*.npz'):
            removed.append(dict(path=str(p.relative_to(work)),sha256=digest(p),bytes=p.stat().st_size,reason='derived intermediate; geometry, feature tables and checkpoints retained'))
            p.unlink()
    for path in (work/'results').rglob('*.csv'):
        target=path.with_suffix('.csv.gz')
        with path.open('rb') as f,gzip.open(target,'wb',compresslevel=6) as out: shutil.copyfileobj(f,out)
        if gzip.open(target,'rb').read()!=path.read_bytes(): raise ValueError('Lossy CSV compression')
        path.unlink()
    write(work/'retention.json',dict(removed=removed,removed_bytes=sum(r['bytes'] for r in removed),
        policy='All stages verified before cleanup; original protocol hashes refer to deliberately retired caches. Expand CSV gzip for table analysis. Raw representations require CNN re-inference; no retraining needed.'))
    retained={str(p.relative_to(work)):digest(p) for folder in ('results','training') for p in (work/folder).rglob('*') if p.is_file()}
    write(work/'compact_complete.json',dict(retained=retained,retained_bytes=sum((work/p).stat().st_size for p in retained)))


def internal(stage):
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(2); suppress_plots()
    work=Path.cwd()
    actions=dict(train=train,baseline=baseline,extensions=extensions,exploratory=exploratory,adaptation=adaptation,member_metrics=member_metrics,analysis=analysis,compact=compact)
    with threadpool_limits(2): actions[stage](work)
    if stage!='compact': write(work/f'{stage}_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()))


def aggregate(root):
    import numpy as np
    import pandas as pd
    records=[]; status=[]
    for index in range(1,11):
        work=root/f'ensemble_{index:02d}'
        status.append(dict(ensemble=index,complete=(work/'compact_complete.json').exists()))
        for study,path in [('replication_analysis',work/'results/replication_bootstrap.csv'),('original',work/'results/final/clean_bootstrap.csv'),('h7',work/'results/extensions/h7/bootstrap.csv'),('h8',work/'results/extensions/h8/bootstrap.csv'),('adaptation',work/'results/adaptation/bootstrap.csv')]:
            source=path if path.exists() else path.with_suffix('.csv.gz')
            if not source.exists(): continue
            frame=pd.read_csv(source); frame['ensemble']=index; frame['study']=study
            if 'condition' not in frame: frame['condition']='clean'
            if 'estimate' in frame:
                frame=frame[frame.metric.eq('aurc')].copy()
                frame['delta_aurc']=frame['estimate']
            records.append(frame)
    pd.DataFrame(status).to_csv(root/'status.csv',index=False)
    if not records: return
    joined=pd.concat(records,ignore_index=True)
    joined.to_csv(root/'all_ensemble_contrasts.csv',index=False)
    # Summarize every contrast/condition as well as the central clean comparisons.
    detail=joined.copy()
    if 'family' not in detail: detail['family']='unspecified'
    detail['family']=detail['family'].fillna('encoded_in_comparison')
    if 'subset' not in detail: detail['subset']='all'
    detail['subset']=detail['subset'].fillna('all')
    effects=[]
    rng_detail=np.random.default_rng(20261109)
    for keys,g in detail.dropna(subset=['delta_aurc']).groupby(['study','condition','subset','family','comparison'],dropna=False):
        values=g.delta_aurc.to_numpy()
        draws=rng_detail.choice(values,size=(10000,len(values)),replace=True).mean(1)
        effects.append(dict(zip(['study','condition','subset','family','comparison'],keys),
            completed_ensembles=len(values),mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else None,
            median=float(np.median(values)),fraction_negative=float((values<0).mean()),
            ci_low=float(np.quantile(draws,.025)) if len(values)>1 else None,ci_high=float(np.quantile(draws,.975)) if len(values)>1 else None))
    pd.DataFrame(effects).to_csv(root/'all_effect_summaries.csv',index=False)
    metric_frames=[]
    for index in range(1,11):
        work=root/f'ensemble_{index:02d}'
        for study,rel in [('baseline','results/final/all_metrics.csv'),('h7','results/extensions/h7/metrics.csv'),('h8','results/extensions/h8/metrics.csv'),('adaptation','results/adaptation/metrics.csv'),('single_cnn','results/single_member_metrics.csv')]:
            path=work/rel
            path=path if path.exists() else path.with_suffix('.csv.gz')
            if path.exists():
                frame=pd.read_csv(path); frame['ensemble']=index; frame['study']=study; metric_frames.append(frame)
    if metric_frames: pd.concat(metric_frames,ignore_index=True).to_csv(root/'all_ensemble_metrics.csv',index=False)
    base=joined[joined.study.eq('original') & joined.comparison.isin(['B_minus_A','C_minus_B'])]
    summary=[]; rng=np.random.default_rng(20261108)
    for (family,comparison),g in base.groupby(['family','comparison']):
        values=g.delta_aurc.to_numpy(); draws=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
        summary.append(dict(family=family,comparison=comparison,completed_ensembles=len(values),mean=values.mean(),
            std=values.std(ddof=1) if len(values)>1 else None,median=np.median(values),fraction_negative=float((values<0).mean()),
            ensemble_bootstrap_ci_low=float(np.quantile(draws,.025)) if len(values)>1 else None,
            ensemble_bootstrap_ci_high=float(np.quantile(draws,.975)) if len(values)>1 else None))
    pd.DataFrame(summary).to_csv(root/'ensemble_summary.csv',index=False)
    (root/'SUMMARY.md').write_text('# Ensemble replication\n\n'+str(sum(x['complete'] for x in status))+'/10 completed. All per-ensemble contrasts are retained in all_ensemble_contrasts.csv; ensemble_summary.csv summarizes the ten new ensembles only. Partial results are provisional; no ensemble is selected by test performance. Ensemble-resampling intervals address training randomness conditional on the shared split and dataset, not new-data uncertainty. Original results remain separate.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','worker','internal','aggregate'])
    parser.add_argument('stage',nargs='?')
    parser.add_argument('--root',type=Path,default=Path('/common/users/ns1788/beyond-ensemble-agreement/replication_v1'))
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command=='init': initialize(args.repo,args.root)
    elif args.command=='worker': worker(args.root,args.index)
    elif args.command=='internal': internal(args.stage)
    else: aggregate(args.root)


if __name__=='__main__': main()
