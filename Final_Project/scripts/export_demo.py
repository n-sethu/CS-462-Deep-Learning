"""Export cached evaluation artifacts; never run CNN inference or bootstrap."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demo'))
from calculations import ranked, choose_examples

CONDITIONS=['clean']+[f'{c}_{s}' for c in ['gaussian_noise','motion_blur','jpeg_compression'] for s in [1,3,5]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/replication'));p.add_argument('--out',type=Path,default=Path('demo/assets'));p.add_argument('--data',type=Path,default=Path('data'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    hashes={};checks=[];thumbnail_ids={c:set() for c in CONDITIONS}
    def read(path):
        if not path.exists():path=path.with_suffix('.csv.gz')
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    for ens in range(1,11):
        work=a.root/f'ensemble_{ens:02d}/results';saved=read(work/'adaptation/metrics.csv')
        for condition in CONDITIONS:
            f=read(work/f'final/{condition}/features.csv');s=read(work/f'adaptation/evaluation/{condition}/scores.csv')
            np.testing.assert_array_equal(f.id,s.id);np.testing.assert_array_equal(f.error,s.error)
            np.testing.assert_array_equal(f.error,(f.label!=f.prediction).astype(int))
            assert len(f)==10000 and f.id.is_unique and set(f.id)==set(range(10000))
            arrays=dict(id=f.id.to_numpy(dtype=np.uint16),label=f.label.to_numpy(dtype=np.uint8),prediction=f.prediction.to_numpy(dtype=np.uint8),error=f.error.to_numpy(dtype=np.uint8),confidence=f.confidence.to_numpy(dtype=np.float64))
            for regime,prefix in [('clean_trained','original'),('adapted','adapted')]:
                for key in 'AB':
                    name=f'{prefix}_{key}_mlp_selected';score=s[name].to_numpy(dtype=np.float64);arrays[f'{regime}_{key}']=score
                    expected=saved[saved.condition.eq(condition)&saved.subset.eq('all')&saved.detector.eq(name)]
                    assert len(expected)==1
                    for cov in [.8,.9]:
                        v=ranked(f.error,score,f.id,cov);row=expected.iloc[0];pct=int(cov*100)
                        assert v['errors']==int(row[f'accepted_errors{pct}'])
                        assert v['kept']==int(row[f'accepted{pct}'])
                        np.testing.assert_allclose(v['risk'],row[f'risk{pct}'],atol=1e-12)
                        np.testing.assert_allclose(v['aurc'],row.aurc,atol=1e-12)
                        np.testing.assert_allclose(roc_auc_score(f.error,score),row.auroc,atol=1e-12)
                        checks.append(dict(ensemble=ens,condition=condition,regime=regime,detector=key,coverage=cov,accepted=v['kept'],accepted_errors=v['errors']))
                    # Curves contain every integer-percent operating point and finer context.
                    idx=np.unique(np.r_[0,np.arange(19,10000,20),np.arange(99,10000,100),9999])
                    arrays['curve_coverage']=(idx+1)/10000
                    arrays[f'{regime}_{key}_curve']=v['curve'][idx]
                for percent in range(50,101):
                    decisions=[ranked(f.error,arrays[f'{regime}_{key}'],f.id,percent/100)['accepted'] for key in 'AB']
                    for _,j in choose_examples(f.error.to_numpy(),f.id.to_numpy(),*decisions):
                        if j is not None:thumbnail_ids[condition].add(int(f.id.iloc[j]))
            np.savez_compressed(a.out/f'ensemble_{ens:02d}_{condition}.npz',**arrays)
        print('Verified and exported ensemble',ens,flush=True)
    # Ten shared image atlases, independent of ensemble; IDs select original source images.
    with (a.data/'cifar-10-batches-py/test_batch').open('rb') as stream:raw=pickle.load(stream,encoding='bytes')
    clean=raw[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    with (a.data/'cifar-10-batches-py/batches.meta').open('rb') as stream:meta=pickle.load(stream,encoding='bytes')
    labels=[x.decode() for x in meta[b'label_names']]
    positions={}
    for condition in CONDITIONS:
        if condition=='clean':images=clean
        else:
            name,severity=condition.rsplit('_',1);images=np.load(a.data/f'CIFAR-10-C/{name}.npy',mmap_mode='r')[(int(severity)-1)*10000:int(severity)*10000]
        chosen=sorted(thumbnail_ids[condition]);positions[condition]={str(image_id):j for j,image_id in enumerate(chosen)}
        atlas=Image.new('RGB',(3200,32*max(1,int(np.ceil(len(chosen)/100)))))
        for j,image_id in enumerate(chosen):atlas.paste(Image.fromarray(images[image_id]),((j%100)*32,(j//100)*32))
        atlas.save(a.out/f'{condition}.webp',quality=85,method=6)
    summary=read(Path('results/replication_analysis/effect_summary.csv'))
    summary[~summary.condition.eq('pooled_source_clustered')].to_csv(a.out/'replication_effects.csv',index=False)
    individual=read(Path('results/replication_analysis/per_ensemble_effects.csv'))
    individual[~individual.condition.eq('pooled_source_clustered')][['study','condition','subset','family','comparison','ensemble','delta_aurc']].to_csv(a.out/'replication_individual.csv',index=False)
    pd.DataFrame(checks).to_csv(a.out/'verification.csv',index=False)
    files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(a.out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (a.out/'manifest.json').write_text(json.dumps(dict(default_ensemble=1,ensemble_count=10,conditions=CONDITIONS,labels=labels,thumbnail_positions=positions,detector_family='validation-selected MLP',scores='float64, no rounding; increasing risk; ties by ascending source image ID',coverage='ceil(coverage*N) accepted',thumbnail_note='Only gallery thumbnails needed for the 50–100% integer coverage controls are exported, from actual CIFAR-10 / CIFAR-10-C source images; lossy WebP, original IDs retained',verification_checks=len(checks),files=files,source_hashes=hashes),indent=2))
    print('Export verified:',len(checks),'count checks;',sum(x['bytes'] for x in files.values())/1e6,'MB')

if __name__=='__main__':main()
