"""Rebuild three figures and a verified adaptation interpretation from caches only.

Run: python results/adaptation/analyze.py
No model fitting, inference, new selection, or new bootstrap tests are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONDITIONS=['clean']+[f'{c}_{s}' for c in ('gaussian_noise','motion_blur','jpeg_compression') for s in (1,3,5)]
LABELS=['Clean','Noise 1','Noise 3','Noise 5','Blur 1','Blur 3','Blur 5','JPEG 1','JPEG 3','JPEG 5']
FAMILIES=['lr','mlp_selected']
COLORS={'original':'#4575b4','adapted':'#d95f02'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def direction(low,high,negative_better=True):
    if high<0:return 'benefit' if negative_better else 'harm'
    if low>0:return 'harm' if negative_better else 'benefit'
    return 'inconclusive'


def ranked_metrics(errors,score,ids):
    errors,score,ids=map(np.asarray,(errors,score,ids))
    if not (errors.shape==score.shape==ids.shape) or not np.isfinite(score).all():raise ValueError('Unaligned/nonfinite scores')
    order=np.lexsort((ids,score)); ordered=errors[order]
    risk=np.cumsum(ordered)/np.arange(1,len(ordered)+1)
    values=dict(aurc=float(risk.mean()),auroc=float(roc_auc_score(errors,score)) if len(np.unique(errors))==2 else None)
    for c in (80,90):
        n=int(np.ceil(c/100*len(ordered)))
        values.update({f'accepted{c}':n,f'accepted_errors{c}':int(ordered[:n].sum()),f'risk{c}':float(risk[n-1])})
    return values


def load_and_verify(root):
    repo=Path(__file__).resolve().parents[2]
    files={name:root/name for name in ['bootstrap.csv','metrics.csv','threshold_metrics.csv','seed_sensitivity.csv',
        'training_history.csv','validation_metrics.csv','validation_scores.csv','frozen.json','registration.json','environment.json','evaluated.json','evaluation_inputs.json']}
    frames={name[:-4]:pd.read_csv(path) for name,path in files.items() if name.endswith('.csv')}
    complete=json.loads(files['evaluated.json'].read_text())
    for path,digest in complete['files'].items():
        p=Path(path); p=p if p.is_absolute() else repo/p
        if sha(p)!=digest:raise ValueError(f'Evaluated artifact changed: {p}')
    if sha(files['frozen.json'])!=complete['frozen_sha256']:raise ValueError('Frozen detector record changed')
    m,b,t=frames['metrics'],frames['bootstrap'],frames['threshold_metrics']
    if m.duplicated(['condition','subset','detector']).any() or b.duplicated(['condition','family','comparison']).any():raise ValueError('Duplicate result rows')
    if set(m.condition)!=set(CONDITIONS):raise ValueError('Missing/extra conditions')
    for condition in CONDITIONS:
        p=root/'evaluation'/condition/'scores.csv'; files[f'scores_{condition}']=p
        scores=pd.read_csv(p)
        if not scores.id.is_unique:raise ValueError('Duplicate example IDs')
        actual=m[m.condition.eq(condition)&m.subset.eq('all')].set_index('detector')
        for detector,row in actual.iterrows():
            values=ranked_metrics(scores.error,scores[detector],scores.id)
            for key,value in values.items():
                if value is not None:np.testing.assert_allclose(value,row[key],rtol=1e-7,atol=1e-10)
        for family in FAMILIES:
            selected=b[b.condition.eq(condition)&b.family.eq(family)].set_index('comparison')
            original=actual.loc[f'original_B_{family}','aurc']-actual.loc[f'original_A_{family}','aurc']
            adapted=actual.loc[f'adapted_B_{family}','aurc']-actual.loc[f'adapted_A_{family}','aurc']
            for comparison,value in [('original_B_minus_A',original),('adapted_B_minus_A',adapted),('Gamma',adapted-original)]:
                np.testing.assert_allclose(selected.loc[comparison,'estimate'],value,atol=1e-12)
            for regime in ('original','adapted'):
                for coverage in (80,90):
                    difference=actual.loc[f'{regime}_A_{family}',f'accepted_errors{coverage}']-actual.loc[f'{regime}_B_{family}',f'accepted_errors{coverage}']
                    np.testing.assert_allclose(selected.loc[f'{regime}_additional_errors_caught_{coverage/100}','estimate'],difference)
        for row in t[t.condition.eq(condition)&t.subset.eq('all')].itertuples():
            accept=scores[row.detector].le(row.threshold)
            assert int(accept.sum())==row.accepted
            assert int(scores.error[accept].sum())==row.accepted_errors
            np.testing.assert_allclose(accept.mean(),row.actual_coverage,atol=1e-12)
            if accept.any():np.testing.assert_allclose(scores.error[accept].mean(),row.risk,atol=1e-12)
    # Subset rows retain their original construction; verify internal risk/count arithmetic.
    for c in (80,90):
        valid=m[f'accepted{c}'].gt(0)
        np.testing.assert_allclose(m.loc[valid,f'accepted_errors{c}']/m.loc[valid,f'accepted{c}'],m.loc[valid,f'risk{c}'],atol=1e-12)
    return frames,files


def interval_plot(ax,frame,xscale=1,offset=0,label=None,color=None):
    d=frame.set_index('condition').loc[CONDITIONS]
    # Draw endpoints directly: a percentile interval need not contain its point estimate.
    y=np.arange(len(CONDITIONS))+offset
    ax.hlines(y,d.ci_low*xscale,d.ci_high*xscale,color=color,lw=1.6)
    ax.scatter(d.estimate*xscale,y,s=27,color=color,label=label,zorder=3)


def figures(out,m,b):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
    for ax,family,title in zip(axes,FAMILIES,['Logistic regression','Validation-selected MLP']):
        for regime,offset in [('original',-.12),('adapted',.12)]:
            d=b[b.family.eq(family)&b.comparison.eq(f'{regime}_B_minus_A')]
            interval_plot(ax,d,1000,offset,'Clean-trained' if regime=='original' else 'Noise-adapted',COLORS[regime])
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(10),LABELS);ax.invert_yaxis() if not ax.yaxis_inverted() else None
        ax.set(title=title,xlabel='(AURC B − A) × 1,000; negative favors novelty');ax.grid(axis='x',alpha=.2)
    axes[0].legend(loc='best');fig.suptitle('1. Noise adaptation recovers novelty benefit only in specific settings\nPoints and existing paired 95% percentile intervals',fontsize=12)
    fig.tight_layout();fig.savefig(out/'01_incremental_benefit.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
    for ax,family,title in zip(axes,FAMILIES,['Logistic regression','Validation-selected MLP']):
        d=m[m.subset.eq('all')].pivot(index='condition',columns='detector',values='aurc').loc[CONDITIONS]
        for key,offset,color in [('A',-.12,'#666666'),('B',.12,'#1b9e77')]:
            change=d[f'adapted_{key}_{family}']-d[f'original_{key}_{family}']
            ax.scatter(change*1000,np.arange(10)+offset,label=f'Detector {key}',color=color,s=30)
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(10),LABELS)
        if not ax.yaxis_inverted():ax.invert_yaxis()
        ax.set(title=title,xlabel='(Adapted − original AURC) × 1,000; negative improves');ax.grid(axis='x',alpha=.2)
    axes[0].legend();fig.suptitle('2. Does adaptation improve the detector itself?\nPoint estimates only; no direct adaptation-effect intervals were computed',fontsize=12)
    fig.tight_layout();fig.savefig(out/'02_adaptation_effect.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,10),sharey=True)
    for j,family in enumerate(FAMILIES):
        for i,coverage in enumerate((.8,.9)):
            ax=axes[i,j]
            for regime,offset in [('original',-.12),('adapted',.12)]:
                d=b[b.family.eq(family)&b.comparison.eq(f'{regime}_additional_errors_caught_{coverage}')]
                interval_plot(ax,d,1,offset,'Clean-trained' if regime=='original' else 'Noise-adapted',COLORS[regime])
            ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(10),LABELS)
            if not ax.yaxis_inverted():ax.invert_yaxis()
            ax.set(title=f'{"LR" if j==0 else "MLP"}: {int(coverage*100)}% coverage',xlabel='Additional errors rejected by B vs A; positive favors novelty');ax.grid(axis='x',alpha=.2)
    axes[0,0].legend();fig.suptitle('3. Matched coverage: same number accepted, different errors retained\nExisting paired 95% intervals; counts concern B vs A within each training regime',fontsize=12)
    fig.tight_layout();fig.savefig(out/'03_matched_coverage.png',dpi=180);plt.close(fig)


def md_table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+['']


def report(out,frames):
    m,b,t=frames['metrics'],frames['bootstrap'],frames['threshold_metrics']
    classify=b[b.metric.eq('aurc')].copy()
    classify['interpretation']=[direction(x.ci_low,x.ci_high) for x in classify.itertuples()]
    classify.to_csv(out/'interval_interpretations.csv',index=False)
    pivot=m[m.subset.eq('all')].pivot(index='condition',columns='detector',values='aurc').loc[CONDITIONS]
    effects=[]
    for condition in CONDITIONS:
        for family in FAMILIES:
            effects.append(dict(condition=condition,family=family,**{f'{key}_adapted_minus_original':pivot.loc[condition,f'adapted_{key}_{family}']-pivot.loc[condition,f'original_{key}_{family}'] for key in 'AB'}))
    pd.DataFrame(effects).to_csv(out/'absolute_adaptation_effects.csv',index=False)
    lines=['# What Gaussian-noise detector adaptation establishes','',
        '**Main finding:** noise adaptation recovers mean novelty’s incremental error-detection benefit for the MLP under Gaussian noise severities 3 and 5. It does not recover that benefit under mild noise or with logistic regression. The MLP gain comes with worse clean performance and worse transfer to every tested blur/JPEG condition. This is conditional evidence of a usable shifted feature relationship, not a universal fix or a causal mechanism.','',
        '## Design and evidence checks','',
        'A uses four output features; B adds mean predicted-class novelty. Original detectors trained on clean detector-train images. Adapted detectors trained on equally weighted clean/noise-1/noise-3/noise-5 versions of the same 10,000 detector-training sources. Corresponding versions of 5,000 validation sources set checkpoint and seed selection by equal-weight mean per-condition AURC. CNN weights, centroids, clean novelty standardization, predictions and test error targets were fixed. Adapted feature scalers used the training mixture. Blur and JPEG were excluded from adaptation development.','',
        'This script reads every evaluation score table and the metric, bootstrap, threshold, seed-sensitivity, training-history and validation tables. It verifies the completed evaluation manifest, all-example AUROC/AURC and coverage counts against cached scores, contrast arithmetic, and frozen-threshold decisions. It does not fit models, alter selections, or recompute bootstrap intervals. The figures use existing 2,000-resample paired percentile intervals; no multiplicity correction or pooling is applied.','',
        '## 1. Novelty benefit is recovered selectively','',
        'Define Δ = AURC(B) − AURC(A). Negative favors novelty. Gamma = Δadapted − Δoriginal; negative Gamma means the incremental effect improved, but recovery additionally requires negative adapted Δ with its interval below zero.','']
    rows=[]
    for condition in CONDITIONS:
        row=[condition]
        for family in FAMILIES:
            x=classify[classify.condition.eq(condition)&classify.family.eq(family)&classify.comparison.eq('adapted_B_minus_A')].iloc[0]
            row.append(f'{x.estimate:+.6f} [{x.ci_low:+.6f}, {x.ci_high:+.6f}]; {x.interpretation}')
        rows.append(row)
    lines+=md_table(['Condition','Adapted LR B−A [95% CI]','Adapted MLP B−A [95% CI]'],rows)
    lines+=['MLP noise-3 Δ moves from +0.006791 to −0.014348, with Gamma −0.021140 [−0.024770, −0.017489]. Noise-5 Δ moves from +0.002850 to −0.009998, with Gamma −0.012848 [−0.016783, −0.008999]. Both satisfy the recovery criterion under the registered exploratory procedure. Mild noise instead worsens: adapted Δ=+0.002873.','',
        'LR remains harmful under all three noise severities. Its negative Gamma at mild noise is an instructive counterexample: the harm shrinks but does not become a benefit. LR retains benefits on clean and all blur conditions, and JPEG 1/5; JPEG 3 is inconclusive. Thus adaptation is not intrinsically harmful, and the recovery result depends on the detector family. This experiment alone does not isolate nonlinearity as its cause.','',
        '![Incremental novelty benefit](01_incremental_benefit.png)','',
        '## 2. Recovery reflects better B performance, not merely a worse A','',
        'Absolute selected-MLP AURCs below retain all four systems. Differences across corruption conditions also reflect different error prevalence; compare systems within each row.','']
    lines+=md_table(['Condition','Original A','Original B','Adapted A','Adapted B'],[[c]+[f'{pivot.loc[c,f"{r}_{k}_mlp_selected"]:.6f}' for r,k in [('original','A'),('original','B'),('adapted','A'),('adapted','B')]] for c in CONDITIONS])
    lines+=['Under noise 3, adapted B improves from 0.473270 to 0.449849 while A also improves slightly (0.466479→0.464198). Under noise 5, B improves 0.634157→0.615542 while A improves 0.631307→0.625540. Therefore the recovered B−A effect is not produced solely by degrading A. Direct adapted-versus-original A/B differences are point estimates; the stored intervals test B−A and Gamma, not each direct adaptation effect.','',
        'On clean images, adapted B worsens from 0.004955 to 0.006253 (about 26.2% higher AURC), and AUROC falls from 0.9401 to 0.9190. Under blur/JPEG, adapted B also worsens in every condition, while A changes comparatively little. The formerly helpful mean-novelty input becomes harmful for the selected adapted MLP. This supports a specialization/tradeoff interpretation rather than preserved cross-corruption usefulness.','',
        '![Direct adaptation effects](02_adaptation_effect.png)','',
        '## 3. Equal coverage confirms practical gains and losses','',
        'At 90% coverage all systems keep exactly 9,000 predictions. Positive additional errors caught = accepted errors(A) − accepted errors(B). The table uses adapted MLPs; paired count intervals are from the original bootstrap file.','']
    rows=[]
    for c in CONDITIONS:
        mm=m[m.condition.eq(c)&m.subset.eq('all')].set_index('detector')
        x=b[b.condition.eq(c)&b.family.eq('mlp_selected')&b.comparison.eq('adapted_additional_errors_caught_0.9')].iloc[0]
        rows.append([c,int(mm.loc['adapted_A_mlp_selected','accepted_errors90']),int(mm.loc['adapted_B_mlp_selected','accepted_errors90']),f'{x.estimate:+.0f} [{x.ci_low:+.1f}, {x.ci_high:+.1f}]'])
    lines+=md_table(['Condition','Adapted A errors','Adapted B errors','Additional caught [95% CI]'],rows)
    lines+=['At noise 3, B catches 65 more errors than adapted A at 90% coverage; at noise 5 it catches 50 more, and both count intervals exclude zero. At 80%, the corresponding gains are 110 and 72. Clean adaptation reverses the original benefit: original B catches nine more errors than original A at 90%, while adapted B leaves 35 more errors than adapted A. Blur/JPEG all show negative adapted-MLP count differences at 90%, with intervals below zero. These are benefits/losses at matched rejection budgets, not merely effects of rejecting more images.','',
        '![Matched coverage](03_matched_coverage.png)','',
        '## 4. Frozen thresholds improve severe-noise risk but do not make it safe','',
        'Each system retains its own clean-validation threshold targeting 90% coverage. Cells show actual coverage / error rate among accepted images; they are not exact-coverage comparisons.','']
    rows=[]
    for c in CONDITIONS:
        tt=t[t.condition.eq(c)&t.subset.eq('all')&np.isclose(t.validation_coverage,.9)].set_index('detector')
        rows.append([c]+[f"{100*tt.loc[f'{r}_{k}_mlp_selected','actual_coverage']:.2f}% / {100*tt.loc[f'{r}_{k}_mlp_selected','risk']:.2f}%" for r,k in [('original','B'),('adapted','A'),('adapted','B')]])
    lines+=md_table(['Condition','Original B','Adapted A','Adapted B'],rows)
    lines+=['At noise 5, adapted B accepts only 34.80% of images and 62.21% of those are wrong; original B accepts 46.60% with 68.80% wrong. Lower accepted risk therefore partly accompanies more abstention. The equal-coverage gains above show a real ranking improvement, but neither result implies reliable classification under severe noise or a risk guarantee under shift.','',
        'Within the clean confident-unanimous subset (8,850 images, 131 errors), original A/B retain 49/45 errors at 90% subset coverage, versus adapted A/B 48/64. Adapted B also has worse subset AUROC (0.8650 versus adapted A 0.8854). Noise adaptation sacrifices part of the original motivating clean confident-error benefit. Subset metrics are descriptive; the stored bootstrap intervals concern all-example comparisons, not these subsets.','',
        '## 5. Seed sensitivity, validation selection and training budget','']
    frozen=json.loads((out.parent/'frozen.json').read_text())
    lines += [f"Validation selected MLP seeds A={frozen['selected_mlp_seeds']['A']}, B={frozen['selected_mlp_seeds']['B']}; these choices are retained, not revised using test performance.",'']
    sensitivity=frames['seed_sensitivity'];seed_rows=[]
    for c in CONDITIONS:
        a=np.array(json.loads(sensitivity[(sensitivity.condition.eq(c))&(sensitivity.feature_set.eq('A'))&(sensitivity.metric.eq('aurc'))].iloc[0]['values']))
        bb=np.array(json.loads(sensitivity[(sensitivity.condition.eq(c))&(sensitivity.feature_set.eq('B'))&(sensitivity.metric.eq('aurc'))].iloc[0]['values']))
        paired=bb-a;cross=(bb[:,None]-a[None,:]).ravel()
        seed_rows.append(dict(condition=c,same_seed_delta_min=paired.min(),same_seed_delta_max=paired.max(),all_nine_pairings_delta_min=cross.min(),all_nine_pairings_delta_max=cross.max()))
    pd.DataFrame(seed_rows).to_csv(out/'seed_effect_ranges.csv',index=False)
    lines+=['All three same-seed comparisons and all nine A/B seed pairings have negative adapted B−A under noise 3/5, and positive B−A on clean, mild noise, blur and JPEG. Thus the sign pattern is not confined to the two validation-selected detector seeds. These dependent seed pairings are a sensitivity check, not nine independent experiments or evidence across independently trained CNN ensembles.','']
    lines+=md_table(['Feature set','Seed','Best epoch','Epochs run','Optimizer steps'],[[x.feature_set,x.seed,x.best_epoch,x.epochs_run,x.optimizer_steps] for x in frames['training_history'].itertuples()])
    lines+=['All B seeds reach the 200-epoch cap with best epochs 189/195/194. A uses best epochs 62/188/65. This documents the executed budget; it does not justify extending training after inspecting test outcomes. Mixture epochs process four times as many rows as the original clean-only epoch, and the adaptation regime changes scalers and validation selection objective. Neither noise exposure, compute nor nonlinear architecture is isolated as the unique cause. Validation performance guided selection; it is not an independent evaluation result.','',
        '## Interpretation for the paper','',
        '> With the CNN ensemble and representation novelty reference held fixed, Gaussian-noise detector adaptation recovered a mean-novelty benefit beyond output features for the MLP at moderate and severe noise, but not at mild noise or for logistic regression. Recovery coincided with poorer clean and blur/JPEG performance, indicating a condition-dependent adaptation tradeoff.','',
        'Because fixed novelty features become useful in these shifted settings under the adapted MLP, the original failure cannot be interpreted as complete absence of usable novelty information. This supports a detector-transfer limitation for those conditions, while allowing reduced feature discrimination to coexist. It does not demonstrate a causal neural mechanism, prove that feature information is universally independent of outputs, or establish broad transfer.','',
        'Every condition was previously inspected; this is exploratory, with many unadjusted comparisons and one CNN ensemble. Blur/JPEG were withheld from adaptation development but are not untouched confirmatory tests. Bootstrap intervals condition on the selected models; detector seeds do not capture CNN-training variability. The original H1 tests novelty variation, whereas this experiment compares A versus B using mean novelty: its H1 outcome is unchanged.','',
        'The clean original LR B−A interval here barely crosses zero, while the earlier final-evaluation interval barely excluded zero. The point estimate is identical; the bootstrap seed changed (adaptation 20261107 versus original 20261103). With 2,000 draws and a near-zero endpoint, this boundary sensitivity is expected Monte Carlo variability. Report the registered intervals for each analysis and do not select whichever gives a desired significance label.','',
        'Source tables are in the parent folder: bootstrap.csv, metrics.csv, threshold_metrics.csv, seed_sensitivity.csv, training_history.csv, validation_metrics.csv and validation_scores.csv. Registration/environment/freeze/completion records provide provenance; cached per-condition scores reproduce ranking and threshold results. No new statistical tests or training were performed for this report.']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.root.resolve();out=root/'analysis'
    frames,files=load_and_verify(root)
    before={str(p):sha(p) for p in files.values()}
    out.mkdir(exist_ok=True)
    figures(out,frames['metrics'],frames['bootstrap']);report(out,frames)
    assert all(sha(Path(p))==digest for p,digest in before.items()),'An input artifact changed during analysis'
    (out/'analysis_manifest.json').write_text(json.dumps(dict(source_sha256=before,script_sha256=sha(Path(__file__)),
        interpretation='exploratory, existing paired intervals; no refitting or additional tests',figures=3),indent=2)+'\n')
    print(f'Verified source results; wrote three figures and {out / "ANALYSIS.md"}')


if __name__=='__main__':main()
