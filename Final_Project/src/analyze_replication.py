"""Read-only ten-ensemble audit. Run: python -m src.analyze_replication.

Writes a separate report; never selects detectors or modifies run artifacts.
Intervals resample ensembles conditional on the fixed shared images/splits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t, binomtest
from .metrics import metrics, risk_curve

CONDITIONS=['clean']+[f'{c}_{s}' for c in ['gaussian_noise','motion_blur','jpeg_compression'] for s in [1,3,5]]
LABELS=['Clean','Noise 1','Noise 3','Noise 5','Blur 1','Blur 3','Blur 5','JPEG 1','JPEG 3','JPEG 5']
FAMILIES=['lr','mlp_selected']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/replication'));p.add_argument('--out',type=Path,default=Path('results/replication_analysis'));a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    sources={}; audit=[]
    def read(path):
        path=path if path.exists() else path.with_suffix(path.suffix+'.gz')
        sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    def collect(rel):
        frames=[]
        for i in range(1,11):
            d=read(a.root/f'ensemble_{i:02d}/results'/rel);d['ensemble']=i;frames.append(d)
        return pd.concat(frames,ignore_index=True)
    rows=[]
    for study,rel in [('replication','replication_bootstrap.csv'),('h7','extensions/h7/bootstrap.csv'),('h8','extensions/h8/bootstrap.csv'),('adaptation','adaptation/bootstrap.csv')]:
        d=collect(rel)
        if 'estimate' in d:
            if 'delta_aurc' not in d:d['delta_aurc']=np.nan
            d=d[d.delta_aurc.notna()|d.metric.eq('aurc')].copy()
            d['delta_aurc']=d.delta_aurc.fillna(d.estimate)
        if 'subset' not in d:d['subset']='all'
        d['subset']=d['subset'].fillna('all')
        if 'family' not in d:d['family']=d.comparison.str.split(':').str[0]
        d['study']=study;rows.append(d)
    effects=pd.concat(rows,ignore_index=True)
    keys=['study','condition','subset','family','comparison']
    assert not effects.duplicated(keys+['ensemble']).any()
    effects.to_csv(a.out/'per_ensemble_effects.csv',index=False)
    rng=np.random.default_rng(20261109);summary=[]
    for key,g in effects.groupby(keys):
        x=g.sort_values('ensemble').delta_aurc.to_numpy();assert len(x)==10,(key,len(x))
        draws=rng.choice(x,(20000,10),replace=True).mean(1);se=x.std(ddof=1)/np.sqrt(10)
        summary.append(dict(zip(keys,key),n=10,mean=x.mean(),std=x.std(ddof=1),median=np.median(x),negative=int((x<0).sum()),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975),t_low=x.mean()-t.ppf(.975,9)*se,t_high=x.mean()+t.ppf(.975,9)*se,sign_p=binomtest(int((x<0).sum()),int((x!=0).sum()),.5).pvalue))
    summary=pd.DataFrame(summary);summary.to_csv(a.out/'effect_summary.csv',index=False)
    base=collect('final/all_metrics.csv');adapt=collect('adaptation/metrics.csv')
    counts=collect('matched_coverage.csv'); thresholds=collect('adaptation/threshold_metrics.csv')
    # Verify all primary clean metrics and matched-coverage counts from actual scores.
    curves={k:[] for k in ['confidence','margin','A_mlp_selected','B_mlp_selected','C_mlp_selected','D_mlp_selected']}
    for i in range(1,11):
        work=a.root/f'ensemble_{i:02d}'
        assert (work/'compact_complete.json').exists()
        d=read(work/'results/final/clean/scores.csv');f=read(work/'results/final/clean/features.csv')
        assert d.id.is_unique and len(d)==10000
        np.testing.assert_array_equal(d.id,f.id);np.testing.assert_array_equal(d.error,f.error)
        np.testing.assert_array_equal(f.error,(f.prediction!=f.label).astype(int))
        for name in curves:
            got=metrics(d.error,d[name],d.id)
            expected=base[(base.ensemble==i)&base.condition.eq('clean')&base.subset.eq('all')&base.detector.eq(name)].iloc[0]
            for metric in ['aurc','auroc','risk80','risk90']:np.testing.assert_allclose(got[metric],expected[metric],rtol=1e-10,atol=1e-12)
            curves[name].append(risk_curve(d.error,d[name],d.id))
        for fam in FAMILIES:
            s=summary[(summary.study=='replication')&(summary.condition=='clean')&(summary.subset=='all')&(summary.comparison==f'{fam}:C_minus_B')]
            assert len(s)==1
            for left,right in [('B','A'),('C','B'),('D','C')]:
                v=metrics(d.error,d[f'{left}_{fam}'],d.id)['aurc']-metrics(d.error,d[f'{right}_{fam}'],d.id)['aurc']
                stored=effects[(effects.ensemble==i)&effects.study.eq('replication')&effects.condition.eq('clean')&effects.subset.eq('all')&effects.comparison.eq(f'{fam}:{left}_minus_{right}')].delta_aurc.item()
                np.testing.assert_allclose(v,stored,atol=1e-12)
        audit.append(dict(ensemble=i,clean_rows=len(d),errors=int(d.error.sum()),accuracy=1-d.error.mean(),unanimous=int(f.unanimous.sum()),confident_unanimous=int((f.unanimous&f.confidence.ge(.9)).sum()),confident_unanimous_errors=int(f.loc[f.unanimous&f.confidence.ge(.9),'error'].sum())))
    pd.DataFrame(audit).to_csv(a.out/'integrity_audit.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    figures=[]
    def save(fig,name,caption):
        fig.tight_layout();fig.savefig(a.out/(name+'.png'),bbox_inches='tight');fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight');plt.close(fig);figures.append((name,caption))
    def forest(ax,groups,labels,title,scale=1000):
        for j,g in enumerate(groups):
            x=g.sort_values('ensemble').delta_aurc.to_numpy()*scale
            assert len(x)==10
            ax.scatter(x,j+np.linspace(-.13,.13,10),s=17,alpha=.65)
            draws=np.random.default_rng(42).choice(x,(20000,10),replace=True).mean(1)
            lo,hi=np.quantile(draws,[.025,.975]);ax.plot([lo,hi],[j,j],color='black',lw=2);ax.scatter(x.mean(),j,marker='D',c='black',s=30)
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('ΔAURC × 1,000 (negative favors added feature)')
    def groups(study,comparisons,condition='clean',subset='all'):
        return [effects[effects.study.eq(study)&effects.condition.eq(condition)&effects.subset.eq(subset)&effects.comparison.eq(c)] for c in comparisons]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,fam in zip(axs,FAMILIES):forest(ax,groups('replication',[f'{fam}:{c}' for c in ['B_minus_A','C_minus_B','D_minus_C']]),['H2: mean (B−A)','H1: variation (C−B)','H3: max (D−C)'],fam)
    save(fig,'01_primary_ablation','Ten independent ensembles; dots are paired ensemble effects; diamonds and bars are mean and 95% ensemble-bootstrap interval. H1 is C−B; negative favors the added feature. Original ensemble excluded.')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for name,values in curves.items():
        v=np.array(values);x=np.arange(1,10001)/10000;axs[0].plot(x,v.mean(0),label=name.replace('_mlp_selected',''));axs[0].fill_between(x,*np.quantile(v,[.1,.9],axis=0),alpha=.07)
    axs[0].set(xlabel='Coverage',ylabel='Selective risk',title='Clean risk–coverage; mean and 10–90% spread');axs[0].legend(fontsize=8)
    selected=['confidence','margin','entropy','mutual_information','temperature_confidence']+[f'{k}_{fam}' for fam in FAMILIES for k in 'ABCD']
    for j,name in enumerate(selected):
        g=base[base.condition.eq('clean')&base.subset.eq('all')&base.detector.eq(name)]
        axs[1].scatter(g.aurc*1000,np.full(len(g),j),s=15)
    axs[1].set_yticks(range(len(selected)),selected);axs[1].set(xlabel='AURC × 1,000',title='Strong baselines and A–D')
    save(fig,'02_clean_baselines','Clean rankings and detectors on identical predictions. Shading is ensemble spread, not a confidence band. AURC magnitudes depend on error prevalence.')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,fam in zip(axs,FAMILIES):
        gs=[];labs=[]
        for subset in ['all','unanimous','confident_unanimous']:
            for c in ['B_minus_A','C_minus_B']:gs+=groups('replication',[f'{fam}:{c}'],subset=subset);labs.append(subset+' '+c)
        forest(ax,gs,labs,fam)
    save(fig,'03_agreement','Subset-specific paired comparisons; each ensemble defines its own unanimous and confidence ≥0.9 subset. Counts are in integrity_audit.csv; these follow-up comparisons are exploratory.')
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for ax,(fam,c) in zip(axs.flat,[(f,c) for f in FAMILIES for c in ['B_minus_A','C_minus_B']]):
        forest(ax,[groups('replication',[f'{fam}:{c}'],condition=k)[0] for k in CONDITIONS],LABELS,fam+' '+c)
    save(fig,'04_shift','Frozen clean-trained detectors across every condition. Within-condition contrasts avoid confusing error prevalence with detector quality. Conditions are not independent replications.')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,fam in zip(axs,FAMILIES):
        for offset,comparison,color in [(-.2,'original_B_minus_A','tab:blue'),(0,'adapted_B_minus_A','tab:orange'),(.2,'Gamma','tab:green')]:
            g=summary[summary.study.eq('adaptation')&summary.family.eq(fam)&summary.subset.eq('all')&summary.comparison.eq(comparison)].set_index('condition').loc[CONDITIONS]
            ax.errorbar(g['mean']*1000,np.arange(10)+offset,xerr=np.array([g['mean']-g.ci_low,g.ci_high-g['mean']])*1000,fmt='o',label=comparison,color=color,capsize=2)
        ax.axvline(0,c='gray',ls='--');ax.set_yticks(range(10),LABELS);ax.invert_yaxis();ax.set_title(fam);ax.set_xlabel('ΔAURC × 1,000');ax.legend(fontsize=8)
    save(fig,'05_adaptation','Original B−A, adapted B−A and their paired difference Gamma. Recovery requires negative adapted B−A, not merely negative Gamma. Intervals resample ensembles.')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,coverage in zip(axs,[.8,.9]):
        for shift,prefix in [(-.12,'original'),(.12,'adapted')]:
            g=adapt[adapt.subset.eq('all')&adapt.detector.isin([f'{prefix}_A_mlp_selected',f'{prefix}_B_mlp_selected'])].pivot(index=['ensemble','condition'],columns='detector',values=f'accepted_errors{int(coverage*100)}')
            diff=g[f'{prefix}_A_mlp_selected']-g[f'{prefix}_B_mlp_selected']
            for j,c in enumerate(CONDITIONS):ax.scatter(diff.xs(c,level='condition'),np.full(10,j+shift),s=17,c='tab:blue' if prefix=='original' else 'tab:orange',label=prefix if j==0 else None)
        ax.axvline(0,c='gray',ls='--');ax.set_yticks(range(10),LABELS);ax.invert_yaxis();ax.set_title(f'MLP at exact {coverage:.0%} coverage');ax.set_xlabel('Additional errors rejected by B vs A (positive better)');ax.legend()
    save(fig,'06_accepted_errors','Original and adapted MLP A/B at equal rejection budgets; all ten paired counts. Counts and risks at both coverages and both families are retained in companion tables.')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for name in ['original_A_mlp_selected','original_B_mlp_selected','adapted_A_mlp_selected','adapted_B_mlp_selected']:
        g=thresholds[thresholds.subset.eq('all')&thresholds.validation_coverage.eq(.9)&thresholds.detector.eq(name)]
        for ax,value in zip(axs,['actual_coverage','risk']):
            v=g.groupby('condition')[value].agg(['mean','std']).loc[CONDITIONS];ax.errorbar(range(10),v['mean'],yerr=v['std'],label=name,marker='o',capsize=2)
    for ax,title in zip(axs,['Actual coverage','Error rate among accepted']):ax.set_xticks(range(10),LABELS,rotation=50,ha='right');ax.set_title(title);ax.legend(fontsize=7)
    axs[0].axhline(.9,c='gray',ls='--')
    save(fig,'07_frozen_thresholds','Clean-validation thresholds targeting 90% coverage; mean ± sample SD across ensembles. Unequal actual coverage precludes interpreting lower accepted risk as equal-budget superiority.')
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,study,c,title in [(axs[0],'h7','mlp_selected:mahalanobis_B_minus_original_B','Mahalanobis B − Euclidean B'),(axs[1],'h8','mlp_selected:stage1_B_minus_original_B','Stage 1 B − Stage 4 B'),(axs[2],'h8','mlp_selected:stage2_B_minus_original_B','Stage 2 B − Stage 4 B')]:
        forest(ax,[groups(study,[c],condition=k)[0] for k in CONDITIONS],LABELS,title)
    save(fig,'08_geometry_depth','Exploratory geometry and depth comparisons for MLP B. Negative favors the extension; original means that ensemble’s Euclidean stage-4 system. LR and A–D contrasts remain in effect_summary.csv.')
    novelty=collect('exploratory/novelty_discrimination.csv');trans=collect('exploratory/paired_novelty_changes.csv');perm=collect('final/permutation_diagnostic.csv')
    for rel,name in [('exploratory/confidence_novelty.csv','confidence_bins'),('exploratory/confidence_detector_comparisons.csv','confidence_detector_comparisons'),('validation/feature_correlations.csv','feature_correlations'),('final/seed_sensitivity.csv','original_seed_sensitivity'),('adaptation/seed_sensitivity.csv','adapted_seed_sensitivity'),('adaptation/training_history.csv','adapted_training_history'),('single_member_metrics.csv','single_member_metrics'),('extensions/h7/metrics.csv','geometry_metrics'),('extensions/h8/metrics.csv','depth_metrics')]:
        collect(rel).to_csv(a.out/(name+'.csv'),index=False)
    novelty.to_csv(a.out/'novelty_discrimination.csv',index=False);trans.to_csv(a.out/'paired_novelty_changes.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for feature in ['mean_novelty','novelty_variation']:
        g=novelty[novelty.feature.eq(feature)].groupby('condition').auroc.agg(['mean','std']).reindex(CONDITIONS[1:]);axs[0].errorbar(range(9),g['mean'],yerr=g['std'],label=feature,marker='o')
    axs[0].axhline(.5,c='gray',ls='--');axs[0].set_xticks(range(9),LABELS[1:],rotation=60,ha='right');axs[0].set_title('Novelty-only error AUROC');axs[0].legend(fontsize=7)
    for transition in ['correct_correct','correct_wrong']:
        g=trans[trans.feature.eq('mean_novelty')&trans.restriction.eq('all')&trans.transition.eq(transition)].groupby('condition')['mean'].mean().reindex(CONDITIONS[1:]);axs[1].plot(range(9),g,marker='o',label=transition)
    axs[1].set_xticks(range(9),LABELS[1:],rotation=60,ha='right');axs[1].set_title('Paired mean-novelty change');axs[1].legend(fontsize=7)
    pg=perm.groupby(['ensemble','family']).delta_aurc.mean().reset_index()
    for j,fam in enumerate(FAMILIES):axs[2].scatter(np.full(10,j),pg[pg.family.eq(fam)].delta_aurc*1000)
    axs[2].axhline(0,c='gray',ls='--');axs[2].set_xticks([0,1],['LR','MLP']);axs[2].set_title('Permutation reliance on variation');axs[2].set_ylabel('Mean permuted − intact AURC × 1,000')
    save(fig,'09_diagnostics','Novelty discrimination and clean→corruption transitions are descriptive, not causal. Permutation points average repeats within each ensemble: reliance does not demonstrate incremental value.')
    seed=collect('paired_seed_sensitivity.csv');latency=collect('final/latency.csv')
    seed.to_csv(a.out/'detector_seed_sensitivity.csv',index=False);latency.to_csv(a.out/'latency.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for j,s in enumerate([11,22,33]):axs[0].scatter(seed[seed.condition.eq('clean')&seed.seed.eq(s)].ensemble,seed[seed.condition.eq('clean')&seed.seed.eq(s)].C_minus_B*1000,label=str(s),marker=['o','s','^'][j])
    axs[0].axhline(0,c='gray',ls='--');axs[0].set(xlabel='Ensemble',ylabel='C−B AURC × 1,000',title='H1 sensitivity to detector initialization');axs[0].legend()
    g=latency[latency.batch_size.eq(1)]
    for j,(name,v) in enumerate(g.groupby('operation')):axs[1].scatter(v.median_ms,np.full(len(v),j));axs[1].text(v.median_ms.max(),j,name,fontsize=7,va='bottom')
    axs[1].set_yticks([]);axs[1].set(xlabel='Batch-1 median latency (ms)',title='Hardware-specific cost across runs')
    save(fig,'10_seed_cost','Detector seeds are sensitivity checks, not additional independent ensembles. Latency dots are run medians; shared GPU timing can vary with hardware/load. All batch-size results are retained.')
    base.to_csv(a.out/'baseline_metrics.csv',index=False);adapt.to_csv(a.out/'adaptation_metrics.csv',index=False);thresholds.to_csv(a.out/'threshold_metrics.csv',index=False);counts.to_csv(a.out/'matched_coverage.csv',index=False)
    def table(d):
        cols=['condition','subset','family','comparison','mean','ci_low','ci_high','t_low','t_high','negative']
        lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for row in d[cols].itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.7f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(lines)
    primary=summary[summary.study.eq('replication')&summary.condition.eq('clean')&summary.subset.eq('all')]
    metric_rows=[]
    for study,frame in [('baseline',base),('adaptation',adapt)]:
        for key,g in frame.groupby(['condition','subset','detector']):
            assert g.ensemble.nunique()==10
            for metric in ['aurc','auroc','risk80','risk90']:
                metric_rows.append(dict(study=study,condition=key[0],subset=key[1],detector=key[2],metric=metric,mean=g[metric].mean(),std=g[metric].std(ddof=1),minimum=g[metric].min(),maximum=g[metric].max()))
    pd.DataFrame(metric_rows).to_csv(a.out/'metric_summary.csv',index=False)
    conclusions='''## Concise findings

- **H1 does not replicate as a benefit.** Clean MLP C−B averages +0.0000372; only 2/10 ensembles improve. The percentile interval barely excludes zero on the harmful side, but the Student-t interval spans zero. Say “no robust variation benefit,” rather than claiming definitive harm or equivalence. LR is also inconclusive.
- **H2 is modestly supported for MLP.** Mean novelty improves clean AURC in 8/10 ensembles: mean B−A −0.0001612, 95% ensemble-bootstrap CI approximately [−0.0002769, −0.0000527]. Average A/B AURCs are 0.005339/0.005178, about a 3.0% reduction and five fewer accepted errors at 90% coverage. The original discovery effect was larger (about 8.8%). LR's ensemble interval spans zero. B also has lower mean AURC than confidence/margin; paired baseline-superiority intervals remain a follow-up requirement.
- **H3 has no clear clean MLP benefit. H6 changes:** selected MLP B beats LR B in all ten ensembles, with mean difference −0.0000777. This is a small, repeatable family effect under the executed training recipes, not proof that nonlinearity alone causes the difference.
- **H4 is weaker than the original motivating example.** Mean novelty improves unanimous-subset MLP AURC on average, but confident-unanimous B−A improves in 6/10 and its ensemble interval includes zero. Do not claim reliable confident-unanimous recovery across ensembles.
- **H5 is condition-dependent and highly repeatable.** Clean-trained MLP B beats A for every blur/JPEG condition in all ten ensembles. Mean B−A instead increases under all noise severities; noise-3 harms all ten. Variation has occasional noise-specific gains, which do not rescue clean H1.
- **Adaptation's main recovery repeats:** adapted MLP B−A is negative in 10/10 at noise 3 and 5, averaging −0.005184 and −0.003661; both ensemble intervals exclude zero. At 90% coverage B rejects on average 38.1/13.0 more errors than adapted A. Gamma is negative in all ten. LR still has positive average noise B−A.
- **The tradeoff repeats more clearly than universal harm.** MLP Gamma is positive on clean and every blur/JPEG condition: adaptation erodes novelty's incremental benefit. However, adapted B−A intervals include zero on clean, mild noise and each blur/JPEG condition. The original claim that adapted novelty is demonstrably harmful everywhere outside severe noise is too strong across ensembles. Matched-coverage counts and integrated AURC can differ.
- **H7/H8 remain conditional:** Mahalanobis B is worse than Euclidean B on clean in 10/10, but improves severe-noise averages. Stage-1/2 B are worse than stage-4 on clean in 10/10 and on every blur/JPEG condition in 10/10. Stage-1 improves noise-3 averages; other severe-noise depth contrasts are less certain. Stronger novelty response does not guarantee better error ranking.

The paper's strongest framing is a replicated audit of **small clean mean-novelty gains, lack of robust clean novelty-variation benefit, and a repeatable noise-adaptation/transfer tradeoff**. Ten fresh ensembles substantially strengthen evidence about training randomness; they do not establish architecture or dataset generality.
'''
    sections=['# Ten-ensemble replication analysis\n',
    'All ten additional ensembles are included; the original discovery ensemble is excluded from estimates. Each effect is computed within an ensemble on identical predictions and images. No detector is selected or refit here.\n',
    conclusions,
    '## Main hypotheses\n\n'+table(primary),
    '## Adaptation\n\n'+table(summary[summary.study.eq('adaptation')&summary.family.eq('mlp_selected')&summary.subset.eq('all')]),
    '## Geometry and depth (clean)\n\n'+table(summary[summary.study.isin(['h7','h8'])&summary.condition.eq('clean')&summary.subset.eq('all')]),
    '## Figures\n']
    for name,caption in figures:sections.append(f'### {name}\n\n![{name}]({name}.png)\n\n{caption}\n')
    sections.append('## Statistical interpretation and paper priorities\n\nThe unit of replication is the three-CNN ensemble (n=10), not 30 members, 30 detector seeds, 100,000 independent test images, or corruption conditions. Report mean, sample SD, median, 20,000-draw percentile ensemble-bootstrap CI and negative-effect count. The companion table includes Student-t intervals and exact two-sided sign-test p-values as sensitivity checks. A near-zero endpoint is fragile; do not infer equivalence from a non-significant effect. All intervals are pointwise and unadjusted. H1 is the prespecified primary question; treat secondary tests and extensions as exploratory and disclose multiplicity.\n\nThese intervals address training randomness conditional on one shared dataset, split, architecture and test set; they do not include new-image uncertainty. Per-ensemble image-bootstrap intervals answer a different conditional question. For a population-over-images claim, add a crossed bootstrap using the same resampled source IDs across all ensembles (and all corruption copies), together with resampled ensembles; do not independently bootstrap shared images per model. No pooled corruption result is needed.\n\nFor the paper: lead with H1 and H2, preserve the original hypothesis and negative outcomes, then show condition-specific transfer and adaptation. Use figures 1, 4 and 5 in the main text, with figure 6 for practical effect sizes; the others fit the supplement. Supply AUROC, risk80/90 and error counts alongside AURC, plus strongest single-score baselines, every condition, all seeds and freeze/provenance records. The original ensemble is a separate discovery example, not an eleventh confirmatory replicate. Existing extension/adaptation questions remain exploratory, even when repeated on fresh ensembles.\n\nBefore a stronger publication claim, verify snapshot/split/freeze and retirement manifests against retained artifacts, add crossed image/ensemble uncertainty for chosen principal comparisons, and explicitly limit conclusions to CIFAR-10/ResNet-18 and these three corruptions. A second architecture or independent data split is a future experiment requiring a separately fixed protocol, not something demonstrated here. Depth is confounded by dimensionality/pooling; adaptation changes compute, scaler and selection objective. Retain the validation-loss CNN checkpoint deviation. No causal explanation or universal novelty-monitoring claim follows.\n\nIntegrity checks here recompute every clean selected-detector/baseline metric and primary contrast from scores, check IDs/error targets and ten completion markers. They do not constitute a full retraining or provenance audit. Root aggregation had omitted mixed-schema replication AURC rows; this report reads their source tables directly. No original run artifacts were modified.\n')
    (a.out/'ANALYSIS.md').write_text('\n\n'.join(sections))
    (a.out/'SUMMARY.md').write_text('# Replication: concise evidence summary\n\n'+conclusions+'\nSee [full analysis and ten figures](ANALYSIS.md). Reproduce with `python -m src.analyze_replication`. Numerical intervals above condition on the shared images/splits; all secondary/extension intervals are pointwise and unadjusted.\n')
    (a.out/'manifest.json').write_text(json.dumps(dict(root=str(a.root.resolve()),ensembles=10,bootstrap_draws=20000,seed=20261109,source_hashes=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=[f[0] for f in figures]),indent=2))
    print(primary.to_string(index=False));print('Saved ten PNG/PDF figures and report to',a.out)

if __name__=='__main__':main()
