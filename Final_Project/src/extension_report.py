"""Standalone H7/H8 reporting; reads existing experiment artifacts without editing them."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import risk_curve


def table(frame, columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        def fmt(value):
            if isinstance(value,(float,np.floating)):
                return f'{value:.6f}' if np.isfinite(value) else '—'
            return str(value)
        lines.append('| '+' | '.join(fmt(row[col]) for col in columns)+' |')
    return lines


def report(root, study):
    from .run_extensions import verify_freeze, verify_hashes, audit_originals, read_csv, variants
    path=Path(root)/study
    verify_freeze(path)
    completion=json.loads((path/'evaluated.json').read_text())
    verify_hashes(completion['files'])
    if (path/'summary.md').exists():
        raise FileExistsError('Report already exists; use a new report directory/version rather than overwrite')
    results=read_csv(path/'metrics.csv')
    ci=read_csv(path/'bootstrap.csv')
    selected=[f'{key}_{family}' for key in 'ABCD' for family in ('lr','mlp_selected')]
    clean=results[results.condition.eq('clean') & results.subset.eq('all') & results.detector.isin(selected)]
    lines=[f'# {study.upper()}: exploratory extension','',
           '**Exploratory:** the original test findings were known before this study. Settings were selected only on clean validation and frozen before evaluating this extension. Original experiment files and CNN checkpoints were preserved.','']
    focused=ci[ci.condition.eq('clean') & ci.comparison.str.startswith('mlp_selected:')]
    if study=='h7':
        lines += ['Question: does accounting for feature covariance change the usefulness of mean novelty or novelty variation?','',
                  'Method: one Ledoit–Wolf covariance per CNN, fitted to CNN-training features after subtracting each training image’s true-class centroid. Evaluation always uses the ensemble-predicted class. Detector-training distances set normalization. Cholesky solves compute Mahalanobis distances; no raw inverse is used.','']
        wanted=['mlp_selected:mahalanobis_B_minus_original_B','mlp_selected:mahalanobis_C_minus_original_C','mlp_selected:mahalanobis_C_minus_B','mlp_selected:geometry_change_in_C_minus_B']
    else:
        lines += ['Question: do earlier representations react more strongly to corruption, while later representations better identify classification errors?','',
                  'Method: spatially averaged stages 1, 2, and 4 (64/128/512 dimensions), using Euclidean distance at every stage. Stage 4 reuses original artifacts. Early-stage extraction verifies cached logits, predictions and stage-4 features. No CNN is retrained.','']
        wanted=[f'mlp_selected:{v}_{key}_minus_original_{key}' for v in variants(study) for key in 'BC']
    lines += ['## Main clean-test comparisons','', 'Negative differences favor the first method. Intervals are 95% percentile intervals from 2,000 paired image resamples.','']
    lines += table(focused[focused.comparison.isin(wanted)],['comparison','delta_aurc','ci_low','ci_high'])
    for _,row in focused[focused.comparison.isin(wanted)].iterrows():
        if row.ci_high<0:
            interpretation='lower AURC, with an interval entirely below zero'
        elif row.ci_low>0:
            interpretation='higher AURC, with an interval entirely above zero'
        else:
            interpretation='inconclusive; the interval includes zero'
        lines += ['',f'- {row.comparison}: {interpretation}.']
    lines += ['', '## Clean metrics','']+table(clean,['variant','detector','auroc','aurc','risk80','risk90'])
    with (path/'main_table.csv').open('x') as handle:
        clean.to_csv(handle,index=False)
    subset=results[results.condition.eq('clean') & ~results.subset.eq('all') & results.detector.isin(selected)]
    lines += ['', '## Agreement subsets','', 'AUROC is suppressed if fewer than 20 errors or correct examples are present; counts above that threshold do not guarantee adequate power.','']
    lines += table(subset,['variant','subset','detector','count','errors','correct','auroc','aurc'])
    lines += ['', '## Corrupted images','',
              'All nine original conditions are retained: Gaussian noise, motion blur, JPEG compression at severities 1/3/5. No settings were selected on corruptions. Full AUROC/AURC/risk tables are in [metrics.csv](metrics.csv).','',
              'Selected-MLP C-minus-B comparisons within each measurement:','']
    within=ci[ci.comparison.str.startswith('mlp_selected:') & ci.comparison.str.endswith('_C_minus_B')]
    lines += table(within,['condition','comparison','delta_aurc','ci_low','ci_high'])
    lines += ['', 'Pooled AURC ranks all 90,000 corrupted copies jointly rather than averaging condition AURCs. Pooled bootstrap draws original-image IDs with all nine copies kept together. Deterministic ties use condition-major original example indices.','']
    sensitivity=read_csv(path/'seed_sensitivity.csv')
    lines += ['## Three-seed sensitivity','', 'Population mean/std over MLP seeds 11/22/33. Seeds vary detector initialization and minibatch order, not the CNN ensemble.','']
    lines += table(sensitivity[sensitivity.condition.eq('clean')],['variant','configuration','metric','mean','std'])
    lines += ['', '[Same-seed C-minus-B differences](paired_seed_sensitivity.csv) and [all predeclared intervals](bootstrap.csv) are retained, including unfavorable results. The geometry interaction subtracts Euclidean C-minus-B from Mahalanobis C-minus-B using the same bootstrap draws.','']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key in zip(axes,'BC'):
        for v in ['original']+variants(study):
            score_path=Path('results/final/clean/scores.csv') if v=='original' else path/v/'clean/scores.csv'
            scores=read_csv(score_path)
            curve=risk_curve(scores.error,scores[f'{key}_mlp_selected'],scores.id)
            ax.plot(np.arange(1,len(curve)+1)/len(curve),curve,label='stage4 Euclidean' if v=='original' else v)
        ax.set(xlabel='Coverage',ylabel='Selective error risk',title=f'Clean test: {key}, selected MLP')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path/'clean_risk_coverage.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    conditions=json.loads((path/'registration.json').read_text())['protocol']['evaluation']['conditions']
    for offset,v in enumerate(variants(study)):
        values=ci[ci.comparison.eq(f'mlp_selected:{v}_C_minus_original_C')].set_index('condition').loc[conditions]
        y=np.arange(len(values))+(offset-.5)*.16
        # Draw endpoints directly: a percentile interval need not contain the point estimate.
        ax.hlines(y,values.ci_low,values.ci_high)
        ax.plot(values.delta_aurc,y,'o',label=v)
    ax.set_yticks(np.arange(len(conditions)),conditions)
    ax.axvline(0,color='gray',linestyle='--')
    ax.set(xlabel='C AURC minus original stage4 Euclidean C')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path/'C_comparison.png',dpi=160)
    plt.close(fig)
    if study=='h8':
        response=read_csv(path/'corruption_response.csv')
        lines += ['## Response to corruption','',
                  'Paired changes subtract each source image’s clean standardized mean novelty from its corrupted value. This measures response to corruption, not the ability to tell mistakes from correct predictions.','',
                  'All-image mean change (full correctness-transition breakdown in [corruption_response.csv](corruption_response.csv)):','']
        lines += table(response[response.transition.eq('all')],['condition','stage','count','mean_delta','median_delta'])
        fig,ax=plt.subplots(figsize=(10,4))
        for stage in ('stage1','stage2','stage4'):
            frame=response[response.transition.eq('all') & response.stage.eq(stage)]
            ax.plot(frame.condition,frame.mean_delta,'o-',label=stage)
        ax.set(ylabel='Mean paired change in standardized novelty')
        ax.tick_params(axis='x',rotation=35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path/'depth_corruption_response.png',dpi=160)
        plt.close(fig)
        lines += ['', 'Depth also changes feature dimension and spatial detail; this simple pooled-distance study cannot uniquely attribute a difference to layer depth. The supplied local PDFs were unavailable, so no paper-specific method or result is attributed to them.','']
    count=audit_originals(Path(root))
    lines += ['## Reproduction and limitations','',
              f'{count} pre-existing files passed the final unchanged-file checksum audit. See the new [extension instructions](../../../../../../common/home/ns1788/DeepLearning/CS-462-Deep-Learning/Final_Project/EXTENSIONS_H7_H8.md) for commands; registration, freeze, and evaluation records are saved alongside this report.','',
              'All intervals are unadjusted exploratory comparisons, conditional on one trained ensemble. No universal conclusion about whether representations contain useful information follows. Existing CNN checkpoints retain the documented validation-loss selection deviation.','',
              'Data tables: [threshold decisions](threshold_metrics.csv), [seed sensitivity](seed_sensitivity.csv), [full bootstrap](bootstrap.csv).']
    # A plain repository-relative reference is more portable than a host-specific traversal.
    lines=[line.replace('[extension instructions](../../../../../../common/home/ns1788/DeepLearning/CS-462-Deep-Learning/Final_Project/EXTENSIONS_H7_H8.md)','`EXTENSIONS_H7_H8.md` in the repository root') for line in lines]
    with (path/'summary.md').open('x') as handle:
        handle.write('\n'.join(lines)+'\n')
    with (path/'report_audit.json').open('x') as handle:
        json.dump(dict(original_files_unchanged=count,exploratory=True,evaluation_verified=True),handle,indent=2)
    print(f'{study}: report written to {path / "summary.md"}',flush=True)
