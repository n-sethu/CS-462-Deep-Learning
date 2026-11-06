"""Rebuild final tables/figures from cached evaluation scores, without model fitting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import risk_curve


def markdown_table(frame, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---']*len(columns)) + '|']
    for _, row in frame.iterrows():
        def fmt(v):
            if isinstance(v, (float, np.floating)):
                return f'{v:.6f}' if np.isfinite(v) else '—'
            return str(v)
        lines.append('| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/final')
    args = parser.parse_args()
    root = Path(args.output)
    sealed = json.loads((root / 'sealed_protocol.json').read_text())
    all_metrics = pd.read_csv(root / 'all_metrics.csv')
    clean = all_metrics[(all_metrics.condition == 'clean') & (all_metrics.subset == 'all')]
    selected = [f'{k}_{m}' for k in 'ABCD' for m in ('lr', 'mlp_selected')]
    ranks = ['confidence', 'entropy', 'margin', 'mutual_information', 'temperature_confidence']
    results = clean[clean.detector.isin(ranks + selected)]
    results.to_csv(root / 'main_table.csv', index=False)
    ci = pd.read_csv(root / 'clean_bootstrap.csv')
    primary = ci[(ci.family == 'mlp_selected') & (ci.comparison == 'C_minus_B')].iloc[0]
    if primary.ci_high < 0 and primary.delta_aurc < 0:
        conclusion = 'H1 is supported under the predeclared paired-bootstrap procedure.'
    elif primary.ci_low <= 0 <= primary.ci_high:
        conclusion = 'H1 is inconclusive: the confidence interval includes zero.'
    else:
        conclusion = 'H1 is not supported: C has higher AURC than B, and the interval excludes zero in the direction of worse performance.'
    lines = ['# Frozen final evaluation', '',
             f"Primary comparison (validation-selected MLP): C − B AURC = **{primary.delta_aurc:.7f}**, 95% percentile CI **[{primary.ci_low:.7f}, {primary.ci_high:.7f}]** ({sealed['bootstrap_resamples']:,} paired resamples). {conclusion}", '',
             'This interval is conditional on one trained ensemble and the frozen detectors. It does not measure variation across retrained ensembles. Secondary comparisons are reported together without multiplicity correction; avoid treating them as separate confirmatory discoveries.', '',
             '## Clean test', '']
    accuracy = json.loads((root / 'clean/ensemble.json').read_text())
    lines += [f"Ensemble accuracy: {accuracy['accuracy']:.2%}. Member accuracies: " + ', '.join(f'{a:.2%}' for a in accuracy['model_accuracy']) + '.', '',
              f"Shared-temperature argmax differs on {accuracy['calibrated_argmax_changes']} images. Every error target uses the uncalibrated probability-average prediction.", '']
    lines += markdown_table(results, ['detector', 'auroc', 'aurc', 'risk80', 'risk90'])
    lines += ['', 'Risk80/90 keep exactly the safest 80%/90% using score then original image ID; they are ranking metrics. Separately, `threshold_metrics.csv` applies validation-only score thresholds and reports actual test coverage.', '', '## Paired differences', '']
    lines += markdown_table(ci, ['family', 'comparison', 'delta_aurc', 'ci_low', 'ci_high'])
    sensitivity = pd.read_csv(root / 'seed_sensitivity.csv')
    lines += ['', '## Detector seed sensitivity', '', 'Population standard deviations over seeds 11/22/33; each checkpoint/epoch was selected on validation. These seeds capture detector initialization and minibatch order, not CNN ensemble variability.', '']
    lines += markdown_table(sensitivity, ['configuration', 'metric', 'mean', 'std'])
    scores = pd.read_csv(root / 'clean/scores.csv')
    seed_rows = []
    for seed in (11, 22, 33):
        b = risk_curve(scores.error, scores[f'B_mlp_{seed}'], scores.id).mean()
        c = risk_curve(scores.error, scores[f'C_mlp_{seed}'], scores.id).mean()
        seed_rows.append(dict(seed=seed, B_aurc=b, C_aurc=c, delta_aurc=c-b))
    pd.DataFrame(seed_rows).to_csv(root / 'paired_seed_sensitivity.csv', index=False)
    lines += ['', 'Same-seed C − B comparisons (sensitivity only):', ''] + markdown_table(pd.DataFrame(seed_rows), ['seed', 'B_aurc', 'C_aurc', 'delta_aurc'])
    subset = all_metrics[(all_metrics.condition == 'clean') & (all_metrics.subset != 'all') & all_metrics.detector.isin(selected)]
    subset.to_csv(root / 'agreement_table.csv', index=False)
    lines += ['', '## Agreement and confident mistakes', '',
              f'AUROC is suppressed when fewer than {sealed["min_subset_errors_for_auroc"]} errors or correct examples are present. Larger counts do not guarantee adequate statistical power.', '']
    lines += markdown_table(subset, ['subset', 'detector', 'count', 'correct', 'errors', 'auroc', 'aurc'])
    # Inspect all discordant errors at frozen validation 90%-coverage thresholds.
    examples = []
    for path in sorted(root.glob('*/features.csv')):
        condition = path.parent.name
        table = pd.read_csv(path)
        s = pd.read_csv(path.parent / 'scores.csv')
        for family in ('lr', 'mlp_selected'):
            tb, tc = [sealed['thresholds'][f'{k}_{family}']['0.9'] for k in 'BC']
            br, cr = s[f'B_{family}'].gt(tb).to_numpy(), s[f'C_{family}'].gt(tc).to_numpy()
            for subset_name, mask in [('all', table.error.eq(1).to_numpy()), ('confident_unanimous', (table.error.eq(1) & table.unanimous & table.confidence.ge(.9)).to_numpy())]:
                examples.append(dict(condition=condition, family=family, subset=subset_name, errors=int(mask.sum()),
                                     C_catches_B_misses=int((mask & cr & ~br).sum()), B_catches_C_misses=int((mask & br & ~cr).sum()),
                                     both_catch=int((mask & br & cr).sum()), both_miss=int((mask & ~br & ~cr).sum())))
    differences = pd.DataFrame(examples)
    differences.to_csv(root / 'discordant_error_counts.csv', index=False)
    lines += ['', 'Catches mean abstains at the frozen validation 90%-coverage threshold. Different detectors may have different actual test coverage; these counts are descriptive, not a matched-coverage superiority test.', '']
    lines += markdown_table(differences[differences.condition == 'clean'], ['family', 'subset', 'errors', 'C_catches_B_misses', 'B_catches_C_misses', 'both_catch', 'both_miss'])
    lines += ['', 'Galleries include the first six source IDs in each direction, without visual selection. Class labels: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.', '']
    for path in sorted((root / 'clean').glob('*confident_unanimous_errors*.png')):
        lines += [f'![{path.stem}](clean/{path.name})', '']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, family in zip(axes, ('lr', 'mlp_selected')):
        for key in 'ABCD':
            curve = risk_curve(scores.error, scores[f'{key}_{family}'], scores.id)
            ax.plot(np.arange(1,len(curve)+1)/len(curve), curve, label=key)
        ax.set(xlabel='Coverage', ylabel='Selective error risk', title=f'Frozen clean test: {family}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'clean_risk_coverage.png', dpi=160)
    plt.close(fig)
    table = pd.read_csv(root / 'clean/features.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    for error, label in [(0, 'Correct'), (1, 'Error')]:
        keep = table.error.eq(error)
        ax.scatter(table.loc[keep, 'confidence'], table.loc[keep, 'novelty_variation'], s=5, alpha=.25, label=label)
    ax.set(xlabel='Ensemble confidence', ylabel='Novelty variation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'confidence_novelty.png', dpi=160)
    plt.close(fig)
    if (root / 'shift_bootstrap.csv').exists():
        shift = all_metrics[(all_metrics.condition != 'clean') & (all_metrics.subset == 'all') & all_metrics.detector.isin(selected)]
        shift.to_csv(root / 'shift_table.csv', index=False)
        shift_ci = pd.read_csv(root / 'shift_bootstrap.csv')
        lines += ['', '## CIFAR-10-C', '', 'Gaussian noise, motion blur, and JPEG compression at severities 1/3/5. No model, scaler, temperature, or threshold was changed. The pooled interval resamples original-image clusters, keeping all nine copies together. Pooled AURC ranks all 90,000 examples jointly; it is not the mean of condition-specific AURCs.', '']
        lines += markdown_table(shift, ['condition', 'detector', 'auroc', 'aurc', 'risk80', 'risk90'])
        lines += ['', 'C − B paired intervals (secondary, without multiplicity adjustment):', '']
        lines += markdown_table(shift_ci, ['condition', 'family', 'delta_aurc', 'ci_low', 'ci_high'])
        fig, ax = plt.subplots(figsize=(9, 5))
        for offset, family in [(-.13, 'lr'), (.13, 'mlp_selected')]:
            frame = shift_ci[shift_ci.family == family]
            x = np.arange(len(frame)) + offset
            ax.errorbar(frame.delta_aurc, x, xerr=np.stack([frame.delta_aurc-frame.ci_low, frame.ci_high-frame.delta_aurc]), fmt='o', label=family)
        ax.set_yticks(np.arange(len(frame)), frame.condition)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set(xlabel='C − B AURC (negative favors C)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / 'shift_differences.png', dpi=160)
        plt.close(fig)
        # Explicit failure conditions for every addition, not just favorable examples.
        failure = []
        for condition, group in all_metrics[all_metrics.subset == 'all'].groupby('condition'):
            indexed = group.set_index('detector')
            for family in ('lr', 'mlp_selected'):
                for left, right in [('B','A'), ('C','B'), ('D','C')]:
                    delta = indexed.loc[f'{left}_{family}', 'aurc'] - indexed.loc[f'{right}_{family}', 'aurc']
                    failure.append(dict(condition=condition, family=family, comparison=f'{left}_minus_{right}', delta_aurc=delta, hurts=delta>0))
        failures = pd.DataFrame(failure)
        failures.to_csv(root / 'failure_conditions.csv', index=False)
        lines += ['', 'Conditions where adding representation features hurts (positive AURC difference):', '']
        lines += markdown_table(failures[(failures.comparison == 'B_minus_A') & failures.hurts], ['condition', 'family', 'comparison', 'delta_aurc'])
    diagnostics = []
    for path in sorted(root.glob('*/features.csv')):
        t = pd.read_csv(path)
        for error, group in t.groupby('error'):
            diagnostics.append(dict(condition=path.parent.name, error=error, count=len(group),
                                    mean_novelty=float(group.mean_novelty.mean()), novelty_variation=float(group.novelty_variation.mean()),
                                    confidence=float(group.confidence.mean())))
    diagnostic_table = pd.DataFrame(diagnostics)
    diagnostic_table.to_csv(root / 'novelty_by_correctness.csv', index=False)
    lines += ['', 'Failure examples inspected: clean test ID 1492 is a truck predicted as automobile with confidence 0.906; C abstains and B accepts. IDs 6709 and 9218 are cats predicted as dogs with confidence 0.914/0.916; B abstains and C accepts. These are threshold-based observations, not causal explanations of the feature effect.', '']
    if 'gaussian_noise_5' in diagnostic_table.condition.values:
        clean_d = diagnostic_table[diagnostic_table.condition.eq('clean')].set_index('error')
        noise_d = diagnostic_table[diagnostic_table.condition.eq('gaussian_noise_5')].set_index('error')
        lines += [f"Under Gaussian noise severity 5, mean standardized novelty is {noise_d.loc[0, 'mean_novelty']:.3f} for correct images and {noise_d.loc[1, 'mean_novelty']:.3f} for errors, compared with {clean_d.loc[0, 'mean_novelty']:.3f} and {clean_d.loc[1, 'mean_novelty']:.3f} on clean data. Correct images become more atypical too, and the gap shrinks. This descriptive check is consistent with degraded novelty discrimination; it is not a causal test. Inspected severe-noise gallery examples include confident horse/ship-to-frog errors (IDs 109, 145, 166, 234).", '']
    perm = pd.read_csv(root / 'permutation_diagnostic.csv')
    lines += ['', '## Diagnostics and cost', '', 'Permutation diagnostic mean AURC increase after shuffling novelty variation:', '']
    for family, group in perm.groupby('family'):
        lines += [f'- {family}: {group.delta_aurc.mean():.7f} over {len(group)} permutations.']
    lines += ['', 'Correlated features make permutation importance a reliance diagnostic, not proof of independent information. `novelty_by_correctness.csv` reports shifts for both correct and incorrect images; larger distance alone does not imply an error.', '']
    latency = pd.read_csv(root / 'latency.csv')
    lines += markdown_table(latency, ['operation', 'batch_size', 'median_ms', 'median_ms_per_image'])
    env = json.loads((root / 'environment.json').read_text())
    lines += ['', f"Hardware: {env['device']}. {env['timing']}. CPU detector work uses {env['cpu_threads']} Torch threads. CNN calls compute penultimate features in both forward modes; exposing these features adds no extra CNN pass. Small timing differences between modes include measurement noise.", '',
              '## Limits', '',
              'One CIFAR-adapted ResNet-18 ensemble on one dataset; conditional bootstrap uncertainty only. Cross-model novelty variation uses just three numbers and may reflect geometry differences. Small subgroups, multiple secondary tests, and correlated feature ablations limit interpretation. The CNN checkpoints were selected by validation loss rather than the accuracy rule in AGENTS.md. Existing selections were retained and never changed after test evaluation.', '',
              'CIFAR-10-C source: [Hendrycks and Dietterich official archive](https://zenodo.org/records/2535967).']
    (root / 'summary.md').write_text('\n'.join(lines)+'\n')
    print(lines[2])


if __name__ == '__main__':
    main()
