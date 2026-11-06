"""Summarize existing validation artifacts without inference or fitting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import risk_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/validation')
    args = parser.parse_args()
    root = Path(args.output)
    table = pd.read_csv(root / 'val_features.csv')
    results = pd.read_csv(root / 'validation_metrics.csv')
    scores = pd.read_csv(root / 'validation_scores.csv')
    verification = json.loads((root / 'ensemble_verification.json').read_text())
    protocol = json.loads((root / 'frozen_protocol.json').read_text())
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ['confidence', 'temperature_confidence', 'A_lr', 'A_mlp_selected', 'B_mlp_selected', 'C_mlp_selected']:
        curve = risk_curve(scores.error, scores[name], scores.id)
        ax.plot(np.arange(1, len(curve) + 1) / len(curve), curve, label=name)
    ax.set(xlabel='Coverage', ylabel='Selective error risk', title='Validation (used for selection)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'risk_coverage.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for error, name in [(0, 'Correct'), (1, 'Incorrect')]:
        ax.hist(table.loc[table.error == error, 'confidence'], bins=np.linspace(0, 1, 31), density=True, alpha=.5, label=name)
    ax.set(xlabel='Ensemble confidence', ylabel='Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'confidence_distribution.png', dpi=160)
    plt.close(fig)
    keep = ['confidence', 'entropy', 'margin', 'mutual_information', 'temperature_confidence'] + [f'{k}_{method}' for k in 'ABCD' for method in ('lr', 'mlp_selected')]
    lines = ['# Validation results', '', 'Validation is used for fitting temperature and selecting detectors. These are not final test estimates.', '',
             'Model accuracies: ' + ', '.join(f'{a:.2%}' for a in verification['model_accuracy']) + f"; ensemble: {verification['ensemble_accuracy']:.2%}.", '',
             f"Unanimous agreement: {verification['unanimous_fraction']:.2%}. Pairwise disagreement counts: {verification['pairwise_disagreements']}.", '',
             f"Shared temperature: {protocol['temperature']:.6f}; calibrated argmax changed on {protocol['temperature_argmax_changes']} / {len(table)} validation images. Fixed T=1 predictions define all errors.", '',
             '| Ranking/detector | AUROC | AURC | Risk 80% | Risk 90% |', '|---|---:|---:|---:|---:|']
    for name in keep:
        row = results.set_index('detector').loc[name]
        lines.append(f"| {name} | {row.auroc:.6f} | {row.aurc:.6f} | {row.risk80:.6f} | {row.risk90:.6f} |")
    lines += ['', 'Selected MLP seeds: ' + str(protocol['selected_mlp_seeds']) + '.', '', 'Three-seed sensitivity on validation (mean ± population std):', '']
    for key in 'ABCD':
        rows = results[results.detector.isin([f'{key}_mlp_{s}' for s in (11, 22, 33)])]
        lines.append(f"- {key}: AURC {rows.aurc.mean():.6f} ± {rows.aurc.std(ddof=0):.6f}; AUROC {rows.auroc.mean():.6f} ± {rows.auroc.std(ddof=0):.6f}.")
    lines += ['', 'CNN checkpoint deviation: saved models were selected by minimum validation loss. Test remains untouched; no primary H1 conclusion or test bootstrap interval is reported here.']
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
