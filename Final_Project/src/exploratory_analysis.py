"""Cache-only exploratory analyses; never fits or changes any model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import metrics

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
OUT = ROOT / "results/exploratory"
BINS = [0, .5, .7, .8, .9, .95, 1]
FEATURES = ["mean_novelty", "novelty_variation"]
CONDITIONS = [f"{c}_{s}" for c in ("gaussian_noise", "motion_blur", "jpeg_compression") for s in (1, 3, 5)]


def load(condition):
    table = pd.read_csv(FINAL / condition / "features.csv").sort_values("id").reset_index(drop=True)
    scores = pd.read_csv(FINAL / condition / "scores.csv").sort_values("id").reset_index(drop=True)
    assert table.id.is_unique and scores.id.is_unique
    np.testing.assert_array_equal(table.id, scores.id)
    np.testing.assert_array_equal(table.error, scores.error)
    assert np.array_equal(table.error, (table.prediction != table.label).astype(int))
    return table, scores


def confidence_analysis(clean, scores):
    rows, comparisons = [], []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = pd.cut(clean.confidence, BINS, right=False, include_lowest=True)
    bins = bins.astype(object)
    bins.loc[clean.confidence.eq(1)] = pd.Interval(.95, 1, closed="left")
    for subset, mask in {"all": np.ones(len(clean), bool), "unanimous": clean.unanimous.to_numpy(bool)}.items():
        for i, (low, high) in enumerate(zip(BINS[:-1], BINS[1:])):
            keep = mask & clean.confidence.ge(low) & (clean.confidence.le(high) if high == 1 else clean.confidence.lt(high))
            group = clean.loc[keep]
            n, errors = len(group), int(group.error.sum())
            for feature in FEATURES:
                auc = metrics(group.error, group[feature], group.id)["auroc"] if min(errors, n-errors) >= 20 else None
                for error in (0, 1):
                    values = group.loc[group.error.eq(error), feature]
                    rows.append(dict(subset=subset, bin=i, low=low, high=high, feature=feature, error=error,
                                     count=len(values), median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75), auroc=auc))
            if n:
                for family in ("lr", "mlp_selected"):
                    m = {k: metrics(group.error, scores.loc[keep, f"{k}_{family}"], group.id) for k in "ABC"}
                    comparisons.append(dict(subset=subset, low=low, high=high, count=n, errors=errors, family=family,
                                            B_minus_A=m["B"]["aurc"]-m["A"]["aurc"], C_minus_B=m["C"]["aurc"]-m["B"]["aurc"]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "confidence_novelty.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "confidence_detector_comparisons.csv", index=False)
    for ax, feature in zip(axes, FEATURES):
        for error, color in [(0, "tab:blue"), (1, "tab:orange")]:
            d = table.query("subset == 'all' and feature == @feature and error == @error").sort_values("bin")
            ax.errorbar(d.bin + (.08 if error else -.08), d["median"], yerr=[d["median"]-d.q25, d.q75-d["median"]], fmt="o", color=color, label="Error" if error else "Correct")
            for r in d.itertuples():
                ax.annotate(str(r.count), (r.bin, r.q75), fontsize=7)
        ax.set_xticks(range(6), [f"{a:g}–{b:g}" for a,b in zip(BINS[:-1], BINS[1:])], rotation=30)
        ax.set(xlabel="Ensemble confidence", ylabel=feature, title="Median and interquartile range; labels = counts")
        ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "confidence_novelty.png", dpi=160); plt.close(fig)


def corruption_analysis(clean):
    rows, discrimination, comparisons, catches = [], [], [], []
    for condition in CONDITIONS:
        table, scores = load(condition)
        np.testing.assert_array_equal(clean.id, table.id)
        np.testing.assert_array_equal(clean.label, table.label)
        changed = clean.prediction.ne(table.prediction)
        transitions = {"correct_correct": clean.error.eq(0)&table.error.eq(0), "correct_wrong": clean.error.eq(0)&table.error.eq(1),
                       "wrong_wrong": clean.error.eq(1)&table.error.eq(1), "wrong_correct": clean.error.eq(1)&table.error.eq(0)}
        assert sum(int(x.sum()) for x in transitions.values()) == len(clean)
        for name, mask in transitions.items():
            for restriction, keep in {"all": mask, "unchanged_prediction": mask & ~changed}.items():
                for feature in FEATURES:
                    delta = (table[feature]-clean[feature])[keep]
                    rows.append(dict(condition=condition, transition=name, restriction=restriction, feature=feature, count=len(delta),
                                     prediction_change_rate=changed[mask].mean(), mean=delta.mean(), median=delta.median(), q25=delta.quantile(.25), q75=delta.quantile(.75)))
        for feature in FEATURES:
            discrimination.append(dict(condition=condition, feature=feature, **metrics(table.error, table[feature], table.id)))
        for family in ("lr", "mlp_selected"):
            m = {k: metrics(table.error, scores[f"{k}_{family}"], table.id) for k in "ABC"}
            for a,b in [("B","A"),("C","B")]:
                comparisons.append(dict(condition=condition, family=family, comparison=f"{a}_minus_{b}", delta_aurc=m[a]["aurc"]-m[b]["aurc"]))
            for coverage in (.8,.9):
                accepted = {}
                for k in "ABC":
                    order=np.lexsort((table.id.to_numpy(), scores[f"{k}_{family}"].to_numpy()))
                    accepted[k]=int(table.error.iloc[order[:int(np.ceil(coverage*len(table)))]].sum())
                catches.append(dict(condition=condition, family=family, coverage=coverage, accepted_A=accepted["A"], accepted_B=accepted["B"], accepted_C=accepted["C"], additional_B_vs_A=accepted["A"]-accepted["B"], additional_C_vs_B=accepted["B"]-accepted["C"]))
    response=pd.DataFrame(rows); response.to_csv(OUT/"paired_novelty_changes.csv", index=False)
    pd.DataFrame(discrimination).to_csv(OUT/"novelty_discrimination.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"corruption_detector_benefit.csv", index=False)
    pd.DataFrame(catches).to_csv(OUT/"matched_coverage_errors.csv", index=False)
    thresholds=[]
    for condition in ["clean"]+CONDITIONS:
        d=pd.read_csv(FINAL/condition/"threshold_metrics.csv"); d["condition"]=condition; thresholds.append(d)
    pd.concat(thresholds).to_csv(OUT/"frozen_threshold_metrics.csv", index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for j,corruption in enumerate(("gaussian_noise","motion_blur","jpeg_compression")):
        for ax,feature in zip(axes[:,j], FEATURES):
            for t,name in enumerate(transitions):
                d=response[(response.feature==feature)&(response.transition==name)&response.restriction.eq("all")&response.condition.str.startswith(corruption)].copy()
                x=np.arange(3)+(t-1.5)*.16
                ax.errorbar(x,d["median"],yerr=[d["median"]-d.q25,d.q75-d["median"]],fmt="o",label=name)
            ax.axhline(0,color="gray",lw=.7); ax.set_xticks(range(3),[1,3,5]); ax.set(title=corruption, xlabel="Severity",ylabel="Paired change: "+feature)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/"paired_novelty_changes.png",dpi=160); plt.close(fig)
    # Reuse the existing frozen paired-bootstrap intervals, without new model selection.
    bootstrap=pd.read_csv(FINAL/"shift_bootstrap.csv")
    bootstrap.to_csv(OUT/"existing_shift_bootstrap.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    data=pd.DataFrame(comparisons)
    for ax,family in zip(axes,("lr","mlp_selected")):
        for offset,comparison in [(-.1,"B_minus_A"),(.1,"C_minus_B")]:
            d=data[(data.family==family)&(data.comparison==comparison)].set_index("condition").loc[CONDITIONS]
            ax.plot(np.arange(9)+offset,d.delta_aurc,"o",label=comparison)
        ax.axhline(0,color="gray"); ax.set_xticks(range(9),CONDITIONS,rotation=70,fontsize=7); ax.set(title=family,ylabel="ΔAURC (negative = improvement)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"corruption_detector_benefit.png",dpi=160); plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[p for c in ["clean"]+CONDITIONS for p in (FINAL/c/"features.csv",FINAL/c/"scores.csv",FINAL/c/"threshold_metrics.csv")]
    protocol=dict(status="Exploratory: final results already inspected", confidence_bins=BINS, minimum_per_class_for_auc=20,
                  conditions=CONDITIONS, comparisons=["B-A","C-B"], families=["lr","mlp_selected"], coverage=[.8,.9],
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/"analysis_settings.json").write_text(json.dumps(protocol,indent=2)+"\n")
    clean,scores=load("clean")
    confidence_analysis(clean,scores); corruption_analysis(clean)
    (OUT/"README.md").write_text("""# Exploratory analysis: confidence and corruption response

Run `python -m src.exploratory_analysis` from the repository root. Uses cached CSVs only; no fitting, inference, threshold changes, or alteration of final results.

These settings were recorded after prior final-result inspection. All results are exploratory, conditional on one trained ensemble. Confidence bins leave residual confidence differences. Counts and interquartile ranges are descriptive, not confidence intervals. Within-bin AURC comparisons are descriptive and unadjusted for multiple comparisons. AUROC is suppressed below 20 errors or correct examples.

Figures: confidence_novelty.png (correct/error medians and IQRs), paired_novelty_changes.png (corrupted minus clean, four correctness transitions), corruption_detector_benefit.png (frozen A-B-C ranking comparisons). The third figure shows point estimates; existing paired-bootstrap results are copied separately to existing_shift_bootstrap.csv.

paired_novelty_changes.csv includes unchanged-prediction restrictions and group counts. Predicted-class centroid changes can contribute to novelty changes when the prediction changes. Clean-fitted novelty standardization is retained. These transitions do not establish causation. novelty_discrimination.csv assesses separation within each condition; compare it with changes for correct→correct images.

matched_coverage_errors.csv counts accepted errors at exactly ceil(coverage*N), ordered by risk then original ID. Positive additional-errors-caught values favor augmentation. frozen_threshold_metrics.csv separately retains actual coverage and accepted risk at validation thresholds. Never infer superiority from rejection counts without matching coverage.

No new detectors are selected. No pooled inference is performed; pooled uncertainty would require source-image cluster bootstrap. Existing H1 conclusions remain unchanged.
""")
    print(f"Saved exploratory outputs to {OUT}")


if __name__ == "__main__":
    main()
