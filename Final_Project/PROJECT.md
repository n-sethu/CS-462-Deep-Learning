# Beyond Ensemble Agreement

**Detecting Confident Neural Network Failures Using Internal Representations**

CS 462 final project (individual). Start: Oct 30, 2026. Presentation: Nov 25, 2026 (max 3 slides, 3 minutes). Final report: Dec 7, 2026. Budget: about 8-10 hours/week.

> Source of truth for the AI coding agent working in this repo. Read fully before writing code. Follow the scope, split rules, leakage rules, and priority order exactly. When something is ambiguous, pick the simplest option that respects the rules, and note the choice in the README or config.

---

## 1. Problem

Deep networks can be confidently wrong. Deep ensembles improve uncertainty estimates, but disagreement-based measures fail when **all members confidently predict the same incorrect class**. Models that agree on the *label* may still differ in how *typical* they consider the input. This project tests whether that difference is useful for detecting errors.

## 2. Research objective and question

Determine whether **variation in class-conditional representation novelty across independently trained networks** improves error detection beyond (a) output uncertainty and (b) average representation novelty.

Main question: can models that agree on a class prediction still reveal useful differences in how typical they consider an input?

## 3. Notation

- `M = 3` ensemble members. `h_m(x)` = penultimate representation of model `m`. `y_hat(x)` = ensemble prediction.
- `d~_m(x)` = standardized class-conditional novelty of model `m` (defined in Section 7).
- Output features `Z_out = [c, H, M_margin, I]` (confidence, entropy, margin, mutual information).
- `d_bar(x) = (1/M) * sum_m d~_m(x)` (mean novelty)
- `D_novelty(x) = std_m d~_m(x)` (novelty variation across models)
- `d_max(x) = max_m d~_m(x)` (maximum novelty)

## 4. Hypotheses

**H1 (primary): cross-model novelty variation adds value beyond mean novelty.**
`AURC(Z_out, d_bar, D_novelty) < AURC(Z_out, d_bar)`
- **Null (H0):** adding `D_novelty` does not reduce AURC (the difference is zero or positive).
- Supported only if `ΔAURC = AURC(C) - AURC(B) < 0` AND the paired bootstrap CI excludes zero. A negative point estimate with a CI spanning zero is *not* support; report it as inconclusive.

Secondary hypotheses (each with its own null of "no improvement"):
- **H2:** mean novelty improves on output-only. `AURC(B) < AURC(A)`.
- **H3:** maximum novelty adds value beyond mean + variation. `AURC(D) < AURC(C)`.
- **H4 (unanimous agreement):** the representation features help detect errors *within* the subset where all 3 models predict the same class, including the confident subset (`c >= 0.9`).
- **H5 (distribution shift):** representation features remain useful on CIFAR-10-C even though the detector is trained only on clean data. This may fail: distances can rise for both correct and incorrect corrupted images, destroying discriminative power.
- **H6 (nonlinearity):** the MLP detector beats logistic regression on the same features. If not, say so; linear is then sufficient.

Optional hypotheses (only after the primary experiment is done; see Section 9):
- **H7:** covariance-aware (Mahalanobis) novelty beats Euclidean novelty.
- **H8:** representation depth matters: later layers detect decision-related errors better; earlier layers detect low-level corruptions better.

H1 is the primary experiment. All others are extensions and **must not delay H1**.

## 5. Contribution framing (be honest)

- Not inventing deep ensembles, representation-based uncertainty, or selective classification. Lee et al. (2018) already uses intermediate representations for confidence, so representation distance alone is **not novel**.
- The defensible contribution is a **controlled ablation** isolating (i) mean novelty, (ii) cross-model novelty variation, and (iii) max novelty on top of strong output-only baselines, plus targeted unanimous-error and distribution-shift analysis.
- Incremental value from latent features is **not guaranteed**. Cite *Auditing Latent-Space Monitors for Autonomous Driving (2026)* as an important related-work comparison and acknowledge this explicitly.
- Outcomes: positive result may motivate a new method; negative result supports a careful empirical audit; mixed result may identify conditions where representation monitoring works. All are valid. Never overclaim.
- Scope limit: one trained ensemble on one dataset is a course-project foundation, not a publication-level claim. A stronger paper would need multiple independently trained ensembles, stronger representation baselines, and a clearer account of when latent monitoring succeeds or fails. Say this in the limitations. Also keep open that `D_novelty` may just reflect differences in feature geometry across networks rather than useful error information.

Related work to read/cite: Lakshminarayanan 2017 (Deep Ensembles), Guo 2017 (Calibration / temperature scaling), Cattelan & Silva 2024 (simple confidence baselines are strong), Lee 2018 (Mahalanobis confidence), Papernot & McDaniel 2018 (Deep k-NN), Geifman & El-Yaniv 2017 (selective classification), Hendrycks & Dietterich 2019 (CIFAR-10-C), Ovadia 2019 (uncertainty under dataset shift), Auditing Latent-Space Monitors (2026).

---

## 6. Data splits (fixed, stratified, seeded)

Use CIFAR-10 only. Save split indices to disk and never regenerate them differently.

| Split | Images | Source | Used for |
|---|---|---|---|
| **Train (CNN)** | 35,000 | CIFAR-10 train set | Train the 3 CNNs; compute class centroids |
| **Detector-train** | 10,000 | CIFAR-10 train set (held out from CNN training) | Fit novelty-standardization stats (label-free); train the error detectors (LR and MLP) |
| **Validation** | 5,000 | CIFAR-10 train set (held out) | Early stopping, hyperparameters, temperature scaling, abstain thresholds, model selection |
| **Test** | 10,000 | Official CIFAR-10 test set | Final evaluation only, touched once at the end per experiment |

**Why the detector gets its own held-out split:** CNNs are overconfident and overfit on their own training data, so error labels computed there would be nearly all zero and unrepresentative. The detector must learn from predictions on images the CNNs never trained on.

## 7. Component A: Deep ensemble + output uncertainty

- 3 CIFAR-adapted ResNet-18 models (replace the large initial conv and maxpool with a small-image stem). Input `3x32x32`. Each yields logits (10) and a 512-d penultimate representation.
- Same architecture and training recipe, **different random seeds** (changing both init and data order). Cross-entropy loss. Augmentation on the CNN train split only; deterministic preprocessing for all evaluation/feature extraction.
- Record: optimizer, LR schedule, batch size, epochs, augmentation, checkpoint selection rule (best val accuracy), seeds.
- If training is too slow, shrink the CNN rather than cutting later stages.

Ensemble: `p_m = softmax(l_m)`, `p_bar = mean_m p_m`, `y_hat = argmax p_bar`.

Output features (on `p_bar`):
- `c(x) = max_k p_bar_k` (max confidence)
- `H(x) = -sum_k p_bar_k log p_bar_k` (predictive entropy)
- `M_margin(x) = p_bar_(1) - p_bar_(2)` (top-1 minus top-2 probability)
- `I(x) = H(p_bar) - (1/3) sum_m H(p_m)` (ensemble mutual information)

Also compute the fraction of examples where all 3 models agree.

**Strong output-only baselines** (the bar the proposed method must beat):
- Rankings by max-softmax confidence, margin, entropy, mutual information.
- Temperature scaling (Guo 2017), defined precisely: fit a **single shared temperature `T`** on the **validation** split only (never test). Apply it to every member's logits *before* averaging: `p_m^(T) = softmax(l_m / T)`, `p_bar^(T) = (1/M) * sum_m p_m^(T)`. Use this one definition everywhere a calibrated baseline is needed.
  - Caveat: scaling logits before averaging can change the ensemble's argmax. To keep every comparison on identical predictions, the fixed base prediction `y_hat` (and therefore the error labels `e`) always comes from the **uncalibrated (T=1) ensemble**. Calibrated scores are used only to *rank* those fixed predictions. Report how often the calibrated argmax differs from `y_hat`.
- Output-only logistic regression and output-only MLP (detector A, below).

**Validation checks:** the 3 models are not identical (different weights, different predictions on some inputs); ensemble predictions are computed correctly (sanity test on a toy example); per-model and ensemble accuracy reported; one training step, one eval step, and checkpoint save/load verified before long runs.

**Deliverables:** 3 checkpoints; config + seeds; per-model and ensemble accuracy; cached logits and probabilities; feature table (output uncertainty + correctness); baseline AUROC and AURC; risk-coverage curves; confidence distributions for correct vs. incorrect; agreement statistics.

**Completion criterion:** run inference on one image, get all 3 predictions, compute all 4 output features.

## 8. Component B: Representation novelty (central component)

**Why not compare raw feature vectors across models:** independently trained networks can encode the same information in different coordinate systems, so `||h_1(x) - h_2(x)||` is not meaningful disagreement. Evaluate each model's representation **within its own feature space**.

**B1 (required): class-conditional Euclidean novelty**
- Centroids per model `m` and class `k` from the **train (CNN)** split only: `mu_{m,k} = mean of h_m(x_i) over train images with label k`.
- Distance to the centroid of the **ensemble-predicted** class (never the true label): `d_m(x) = ||h_m(x) - mu_{m,y_hat(x)}||_2`.
- Standardize per model: `d~_m(x) = (d_m(x) - mu_{d_m}) / (sigma_{d_m} + eps)`. Compute `mu_{d_m}` and `sigma_{d_m}` from distances on the **detector-train** split (not the CNN train split). Reason: centroids are built from CNN-train images, so those images sit systematically closer to their centroids than unseen images do, which would give a miscalibrated reference distribution. These stats use only predicted classes, never labels, so there is no label leakage. (Alternative if ever needed: cross-fitting to get out-of-sample train distances. Default is detector-train.)
- Aggregate across models into `d_bar`, `D_novelty`, `d_max`.
- Interpretations: `d_bar` = unusual across the ensemble? `D_novelty` = do models disagree about how unusual it is? `d_max` = does at least one model find it especially unusual? `D_novelty` is variation in novelty, **not** disagreement between representation vectors; state this in the report.
- A large distance means "atypical for the class," not necessarily "wrong."
- Caveats to check and report: `D_novelty` is the std of only 3 numbers (noisy); check correlation between `d_bar`, `D_novelty`, `d_max` (collinearity) and between `D_novelty` and output disagreement (`I`), since a feature that merely proxies existing signals will not add value.

**B2 (optional, only after Exp 1 is complete): Mahalanobis novelty**
- `d_Mah_m(x) = sqrt((h_m(x) - mu_{m,y_hat})^T Sigma_m^{-1} (h_m(x) - mu_{m,y_hat}))`, with `Sigma_m` from train-split features.
- Use a regularized/shrinkage covariance (e.g. Ledoit-Wolf), never a raw inverse of a 512-d sample covariance.
- Standardize and aggregate exactly like B1. Tests H7. Planned slot: Nov 17 if primary experiments are done; otherwise use that day to investigate unexpected results.

**B3 (optional, only if time permits): representation depth**
- Extract representations from several ResNet stages and compute novelty per stage; compare error detection by depth and by clean vs. corrupted data. Tests H8.

**Other optional items (after primary results):** MC Dropout as an extra uncertainty baseline; additional architectures. These are optional extras, not required.

**Deliverables:** cached representations for train / detector-train / val / test; centroids per network; standardized novelty (3 scores per input); `d_bar`, `D_novelty`, `d_max`; novelty distributions for correct vs. incorrect predictions; diagnostic plots (including confidence-vs-novelty scatter).

**Completion criterion:** given an image, produce the novelty features **without needing its true label**.

## 9. Component C: Error detector and controlled ablation

Target: `e_i = 1[y_hat(x_i) != y_i]`. Detector estimates `P(e = 1 | features)`. It **never changes the prediction**; it only scores risk and decides accept/abstain.

**Four feature configurations (the controlled ablation):**

| Model | Input features |
|---|---|
| A | `Z_out` (output uncertainty) |
| B | `Z_out` + `d_bar` |
| C | `Z_out` + `d_bar` + `D_novelty` |
| D | `Z_out` + `d_bar` + `D_novelty` + `d_max` |

Each configuration is trained as **both** logistic regression and an MLP = **8 detectors total**. MLP: `Linear(d, 32) -> ReLU -> Linear(32, 1) -> sigmoid`, BCE loss; identical architecture and training procedure across A-D (only the input dimension `d` changes: 4, 5, 6, 7).

- Train on **detector-train**; early stopping and hyperparameters on **validation**.
- Fit feature scalers on detector-train only; save them with the checkpoints.
- Accept/abstain threshold chosen on validation.
- Logistic regression is required: without it you cannot claim the nonlinear detector adds anything (H6).

**Deliverables:** 8 trained detectors; saved checkpoints + scalers; predicted error probabilities; accept/abstain decisions; reproducible evaluation scripts.

**Completion criterion:** the system estimates prediction risk and can reject uncertain classifications.

### Leakage rules (non-negotiable; add tests/asserts)

1. Features never use the true label of the evaluated image; novelty uses the **predicted** class.
2. Centroids come **only** from the Train (CNN) split. Novelty-standardization stats come **only** from the detector-train split (label-free).
3. Detectors train **only** on detector-train.
4. Hyperparameters, temperature scaling, thresholds, and model selection use **only** the validation split.
5. The **test set is untouched** until final evaluation. **Never tune on test or CIFAR-10-C.** The detector is frozen for shift evaluation.
6. Compare detectors on identical test examples and identical ensemble predictions (the ensemble and its predictions are fixed; only detector inputs differ).

---

## 10. Experiments

**Exp 1: Primary evaluation (clean test set), required.**
Evaluate all detectors plus the single-score baselines on the test set. Metrics:
- Error-detection **AUROC**
- **AURC** (lower is better), defined exactly: sort the `N` predictions from lowest to highest predicted error risk, with `e_(i)` the 0/1 error of the i-th sorted prediction. Selective risk at k kept predictions is `r_k = (1/k) * sum_{i=1..k} e_(i)`, and `AURC = (1/N) * sum_{k=1..N} r_k`. Use **deterministic tie-breaking** (stable sort by score, then by example index) and document it. Unit-test this function (perfect ranker, random ranker, all-tied scores, hand-computed small case).
- Selective risk at **80%** and **90%** coverage.
Primary results: `ΔAURC = AURC(C) - AURC(B)` (H1), plus `AURC(B) - AURC(A)` (H2) and `AURC(D) - AURC(C)` (H3), each with paired bootstrap CIs. Compare LR vs. MLP (H6).

**Primary-detector protocol (so H1 has one unambiguous test):** for each MLP configuration, train 3 seeds, **select the seed with the best validation AURC**, and freeze it before any test evaluation. The primary comparison and bootstrap CI use only those frozen detectors. The 3-seed mean/std is reported separately as a sensitivity analysis.

**Diagnostic (not a primary test): permutation importance of `D_novelty`.** After training detector C, randomly permute `D_novelty` across test examples (other features unchanged, fixed seed, average over many permutations) and report `Δ_perm = AURC(C_permuted) - AURC(C)`. A large increase means the detector relies on `D_novelty`. Caveat: features are correlated, so this does **not** prove `D_novelty` carries independent information; interpret it alongside the B vs. C comparison and the correlation checks.

Deliverables: main comparison table and risk-coverage curves.

**Exp 2: Unanimous agreement and confident errors, required.**
- `A = {x : y_hat_1 = y_hat_2 = y_hat_3}`. Evaluate all detectors on this subset: AUROC and AURC. Report counts of unanimously correct and unanimously incorrect examples.
- Confident subset `x in A, c(x) >= 0.9`: compare output-only vs. representation-augmented risk scores. Does high `D_novelty` flag failures missed by conventional uncertainty?
- If a subset has too few errors, **do not interpret its AUROC**. Report counts and show a gallery of real confident-error examples instead.

**Exp 3: CIFAR-10-C distribution shift, required.**
- Corruptions fixed in advance: Gaussian noise, motion blur, JPEG compression; severities 1, 3, 5 (9 conditions).
- Train CNNs and detectors on clean data; evaluate the **frozen** system on corrupted images. No retraining or retuning on corrupted data.
- Report AURC and AUROC per condition and compare detectors A-D vs. clean performance.
- Corrupted copies of the same source image are correlated: when pooling, **bootstrap at the original-image level**.

**Exp 4: Cost and failure analysis, required.**
- Inference latency, feature-extraction overhead, detector overhead (3 networks + detector vs. single network).
- Failure cases where representation features **reduce** performance; limitations analysis.

**Exp 5 (optional): Mahalanobis vs. Euclidean (H7), depth comparison (H8), MC Dropout, extra architectures.** Only after Exp 1-4.

## 11. Statistical rigor and reproducibility

- Train each MLP detector with **3 random seeds**. Pick the primary seed by **validation AURC**, freeze it, and use it for the primary test (Exp 1). Report 3-seed mean/std of AURC and AUROC separately as a sensitivity analysis. Seeds capture detector-init variability only, **not** variability across independently trained ensembles; say so.
- The paired bootstrap resamples test examples conditional on the trained ensemble and detectors. A CI excluding zero is evidence against the null under this procedure; it does **not** show the improvement generalizes across independently trained ensembles. State this in the report.
- **Paired bootstrap** (resample test examples, same resample for both detectors compared) for CIs on all `ΔAURC` comparisons. Use enough resamples (e.g. 1,000+), fixed bootstrap seed.
- Note multiple comparisons (several ΔAURC tests); don't cherry-pick the best one.
- Do not overstate: results are limited to this architecture, dataset, corruptions, and training protocol.
- Record and version: split indices, all seeds, architecture, optimizer, LR, batch size, epochs, augmentation, checkpoint rule, feature normalization, detector hyperparameters. Pin dependency versions.
- Cache logits/representations/features to disk so evaluation never needs retraining or CNN re-inference. All experiments reproducible from saved checkpoints via scripts.

## 12. ML research-engineer mindset (best practices)

- **Baselines first.** Build strong output-only baselines (calibrated confidence, LR, MLP) before the proposed idea; judge the idea against them.
- **Change one thing at a time.** A-D differ only in input features; same data, architecture, training recipe, predictions, and test examples.
- **Strict train/val/test discipline.** Test is for reporting, not decisions. No peeking; no tuning on test or CIFAR-10-C.
- **Pre-commit decisions.** Fix corruptions, metrics, splits, and hypotheses before seeing results. No p-hacking.
- **Sanity-check everything.** Overfit a tiny batch, verify models differ, unit-test metrics (e.g. AURC on a perfect/random ranker), assert no label use in feature code.
- **Quantify uncertainty.** Multiple seeds and bootstrap CIs; be skeptical of small subsets.
- **Report negatives and explain them.** A clean null result is a valid, valuable outcome.
- **Do failure analysis.** Inspect when the detector is wrong or worse than baseline; show detected and missed errors.
- **Modular, cached pipeline.** Training, feature extraction, detection, evaluation, and demo are separate stages connected by cached artifacts and configs; reproducible with one command.
- **Be ready to explain:** why representations might hold information absent from outputs; how the contribution is isolated; whether an AURC gain is meaningful; behavior under shift; inference cost; when the detector hurts.
- **Resist scope creep.** Finish the primary result first.

## 13. Demo (Streamlit)

Show for a selected cached example: input image, predictions from all 3 CNNs, ensemble confidence and disagreement, mean novelty, cross-model novelty variation, estimated error probability, accept/abstain decision. Include examples of confident correct agreement, **confident incorrect agreement**, ensemble disagreement, and corrupted inputs. Also a results view: risk-coverage curves, AURC/AUROC table, confidence-vs-novelty scatter, gallery of detected and missed errors. Use cached predictions so evaluation never requires live inference.

## 14. Repo structure

```
beyond-ensemble-agreement/
├── configs/        ensemble.yaml, detector.yaml
├── src/            data.py models.py train_ensemble.py extract_features.py
│                   representation.py train_detector.py evaluate.py metrics.py
├── app/            streamlit_app.py
├── scripts/        run_baselines.sh run_experiments.sh
├── results/        tables/ figures/
├── reports/        presentation.pdf final_report.pdf
├── requirements.txt
└── README.md
```

Do not commit datasets or checkpoints to Git. The README must explain how to reproduce experiments and launch the demo.

## 15. Timeline (8-10 hrs/week)

**Week 1 (Oct 30 - Nov 5): Component A. Ensemble and output-only baselines.**
- Oct 30: read Lakshminarayanan, Guo, Cattelan & Silva; create repo/configs; download CIFAR-10; make stratified splits; implement CIFAR ResNet-18; verify train step, eval step, checkpointing.
- Oct 31 - Nov 2: train 3 ResNet-18s (different seeds); report accuracies; cache logits/probabilities; verify models differ and ensemble math is correct.
- Nov 3: output features `Z_out`; agreement fraction; feature table with correctness.
- Nov 4: baselines (4 rankings, output-only LR and MLP, temperature scaling fit on validation); baseline AUROC/AURC.
- Nov 5: risk-coverage curves, confidence distributions (correct vs. incorrect), agreement statistics.
- *Milestone:* working ensemble with reproducible output-only baselines.

**Week 2 (Nov 6 - Nov 12): Components B and C. Primary experiment.**
- Nov 6: read Lee 2018 and Papernot & McDaniel 2018; extract and cache penultimate representations for all splits.
- Nov 7: centroids, standardized predicted-class novelty (3 scores per input).
- Nov 8: `d_bar`, `D_novelty`, `d_max`; novelty distribution plots.
- Nov 9: train detectors A-D, each as LR and MLP (8 total).
- Nov 10: primary test-set evaluation (AUROC, AURC, risk at 80/90% coverage); main table and curves.
- Nov 11: 3-seed MLP repeats; select primary seeds by validation AURC and freeze; paired bootstrap CIs for `ΔAURC = AURC(C) - AURC(B)` on the frozen detectors; 3-seed results as sensitivity analysis; permutation-importance diagnostic for `D_novelty`.
- Nov 12: interpret: does mean novelty help (H2)? does variation add value (H1)? does max help (H3)? is LR as good as MLP (H6)?
- *Milestone (critical, Nov 12):* primary hypothesis tested.

**Week 3 (Nov 13 - Nov 19): Failure analysis and distribution shift.**
- Nov 13: unanimous-agreement analysis (all detectors).
- Nov 14: confident-error analysis (`c >= 0.9`), example gallery.
- Nov 15-16: read Hendrycks & Dietterich and Ovadia; CIFAR-10-C evaluation (3 corruptions x severities 1, 3, 5), no detector retraining.
- Nov 17: optional Mahalanobis vs. Euclidean, or investigate unexpected results.
- Nov 18: latency/overhead table; failure cases where representation features hurt.
- Nov 19: freeze result tables and figures; identify strongest supported conclusion.
- *Milestone:* complete empirical evaluation.

**Week 4 (Nov 20 - Nov 25): Demo and presentation.**
- Nov 20-21: Streamlit demo.
- Nov 22: final figures: risk-coverage comparison; novelty variation vs. correctness; unanimous-agreement results; distribution-shift comparison.
- Nov 23: 3 slides. (1) Motivation: ensembles can confidently agree on wrong answers; can novelty variation reveal them? (2) Method: 3 CNNs, output + novelty features, learned detector, controlled ablation. (3) Results: primary AURC comparison, unanimous-error findings, one representative example. Do not explain every equation.
- Nov 24: reproduce main results from checkpoints, test demo, verify figures/stats, rehearse, prepare limitations and negative-result answers.
- Nov 25: present (3 minutes).

**Nov 26 - Dec 7: Final report.** Sections: abstract; introduction; related work (ensembles, representation novelty, selective classification, plus the 2026 latent-space monitor audit); method; experimental setup; results and ablations; analysis (unanimous errors, shift); limitations (statistical power, cost, generalization); conclusion. Prioritize exam prep before Dec 4.

## 16. Scope

| Item | Status |
|---|---|
| 3x ResNet-18 ensemble, output baselines (rankings, temperature scaling, LR, MLP) | Required |
| Representation novelty (`d_bar`, `D_novelty`, `d_max`), detectors A-D x {LR, MLP} | Required |
| Clean CIFAR-10 evaluation, unanimous-error analysis, CIFAR-10-C (3 corruptions x 3 severities) | Required |
| Cost/failure analysis, Streamlit demo, 3 slides, reproducible repo | Required |
| Mahalanobis novelty, multi-layer representations, MC Dropout, additional architectures | Optional, only after primary results |
| Additional datasets | Do not add |

### Priorities if time runs short

This project is about error detection, **not** training a high-accuracy classifier. The biggest risk is burning week 1 debugging ResNet training. If training is slow, simplify the architecture (smaller CNN, fewer epochs) before sacrificing the primary experiment.

| Priority | Component | Why |
|---|---|---|
| P0 | Working ensemble + cached predictions | Everything depends on it |
| P0 | Output-only baselines | Sets the bar to beat |
| P0 | Novelty features and B vs. C | Tests the primary hypothesis |
| P1 | Statistical evaluation (frozen seeds, bootstrap, AURC tests) | Makes results credible |
| P1 | Unanimous-agreement analysis | Tests the motivating failure mode |
| P1 | CIFAR-10-C | Robustness |
| P2 | Full interactive demo | Communication, not science |
| P3 | Optional novelty metrics (Mahalanobis, depth, MC Dropout) | Only after primary results |

## 17. Definition of success

Answer three questions with reproducible evidence:
1. Does representation novelty improve error detection beyond output uncertainty?
2. Does cross-model novelty variation provide information beyond mean novelty?
3. Do these findings hold for unanimously agreeing ensembles and corrupted inputs?

The critical milestone is **Nov 12**. Everything after it should strengthen the evidence, analyze failures, and communicate results, not start a fundamentally new project.