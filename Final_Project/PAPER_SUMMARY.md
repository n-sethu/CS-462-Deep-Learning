# Beyond Ensemble Agreement: methods, findings, and validity

**Research question:** When does representation novelty improve error detection beyond output uncertainty, and when does distribution shift undermine that benefit?

**Evidence snapshot:** original evaluation, geometry/depth extensions, and cache-based exploratory analyses are complete. Gaussian-noise detector adaptation is implemented and registered but has no results; job 1047614 is queued. This document summarizes evidence for a paper rather than constituting a literature review or a claim of publication readiness.

## Shared methodology

CIFAR-10 uses fixed, stratified splits (split seed 42): 35,000 CNN-training images, 10,000 detector-training images, 5,000 validation images, and the separate official 10,000-image test set. Three CIFAR-adapted ResNet-18s (seeds 1/2/3) use a 3×3 stride-1 stem, no initial maxpool, and 512-dimensional penultimate features. The recorded recipe is 100 epochs, SGD (LR 0.1, momentum 0.9, weight decay 0.0005), cosine scheduling, batch 128, random crop/horizontal flip on CNN training only. Evaluation preprocessing is deterministic.

The ensemble averages member softmax probabilities. Its uncalibrated argmax fixes predictions and error targets throughout. Output features are confidence, predictive entropy, top-two margin, and mutual information. Within each network's own feature space, Euclidean distance is measured to its centroid for the **ensemble-predicted class**. Centroids use CNN-training labels only; per-network distance mean/std use detector-training predictions without labels. Standardized distances yield mean novelty, population standard deviation across members (novelty variation), and maximum novelty. Raw representation vectors are never compared across networks.

| Detector | Features |
| --- | --- |
| A | Four output features |
| B | A + mean novelty |
| C | B + novelty variation |
| D | C + maximum novelty |

Each feature set has logistic regression (C=1, max_iter=2000) and an MLP: Linear(d,32), ReLU, Linear(32,1), sigmoid risk at inference. MLP training uses BCEWithLogits, Adam 0.001, batch 256, up to 200 epochs, patience 20. Scalers fit detector-training data only. MLP seeds 11/22/33 are each early-stopped on validation AURC; the lowest validation-AURC seed is frozen (A/B/C=33, D=22). Three-seed sensitivity is separate from selected-model evaluation.

Metrics: error AUROC (higher better), AURC (lower better), and selective risk at exact 80%/90% coverage. AURC averages cumulative error fractions over all accepted-prefix sizes after sorting by increasing risk, then original example ID. Validation-only thresholds separately measure actual coverage and accepted risk. Paired 95% percentile bootstrap intervals use 2,000 resamples; pooled corruption analyses resample original-image clusters. These quantify uncertainty conditional on the fitted system, not across independently trained ensembles.

**Protocol deviation:** CNN checkpoints were selected by minimum validation loss, although the original specification requested maximum validation accuracy. Existing selections were preserved after test evaluation. Sources: [ensemble configuration](configs/ensemble.json), [detector protocol](results/validation/frozen_protocol.json), [evaluation methods](FINAL_EVALUATION.md).

## 1. Ensemble and strong output baselines — completed

**Method:** compare confidence, margin, entropy, mutual-information rankings, calibrated confidence, and learned A. Fit a single shared temperature on validation, apply it to each member before probability averaging, and retain original uncalibrated error targets.

**Results:** member accuracies are 94.01%, 94.05%, and 93.86%; ensemble accuracy is 94.77%. Clean AURC is 0.005427 for confidence, 0.005406 for margin, 0.005522 for entropy, 0.005828 for mutual information, and 0.005481 for calibrated confidence. Temperature T=1.3149 changes the test ensemble argmax on 17 images, but those changed predictions are not used as targets. Learned output-only MLP AURC is 0.005435: simple output scores are strong competitors.

**Validity:** members have distinct weights/predictions; cached scores support fixed-prediction comparisons. Temperature ranking is not a demonstration of improved calibration; no calibration-solving claim follows. [Results](results/final/main_table.csv).

## 2. Controlled novelty ablation and detector nonlinearity — completed

Selected-MLP clean results:

| Configuration | AUROC | AURC | Accepted errors at 90% coverage |
| --- | --- | --- | --- |
| A | 0.9363 | 0.005435 | 149 |
| B | 0.9401 | 0.004955 | 140 |
| C | 0.9366 | 0.005160 | 155 |
| D | 0.9390 | 0.005018 | 140 |

**H1, primary:** C−B AURC = +0.0002053, CI [+0.0000870, +0.0003528]. Novelty variation worsens the primary comparison; H1 is unsupported. LR C−B is inconclusive (+0.0000180, CI [−0.0000704, +0.0000922]).

**H2:** B−A is −0.0004802, CI [−0.0010193, −0.0000757], for MLP; −0.0004433, CI [−0.0010320, −0.0000057], for LR. Mean novelty provides conditional evidence of improvement, about 8.8% relative MLP AURC reduction and nine additional errors rejected at equal 90% coverage. H2 is secondary and unadjusted.

**H3:** MLP D−C = −0.0001419, CI [−0.0003971, +0.0000427]; maximum novelty has no clear incremental benefit. **H6:** clean MLP−LR intervals span zero for every feature set; nonlinear superiority is not established.

**Sensitivity/diagnostics:** three-seed AURC mean±population std is 0.004952±0.000003 for B versus 0.005126±0.000055 for C. Permuting variation over 100 permutations raises AURC by 0.000145 (LR) / 0.000339 (MLP): this demonstrates reliance, not independent useful information. Validation correlations include mean/max novelty 0.976, variation/mean 0.646, and variation/mutual information 0.673; collinearity complicates interpretation. Sources: [clean intervals](results/final/clean_bootstrap.csv), [seed sensitivity](results/final/seed_sensitivity.csv), [correlations](results/validation/feature_correlations.csv).

## 3. Unanimous predictions and confident failures — completed

**Method:** evaluate unchanged detectors on unanimous predictions and on unanimous predictions with uncalibrated confidence ≥0.9. Suppress AUROC when either class has fewer than 20 examples.

**Results:** 9,274 test predictions are unanimous (9,050 correct, 224 wrong); 8,850 are confident and unanimous (8,719 correct, 131 wrong). Within the confident-unanimous subset, A/B/C MLP AURCs are 0.002428/0.001947/0.001992; AUROCs are 0.8858/0.8978/0.8961. At 90% **subset** coverage they accept 49/45/42 errors; at 80% they accept 21/22/20.

**Validity:** novelty helps some confident errors, but B is not better at every operating point. C can beat B at a chosen coverage while having worse integrated AURC. Subset point estimates have no new paired intervals in this analysis. Frozen-threshold galleries show both successes and regressions; they are descriptive, not causal explanations. [Subset metrics and examples](results/final/clean/metrics.csv).

## 4. Frozen clean-to-corruption transfer — completed

**Method:** evaluate unchanged CNNs, detectors, scalers and thresholds on CIFAR-10-C Gaussian noise, motion blur, and JPEG compression, each at severities 1/3/5. Report each condition separately.

| Corruption | B−A MLP AURC at severities 1 / 3 / 5 | Interpretation |
| --- | --- | --- |
| Gaussian noise | +0.001032 / +0.006791 / +0.002850 | Mean novelty hurts integrated ranking |
| Motion blur | −0.000995 / −0.004327 / −0.006499 | Favorable mean-novelty point estimates |
| JPEG | −0.001264 / −0.000409 / −0.003004 | Favorable mean-novelty point estimates |

Both detector families show these B−A directions, but the completed original shift intervals cover C−B, not B−A. H5 therefore has mixed, condition-dependent evidence. C−B MLP improves under noise severity 3 (−0.004635, CI [−0.005911, −0.003439]) and 5 (−0.002670, CI [−0.004127, −0.001318]), but worsens under mild noise and all blur/JPEG conditions. Even at noise 3/5, C does not beat A on integrated AURC. This secondary exception does not rescue clean H1.

**Deployment:** clean-validation 90%-coverage thresholds do not retain 90% coverage under shift. At noise severity 5, B accepts 46.60% of images with 68.80% accepted error rate. Ranking gains and threshold reliability must be reported separately. Pooled C−B can conceal opposite per-condition directions. [Shift intervals](results/final/shift_bootstrap.csv), [thresholds and matched coverage](results/exploratory/ANALYSIS.md).

## 5. Cost and qualitative failure analysis — completed

**Method:** RTX A4500, synchronized CUDA, five warmups and 30 timing repeats, batch sizes 1/128; exclude disk I/O and preprocessing. Compare single CNN, ensemble, feature extraction and detectors. Inspect fixed-source-ID galleries in both directions.

**Results:** batch-1 median latency is 1.95 ms for one CNN, 5.83 ms for ensemble logits, and 7.45 ms for ensemble features plus selected C MLP; the latter is approximately 3.8× single-CNN latency. At batch 128 it is 0.250 ms/image versus 0.079 for one CNN. Penultimate features require no additional CNN pass. Clean example 1492 is an incorrect truck→automobile prediction caught by C but accepted by B; examples 6709/9218 are incorrect cat→dog predictions caught by B but accepted by C.

**Validity:** timing is hardware/implementation specific; gallery catches use frozen thresholds with potentially unequal actual coverage. They neither establish equal-coverage superiority nor explain a causal mechanism. [Latency](results/final/latency.csv), [failure analysis](results/final/summary.md).

## 6. Mahalanobis geometry (H7) — completed, exploratory

**Method:** retain stage-4 features and centroids; fit one Ledoit–Wolf covariance per CNN to true-class-centered CNN-training residuals. Use Cholesky solves for predicted-class Mahalanobis distance. Fit normalization on detector-train and refit B–D with the original detector recipe; reuse A. Freeze on clean validation before evaluating.

**Results:** clean B MLP AURC is 0.005112, worse than Euclidean B by +0.000157 (CI approximately [+0.000001, +0.000307]). Mahalanobis C−B is +0.000037, CI [−0.000070, +0.000148]: no recovered variation benefit. Mahalanobis B−A is also inconclusive. Its B improves over Euclidean B under noise 3/5, but worsens under JPEG 1/3/5 and blur 3 with intervals excluding zero. Geometry changes where novelty succeeds rather than universally improving it.

**Validity:** all extension comparisons are exploratory and unadjusted. A favorable change in C−B relative to Euclidean does not establish negative C−B within Mahalanobis. [Full results and intervals](results/extensions_h7_h8/h7/summary.md).

## 7. Representation depth (H8) — completed, exploratory

**Method:** spatially average stages 1/2/4 (64/128/512 dimensions), using Euclidean novelty at every depth and identical detector procedures. Verify logits, predictions and stage-4 features against original caches during early-feature extraction. Reuse stage 4; no CNN retraining.

**Results:** stage-1/stage-2 B MLP clean AURCs are 0.006055/0.005490 versus stage-4 0.004955. Differences are +0.001100 (CI [+0.000259, +0.002132]) and +0.000536 ([+0.000154, +0.001018]). Stage-4 B is better on clean data. Earlier B improves over stage-4 B under noise 3/5, while stage-1 B worsens under all blur/JPEG conditions. Stage-2 C is frequently worse; early representations are not generally superior shift monitors. For correct→correct noise-5 images, mean standardized novelty increases by 3.979/3.514/1.255 at stages 1/2/4: a stronger corruption response does not itself imply better error detection.

**Validity:** depth is confounded with dimension and spatial pooling; responses use stage-specific standardization. Results suggest task/condition dependence, not a universal layer rule. Extension reports record unchanged-file audits of 247 pre-existing files. [Depth results](results/extensions_h7_h8/h8/summary.md).

## 8. Confidence-conditioned and paired-transition analysis — completed, exploratory

**Method:** reuse cached features/scores in fixed confidence bins [0,.5,.7,.8,.9,.95,1], and pair each clean image with its corrupted version. Compare correct→correct, correct→wrong, wrong→wrong and wrong→correct transitions; retain unchanged-prediction checks. Report matched-coverage counts separately from frozen-threshold behavior.

**Results:** mean novelty is higher for errors in every confidence bin. At confidence ≥0.95, 8,481 correct and 92 wrong predictions have median novelty −0.402 and +0.745; novelty-only AUROC is 0.878. Variation AUROC is below 0.5 in every lower-confidence bin and 0.681 in the highest. Correct→correct novelty rises in all nine corruptions. Under noise 3/5, its mean increase is +1.280/+1.255, close to +1.495/+1.501 for new errors. Mean-novelty AUROC falls to 0.640/0.633, compared with 0.784 for blur 5 and 0.827 for JPEG 5.

**Validity:** confidence varies within bins; this is not exact confidence matching or proof of conditional independence. Correct images also become atypical under blur/JPEG, where B still helps, so rising correct-image novelty alone is an incomplete failure explanation. Prediction changes switch the reference centroid; transition differences are not causal effects. These analyses do not separate lost novelty discrimination from detector-transfer failure. [Complete exploratory analysis and CSV inventory](results/exploratory/ANALYSIS.md).

## 9. Gaussian-noise detector adaptation — registered; results pending

**Method:** keep CNNs, centroids and clean novelty normalization fixed. Fit new A/B LR and three-seed MLPs on clean + noise 1/3/5 versions of detector-training sources only; use corresponding validation sources for equal-weight mean per-condition AURC selection. Fit feature scalers only on the training mixture. Freeze clean-validation thresholds before evaluating clean and all nine existing corruption conditions. Blur/JPEG are excluded from adaptation development.

**Planned evidence:** original B−A, adapted B−A, and Gamma=(adapted B−A)−(original B−A), with shared paired-bootstrap draws, matched-coverage error differences, seed sensitivity and frozen-threshold coverage/risk. Recovered novelty benefit requires adapted B beating adapted A; improvement of B alone or negative Gamma is insufficient.

**Status/validity:** implementation passed five adaptation and six existing evaluation tests; registration exists, but no training/evaluation completion record exists. Job 1047614 is queued at this snapshot. Do not report an adaptation outcome. All evaluations remain exploratory because the existing test conditions were inspected. Mixture training also changes optimizer exposure, scaler fitting and selection objective; recovery would support a transfer explanation without proving a mechanism. [Protocol](configs/adaptation.json), [instructions](ADAPTATION.md).

## Paper framing and remaining limits

The defensible contribution is a controlled empirical audit: **mean representation novelty adds modest clean error-detection value, including among confident unanimous predictions; novelty variation has no primary clean benefit; geometry, depth, corruption and operating point determine usefulness.** Representation distance is an established approach, not a new invention here.

There is one independently trained three-member ensemble, not multiple ensemble replications. Detector seeds do not address that limitation. Bootstrap intervals are conditional, secondary comparisons are numerous and unadjusted, extension questions were chosen after original results, and no causal mechanism or calibration solution is established. Confidence bins retain residual confounding, and only three corruptions on one dataset/architecture are studied. Preserve the documented checkpoint-selection deviation.

No independent-ensemble replication, alternative-architecture study, SimCLR, graph clustering, MC-dropout comparison, or completed demo is evidenced by the reviewed experiment artifacts. These should not appear as completed work. For the paper, cite the established ensemble, calibration, selective-classification, representation-distance and dataset-shift literature listed in AGENTS.md after verifying the papers; this summary does not substitute for a verified bibliography.
