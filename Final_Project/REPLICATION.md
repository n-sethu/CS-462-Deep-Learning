# Ten additional ensemble replications

The protocol in `configs/replication.json` fixes 30 new CNN seeds, grouped into ten ensembles. The original CNN architecture, saved data partition and executed validation-loss checkpoint recipe are preserved. Each ensemble trains its own detectors and computes its own novelty geometry and normalization. No test-based seed/ensemble selection is permitted.

Scope per ensemble: A–D LR and three-seed MLPs (D is needed for H3), output rankings/shared temperature, clean and all nine corruptions, all/unanimous/confident-unanimous subsets, paired ablation intervals and accepted-error counts, all detector seed sensitivity, permutation reliance diagnostic, latency, Mahalanobis H7, depth H8, confidence-bin and paired-transition analyses, and the separate noise-adaptation experiment. Individual CNNs get accuracy and output-ranking AUROC/AURC/selective risk; cross-member H1 is defined only for ensembles.

Storage is under `/common/users/ns1788/beyond-ensemble-agreement/replication_v1`, which has more quota than home. A frozen copy of source/config/splits is created before submission. Data is shared read-only; each ensemble has its own training/results directories. The member-slot directories `seed_1/2/3` contain weights trained with the unique actual seeds in identity.json/checkpoint config, allowing reuse of the original pipeline without changing its implementation.

One GPU per task, at most two ensembles concurrently. Each task trains its three CNNs sequentially, then completes the analysis stages. The requested wall-time ceiling is 24 hours per ensemble task, not a runtime estimate. These jobs join the existing unlimited iLab queue and do not cancel or alter other jobs. Failures remain visible in logs; interrupted stages are retained for diagnosis rather than silently replacing partially fitted CNNs. An after-any dependent CPU job aggregates all available ensembles and marks completion count, including partial runs.

```bash
/common/system/venv/python312/bin/python -m unittest discover -s tests -p test_replication.py -v
/common/system/venv/python312/bin/python -m src.replicate init
sbatch scripts/replicate.slurm
# Use the returned array job ID:
sbatch --dependency=afterany:ARRAY_JOB_ID scripts/replicate_aggregate.slurm
# Read-only provisional aggregation at any time:
/common/system/venv/python312/bin/python -m src.replicate aggregate
```

No plots/galleries are saved. Large train/test/early-layer raw representation caches are temporary; all dependent extensions/adaptation and integrity checks run before cleanup. Retained artifacts include model-only CNN checkpoints, detectors/scalers/geometry, compact per-member logits, original labels/IDs, compressed CSV feature tables and risk scores, metrics/contrasts, training histories, environment/source hashes and retention manifests. Gzip is lossless. Expect checkpoint storage alone to be roughly 1.3 GB across 30 float32 ResNet-18s, plus detectors, compact features/scores, and temporary caches; actual bytes are recorded per run. Only up to two ensembles generate intermediate caches concurrently. This is compact retention, not deletion of the evidence needed to redo tabular evaluation. Metadata hashes for retired raw caches remain historical provenance; future raw-feature computations require CNN re-inference. Existing original artifacts are untouched.

Outputs: each `ensemble_XX/results/` holds studies and compressed tables; root `all_ensemble_contrasts.csv` retains every reported comparison; `ensemble_summary.csv` summarizes clean B−A/C−B across the ten new ensembles (mean, sample std, median, negative-effect fraction, 10,000 ensemble-resampled mean-effect draws). Root status.csv and SUMMARY.md record availability. Do not interpret partial aggregates as final or exclude unfavorable ensembles. Per-ensemble image bootstrap intervals and across-ensemble seed intervals answer different questions; fixed shared images and one split limit generalization. No architecture-generalization or new-data confirmation claim follows from this run.

Submitted ensemble array: **1047647** (ten tasks, at most two concurrent). Scheduler CPU-count directives were omitted per site policy; computational libraries are limited to two threads. The source/configuration snapshot and split hash were registered before submission.

Dependent summary job: **1047648** (`afterany:1047647`). Live output link: `results/replication/`. Job IDs are also recorded in `submission.json`.
