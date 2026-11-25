# Beyond Ensemble Agreement — results explorer

**Interactive exploration of precomputed experiment results.** A CPU-only, single-page Streamlit demo comparing output-only risk scoring (A) with mean representation novelty (B). No database, CNN inference, GPU, training, or runtime bootstrap is required.

## Run locally

Use Python 3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run checks with `python -m unittest test_demo -v`. The `assets/` directory is self-contained; there are no links back to cluster storage. On a remote machine, forward port 8501 to your presentation computer. Keep a local copy and test it before presenting.

## 30-second presentation

1. **0–7 seconds · Clean, clean-trained, 90%.** “These are cached results from a fixed ensemble. Adding mean novelty catches seven more errors at the same rejection budget.” Ensemble 01 is the disclosed default, selected by ID, not outcome.
2. **7–14 seconds · Gaussian noise, severity 3, clean-trained.** “That clean-trained novelty advantage does not transfer reliably to Gaussian noise.” Read the current counts rather than claiming every metric agrees.
3. **14–21 seconds · Noise-adapted.** “Training the detector on noise can recover an incremental novelty benefit; the CNNs and novelty reference stay fixed.”
4. **21–27 seconds · Motion blur, severity 3, noise-adapted.** “Adaptation specializes the detector and can sacrifice transfer elsewhere.”
5. **27–30 seconds · Replication evidence.** “Severe-noise recovery repeats across ten ensembles; the original clean variation hypothesis is unsupported.”

The silent [backup recording](presentation/backup.mp4) captures this sequence. These are the ensemble 01 values at 90% coverage.

| Condition | Training | A errors | B errors | A AURC | B AURC |
|---|---|---:|---:|---:|---:|
| clean | clean_trained | 140 | 133 | 0.005544 | 0.005022 |
| gaussian_noise_3 | clean_trained | 5235 | 5229 | 0.422754 | 0.427671 |
| gaussian_noise_3 | adapted | 5216 | 5188 | 0.419832 | 0.418096 |
| motion_blur_3 | adapted | 1837 | 1886 | 0.085318 | 0.084664 |

At noise severity 3, clean-trained B catches six more errors at this coverage and has higher integrated AURC. Use both measurements in the walkthrough.

## Deploy to Streamlit Community Cloud

Select the repository and the branch containing the demo. Set the entrypoint to `Final_Project/demo/streamlit_app.py` and use Python 3.12. The requirements beside the entrypoint cover the app. For a standalone deployment of the demo directory, use `streamlit_app.py` as the entrypoint and include `assets/`, the calculations module, `requirements.txt`, and `.streamlit/config.toml`. The app requires no secrets.

## Data and interpretation

- All ten additional ensembles, all ten conditions, clean-trained and noise-adapted validation-selected MLP A/B scores are exported. Original discovery-ensemble results are excluded from the aggregate evidence.
- Scores and confidence retain float64 precision. IDs are original CIFAR-10 test IDs. Sorting uses increasing risk then ascending source ID. Coverage keeps exactly `ceil(coverage*N)` predictions. Every 80%/90% accepted-error count, risk, AURC and AUROC was verified offline against saved evaluation tables (800 operating-point checks).
- Full scores support exact interactive counts. Exported curve points are subsampled; the selected operating point is inserted exactly. AURC is calculated from all prefixes, not subsampled points.
- The gallery chooses the smallest source ID in each category. Both success and regression categories are shown when available; absence is disclosed. Only thumbnails reachable through the integer 50–100% coverage controls are shipped. CIFAR-10/CIFAR-10-C thumbnails are 32px, enlarged, and WebP-compressed.
- Aggregate pointwise 95% intervals were computed offline by resampling ten ensembles, conditional on shared images/splits. They do not include new-image uncertainty. Student-t sensitivity intervals are included; extensions are exploratory and comparisons are unadjusted.
- Severity 3/5 noise adaptation recovers MLP B−A benefits in 10/10 ensembles. Its erosion of other benefits is more reproducible than a claim of universally harmful adapted B−A. Clean C−B variation is unsupported; its near-zero interval is sensitive to interval method.
- Individual-ensemble example choices never alter the ten-ensemble aggregate. The coverage slider demonstrates matched rejection budgets, not a newly validated deployment threshold.

## Refresh assets offline

From the full research checkout:

```bash
python scripts/export_demo.py --root results/replication --out demo/assets --data data
```

The exporter requires the cluster replication runs, local CIFAR-10/CIFAR-10-C images, and precomputed replication intervals. It writes `manifest.json` with source/output SHA-256 hashes, schema details and thumbnail indices, and `verification.csv` with count checks. It does not modify research artifacts. Its export dependencies are the research environment’s NumPy, pandas, Pillow and scikit-learn; they are not runtime app dependencies.
