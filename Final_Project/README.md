# Beyond Ensemble Agreement

CS 462 final project on detecting prediction errors with internal representations from a three-member CIFAR-10 ensemble. The primary question is whether variation in representation novelty adds value beyond mean novelty and output uncertainty.

The original ensemble and ten additional ensembles have completed evaluations. Across the ten replications, adding mean novelty improved clean MLP AURC in eight ensembles. Adding novelty variation improved it in two, and the mean difference favored the detector without variation. Gaussian-noise adaptation recovered an incremental mean-novelty benefit for the MLP at severities 3 and 5 in all ten ensembles.

## Results and demo

| Start here | Contents |
| --- | --- |
| [Paper summary](PAPER_SUMMARY.md) | Methods, original ensemble results, adaptation, ten-ensemble findings, and limitations |
| [Replication tables and figures](results/replication_analysis/) | All ten ensembles, aggregate intervals, agreement subsets, corruption transfer, geometry/depth, and cost |
| [Original adaptation analysis](results/adaptation/analysis/ANALYSIS.md) | Paired intervals, exact-coverage counts, and frozen-threshold behavior |
| [Streamlit demo](demo/README.md) | Local setup, cached examples, replication evidence, and backup recording |
| [Research protocol](AGENTS.md) | Splits, leakage controls, hypotheses, and scope |

The checkout includes validation tables, original adaptation and exploratory outputs, replication analysis CSVs and PNGs, and the demo assets. Full checkpoints, raw inference caches, and the original final evaluation are on cluster storage. The tracked `results/extensions_h7_h8` and `results/replication` symlinks point to iLab paths and can be unresolved on other machines. Use the tracked analysis tables to read the results locally.

## Reproduce the research

Run commands from `Final_Project` with the dependencies in [requirements.txt](requirements.txt). Use a separate checkout or work directory for a fresh run. The original [ensemble config](configs/ensemble.json) points to iLab training storage. On another machine, use a copy of that config with a local `output_dir` and pass the same copy to training and validation. Keep the saved [split indices](splits/cifar10_seed42.npz). The documented checkpoint rule for reproducing the executed runs is minimum validation loss.

```bash
python -m src.train --config configs/ensemble.json --seed 1 --device cuda
python -m src.train --config configs/ensemble.json --seed 2 --device cuda
python -m src.train --config configs/ensemble.json --seed 3 --device cuda
python -m src.validation --config configs/ensemble.json --device cuda
python scripts/prepare_corruptions.py --archive data/CIFAR-10-C.tar.partial
python -m src.final_evaluation --stage all --device cuda
```

The corruption preparation step takes the downloaded official CIFAR-10-C archive at the specified path, verifies it, and extracts labels, Gaussian noise, motion blur, and JPEG compression. It renames the archive to remove its final suffix after extraction. Final evaluation uses the fitted validation artifacts and produces the original `results/final/`. The [adaptation runner](src/adaptation.py) requires the original validation and final artifacts. Run it with `python -m src.adaptation --stage all --device cuda` in the reproduction workspace.

### Ten-ensemble replication

[Replication configuration](configs/replication.json) fixes 30 new CNN seeds grouped into ten ensembles. Each worker runs baseline evaluation, geometry/depth, exploratory diagnostics, adaptation, and per-member analysis before compacting its artifacts. The `seed_1`, `seed_2`, and `seed_3` directories identify member slots. Actual seeds are recorded in each ensemble’s `identity.json`. The original Slurm array was job 1047647 and the dependent aggregate job was 1047648.

To initialize a fresh replication, use a new output root and a research checkout with CIFAR-10 and CIFAR-10-C prepared. Schedule workers for indices 1–10 with one GPU each and at most two concurrent workers, matching the original replication. The original cluster submit wrappers are external to this checkout.

```bash
python -m src.replicate init --root /path/to/new-replication
# In the GPU array, once per task:
python -m src.replicate worker --root /path/to/new-replication --index "$SLURM_ARRAY_TASK_ID"
# After the workers finish:
python -m src.replicate aggregate --root /path/to/new-replication
python -m src.analyze_replication --root /path/to/new-replication --out /path/to/new-analysis
```

The aggregator writes completion status and provisional summaries if only some workers have finished. The separate analysis requires all ten completed runs. It checks clean metrics against score caches and reads the native per-ensemble contrast tables directly, including mixed-schema contrasts omitted from older aggregates. It writes the cross-ensemble tables, figures, and source hashes without refitting detectors.

Replications retain model-only CNN checkpoints, detectors, scalers, geometry, compact logits, labels and IDs, compressed features and scores, and provenance. Large raw representation caches are retired after dependent stages and integrity checks complete. Rebuilding raw features requires inference from the retained checkpoints.

## Checks

```bash
python -m unittest discover -s tests -v
```
