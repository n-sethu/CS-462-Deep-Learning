# Beyond Ensemble Agreement: first CNN

Run commands from the repository root. Use Rutgers Python 3.12:

```bash
source /common/system/venv/python312/bin/activate
python -m unittest discover -s tests -v
python -m src.train --prepare-data
```

The data preparation command downloads the CIFAR-10 archive and creates
`splits/cifar10_seed42.npz`. Torchvision downloads an archive that includes the
official test files, but this pipeline never opens the official test dataset.
Commit the small split file so every ensemble member uses exactly the same indices.
The saved splits are stratified: 35,000 CNN training, 10,000 detector training,
and 5,000 validation images. The split seed stays 42 for every member.

On a Rutgers GPU server, obtain a GPU allocation through your course's Slurm
partition/account, then run inside that allocation:

```bash
python -m src.train --config configs/ensemble.json --seed 1 --device cuda
```

Do not train on the login node. Allocation flags depend on your course's cluster
access. To inspect available partitions, use `sinfo`.
For a CPU run on an appropriate compute machine, use `--device cpu`.

After each epoch, `results/training/seed_1/` contains:

- `best.pt`: weights, optimizer/scheduler states, config and metrics at the lowest validation loss.
- `metrics.csv`: training/validation losses, accuracies and learning rate per epoch.
- `curves.png`: loss curves and validation accuracy.
- `best_metrics.json`: metrics for the selected checkpoint.
- `config.json` and `environment.json`: resolved settings and runtime versions.

Existing run directories containing a config are protected against overwrite.
Training does not currently support resuming interrupted runs. Use a different
`output_dir` in a copied config for a rerun; do not change the member seed just
to bypass this protection.

The configuration uses SGD with momentum and cosine LR decay over 100 epochs.
Images use fixed CIFAR-10 normalization; only CNN training uses crop/flip
augmentation. Models are initialized from scratch, with a 3x3 stride-1 stem,
no initial max pooling, and ten output logits. `model(x, return_features=True)`
returns `(logits, features)` with feature shape `[batch, 512]`.

Checkpoint selection uses **lowest validation loss**, following the latest task
instructions. This supersedes PROJECT.md's older best-validation-accuracy rule.
Validation accuracy is still logged. Freeze this rule before comparing members.
After the first full run works, repeat with `--seed 2` and `--seed 3`; keep all
other settings and the saved splits identical.

## Concepts to explain

- Cross-entropy fits a categorical distribution over ten mutually exclusive
  classes. Pass raw logits and integer labels; the loss handles log-softmax.
- Validation examples are excluded from gradient updates so they measure
  performance on held-out images and guide checkpoint selection. The test set
  remains reserved for final evaluation.
- Member seeds change initialization, minibatch order and augmentation draws;
  fixed split indices ensure members are compared on the same examples.
- The penultimate representation is the learned 512-dimensional feature vector
  before classification. Later, compute class centroids within each network's
  own feature space.

A Slurm script is included for the currently available `unlimited` partition:

```bash
sbatch scripts/train_first.slurm 1
squeue -u "$USER"
```

Submit from the repository root after data preparation completes. The job requests
one GPU, four CPU cores, 16 GB RAM and a four-hour limit. Adjust the time request
before submission if needed: runtime depends on the assigned GPU, and interrupted
runs cannot yet resume. Logs are written to `cifar10-JOBID.log`. After verifying
the first full run, submit seeds 2 and 3 using the same script argument.
