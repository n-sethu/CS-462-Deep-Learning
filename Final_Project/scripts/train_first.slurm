#!/bin/bash
#SBATCH --job-name=cifar10-seed1
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cifar10-%j.log
#SBATCH --error=cifar10-%j.log

set -euo pipefail
# Capture errors even if a site submission wrapper changes Slurm stderr settings.
exec 2>&1
trap 'status=$?; echo "Job exited with status ${status} at line ${LINENO}"; exit "$status"' ERR
cd "${SLURM_SUBMIT_DIR:?Submit from the Final_Project repository root}"
source /common/system/venv/python312/bin/activate
python -u -X faulthandler -m src.train --config configs/ensemble.json --seed "${1:-1}" --device cuda
