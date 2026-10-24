#!/usr/bin/env bash
#SBATCH --job-name=route-data-curve
#SBATCH --partition=cpu_backfill
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00

# One condition of scripts/run_route_data_curve.py per array task (6 routes,
# 2000 epochs). 2 CPUs: the fit scales poorly with threads (results/thread_bench:
# 1027 / 665 / 583 / 437 s at 1 / 2 / 4 / 8 threads); measured peak RSS 3.8 GB.
# Submit from the project root:
#   sbatch --array=0-467%96 --output=RUN_DIR/logs/%A_%a.out \
#     --export=ALL,RUN_DIR=RUN_DIR scripts/slurm/route_data_curve.sbatch

set -euo pipefail

run_dir="${RUN_DIR:?RUN_DIR is required}"
python_executable="${ROUTE_CURVE_PYTHON:-python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_VISIBLE_DEVICES=""

cd "${SLURM_SUBMIT_DIR}"
"${python_executable}" scripts/run_route_data_curve.py run \
  --run-dir "${run_dir}" --index "${SLURM_ARRAY_TASK_ID}"
