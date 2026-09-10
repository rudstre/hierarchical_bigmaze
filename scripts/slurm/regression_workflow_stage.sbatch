#!/usr/bin/env bash
#SBATCH --job-name=heldout-regression
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

project_root="${REGRESSION_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
python_executable="${REGRESSION_PYTHON:-python}"
config_path="${REGRESSION_CONFIG:?REGRESSION_CONFIG is required}"
output_dir="${REGRESSION_OUTPUT:?REGRESSION_OUTPUT is required}"
stage="${REGRESSION_STAGE:?REGRESSION_STAGE is required}"
task_list="${REGRESSION_TASK_LIST:?REGRESSION_TASK_LIST is required}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"

task_cpus="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${task_cpus}"
export MKL_NUM_THREADS="${task_cpus}"
export OPENBLAS_NUM_THREADS="${task_cpus}"

cd "${project_root}"
"${python_executable}" scripts/run_regression_workflow_task.py \
  --config "${config_path}" \
  --output-dir "${output_dir}" \
  --stage "${stage}" \
  --task-list "${task_list}" \
  --task-id "${task_id}"

