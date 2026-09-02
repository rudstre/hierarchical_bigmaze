#!/usr/bin/env bash
#SBATCH --job-name=adjacent-mlmdp-refit
#SBATCH --array=0-70
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

task_cpus="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${task_cpus}"
export MKL_NUM_THREADS="${task_cpus}"
export OPENBLAS_NUM_THREADS="${task_cpus}"

project_root="${HIERARCHY_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
python_executable="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"
config_path="${HIERARCHY_ADJACENT_CONFIG:-${project_root}/configs/adjacent_mlmdp_regression.json}"
output_dir="${HIERARCHY_ADJACENT_OUTPUT:-${project_root}/output/adjacent_mlmdp_regression/production}"
run_identifier="${HIERARCHY_RUN_IDENTIFIER:-${SLURM_ARRAY_JOB_ID}}"
fold_index="${SLURM_ARRAY_TASK_ID}"

if [[ ! "${run_identifier}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'Invalid run identifier: %s\n' "${run_identifier}" >&2
  exit 2
fi

log_dir="${project_root}/slurm_out/adjacent_mlmdp/job_${run_identifier}"
mkdir -p "${log_dir}"
exec >"${log_dir}/refit-${SLURM_ARRAY_JOB_ID}_${fold_index}.out" 2>&1

cd "${project_root}"
"${python_executable}" scripts/run_adjacent_mlmdp.py refit \
  --config "${config_path}" \
  --output-dir "${output_dir}" \
  --fold-index "${fold_index}"

