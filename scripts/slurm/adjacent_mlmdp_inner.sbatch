#!/usr/bin/env bash
#SBATCH --job-name=adjacent-mlmdp
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
task_mode="${HIERARCHY_ADJACENT_TASK_MODE:-production}"
rank_min="${HIERARCHY_ADJACENT_RANK_MIN:-2}"
rank_max="${HIERARCHY_ADJACENT_RANK_MAX:-49}"
task_list="${HIERARCHY_ADJACENT_TASK_LIST:-}"
run_identifier="${HIERARCHY_RUN_IDENTIFIER:-${SLURM_ARRAY_JOB_ID}}"
task_id="${SLURM_ARRAY_TASK_ID}"

if [[ ! "${run_identifier}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'Invalid run identifier: %s\n' "${run_identifier}" >&2
  exit 2
fi

log_dir="${project_root}/slurm_out/adjacent_mlmdp/job_${run_identifier}"
mkdir -p "${log_dir}"
exec >"${log_dir}/slurm-${SLURM_ARRAY_JOB_ID}_${task_id}.out" 2>&1

cd "${project_root}"

if [[ -n "${task_list}" ]]; then
  # Managed workflow: SLURM_ARRAY_TASK_ID is a local 0..N-1 index into an
  # immutable task-list file written by scripts/slurm/manage_adjacent_mlmdp.py.
  "${python_executable}" scripts/run_adjacent_mlmdp.py inner \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    --task-list "${task_list}" \
    --task-index "${task_id}"
else
  pilot=()
  if [[ "${task_mode}" == "pilot" ]]; then
    pilot=(--pilot)
  fi
  "${python_executable}" scripts/run_adjacent_mlmdp.py inner \
    --config "${config_path}" \
    --output-dir "${output_dir}" \
    --task-index "${task_id}" \
    --rank-min "${rank_min}" \
    --rank-max "${rank_max}" \
    "${pilot[@]}"
fi

