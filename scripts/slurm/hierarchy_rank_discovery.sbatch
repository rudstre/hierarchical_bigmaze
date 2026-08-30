#!/usr/bin/env bash
#SBATCH --job-name=hierarchy-nmf
#SBATCH --array=2-49
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
config_path="${HIERARCHY_SWEEP_CONFIG:-${project_root}/configs/hierarchy_rank_validation_loso.json}"
output_dir="${HIERARCHY_SWEEP_OUTPUT:-${project_root}/output/hierarchy_rank_validation/production_loso}"
discovery_dir="${HIERARCHY_DISCOVERY_OUTPUT:-${output_dir}/discovery}"
max_rank="${HIERARCHY_MAX_RANK:-49}"
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
array_task_id="${SLURM_ARRAY_TASK_ID}"
requested_run_identifier="${HIERARCHY_RUN_IDENTIFIER:-${1:-${array_job_id}}}"
identifier_error=""

if [[ "${requested_run_identifier}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  run_identifier="${requested_run_identifier}"
else
  run_identifier="${array_job_id}"
  identifier_error="Invalid run identifier '${requested_run_identifier}'; use only letters, numbers, periods, underscores, and hyphens."
fi

slurm_log_dir="${project_root}/slurm_out/rank_val/job_${run_identifier}"
slurm_log_path="${slurm_log_dir}/nmf-${array_job_id}_${array_task_id}.out"

mkdir -p "${slurm_log_dir}"
exec >"${slurm_log_path}" 2>&1

if [[ -n "${identifier_error}" ]]; then
  printf '%s\n' "${identifier_error}" >&2
  exit 2
fi

cd "${project_root}"
"${python_executable}" scripts/run_hierarchy_rank_discovery.py \
  --config "${config_path}" \
  --k "${array_task_id}" \
  --max-rank "${max_rank}" \
  --output-dir "${discovery_dir}"
