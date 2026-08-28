#!/usr/bin/env bash
#SBATCH --job-name=hierarchy-k
#SBATCH --array=2-49
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

project_root="${HIERARCHY_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
array_task_id="${SLURM_ARRAY_TASK_ID}"
slurm_log_dir="${project_root}/slurm_out/rank_val/job_${array_job_id}"
slurm_log_path="${slurm_log_dir}/slurm-${array_job_id}_${array_task_id}.out"

mkdir -p "${slurm_log_dir}"
exec >"${slurm_log_path}" 2>&1

cd "${project_root}"
exec bash scripts/slurm/hierarchy_rank_validation.sbatch
