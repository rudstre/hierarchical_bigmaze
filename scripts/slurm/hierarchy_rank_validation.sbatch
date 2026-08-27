#!/usr/bin/env bash
#SBATCH --job-name=hierarchy-k
#SBATCH --array=2-49
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

project_root="${HIERARCHY_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
python_executable="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"
config_path="${HIERARCHY_SWEEP_CONFIG:-${project_root}/configs/hierarchy_rank_validation_production.json}"
output_dir="${HIERARCHY_SWEEP_OUTPUT:-${project_root}/output/hierarchy_rank_validation/production_normalized_threshold}"

cd "${project_root}"
"${python_executable}" scripts/run_hierarchy_rank_validation.py \
  --config "${config_path}" \
  --k "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${output_dir}"
