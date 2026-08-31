#!/usr/bin/env bash
# Submit or safely retry hierarchy-rank validation arrays.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${HIERARCHY_PROJECT_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
python_executable="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"

cd "${project_root}"
exec "${python_executable}" \
  scripts/slurm/manage_hierarchy_rank_validation.py \
  --project-root "${project_root}" \
  "$@"
