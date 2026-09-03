#!/usr/bin/env bash
# Submit or safely resume the adjacent-MLMDP-regression SLURM workflow.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${HIERARCHY_PROJECT_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
python_executable="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"

cd "${project_root}"
exec "${python_executable}" \
  scripts/slurm/manage_adjacent_mlmdp.py \
  --project-root "${project_root}" \
  "$@"
