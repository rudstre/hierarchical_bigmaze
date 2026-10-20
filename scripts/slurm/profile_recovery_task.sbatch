#!/usr/bin/env bash
#SBATCH --job-name=recovery-task
#SBATCH --partition=cpu,cpu_backfill
# cpu_backfill nodes can preempt (requeue) tasks; each task is seeded and
# writes its outputs atomically at the end, so a requeue only costs time.
#SBATCH --exclude=enc1-node3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
# One fit per array task: task 0 is the oracle; task i >= 1 is restart seed
# i-1, or (with RECOVERY_WARM_FROM) the warm start from source rank i-1.
# Expects RECOVERY_RUN_DIR and RECOVERY_ARGS in the environment.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONDONTWRITEBYTECODE=1
PY="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"
task="${SLURM_ARRAY_TASK_ID}"
part="${RECOVERY_RUN_DIR}/parts/task_$(printf '%03d' "${task}")"
if [[ "${task}" == 0 ]]; then
  selection=(--restart-seeds)
elif [[ -n "${RECOVERY_WARM_FROM:-}" && "${RECOVERY_WARM_PER_TASK:-0}" == 1 ]]; then
  # Warm start from the source run's matching task only (aftercorr dependency).
  source_part="${RECOVERY_WARM_FROM}/parts/task_$(printf '%03d' "${task}")"
  selection=(--skip-oracle --restart-seeds --warm-start-from "${source_part}" --warm-start-rank 0)
elif [[ -n "${RECOVERY_WARM_FROM:-}" ]]; then
  selection=(--skip-oracle --restart-seeds --warm-start-from "${RECOVERY_WARM_FROM}" --warm-start-rank "$((task - 1))")
else
  selection=(--skip-oracle --restart-seeds "$((task - 1))")
fi
# shellcheck disable=SC2086
"${PY}" -u scripts/run_profile_recovery.py ${RECOVERY_ARGS} "${selection[@]}" --output-dir "${part}"
