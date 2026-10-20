#!/usr/bin/env bash
# Submit a profile-recovery run as a SLURM array (one fit per task) plus a
# combine-and-plot job that runs only if every task succeeds.
#
#   scripts/slurm/submit_profile_recovery.sh RUN_DIR N_FITS "RUN ARGS" [WARM_FROM] [AFTER_JOB] [PER_TASK]
#
# N_FITS random restarts (seeds 0..N-1) or, with WARM_FROM, warm starts from
# the top N source restarts; task 0 is always the oracle. AFTER_JOB delays the
# array until that job succeeds. With PER_TASK=1, AFTER_JOB must be the source
# *array* id: task i then warm-starts from source task i alone and waits only
# for it (aftercorr), instead of ranking all source restarts.
# Prints "ARRAY_ID COMBINE_ID" for chaining.
set -euo pipefail
run_dir="$1"; n_fits="$2"; run_args="$3"; warm_from="${4:-}"; after="${5:-}"; per_task="${6:-0}"
PY="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"
mkdir -p "${run_dir}/parts"
# Passed through the environment: sbatch --export splits values on commas.
export RECOVERY_RUN_DIR="${run_dir}" RECOVERY_ARGS="${run_args}" RECOVERY_WARM_FROM="${warm_from}"
export RECOVERY_WARM_PER_TASK="${per_task}"
# sbatch can fail transiently when the controller is overloaded ("Slurm
# temporarily unable to accept job"). Retry a few times, then fail loudly so a
# stage is never silently missing from a chain.
submit() {
  local attempt output
  for attempt in 1 2 3 4 5; do
    if output=$(sbatch --parsable "$@"); then
      echo "${output}"
      return 0
    fi
    echo "sbatch attempt ${attempt} failed; retrying in 30s" >&2
    sleep 30
  done
  echo "ERROR: sbatch failed 5 times for ${run_dir}" >&2
  return 1
}
dependency=()
if [[ -n "${after}" && "${per_task}" == 1 ]]; then
  dependency=(--dependency="aftercorr:${after}")
elif [[ -n "${after}" ]]; then
  dependency=(--dependency="afterok:${after}")
fi
array_job=$(submit "${dependency[@]}" --array="0-${n_fits}" \
  --output="${run_dir}/parts/slurm-%A_%a.log" \
  --export=ALL scripts/slurm/profile_recovery_task.sbatch)
combine_job=$(submit --dependency="afterok:${array_job}" \
  --job-name=recovery-combine --partition=cpu,cpu_backfill --exclude=enc1-node3 \
  --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output="${run_dir}/slurm-combine-%j.log" \
  --wrap "export PYTHONDONTWRITEBYTECODE=1; ${PY} scripts/combine_recovery_parts.py ${run_dir} --expected-parts $((n_fits + 1)) && ${PY} scripts/plot_profile_recovery.py ${run_dir}")
echo "${array_job} ${combine_job}"
echo "array ${array_job}, combine ${combine_job}" >&2
