#!/usr/bin/env bash
# Submit once-per-rank NMF and dependent rank-fold validation arrays.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${HIERARCHY_PROJECT_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
python_executable="${HIERARCHY_PYTHON:-/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python}"
config_path="${HIERARCHY_SWEEP_CONFIG:-${project_root}/configs/hierarchy_rank_validation_loso.json}"
output_dir="${HIERARCHY_SWEEP_OUTPUT:-${project_root}/output/hierarchy_rank_validation/production_loso}"
run_identifier="loso"
max_rank=49
max_concurrent=""
memory="12G"
scheduler_arguments=()

usage() {
  printf '%s\n' \
    "usage: $0 [--run-id ID] [--max-rank K] [--max-concurrent N]" \
    "          [--config PATH] [--output-dir PATH] [--mem MEMORY]" \
    "          [--partition NAME] [--time LIMIT] [--account NAME]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      run_identifier="$2"
      shift 2
      ;;
    --max-rank)
      max_rank="$2"
      shift 2
      ;;
    --max-concurrent)
      max_concurrent="$2"
      shift 2
      ;;
    --config)
      config_path="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --mem)
      memory="$2"
      shift 2
      ;;
    --partition|--time|--account)
      scheduler_arguments+=("$1=$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${run_identifier}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'invalid run identifier: %s\n' "${run_identifier}" >&2
  exit 2
fi
if [[ ! "${max_rank}" =~ ^[0-9]+$ ]] || (( max_rank < 2 || max_rank > 49 )); then
  printf 'max rank must be an integer in 2..49\n' >&2
  exit 2
fi
if [[ -n "${max_concurrent}" ]] && {
  [[ ! "${max_concurrent}" =~ ^[0-9]+$ ]] || (( max_concurrent < 1 ))
}; then
  printf 'max concurrent must be a positive integer\n' >&2
  exit 2
fi

cd "${project_root}"
# Production configs record the audited sessions, so submission can determine
# the array size without importing the model stack or reading behavioral TSVs.
if ! fold_count="$("${python_executable}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    dataset = json.load(config_file)["dataset"]
mode = dataset.get("validation_mode", "chronological_holdout")
expected_counts = dataset.get("expected_session_trial_counts", {})
if mode == "chronological_holdout":
    print(1)
elif expected_counts:
    print(len(expected_counts))
else:
    raise SystemExit(1)
' "${config_path}")"; then
  fold_count="$("${python_executable}" scripts/run_hierarchy_rank_validation.py \
    --config "${config_path}" \
    --max-rank "${max_rank}" \
    --print-fold-count)"
fi
if [[ ! "${fold_count}" =~ ^[0-9]+$ ]] || (( fold_count < 1 )); then
  printf 'worker returned invalid fold count: %s\n' "${fold_count}" >&2
  exit 2
fi

rank_count=$((max_rank - 1))
discovery_array="2-${max_rank}"
validation_array="2-${max_rank}"
if [[ -n "${max_concurrent}" ]]; then
  if (( max_concurrent < fold_count )); then
    printf 'max concurrent must be at least the fold count (%s)\n' \
      "${fold_count}" >&2
    exit 2
  fi
  discovery_limit="${max_concurrent}"
  if (( discovery_limit > rank_count )); then
    discovery_limit="${rank_count}"
  fi
  per_fold_limit=$((max_concurrent / fold_count))
  if (( per_fold_limit > rank_count )); then
    per_fold_limit="${rank_count}"
  fi
  discovery_array+="%${discovery_limit}"
  validation_array+="%${per_fold_limit}"
fi

export_values="ALL,HIERARCHY_PROJECT_ROOT=${project_root},HIERARCHY_PYTHON=${python_executable},HIERARCHY_SWEEP_CONFIG=${config_path},HIERARCHY_SWEEP_OUTPUT=${output_dir},HIERARCHY_DISCOVERY_OUTPUT=${output_dir}/discovery,HIERARCHY_RUN_IDENTIFIER=${run_identifier},HIERARCHY_MAX_RANK=${max_rank}"

discovery_submission="$(sbatch --parsable \
  "${scheduler_arguments[@]}" \
  --mem="${memory}" \
  --array="${discovery_array}" \
  --export="${export_values}" \
  scripts/slurm/hierarchy_rank_discovery.sbatch)"
discovery_job_id="${discovery_submission%%;*}"

printf 'discovery_job=%s array=%s\n' "${discovery_submission}" "${discovery_array}"
for ((fold_index = 0; fold_index < fold_count; fold_index++)); do
  fold_export_values="${export_values},HIERARCHY_FOLD_INDEX=${fold_index}"
  validation_submission="$(sbatch --parsable \
    "${scheduler_arguments[@]}" \
    --mem="${memory}" \
    --array="${validation_array}" \
    --dependency="aftercorr:${discovery_job_id}" \
    --kill-on-invalid-dep=yes \
    --export="${fold_export_values}" \
    scripts/slurm/hierarchy_rank_validation.sbatch)"
  printf 'validation_fold=%s job=%s array=%s dependency=aftercorr:%s\n' \
    "${fold_index}" "${validation_submission}" "${validation_array}" \
    "${discovery_job_id}"
done
