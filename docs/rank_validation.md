# Hierarchical rank validation sweep

The production sweep fits one hierarchical model for every integer NMF rank
from 2 through 49. Each worker trains ADAM on the first five selected sessions
and scores the fitted model on the sixth session. The primary validation metric
is pooled over observed movement transitions:

```text
sum(trial log likelihood) / sum(trial movement-transition count)
```

This is not the unweighted mean of per-trial normalized likelihoods. Every
validation trial must receive a finite score for a rank to enter the ranking.

## Run one rank locally

From the repository root:

```bash
python scripts/run_hierarchy_rank_validation.py \
  --config configs/hierarchy_rank_validation_production.json \
  --k 8 \
  --output-dir output/hierarchy_rank_validation/production_normalized_threshold_040
```

The production configuration uses connected KL-NMF seeds 0 through 49 for
every rank and one ADAM initialization. The configured threshold fraction is
`0.4`; each worker resolves the physical initial `core_threshold` as `0.4`
times that rank's structural cap. A compatible existing shard is reused; pass
`--force` to recompute and atomically replace it.

## Submit the SLURM array

Resource and account choices stay at submission time:

```bash
sbatch --partition=PARTITION --time=TIME --mem=MEMORY \
  scripts/slurm/hierarchy_rank_validation.sbatch RUN_IDENTIFIER
```

`RUN_IDENTIFIER` is an optional name containing letters, numbers, periods,
underscores, or hyphens. For example, passing `threshold_040` writes every
array task's combined standard output and error under
`slurm_out/rank_val/job_threshold_040/`. Log filenames retain the numeric
array job ID and rank, for example `slurm-3416000_24.out`, so reusing a run
identifier cannot overwrite logs from another job. If `RUN_IDENTIFIER` is
omitted, the directory falls back to `job_<array-job-id>`.

The wrapper maps `SLURM_ARRAY_TASK_ID` directly to `k` and creates the shared
log directory safely when the array starts.

These optional environment variables customize paths without editing the
script:

- `HIERARCHY_PROJECT_ROOT`
- `HIERARCHY_PYTHON`
- `HIERARCHY_SWEEP_CONFIG`
- `HIERARCHY_SWEEP_OUTPUT`

## Aggregate completed shards

Aggregation can be run before the array finishes. Missing and failed ranks stay
visible and the current winner is marked provisional:

```bash
python scripts/aggregate_hierarchy_rank_validation.py \
  --config configs/hierarchy_rank_validation_production.json \
  --shard-dir output/hierarchy_rank_validation/production_normalized_threshold_040 \
  --output-dir output/hierarchy_rank_validation/production_normalized_threshold_040/aggregate
```

`aggregate.json` contains the complete compatible shards. `rank_summary.csv`
contains the rank scores, NMF selection diagnostics, ADAM convergence fields,
best parameter values, parameter changes from initialization, and the fitted
threshold as a fraction of its structural cap. The CSV includes initial,
best, and last thresholds in both physical and cap-normalized units. The
aggregator writes three figures in both PNG and SVG formats:

- `held_out_log_likelihood_vs_k` compares the pooled held-out and fitted
  training log likelihoods per movement transition. Missing and failed ranks
  remain visible as gaps, and the best available held-out rank is marked.
- `selected_nmf_normalized_kl_vs_k` shows the selected basis's normalized
  generalized KL value. It retains ranks whose NMF discovery succeeded even
  when a later fitting or scoring stage failed.
- `fitted_parameters_vs_k` shows the six best fitted parameters in a 3-by-2
  panel, with `core_threshold` reported as a fraction of its structural cap.

Every plotted value is also present in `rank_summary.csv`.

Every shard records exact configuration, dataset, maze, dependency, Git HEAD,
and worker/model working-tree source fingerprints. Aggregation and plotting
code has a separate fingerprint, so presentation-only changes do not invalidate
existing shards. Aggregation still rejects mixed worker fingerprints and any
configuration, data, maze, or runtime mismatch.


## Threshold-normalized rerun

This schema-v2 sweep is incompatible with the original physical-threshold
shards. Keep the original `production` directory and the prior `0.8`-fraction
`production_normalized_threshold` directory unchanged. The current `0.4`-
fraction sweep writes to `production_normalized_threshold_040`. Do not use
`--force` to mix configurations, schemas, or worker fingerprints between these
directories.
