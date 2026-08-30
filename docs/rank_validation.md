# Hierarchical rank validation sweep

The production workflow compares every integer NMF rank from 2 through a
configurable inclusive maximum (49 by default). Its default validation mode is
leave-one-session-out (LOSO): each chronologically ordered session is held out
once while ADAM fits the other sessions.

For the current six-session dataset, ranks 2–49 produce:

- 48 split-independent NMF discovery tasks; and
- 48 × 6 = 288 rank/fold fitting and scoring tasks.

NMF is fitted only once per rank because discovery uses the flat maze model and
does not depend on the behavioral training split. Every fold records its exact
training and validation sessions and trials.

The held-out metric within a fold remains:

```text
sum(trial log likelihood) / sum(trial movement-transition count)
```

Across LOSO folds, reports use the unweighted session mean. One standard error
is the sample standard deviation across sessions divided by the square root of
the number of sessions. A rank is eligible for ranking only when every expected
fold succeeds.

## Submit the two-stage SLURM sweep

From the repository root:

```bash
scripts/slurm/submit_hierarchy_rank_validation.sh --run-id production_loso
```

The wrapper first submits the NMF rank array, then submits the dependent
rank/fold array after every discovery task finishes. Both stages request 12 GB
per task, the `cpu` partition, and an eight-hour (`08:00:00`) time limit by
default. Submission-time options include:

```bash
scripts/slurm/submit_hierarchy_rank_validation.sh \
  --run-id production_loso \
  --max-rank 49 \
  --max-concurrent 48 \
  --mem 12G \
  --partition PARTITION \
  --time TIME \
  --account ACCOUNT
```

`--max-rank` is inclusive, accepts 2 through 49, and defaults to 49. It
reduces both arrays and the expected aggregation grid. `--max-concurrent`
keeps every requested task but adds SLURM's `%N` concurrency cap. If omitted,
all requested array elements are eligible to run concurrently.

The default configuration is
`configs/hierarchy_rank_validation_loso.json`, and the default result root is
`output/hierarchy_rank_validation/production_loso`. The established
`HIERARCHY_PROJECT_ROOT`, `HIERARCHY_PYTHON`,
`HIERARCHY_SWEEP_CONFIG`, and `HIERARCHY_SWEEP_OUTPUT` environment
overrides remain available.

## Run individual stages locally

Fit or reuse one NMF artifact:

```bash
python scripts/run_hierarchy_rank_discovery.py \
  --config configs/hierarchy_rank_validation_loso.json \
  --k 8 \
  --output-dir output/hierarchy_rank_validation/production_loso/discovery
```

Fit one rank/fold from that artifact:

```bash
python scripts/run_hierarchy_rank_validation.py \
  --config configs/hierarchy_rank_validation_loso.json \
  --k 8 \
  --fold-index 0 \
  --output-dir output/hierarchy_rank_validation/production_loso
```

Compatible artifacts and fold shards are reused. Use `--force` on the
relevant stage to atomically replace that stage's existing result.

## Preserve the chronological holdout option

`configs/hierarchy_rank_validation_production.json` retains the original
first-five-session training and last-session validation split through
`validation_mode: "chronological_holdout"`. It uses one fold per rank but the
same two-stage discovery/fitting machinery.

## Aggregate results

```bash
python scripts/aggregate_hierarchy_rank_validation.py \
  --config configs/hierarchy_rank_validation_loso.json \
  --shard-dir output/hierarchy_rank_validation/production_loso \
  --output-dir output/hierarchy_rank_validation/production_loso/aggregate \
  --max-rank 49
```

Aggregation preserves the expected rank/fold grid and explicitly reports
missing, failed, incompatible, and nonfinite results. Outputs include:

- `aggregate.json` with the complete provenance and ranking;
- `fold_summary.csv` with one row per expected rank/fold;
- `rank_summary.csv` with fold counts, eligibility, means, and standard
  errors;
- held-out and fitted-training likelihood versus rank with mean ±1 SE;
- all six fitted parameters versus rank with mean ±1 SE, including core
  threshold as a fraction of its fold-specific structural cap; and
- the selected once-per-rank NMF normalized generalized KL diagnostic.

Figures are written as PNG and SVG. Aggregation may run while jobs are still
active, but incomplete ranks remain ineligible for ranking. Existing schema-v2
result directories are not modified or mixed with the schema-v3 workflow.
