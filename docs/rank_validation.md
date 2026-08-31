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

The wrapper first submits the NMF rank array, then one rank-indexed array for
each validation fold. Each fold array uses SLURM's `aftercorr` dependency, so
the fits for rank `k` become eligible as soon as NMF task `k` succeeds; they do
not wait for slower higher-rank NMF tasks. Both stages request 12 GB per task,
the `cpu` partition, and an eight-hour (`08:00:00`) time limit by default.
Submission-time options include:

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
keeps every requested task and bounds aggregate fit concurrency by dividing
the cap evenly across the fold arrays; it must therefore be at least the
number of folds (six for production LOSO). If omitted, all requested array
elements are eligible to run concurrently. If an NMF task fails, its
corresponding fit tasks are cancelled by SLURM and remain visibly missing from
aggregation alongside the failed discovery artifact.

The default configuration is
`configs/hierarchy_rank_validation_loso.json`, and the default result root is
`output/hierarchy_rank_validation/production_loso`. The established
`HIERARCHY_PROJECT_ROOT`, `HIERARCHY_PYTHON`,
`HIERARCHY_SWEEP_CONFIG`, and `HIERARCHY_SWEEP_OUTPUT` environment
overrides remain available.

Each new submission writes an atomic manifest under
`OUTPUT_DIR/slurm_runs/RUN_ID.json`. If node launch failures leave array
elements in `JobHeldAdmin`, preview recovery with:

```bash
./scripts/slurm/submit_hierarchy_rank_validation.sh \
  --run-id loso_test1 \
  --retry-missing --cancel-held --dry-run
```

Then perform the recovery without `--dry-run`. Retry inherits the original
resources, cancels only administrator-held elements owned by that manifest,
and leaves ordinary pending, configuring, and running elements alone:

```bash
./scripts/slurm/submit_hierarchy_rank_validation.sh \
  --run-id loso_test1 \
  --retry-missing --cancel-held
```

Successful compatible shards are never resubmitted. Missing work is grouped
into replacement fold arrays. Missing discovery ranks are resubmitted first
and linked to their replacement fits with `aftercorr`. Failed or incompatible
shards are reported and never overwritten. Runs created before manifest
support still require manual recovery.

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
  --output-dir output/hierarchy_rank_validation/production_loso/aggregate
```

When the shard directory contains exactly one matching SLURM submission
manifest, aggregation uses its submitted maximum rank. Pass `--max-rank K` to
override that value; without a matching manifest, the production default is 49.
When run from an interactive terminal, the command also opens the live Plotly
figures in the default browser as soon as aggregation finishes. Use
`--no-show-plots` to suppress this, or `--show-plots` to request it explicitly
when output is redirected.

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

`best_k` uses the one-standard-error rule: find the eligible rank with the
largest mean held-out log likelihood, subtract that rank's held-out standard
error, and select the smallest eligible rank whose mean is at least that
threshold. `aggregate.json` also records the best-mean rank, its standard
error, and the resulting threshold. For one-fold chronological validation,
where a cross-fold standard error is unavailable, this reduces to selecting
the maximum held-out mean.

Figures are written as PNG and SVG. Aggregation may run while jobs are still
active, but incomplete ranks remain ineligible for ranking. Existing schema-v2
result directories are not modified or mixed with the schema-v3 workflow.
