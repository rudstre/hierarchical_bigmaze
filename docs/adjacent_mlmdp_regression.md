# Adjacent-regression hierarchical MLMDP predictor

This workflow adds one four-action hierarchical MLMDP policy to Qin's
full/reduced regression. It is opt-in; the existing Figure 2.19 predictor set
and output filenames are unchanged.

## Statistical partitions

For adjacent subject sessions s_i and s_(i+1):

- the regression is trained on s_i and evaluated on s_(i+1);
- the hierarchical model and Qin route/habit predictors are trained on every
  other session for that subject, including later sessions where applicable;
- k is selected only inside those route-training sessions by leave-one-session-
  out validation;
- after selection, the chosen k is refit on every route-training session;
- predictions are then generated for s_i and s_(i+1), without updating the
  fitted model.

The scientific fold key is the exact tuple of maze, subject, regression-training
session IDs, validation session ID, and route-training session IDs. fold_index
is scheduling metadata and is never used to join predictions.

Each eligible rank must have one terminal successful fit for every inner
validation session. The score is held-out total log likelihood divided by
held-out movement transitions within each session. Rank means and standard
errors are unweighted across sessions. The selected rank is the smallest rank
within one standard error of the best mean.

A deterministic numerical or Adam convergence failure makes only that rank
ineligible and records the failed inner session. A missing or operationally
failed shard keeps the outer fold pending for retry. A fold is unavailable only
when no rank is eligible, the selected refit fails, or its prediction artifact
is absent/incompatible.

## Predictor semantics

Task.movement_predictions performs causal filtering over controller modes and
computes the predictive distribution over every candidate physical destination,
not only the observed departure. Every row is required to sum to one without
post-hoc renormalization, and the product of observed predictive probabilities
matches the trajectory likelihood.

The Qin column is the resulting L1-normalized action probability vector in
east, north, west, south order. Invalid commands retain exact zero before
regression preprocessing. Qin then applies its existing training-fold global
feature standardization; impossible actions are still enforced by its separate
-1e10 mask. Unlike the PCA, forward, and reverse predictors, the first movement
of a trial contains the model's real initial-controller prediction rather than
a zero vector.

Only distributed NMF bases are accepted. Repeated states are not expected in
canonical Doohan movement trials; prediction generation fails if likelihood
collapse would change the canonical decision count.

## Compute and pilot

The JSON configuration specifies an inclusive contiguous range with `rank_min`
and `rank_max`. Explicit `ranks` remains supported for non-contiguous research
or test grids, but a configuration must use exactly one of these two forms.
Changing the configured range requires preparing a new manifest.

The production data create 71 outer folds and 770 inner fits per rank:

- 48 ranks times 770 inner fits = 36,960 inner fits;
- 71 selected-rank refits.

The existing k=2..25 sweep measured 144 fits and 12.63 CPU-hours. Extrapolation
to the full nested grid is approximately 30,000 CPU-hours, so the manifest
records that budget and an initial concurrency cap of 200.

### Configuration

`configs/adjacent_mlmdp_regression.json` is self-contained for everything
that governs the regression itself: `dataset` (which subjects/sessions/dates
feed the actual fits), `adam` (the ADAM optimizer hyperparameters used for
every inner fit and refit), and `ranks`/`discovery_dir`/`slurm`.

It references exactly one other file, `discovery_config` — a
`RankValidationConfig`-shaped JSON such as `hierarchy_rank_validation_loso.json`
(the same shape the standalone rank-validation workflow uses). That file's
*only* two jobs here are: (1) its `discovery` section supplies the NMF
hyperparameters handed directly to the shared discovery worker
(`hierarchy_rank_discovery.sbatch` / `run_hierarchy_rank_discovery.py`,
unchanged from the standalone workflow), and (2) its `dataset.maze_name` is
used to derive the maze-topology fingerprint that keys the NMF-basis cache
(`maze_sha256`). That fingerprint depends only on the maze's fixed topology
lookup (`maze_configs.json`), not on which subjects/sessions/dates
`discovery_config`'s own `dataset` restricts to — so `discovery_config` need
not describe the same subjects or dates as this config's own `dataset`, only
the same `maze_name`. Loading a config whose `discovery_config` names a
different maze raises immediately rather than silently computing an
incompatible fingerprint. `discovery_config`'s `adam` section is not used at
all by the adjacent workflow — only its own `discovery` section is borrowed.

### Automatic SLURM manager (recommended)

Rerun the same idempotent command; it inspects artifacts and `squeue`, then
advances the next safe stage (NMF discovery, then banded inner fits, then
local aggregation, then banded refits) and prints the exact next command:

    scripts/slurm/submit_adjacent_mlmdp.sh \
      --run-id production \
      --config configs/adjacent_mlmdp_regression.json \
      --output-dir output/adjacent_mlmdp_regression/production

It creates the scientific fold manifest automatically, discovers or reuses
compatible NMF bases (an explicit `discovery_dir` in the config is reused
as-is; otherwise it defaults to and caches under
`<data_root>/nmf_bases/<maze_name>/<discovery-compatibility-digest>/`),
submits inner-fit arrays sized to exactly the work that is missing or
retryable in each configured rank band, runs aggregation locally once every
inner shard is terminal, submits selected-rank refits banded by the selected
k, and finally prints the exact `reproduce_figure_2_19_behavior.py` command
below once every predictor is terminal. Resource bands, discovery resources,
partition/account, and concurrency can be overridden with an optional
top-level `"slurm"` object in the config (see
`scripts/slurm/manage_adjacent_mlmdp.py` for the schema); omitting it uses
the bands and defaults documented below. Pass `--dry-run` to preview the next
submission without touching SLURM, and `--cancel-held` to release
administrator-held array elements before retrying.

Every invocation opens with a colored pipeline overview (discovery, inner
fits, rank selection, refits, final command) showing each stage as done
(green), in progress with a percentage (yellow), or not started (grey).
Ordinary retries within an already-started stage submit automatically; the
first time a stage is about to start submitting real work, the command asks
for interactive confirmation (`[Y/n]`) before doing so. The prompt is skipped
automatically when stdin isn't a terminal (cron, scripts) or under
`--dry-run`; pass `--yes`/`-y` to skip it explicitly even in a terminal.

The remainder of this section documents the underlying manual/debugging
commands the manager drives; use them directly only to investigate a single
task or when working outside the managed workflow.

Prepare exact fold identities:

    python scripts/run_adjacent_mlmdp.py prepare \
      --config configs/adjacent_mlmdp_regression.json \
      --output-dir output/adjacent_mlmdp_regression/production

The low-cost pilot contains all inner folds at k={2,5,10}, plus one inner fold
at k={15,20}: 35 fits total. It intentionally does not probe k=35 or k=49.

    sbatch --array=0-34%35 \
      --export=ALL,HIERARCHY_ADJACENT_TASK_MODE=pilot \
      scripts/slurm/adjacent_mlmdp_inner.sbatch

Inspect pilot failures and timings before production. Production should be
submitted in cost bands, initially capped at 200 simultaneous jobs:

| Rank band | Inner tasks |
| --- | ---: |
| 2-12 | 8,470 |
| 13-25 | 10,010 |
| 26-37 | 9,240 |
| 38-49 | 9,240 |

For example, the first band is:

    sbatch --array=0-8469%200 \
      --export=ALL,HIERARCHY_ADJACENT_RANK_MIN=2,HIERARCHY_ADJACENT_RANK_MAX=12 \
      scripts/slurm/adjacent_mlmdp_inner.sbatch

The other bands use the corresponding task count and rank bounds. Missing or
operationally failed tasks may be resubmitted with the same band; successful
and scientific-failure shards are cache hits, while operational-failure shards
are retried. Do not use --force for ordinary retry.

After all inner tasks are terminal:

    python scripts/run_adjacent_mlmdp.py aggregate \
      --config configs/adjacent_mlmdp_regression.json \
      --output-dir output/adjacent_mlmdp_regression/production

Then submit one selected refit per outer fold:

    sbatch --array=0-70%71 scripts/slurm/adjacent_mlmdp_refit.sbatch

Finally run the augmented adjacent regression:

    python doohan_data_interaction/reproduce_figure_2_19_behavior.py \
      --data-root external/GridMaze-mFC-ephys-DATA/data \
      --output-dir results/figure_2_19 \
      --fold-scheme adjacent \
      --include-hierarchical-mlmdp \
      --hierarchical-mlmdp-run-dir output/adjacent_mlmdp_regression/production

The augmented files include with_hierarchical_mlmdp in their names, so they do
not overwrite the original seven-predictor result.

