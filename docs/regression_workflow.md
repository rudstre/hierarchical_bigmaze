# Held-out-session regression workflow

## Statistical contract

For every subject/session predictor partition, the held-out session is excluded
from MLMDP fitting, subgoal-rank selection, PCA/HMM route fitting, and habit
counts. The selected predictor bundle is then fixed. Its reserved-session
features are generated causally with state reset at each trial boundary, and
contiguous whole-trial blocks define regression test folds inside that session.
All preprocessing statistics and coefficients are fitted from the other blocks.

The response is the observed four-way action. The design contains an intercept
and four-action values for vector, optimal, hierarchical MLMDP, route, route
planning, habit, forward, and reverse predictors. Impossible actions retain
Qin's exact `-1e10` additive mask. In k-fold mode, each fold fits Qin's full
conditional-logit model, the intercept-omitted model, and one model omitting
each predictor. Unique predictability is reduced minus full held-out mean NLL
in nats/decision. In learning-curve mode, each split fits only the full model
and reports test log likelihood in nats/decision.

## Configuration

The versioned examples are
[`configs/regression_workflow.json`](../configs/regression_workflow.json) for
k-fold ablation and
[`configs/regression_workflow_learning_curve.json`](../configs/regression_workflow_learning_curve.json)
for the full-model learning curve.

- `heldout_sessions: "last"` creates one partition per subject.
- `heldout_sessions: "all"` creates every possible held-out-session partition.
- A list of `subject_id`/`session_id` records selects exactly one typed session
  per subject without JSON object-key coercion.
- `subgoal_selection.method` is `session_cv`, `training_ll`, or `fixed`.
- `subgoal_selection.rank_range` is always inclusive `[lower, higher]`.
- `predictors.route_family` is `pca` or `hmm`; the unused family's settings do
  not affect predictor compatibility.
- `regression_cv.method: "blocked_trial_kfold"` runs the original full/reduced
  ablation analysis; `n_splits` changes only regression splits and results.
- `regression_cv.method: "blocked_trial_learning_curve"` fits only the full
  regression model. Its sole parameter, `n_subdivisions`, sets the number of
  equal intervals between train-on-one-trial and leave-one-trial-out.

Learning-curve percentages refer to regression-coefficient fitting within the
reserved session. The MLMDP, route, and habit predictors remain fitted from the
complementary sessions. For every distinct requested training size, the workflow
fits all circular placements of the contiguous test block, so a session with
`N` trials performs `N` fits per distinct size. Circular wrapping selects trial
membership only; decision rows remain chronological and trial boundaries remain
intact. Requested sizes that round to the same whole-trial count share their
fits.

`session_cv` fits rank candidates once for each omitted predictor-training
session, selects using only those held-out scores, and refits the winner on all
predictor-training sessions. `training_ll` fits every candidate once on all
predictor-training sessions and selects the greatest total training LL, with a
lower-rank tie break. `fixed` requires `[k, k]` and fits that rank once.

## Commands

```bash
python scripts/run_regression_workflow.py prepare \
  --config configs/regression_workflow.json \
  --output-dir output/regression_workflow/example

python scripts/run_regression_workflow.py complete \
  --config configs/regression_workflow.json \
  --output-dir output/regression_workflow/example

python scripts/run_regression_workflow.py status \
  --config configs/regression_workflow.json \
  --output-dir output/regression_workflow/example
```

Partition stages additionally require `--partition-digest`. Candidate tasks
also require scalar `--k` and, for an inner candidate, a typed JSON value in
`--validation-session-json`. Scalar task identities are not configured rank
ranges.

SLURM preview and resume:

```bash
python scripts/slurm/manage_regression_workflow.py \
  --config configs/regression_workflow.json \
  --run-id example --dry-run

python scripts/slurm/manage_regression_workflow.py \
  --config configs/regression_workflow.json \
  --run-id example --status

python scripts/slurm/manage_regression_workflow.py \
  --config configs/regression_workflow.json \
  --run-id example --retry-missing
```

The manager writes immutable task lists, chunks arrays, assigns inclusive rank
resource bands, records job dependencies and task-list digests, and consults
both `squeue` and `sacct`. Retries contain exact scalar tasks. No production job
is submitted by tests.

## Artifacts and reuse

Candidate roles have separate directories, so an all-training selected-rank
refit can never be mistaken for an inner session-CV candidate. Predictor
compatibility binds data selection, typed partition identity, rank method and
range, fitting settings/code, discovery artifact, route settings, and seeds.
Feature compatibility adds predictor digest, held-out canonical data, predictor
order, and feature code. Regression compatibility adds exact trial blocks,
regression settings, and regression code.

Atomic JSON writes distinguish missing, operational failure, scientific
failure, unavailable, and successful states. I/O and scheduler failures remain
retryable. Explicit optimizer convergence or model-scoring failures are
terminal scientific failures. Schema, alignment, incompatibility, and
unexpected programming/data errors are raised rather than silently classified.

Changing only `n_splits` reuses predictor and feature artifacts. Changing rank
range, selection method, route family/settings, training sessions, discovery
artifact, held-out data, or relevant stage code refuses incompatible reuse.
Historical output directories are never removed by the workflow.

## Reporting

Aggregation writes complete fold, session, subject, and group CSV tables, a
numerical result JSON, and provenance JSON. Plotly is the only reporting
backend. The predictor and session-profile panels show individual observations,
mean and SEM, a zero reference, units, sample counts, stable colors, and hover
metadata. Output includes self-contained HTML plus PNG, SVG, and PDF through
Kaleido. Incomplete grids retain partial diagnostics but have no headline group
estimate.

Learning-curve reporting instead writes split, session, subject, and group CSV
tables and plots training-trial percentage against mean test log likelihood.
Split LL is pooled by decisions within each session and training size; sessions
are averaged equally within subjects and subjects equally within the group.
