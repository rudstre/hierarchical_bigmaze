# Held-out-session regression workflow

## Statistical contract

For every subject/session predictor partition, the held-out session is excluded
from MLMDP fitting, subgoal-rank selection, PCA/HMM route fitting, and habit
counts. The selected predictor bundle is then fixed. Its reserved-session
features are generated causally with state reset at each trial boundary, and
contiguous whole-trial blocks define regression test folds inside that session.
All preprocessing statistics and coefficients are fitted from the other blocks.

The response is the observed four-way action. The design contains an intercept
and four-action values for vector, optimal, the configured MLMDPs, route, route
planning, habit, forward, and reverse predictors. New configurations use
`behavioral_mlmdp`; optional `hierarchical_mlmdp` retains its historical NMF
meaning. Enabling both gives nine predictors in one regression. Impossible actions retain
Qin's exact `-1e10` additive mask. In k-fold mode, each fold fits Qin's full
conditional-logit model, the intercept-omitted model, and one model omitting
each predictor. Unique predictability is reduced minus full held-out mean NLL
in nats/decision. In learning-curve mode, each split fits only the full model
and reports test log likelihood in nats/decision.

K-fold mode runs the same ablation procedure for three prespecified predictor
sets. The combined set contains every predictor. The route-only set removes the
configured MLMDP predictors while retaining the selected PCA or HMM route and
route-planning predictors. The MLMDP-only set removes both route predictors
while retaining every configured MLMDP. Vector, optimal, habit, forward, and
reverse controls remain in all three sets.

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
  Behavioral profiles reject `training_ll`; fixed K still uses inner LOSO for θ.
- `subgoal_selection.rank_range` is always inclusive `[lower, higher]`.
- `predictors.route_family` is `pca` or `hmm`; the unused family's settings do
  not affect predictor compatibility.
- `predictors.pca.n_components` accepts a fixed positive integer or `"auto"`.
  For a partition whose selected MLMDP rank is `K`, `"auto"` retains
  `ceil(K / 2)` signed PCA components, which become twice as many nonnegative
  PCA routes after positive/negative splitting.
- `predictors.hmm.n_routes` accepts a fixed positive integer or `"auto"`.
  Here `"auto"` fits `K + 1` HMM routes. The resolved numeric route settings
  are recorded in each predictor artifact. When behavioral profiles are enabled,
  auto counts use behavioral K*, including when NMF is also fitted.
- Unspecified `predictors.hmm` settings default to Qin's mouse fits: 6 routes,
  `cognitive_constant` 30, `action_cost` 0.13, `reward_value` 1,
  `action_entropy_param` 0.2, `route_entropy_param` 0, `noise` 1 decaying by
  0.997 per policy recalculation, learning rate 0.05, and 2000 epochs. As in her
  `train_entry.run_once`, `validation_fraction` (0.2) of the partition's
  route-training trials are held out with the route seed, validation log
  likelihood is evaluated every `eval_every` (50) epochs, and the parameters at
  the best mean of the last `smooth_k` (5) evaluations are kept. The held-out
  session never enters this split. `validation_fraction: 0` fits all trials and
  keeps the final iterate.
- `predictors.hmm.route_prior_concentration` (default `null`, off) adds the log
  density of a symmetric Dirichlet(α) prior on the non-uniform route weights,
  per observation, to the training objective, as in Qin's 2024 `Route_HMM`
  (which used α = 100). α must be at least 1; larger values hold the route
  weights closer to equal and keep routes from collapsing to zero weight.
  Validation scores and checkpoint selection remain likelihood-only, and
  predictor diagnostics record the per-epoch prior term as
  `training_log_route_prior`. The Streamlit HMM settings expose it as
  "Dirichlet prior on route weights".
- `regression_cv.method: "blocked_trial_kfold"` runs the original full/reduced
  ablation analysis; `n_splits` changes only regression splits and results.
- `regression_cv.method: "blocked_trial_learning_curve"` fits only the full
  regression model. Its sole parameter, `n_subdivisions`, sets the number of
  equal intervals between train-on-one-trial and leave-one-trial-out.
- `slurm.resource_profile: "auto"` derives CPU, memory, time, and array
  concurrency from the active route family and regression method. Optional
  `slurm.stage_overrides` entries may set `cpus_per_task`, `memory`, `time`, and
  `max_concurrent` for a named stage. Resource settings are execution
  provenance and do not enter the scientific configuration signature.
- `slurm.max_concurrent` (default 1200, editable in the GUI as "Max concurrent
  tasks per stage") caps the running tasks of a whole stage. The manager splits
  a stage into SLURM arrays of 1,000 tasks and gives each array
  `ceil(max_concurrent / n_arrays)`, because SLURM's `%N` limit applies per
  array; the cap therefore stays fixed as subjects are added. Runs whose frozen
  plan predates this (no `concurrency_scope`) keep the per-array meaning on
  retry. Each candidate rank band is its own group.

When route settings do not depend on the selected rank (a fixed HMM
`n_routes` or PCA `n_components`), a `route-fit` stage fits routes and habit
counts at the start of the run, alongside MLMDP fitting, and writes
`partitions/<digest>/route_fit.json`. The route fit uses the same training
rows, settings, and seed as before, so the bundle copies it unchanged and
records its digest as `route_fit_artifact`. A route-fit scientific failure is
terminal and carries into the bundle; an operational failure is retried. With
`"auto"` routes the bundle still fits them after rank selection. Runs submitted
before this stage keep fitting routes in the bundle.

The HMM CPU profile (`auto-cpu-v4`) requests 8 CPUs for the route fit and
estimates its time and memory from measured fits on `cpu_backfill`: the slowest of
36 partitions took 11.19 s/epoch with 10 routes (`full_hmm2`, 500 epochs,
peak 8.07 GB) and 4.41 s/epoch with 6 routes (`full_hmm_xiao2`, 2000 epochs,
peak 3.57 GB). Time is linear in epochs; per-epoch time scales as
`(routes + 1)^2.06`, fitted to those two points, and memory as
`(routes + 1)^2`. Requests are twice the estimate, rounded up to whole hours,
with floors of 16 GB and one hour (for example, 6 routes x 2000 epochs requests
5 h, 10 routes x 2000 epochs 13 h). Above 10 routes the time exponent is the
dense-solve bound 3 and the plan is reported as an extrapolated estimate; it
remains launchable, and stage overrides still take precedence. Learned costs
(`learn_costs`) and the route-weight Dirichlet prior were not part of the
measured fits. Eight threads are less efficient per core than two (thread
benchmark: 665 s at 2 threads, 437 s at 8), but there is one route fit per
partition on the critical path, so the profile favors wall time; feature
generation runs no epochs and keeps 2 CPUs. The default partition is
`"cpu_backfill,cpu"`: SLURM starts each task on whichever partition can run it
first, adding the idle protected `cpu` nodes to the preemptible GPU-node pool.
Any partition list that includes `cpu_backfill` permits 96 concurrent HMM
tasks; selecting the protected `cpu` partition alone reduces the automatic HMM
cap to eight. For `n_routes: "auto"`, the manager uses `max_rank + 1` as the
pre-submission bound. The workers hide CUDA devices, so backfill jobs remain CPU-only
on GPU-equipped nodes. No GPU profile is defined.

HMM fitting remains float32. Predictor diagnostics record dtype, device,
thread counts, seed, data counts, completed epochs, the pre-update likelihood
trace, the validation trial keys and evaluation history, the selected epoch,
and a deterministic likelihood on the fitting trials recomputed with the kept
parameters. Do not call `torch.set_num_threads` in processes that run the HMM:
with torch 2.13.0+cpu, batched `torch.linalg.solve` with more than one thread
then fails (MKL `SLASWP` errors) or hangs on every node type tested; thread
counts set through `OMP_NUM_THREADS`, as the SLURM stage script does, work.
Finite-value checks cover the optimizer and the returned policy/state
distributions. Float32 parity against float64 is tested for likelihood,
gradients, policies, and solve residuals; float16 is unsupported by the model's
linear solve and numerical constants.

The precision audit used the production 10-route solve and 3,921 observed
decisions. Float32 versus float64 changed mean observed log likelihood by
`1.49e-7`, the objective by `1.36e-7`, and the gradient by `8.62e-7` in relative
norm; the float32 solve residual was `2.72e-7`. Float32 used 7.74 GB peak memory
versus 14.69 GB for float64, and five matched Adam epochs ended within `2.26e-8`
mean log likelihood. These measurements justify the documented lower-precision
exception while the automated parity test protects the same model equations.

Learning-curve percentages refer to regression-coefficient fitting within the
reserved session. The MLMDP, route, and habit predictors remain fitted from the
complementary sessions. For every distinct requested training size, the workflow
fits all circular placements of the contiguous test block, so a session with
`N` trials performs `N` fits per distinct size. Circular wrapping selects trial
membership only; decision rows remain chronological and trial boundaries remain
intact. Requested sizes that round to the same whole-trial count share their
fits. The manifest stores the chronological trial sequence once and represents
each split by its training count and block start; this compact representation
still determines every train/test assignment exactly.

For the NMF profile source, `session_cv` fits rank candidates once for each omitted predictor-training
session, selects using only those held-out scores, and refits the winner on all
predictor-training sessions. `training_ll` fits every candidate once on all
predictor-training sessions and selects the greatest total training LL, with a
lower-rank tie break. `fixed` requires `[k, k]` and fits that rank once.

## Behavioral profiles and nested θ selection

New configurations explicitly contain:

```json
"mlmdp": {
  "profile_sources": ["behavioral"],
  "behavioral": {"theta_search": "coarse"}
}
```

Use `["behavioral", "nmf"]` to include both MLMDPs in the same regression, or
`["nmf"]` for NMF alone. Historical payloads without `mlmdp` remain NMF-only;
normalization omits the block in that case, preserving their exact configuration
signatures. Historical bundles with `mlmdp_candidate` retain their NMF meaning;
new bundles use `mlmdp_candidates`, indexed by predictor name. The combined
k-fold ablations measure each MLMDP's unique predictability conditional on the
other predictors. Route-only excludes all MLMDPs; MLMDP-only retains both when
both are configured.

The canonical per-cell procedure is
[`fit_cell`](../src/andrew_mlmdp/hierarchy/behavioral_profiles.py). Defaults are
Gaussian centre search (60 fit steps, patience 10, screen top 2, local radius 1)
from one greedy and two k-means++ starts, followed by free-form refinement of
every start (log parameterization, learning rate 0.1, 1500 steps, tolerance
1e-6; Dirichlet 0.1, compactness 0.3, overlap 0). It keeps the lowest eligible
training objective.

A refinement step that collapses a profile onto a goal (its gated support
becomes that goal alone, which the model cannot represent) is projected: the
column's largest off-goal entry is raised to 0.1 log units above the access
gate, and the step is kept. Every projection is recorded with the start.
Previously such steps were rejected with a lower learning rate, which pinned
the fit at the boundary until the rejection limit discarded the start: 26% of
the real-data starts finished before the first full run was cancelled, and 38%
of the simulated K=6 and K=8 starts, ended this way, mostly narrow bumps
centred on a goal pulled in by compactness. On the simulated θ
grid (`results/theta_grid/constraint_test` on the cluster: seed 101, K 6 and
8, three widths, 36 cells), projection made every start eligible, revived the
37/216 cells with no eligible start, never worsened a cell's training
objective, and left the chosen θ unchanged in all six settings. Freezing the
column instead performed almost identically. With projection, greedy plus two
seeded starts chose the nine-start θ in 79% of settings at a mean held-out
cost of 0.001 LL per decision (maximum 0.015); greedy plus four reached 86%.
The rule is `mlmdp.behavioral.infeasible_step` (`"project"`, `"freeze"`, or the
historical `"reject"`), and the start count `n_seeded_starts`. The model uses deferred goal reward, both commitment modes,
radius 2, α=1, interior reward −1, goal reward 0, and fixed access gate 0.05.
The exp(−30) floor is used only for initialization; fitted gated profiles retain
exact zeros. Overrides live beside `theta_search` in `mlmdp.behavioral` and are
fully resolved in normalized configuration records.

For every K and inner validation session, the coarse stage fits D on all other
predictor-training sessions at each of 36 θ cells: λ₁ ∈ {0.3, 0.6, 1.2},
λ₂ ∈ {0.6, 1.2, 2.4}, β ∈ {1, 2, 4, 8}. Strict validation LL per physical
transition is averaged **equally across sessions**. A cell must succeed on every
inner session. θ_K maximizes that mean; ties follow configured grid order.
The unchanged one-standard-error rank rule chooses the smallest K within one
SE of the best θ_K mean.

With `theta_search: "coarse_and_local"`, the same inner folds score 26 local
neighbours at multiplicative √2 spacing around θ at K*. The best mean among
those and the coarse centre chooses θ*. K* stays fixed. With `"coarse"`, the
coarse θ at K* is final. Refit uses all predictor-training sessions at (K*, θ*)
and emits strict training scores and reserved-session action predictions.
The outer session never selects θ, K, starts, or optimizer settings. Its goals
are included in the feasibility check as task-design information.

A coarse fit for outer session h and inner session v trains on the sessions
other than h and v, with goals from all of the subject's sessions, so it is
identical for partition h (inner v) and partition v (inner h). Each such inner
fold is fitted once, stored under `behavioral_folds/<fold digest>/`, and scores
both excluded sessions. A partition reads only the score of its own inner
validation session; the other score belongs to its outer session and is never
read by it. With six sessions per subject this halves the coarse fits.

Seeds derive from `SeedSequence([random_seed, fold_digest_integer, K])`, where
the fold digest identifies the subject, training sessions, and excluded
sessions; they are shared across θ cells of a fold, by both partitions sharing
it, and by its local cells. Start seeds and child
indices are recorded. Every start remains in the artifact, including optimizer
failures and infeasible initializations. All-start failure makes the cell a
`scientific_failure`; its detailed reasons distinguish optimizer and numerical
failure from model infeasibility. Missing or operationally failed cells leave
selection pending. Scientific failures make a cell ineligible and are never
silently omitted. Start checkpoints and cell artifacts validate their scientific
identity and are written atomically; force reruns bypass start checkpoints.

The SLURM stages are `behavioral-cell → behavioral-selection → behavioral-refit`,
or `behavioral-cell → behavioral-selection → behavioral-local-cell →
behavioral-local-selection → behavioral-refit`. Local tasks freeze fold and
offset identities before submission and resolve θ and K* at execution. Each
`behavioral-cell` task fits one coarse θ cell of one (inner fold, K), so the
cells of a fold run in parallel. Seeds depend only on the fold and K, so the
artifacts match those of fitting the fold's cells in sequence. Finished cells
are reused. Without `--theta-cell` a worker fits every cell of the fold in grid
order, as task lists written before this change do. The NMF
`discovery → candidate → selection → selected-refit` chain runs only when NMF
is enabled. `route-fit` (when enabled) starts with no dependency.
`predictor-bundle` waits for all enabled chains and the route fit, followed by features,
regressions, overlap, aggregation, and plots. Coarse cells live under
`behavioral_folds/<fold digest>/`; the other behavioral artifacts live under
`partitions/<digest>/behavioral/`: `selection.json`, `local/`,
`final_selection.json`, and `refit.json`. Coarse workers take `--k`,
`--validation-session-json` (with any partition sharing the fold), and
`--theta-cell`; local workers take `--local-index` instead of `--theta-cell`.

The behavioral resource profile is provisional: one CPU, 3 GB, and 1 h per
start for each coarse or local θ-cell task (3 h for three starts), and 12 h for
the refit. It rests on real-data m2 starts at 1 CPU from the cancelled
`mlmdp_behavior_full` run (median 5–12 min and at most 22 min per start at
K 3–9) and on the real K=6 pilot (SLURM 3827355: 94 min for nine starts, peak
RSS 0.66 GB). Those starts used the earlier `reject` rule; K above 9 and
the `project` rule are unmeasured. Stage overrides take precedence. Cell artifacts record
elapsed seconds and peak RSS. The simulated estimate is about 2.5–3 CPU-hours
per cell: ranks 2–10, five folds, 36 cells require 1620 cells per partition;
36 outer partitions approach 150,000 CPU-hours. Start with one subject,
`heldout_sessions: "last"`, a narrow rank range and a SLURM pilot before a full
run. The checked-in rank range and all-session selection are not a measured
production resource recommendation.

A separate end-to-end real-data SLURM smoke test completed all nine enabled
stages for m2 with the last session held out: five inner cells, selection,
refit, feature generation, all three regression variants, overlap, and reports.
It used fixed K=2, one θ cell, one seeded start, three centre-fit steps and ten
refinement steps, so it checks workflow execution rather than convergence or
default-fit cost. Cells took 32.9–67.9 s; the all-training refit took 57.0 s
with 0.51 GiB process peak RSS. Its configuration and verification record are
in `results/behavioral_pilot/workflow_smoke/`. The completed greedy and first
seeded stored-simulation starts match the original objectives within 6e-17
and D within 5e-15 after column permutation; the remaining starts are pending.

The GUI defaults to behavioral profiles, offers “Also fit NMF MLMDP (both enter
the regression)” and coarse/fine θ controls, and shows θ_K means ± SE, selected
K*, and the inner-LL θ landscape under “Behavioral MLMDP”.

Limitations:

- Behavioral and default NMF models differ in structure as well as D: NMF uses
  inpainted goal reward and fitted α, τ, η. Both use termination-plus-radius-2
  commitment: the repository configs set `adam.commitment_mode: "both"` and
  `adam.commitment_radius: 2` explicitly, because an `adam` block without them
  means termination only (the setting every earlier run used). For a D-only
  comparison match `adam.goal_reward_mode`, commitment mode/radius, access gate
  and parameter constraints explicitly.
- One real inner session per fold gives noisier θ selection than the simulated
  396-trial validation design.
- Maximizing θ at each K makes the inner K curve optimistic; it does not touch
  the outer held-out score.
- Joint θ+D refinement and a validation guard are outside this workflow. The
  experimental simulation scripts remain separate.

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

To add the three k-fold model sets and their reports to a completed historical
run, use its saved feature tensors:

```bash
python scripts/run_regression_workflow.py regression-variants \
  --config configs/regression_workflow.json \
  --output-dir output/regression_workflow/example
```

This command reads `features.json`, preserves the existing combined
`regression.json`, fits only missing route-only and MLMDP-only regressions, and
writes all six likelihood and accuracy figures. It does not refit predictors.

Partition stages additionally require `--partition-digest`. Candidate tasks
also require scalar `--k` and, for an inner candidate, a typed JSON value in
`--validation-session-json`. Scalar task identities are not configured rank
ranges. Workflow commands print concise status and artifact locations by
default. Pass `--json` when a caller needs the complete machine-readable result.

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

python scripts/slurm/manage_regression_workflow.py \
  --config configs/regression_workflow.json \
  --run-id example --cancel
```

The manager writes immutable task lists, chunks arrays, assigns inclusive rank
resource bands, and freezes the complete resolved resource table before any
job is submitted. Every submission records its CPU, memory, time, and
concurrency request, plus the per-array `array_limit` it received; retries
reuse that table. Historical schema-v1 managers
retain their one-CPU retry behavior. The manager records job dependencies and
task-list digests and consults both `squeue` and `sacct`. Status output reports
artifact completion and array task counts for the latest attempt of each stage,
plus allocated CPUs, requested memory, elapsed time, and peak RSS when Slurm
provides them. Prior submissions remain in `slurm_manager.json`. Pass `--json`
with `--status` for machine-readable status.
Retries contain exact scalar tasks. No production job is submitted by tests.
`--cancel` stops only active SLURM arrays recorded for the selected run,
including pending downstream stages. It leaves completed artifacts and the
manager record in place. Add `--dry-run` to preview the job IDs; use
`--retry-missing` later if you want to resume.

Workers refuse to start when the scientific source (`src/`, `scripts/` outside
`slurm/` and `gui/`, and `pyproject.toml`) differs from the fingerprint the run
recorded. This includes tasks that wait in the queue while the code is edited.
`--retry-missing` therefore refuses to submit while the source differs. To let
the rest of the run use the current code, pass
`--rerecord-worker-source "REASON"`. The previous fingerprint, the changed
files, the reason and the time are appended to `worker_source_history` in
`slurm_manager.json`. The override is refused when an existing artifact
recorded a different per-stage source (`candidate`, `feature` or `regression`
scopes), because workers re-check those when reusing artifacts; start a new run
in that case. The GUI Progress page shows the same information and offers the
override behind an acknowledgement and a required reason.

## Artifacts and reuse

Candidate roles have separate directories, so an all-training selected-rank
refit can never be mistaken for an inner session-CV candidate. Predictor
compatibility binds data selection, typed partition identity, rank method and
range, fitting settings/code, discovery artifact, route settings, and seeds.
Feature compatibility adds predictor digest, held-out canonical data, predictor
order, and feature code. Regression compatibility adds exact trial blocks,
regression settings, regression code, the model-set identity, and its predictor
order. A successful k-fold partition contains `regression.json`,
`regression_route_only.json`, and `regression_mlmdp_only.json`. The saved feature
tensor remains the common input to all three fits.

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

Aggregation writes separate fold, session, subject, and group CSV tables,
numerical result JSON, and provenance JSON for the combined, route-only, and
MLMDP-only model sets. Each set has one likelihood figure and one companion
accuracy figure, for six figures total. Plotly is the only reporting backend.
The predictor and session-profile panels show individual observations, mean and
SEM, a zero reference, units, sample counts, stable colors, and hover metadata.
Output includes self-contained HTML plus PNG, SVG, and PDF through Kaleido.
Incomplete grids retain partial diagnostics but have no headline group estimate.

In the Streamlit Results interface, unique-predictability and unique-accuracy
group panels can add significance stars directly from the saved subject table. Each displayed
predictor is tested against zero with a one-sided one-sample t-test for a
positive contribution, and p-values are Holm-corrected across the predictors in
that figure; the likelihood and accuracy figures are separate Holm families. Subjects are the independent observations; held-out sessions have
already been averaged equally within subject. Stars denote adjusted `p < .05`,
`p < .01`, and `p < .001`. Significance is withheld for incomplete results or
when the displayed predictors lack at least two matching, finite subject values
with nonzero between-subject variance. Rendering these annotations does not
refit models or modify historical result artifacts.

Each likelihood figure has a companion `_accuracy` figure generated from the
same held-out folds. Accuracy is the fraction of individual actions for which
the model's highest-probability action matches the observed action; JSON and CSV
store this fraction in `[0, 1]`, while figures display percentages. For the
predictor-ablation report, the plotted quantity is full-model accuracy minus
the accuracy of the model without that predictor, in percentage points. Fold
accuracies are pooled by held-out decision count, sessions are averaged equally
within subjects, and subjects are averaged equally for the group mean and SEM.
No chance correction is applied.

Learning-curve reporting instead writes split, session, subject, and group CSV
tables and plots training-trial percentage against mean test log likelihood.
Split LL is pooled by decisions within each session and training size; sessions
are averaged equally within subjects and subjects equally within the group.
The companion accuracy curve applies the same hierarchy to full-model action
accuracy and uses a fixed 0–100% axis.
Both learning-curve figures include a dashed uniform-policy baseline. At each
decision the baseline assigns equal probability to every valid action. Its log
likelihood is `log(1 / n_valid_actions)`, and its accuracy reference is the
expected correct-action rate `1 / n_valid_actions` under uniform random action
sampling. Baselines use the same decision, session, and contributing-subject
weighting as the plotted group estimate, so the all-decision and 3–4-action
panels have separate data-derived reference lines.
At each training size, the plotted group mean and SEM use only subjects whose
session and split results are complete at that size. The hover data and group
CSV report the contributing and expected subject counts. Failed fits remain in
the diagnostics and keep the overall report status incomplete; incomplete
subject points appear as gaps rather than contributing partial estimates.
The report also reconstructs each stored fold and scores the same fitted model
only at held-out locations with three or four valid actions. This junction-only
likelihood and accuracy curve appears in a separate panel and has separate
`valid-actions-3-4` CSVs. Splits with no matching decisions are marked not
applicable and have no likelihood or accuracy value.
Training still uses every eligible decision. Reconstruction must reproduce the
stored all-decision fold likelihood before junction scores are accepted.

## Observed-decision policy overlap

The `policy-overlap` stage compares all configured regression predictors
on the same held-out movements used to generate their feature tensor. It reads
the four-action values and impossible-action mask from `features.json`; it does
not refit predictors or infer Qin routes a second time. Every saved MLMDP
predictor is checked against its selected candidate's held-out predictions,
whose fit excludes this session. Route and habit predictors come from that
partition's complementary training sessions, and route history in the feature
tensor is causal.

For each navigation movement starting at a location with three or four valid
actions, the stage constructs action distributions from the saved vectors.
Nonnegative vectors are normalized across valid actions. The saved `reverse`
vector marks the action that would undo the previous move; overlap reports its
complement as `dont_reverse`, with equal probability on every other valid
action. The regression feature and fitted coefficient remain unchanged. The
vector predictor instead puts equal mass on the valid directions tied for its
highest stored directional score. In particular, the optimal predictor's
existing tie weights are retained by normalizing its stored vector. A movement
enters the primary analysis only if all configured policies are defined.
Zero-support predictors, trial-start forward/reverse predictors, and blocked
forward/reverse preferences exclude that movement; no uniform distribution is
substituted. Artifacts record eligible and included movement counts, exclusions
by reason, and the
percentage retained. Reason counts can overlap when a movement has more than
one undefined predictor; the distinct excluded count is recorded separately.
Projection uses float64 arithmetic: negative roundoff no smaller than `-1e-12`
and invalid-action mass no larger than `1e-10` are set to zero. Larger
violations raise an error.

For action distributions `p` and `q`, observed-decision policy overlap is
`sum_a min(p_a, q_a) = 1 - TV(p, q)`, from zero (disjoint) to one (identical).
The observed action on each included movement is added as the one-hot
`observed_behavior` distribution. Its overlap with policy `p` is therefore
`p(observed action)`. This is fractional for probabilistic or tied policies; it
is not argmax percentage-correct accuracy, and it is not an estimated
stationary behavioral policy. Each session's (n+1)×(n+1) matrix averages overlap among
the n projected predictors and observed behavior across its included
movements.
The pooled matrix divides the sum of the session overlap numerators by the
total included movement count, so repeated visits and longer sessions have
proportionally more weight. Session artifacts are saved as
`partitions/<digest>/policy_overlap.json`; pooled JSON, CSV, and a Plotly heatmap
are saved in `report/`. A null matrix means no common-valid movement was
available. The pooled matrix is withheld when a session artifact is missing.

Run `python scripts/run_regression_workflow.py policy-overlap --config CONFIG
--output-dir OUTPUT` to backfill a completed run without refitting. Add
`--partition-digest DIGEST` to generate just one session artifact. The
`complete` and SLURM workflows generate overlap artifacts automatically.
After changing an overlap projection or upgrading an existing 8×8 run, add
`--force` once to regenerate only its overlap artifacts and report. Existing
9×9 artifacts with the original `reverse` projection use the prior schema and
are not reused.

The configured model policies are explicit projections of the regression's saved
action **values**; several predictors are not native probability distributions.
Overlap describes their similarity on observed behavioral decisions. It does
not measure regression-coefficient similarity and does not by itself establish
predictor redundancy or unique variance.

## Cluster GUI and saved-feature analyses

Install the optional GUI dependency in the same Python environment as the
workflow. In the current `srun-7day` setup, start Streamlit on the allocated
compute node that runs the VS Code tunnel:

```bash
python -m pip install -r requirements-gui.txt
python -m streamlit run scripts/gui/app.py --server.address 127.0.0.1 --server.port 8501
```

On your laptop, use the SWC compute-node forwarding route (external SSH host,
then HPC gateway, then the allocated compute node):

```bash
ssh -N -J USER@ssh.swc.ucl.ac.uk,USER@hpc-gw2 -L 8501:127.0.0.1:8501 USER@COMPUTE_NODE
```

Then open `http://localhost:8501` in your laptop browser. Replace
`COMPUTE_NODE` with the node reported by `hostname` inside your allocation;
the port forward must end at that node. A tunnel ending on a login node would
forward the login node's localhost instead. The VS Code integrated browser may
also reach `http://localhost:8501` from the remote workspace, but its Ports
view can fail to forward ports through a code tunnel. Streamlit reads artifacts
and launches the existing manager; scientific SLURM jobs continue after the
browser or GUI closes. If the
`srun-7day` allocation ends, start the GUI again on a new allocated node and
forward its port. Select a directory containing `manifest.json` to inspect a
historical run. Reading results does not require its original configuration file.

A new GUI run snapshots its regression and referenced discovery configurations
under `inputs/`. The original discovery directory remains shared. Configuration
changes after launch create a new run; the submitted config is immutable. The
new-workflow form reads session metadata to offer subject checkboxes and start/end
date choices for the selected subjects; applying them updates the configuration
JSON without changing its schema. The manager records content hashes for the
snapshotted regression configuration,
referenced discovery configuration, and workflow manifest. Queued workers verify
these inputs before calculating, so editing one in place stops new work rather
than silently changing the experiment. The
configuration preview resolves subjects, dates, predictor-training sessions, and
held-out sessions before launch. The default k-fold manager run also submits a
lattice stage: all 2ⁿ−1 nonempty combinations of the n configured predictors (255 for eight,
511 for nine), one worker
per held-out partition. Disable this with `--no-precompute-lattice` when invoking
the manager directly, or the equivalent GUI checkbox. Lattice tasks initially
request one CPU, 2 GB, and thirty minutes; override with `--lattice-memory` and
`--lattice-time` after measuring a different workload. Existing manager records
keep their original task set on retry.

Backfill existing runs without refitting predictors:

```bash
python scripts/run_regression_analysis.py lattice \
  --output-dir output/regression_workflow/RUN_ID
python scripts/run_regression_analysis.py scopes \
  --output-dir output/regression_workflow/RUN_ID
python scripts/run_regression_analysis.py overlap \
  --output-dir output/regression_workflow/RUN_ID
```

`--partition-digest DIGEST` narrows a backfill. Each analysis writes immutable
specifications under `analyses/<digest>/`, with distinct execution attempts and
per-partition progress. `--attempt NAME` requests a separate rerun. Scientific
failures remain visible and are not silently retried. A subset fit always
includes the full model, its intercept-omitted model, and one model omitting
each selected predictor. Existing combined, route-only, and MLMDP-only artifacts
keep their original paths and report names.

In Predictor combinations, check the predictors and click Generate plots to
render both likelihood and accuracy figures in the browser. Change the checks
or optional session/location filters and click again to regenerate the figures
from saved scores. For an older k-fold run with saved features, the Results GUI
can submit the existing lattice backfill to SLURM and show its progress; this
computes all 2ⁿ−1 combinations per partition, then the plot button reads the
saved numerical results. Missing scope or overlap calculations still show their
CLI backfill commands. Add `--cancel` to an
analysis command to cancel only the recorded active jobs for that kind and
attempt. The workflow Progress page offers retry and cancellation for jobs
recorded by the workflow manager.

The same action also renders two absolute held-out-performance figures below
the ablation figures: mean log likelihood in nats/decision and action accuracy.
Each shows the selected predictors' singleton regressions followed by a
dark-purple **Combined** category for the regression containing the full
selected set. It uses the same subject circles and mean diamond as the
singletons, with a bold axis label. A one-predictor selection omits this
duplicate combined point. A singleton retains Qin's standard intercept and
impossible-action mask. These figures read the saved full-model scores and do not use
full-minus-reduced contrasts. Each includes the uniform-over-valid-actions
reference computed on the same evaluation locations. Folds are pooled by test
decision count, sessions are averaged equally within subjects, and subjects are
averaged equally for the displayed group mean and SEM. No new fitting is run.

The absolute figures are an unbiased held-out evaluation only when the
predictor combination was specified independently of these scores. Choosing a
combination after inspecting its absolute or ablation plots makes the resulting
comparison exploratory/model-selective rather than a final untouched test.

The location switch changes held-out **evaluation**, not fitting. Both the
all-decision and three-or-four-valid-action regression scores reconstruct the
saved fold before scoring selected decisions. A fold with no selected decisions
is not applicable. Policy overlap uses its own projection: rows with undefined
policies remain excluded, and selecting matrix columns does not change the
common-valid row population. Existing overlap artifacts use the three-or-four
valid-action population. Any newly requested all-location overlap is saved as a
separate analysis.

Progress has an execution state and a scientific completeness state. Finished
jobs may still leave a partial scientific grid, in which case group estimates
remain withheld by the existing reporting rules. On an ambiguous submission
failure, inspect the recorded job IDs in the run manager and check them with
`squeue` or `manage_regression_workflow.py --status` before retrying.
