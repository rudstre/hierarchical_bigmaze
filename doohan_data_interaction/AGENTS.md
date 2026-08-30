# AGENTS.md — Doohan data analysis

Applies to `doohan_data_interaction/`. Follow the repository-root `AGENTS.md`
as well.

These notebooks and scripts are scientific analyses and diagnostic records.
Keep the path from source-data selection to scientific conclusion explicit and
auditable.

## Role of notebooks and helpers

Use notebooks to select and inspect data, configure analyses, call tested
library code, examine diagnostics, compare models, generate figures, and
record scientific reasoning.

Reusable model equations, likelihoods, optimizers, validation rules, and
numerical logic belong in `src/andrew_mlmdp/` with tests. Do not duplicate
substantial model logic in notebook cells.

Small notebook-specific orchestration, serialization, and cache helpers may
remain in this directory when they keep cells declarative and are covered by
focused tests. `fit_workflow.py` is such an analysis helper, not an alternative
implementation of the hierarchy.

## Data selection and trial semantics

For model analyses, prefer `DoohanDataset.from_data_root`. It preserves typed
session metadata, validates that the final selection uses exactly one maze,
and retains data-extraction exclusions.

Make these choices visible near the start of an analysis:

- external data root and relevant data revision;
- subject and session IDs or inclusive date range;
- `maze_name`;
- trial phase and any additional filters; and
- resulting session, trial, transition, and exclusion counts.

Do not independently recode goals, trim trajectories, or reconstruct movement
trials in a notebook when the dataset loader already defines those semantics.
Changes to trial extraction belong in the library with loader tests.

Keep data-extraction exclusions separate from model-scoring exclusions.
Invalid records may be excluded with explicit reasons; a valid trajectory that
is impossible under a model must retain its model-defined score rather than be
reclassified as invalid data.

The raw inspection scripts may use the external GridMaze checkout directly to
inspect one session or trial. They must not become an undocumented parallel
data-processing path for model fitting or comparison.

## Statistical separation

State the role of every data partition: training, validation/model selection,
final held-out evaluation, or exploratory/in-sample analysis.

Do not use held-out sessions to choose fitted parameters, Adam or NMF
restarts, preprocessing, optimizer settings, ranks, or model variants unless
the stated procedure assigns those sessions a validation role.

If no held-out evaluation exists, label reported results as exploratory or
in-sample and do not imply out-of-sample performance.

When fitting uses a subset, record the inclusion rule and the IDs and counts of
omitted trials. In particular, filtering trials because an initial model state
is nonfinite must remain visible in summaries and must not be described as a
fit to the complete dataset.

When comparing ranks, folds, sessions, or models, retain the expected grid of
results. Mark failed, unavailable, and missing entries explicitly rather than
dropping them before aggregation.

Define every reported likelihood normalization, including whether it is a
total, per trial, per transition, or another aggregation.

## Provenance and reproducibility

For expensive or decision-relevant analyses, retain enough information to
recover the result, including as applicable:

- exact session and trial selection plus exclusions;
- train, validation, and held-out assignments;
- rank, basis, gauge, gating, composition, and controller settings;
- random seeds, restart configurations, and selected restart;
- optimizer configuration, termination status, and diagnostics;
- initial, best, and final fitted parameter values;
- external-data, configuration, cache, and relevant source signatures; and
- library/runtime versions when they could affect the result.

Save the selected fitted parameters or basis, not only the seed from which
they were produced. A seed alone is not sufficient provenance.

## Caches and expensive results

Never silently reuse a result whose data selection, scientific configuration,
or relevant implementation is incompatible.

Cache payloads should store the full human-auditable specification as well as
its digest. Validate the signature on load. When the payload schema,
serialization meaning, fitting procedure, or relevant source behavior changes,
update the cache version or signature inputs so stale results cannot match.

Treat missing, failed, truncated, nonfinite, and incompatible cache entries as
distinct states. Do not silently recompute on fewer trials or accept the best
remaining rank or restart after expected entries fail.

Use atomic writes for expensive results. Preserve restart-level diagnostics
even when only the winning fit is materialized for later analysis.

Do not edit cached numerical results by hand. Derived tables and figures must
be reproducible from their stored source result.

## Diagnostics and interpretation

Unexpected diagnostics are evidence to investigate, not nuisances to remove.
Keep optimizer failures, fallbacks, nonfinite states, unavailable ranks,
selection-boundary behavior, and excluded trials visible.

Separate what a diagnostic measured from the proposed explanation. When new
evidence changes an earlier interpretation, revise or clearly supersede the
old notebook commentary so contradictory conclusions do not remain presented
as current.

## Figures

Generate figures from stored or computed numerical results rather than
manually copied values.

Make axes, units, likelihood normalization, aggregation, sample counts, and
uncertainty definitions explicit. Identify training versus held-out quantities
in labels or captions when relevant.

Do not smooth, omit, clip, or rescale results merely to make a plot cleaner.
Document scientifically defined transformations and retain access to the
untransformed values.

## Notebook editing and verification

Keep notebooks readable from top to bottom. Put imports, paths, selections,
and configuration before dependent calculations, and minimize hidden state.

Preserve stable cell IDs when editing existing cells; notebook tests use them
to identify important sections. Avoid unrelated metadata, execution-count, and
output churn.

Expensive Adam-fit and parameter-sweep cells are intentionally stored
unexecuted with empty outputs. Do not execute or persist their outputs merely
to validate an unrelated notebook edit. If a task requires running one, report
the exact cell or workflow, inputs, cache behavior, and result separately.

After notebook edits, compile all affected code cells and execute the safe
affected section when practical. Run:

```bash
pytest tests/test_notebook.py -k doohan
pytest tests/test_doohan_fit_workflow.py
```

For changes to Doohan selection, extraction, exclusions, or reporting, also
run:

```bash
pytest tests/test_doohan_dataset.py tests/test_dataset_likelihood.py
```

State which cells or tests were actually executed and which expensive analyses
remain unverified.
