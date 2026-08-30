# AGENTS.md — scientific model code

Applies to `src/andrew_mlmdp/`. Follow the repository-root `AGENTS.md` as
well. This file contains package-specific rules; do not repeat it in nested
instructions unless a narrower rule differs.

## Mathematical transparency

Keep implementations close to the mathematical objects they represent.

The repository uses these conventions:

- `P[next_state, current_state]` for transition matrices; valid probability
  columns sum to one;
- `D[state, component]` for subgoal profiles; and
- paired NMF factors `D @ W`, with task weights in the rows of `W`.

Document non-obvious shapes, orientations, indexing, and broadcasting at the
point where they matter. Use explicit intermediate quantities when they make
an equation easier to audit. Avoid clever vectorization or implicit
broadcasting when it hides the calculation.

Comments should explain scientific intent, mathematical conventions, or
non-obvious numerical choices. If an optimized implementation is less direct,
retain a concise derivation, reference calculation, or parity test.

## Scientific contracts

Treat these as model semantics rather than implementation details:

- maze coordinates, row-major state IDs, and graph adjacency;
- transition orientation and stochasticity;
- reward, value, and desirability transformations;
- reward gauges and profile normalization gauges;
- exact-zero and impossible-event behavior;
- task-library and basis orientation;
- controller modes, abstract access, and termination semantics; and
- trajectory likelihood and held-out scoring definitions.

Before changing one, inspect the relevant code, tests, and `docs/model.md`.
Update all three when the scientific contract intentionally changes.

Do not introduce a normalization merely for convenience. Preserve immutable
scientific source data, including stored profiles, access profiles, and task
libraries; goal-conditioned calculations must not mutate reusable inputs.

## Numerical integrity and precision

Do not inject epsilon probability or reconstruction mass merely to avoid
zeros, infinities, or optimizer difficulties.

If an event is impossible or a strict KL term is infinite, preserve that
scientific semantics and report or classify the condition explicitly.
Stabilization used inside an optimizer must not silently become the reported
scientific score.

Preserve float64 for model solves, differentiable likelihoods, and fitting.
A lower-precision path requires explicit numerical-parity tests on the
production-relevant workload and must keep failure semantics unchanged.

Do not replace a failed solve with a pseudoinverse, least-squares result,
clipped value, or passive-policy fallback unless that behavior is part of the
documented model contract and is tested directly.

## Optimizers and autodiff

When fitting behaves unexpectedly, distinguish among:

- structural or model infeasibility;
- invalid parameter-domain values;
- bad initialization or zero-locking;
- optimizer failure or failure to converge;
- distinct local optima;
- parameter non-identifiability; and
- ordinary numerical failure.

Do not infer structural infeasibility from optimizer failure without a direct
structural test. An exact differentiable likelihood does not imply identifiable
parameters.

Keep restarts and fallbacks deterministic from recorded seeds and visible in
diagnostics. Give comparable candidates comparable optimization effort unless
the difference is part of a documented procedure.

Avoid maintaining independent mathematical formulas for NumPy and Torch paths
when one shared implementation is practical. Where separate paths exist, add
parity tests for values, gradients when relevant, and failure behavior.

## NMF and basis discovery

Keep raw optimization loss distinct from normalized reporting metrics and
strict final scientific scoring.

Preserve the configured profile gauge exactly. When rescaling a column of
`D`, compensate the corresponding row of `W` so that `D @ W` is unchanged.

Record restart seeds, mask/refit outcomes, convergence, fallback attempts, and
the criterion used to select the retained candidate.

Connectivity, smoothing, sparsity, gating, and other structural restrictions
are scientific assumptions. Keep them explicit and test or quantify their
effect. Forbidden factor entries must remain exact zeros.

## Hierarchical execution and likelihood

Changes to first-hit dynamics, task composition, controller marginalization,
first-departure calculations, abstract access, termination, or trajectory
likelihood require small exact cases or analytical parity checks in addition
to regression tests.

Preserve mutually exclusive latent routes when marginalizing controller
events. Check probability conservation and impossible-route behavior, not
only the final scalar likelihood.

Do not replace exact marginalization or occupancy solves with an approximation
for speed unless the approximation is an explicitly requested model variant
with separate diagnostics and tests.

## Public API and verification

Keep package exports deliberate. When changing a public object, inspect
`src/andrew_mlmdp/__init__.py`, public-API tests, and user-facing documentation.

Run the nearest relevant tests first. Useful routes include:

- flat LMDP behavior: `pytest tests/test_lmdp.py`;
- hierarchy equations and semantics: `pytest tests/test_hierarchy_equations.py
  tests/test_hierarchy_model.py`;
- hierarchy likelihood: `pytest tests/test_hierarchy_likelihood.py`;
- fitting: `pytest tests/test_flat_fitting.py tests/test_hierarchy_fitting.py`;
- NMF discovery: `pytest tests/test_discovery.py`; and
- public contracts: `pytest tests/test_public_api.py
  tests/test_documentation.py`.

Add the smallest test that establishes the scientific invariant, then run the
broader affected group. Run the full suite when shared equations, public APIs,
or cross-module semantics change.

## Performance changes

Do not change model semantics during a performance refactor. Profile the real
fitting or evaluation path, establish a numerical baseline, verify forward and
backward parity when applicable, and then benchmark the same workload.

Follow `benchmarks/AGENTS.md` for benchmark methodology. After a substantial
optimization, re-profile before choosing the next target.
