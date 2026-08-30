# AGENTS.md

## Purpose

This is a scientific research codebase for hierarchical linearly-solvable MDP
models, subgoal discovery, behavioral likelihood fitting, and validation on
maze-navigation data.

Optimize for scientific correctness, auditability, reproducibility, and
clarity to a researcher returning to the code months later.

This is not a production software platform. Prefer simple, explicit research
code over generalized or speculative architecture.

## Instruction scope

This file applies repository-wide.

Before working in a subtree, check for a more specific `AGENTS.md` in or above
the target directory. More specific instructions supplement or override this
file for files in their scope.

## Repository map

- `src/andrew_mlmdp/`: core mathematical and model implementation.
- `tests/`: numerical, behavioral, API, and regression tests.
- `benchmarks/`: profiling and performance experiments.
- `doohan_data_interaction/`: real-data notebooks and diagnostics.
- `docs/model.md`: mathematical documentation and model conventions.
- `README.md`: project overview and basic usage.

Before changing mathematical behavior, read the relevant implementation,
tests, and `docs/model.md`. If they disagree, identify the discrepancy rather
than silently choosing one interpretation.

## Development commands

- Install the library and test dependencies:
  `python -m pip install -e ".[test]"`
- Include notebook dependencies:
  `python -m pip install -e ".[test,notebook]"`
- Run a targeted test:
  `pytest path/to/test_file.py`
- Run the full test suite:
  `pytest`
- Run repository lint checks:
  `ruff check .`

Use Python 3.11 or newer. Consult scoped instructions for benchmark and
data-analysis commands.

## Research-code philosophy

Prefer the simplest implementation that makes the scientific calculation clear
and correct.

Avoid unnecessary abstraction layers, frameworks, configuration machinery,
base classes, indirection, and extensibility for hypothetical future use.

A little repetition is preferable to an abstraction that obscures the
calculation. Introduce a helper or dataclass when it represents a real concept
or materially reduces duplication or error.

Keep public APIs small. Use descriptive scientific names and explicit
intermediate quantities where they make equations or numerical logic easier
to inspect.

Comments should explain why a calculation, constraint, normalization, or
numerical treatment exists. Do not narrate obvious Python.

## Evidence before change

For bugs, numerical anomalies, optimizer behavior, performance problems, or
unexpected scientific results:

1. Separate the observation from the proposed explanation.
2. Inspect the production-relevant code path.
3. Design and run the smallest test that distinguishes plausible explanations.
4. Resolve conflicting evidence before changing behavior.
5. Do not change defaults based on one unvalidated diagnostic or benchmark.

If an earlier conclusion was wrong, state what was measured, why it was
misleading, and which conclusions are affected.

When explicitly asked for a plan, inspect first and return a plan only. Do not
implement until requested.

## Numerical and scientific integrity

Treat numerical conventions as part of the scientific model.

Do not add probability mass, clipping, normalization, regularization, fallback
behavior, or silent approximations merely to make code run.

When such treatment is required, state why, keep it narrow, expose it when
scientifically relevant, and test its effect.

Preserve exact zeros and impossible-event semantics where required.

Distinguish model infeasibility, optimizer failure, and ordinary numerical
failure. Do not silently drop trials, states, transitions, sessions, or failed
fits from scientific comparisons.

Do not use validation or held-out data to choose parameters, restarts,
preprocessing decisions, or model variants unless the procedure explicitly
defines that data as training data.

## Reproducibility

Give every stochastic scientific computation an explicit, recorded seed.

Important fitted or discovered results should retain enough provenance to
identify their configuration, data selection, structural choices, seed or
restart, and relevant optimizer diagnostics.

Do not assume that a random seed guarantees bit-identical results across
environments.

Caches must fail safely. Do not reuse results when the scientific
configuration, selected data, or relevant code is incompatible. Prefer atomic
writes for long-running result shards and caches.

## Testing and performance

Run the smallest relevant tests first, followed by the broader suite when the
change affects shared model semantics, public APIs, or numerical utilities.

Test scientific and numerical invariants rather than only implementation
details. Do not weaken a test unless the scientific contract intentionally
changed.

Profile before optimizing. Validate performance changes on the
production-relevant dtype, workload, and code path, and verify numerical parity
before accepting a speedup.

Detailed testing and benchmarking requirements belong in the relevant scoped
`AGENTS.md`.

## Scope discipline

Make the smallest change that solves the requested problem.

Do not refactor, rename, or clean up adjacent code unless necessary for the
task.

Preserve unrelated dirty-worktree changes. Do not reset, overwrite, or revert
work that is not yours.

Do not commit, create branches, or rewrite Git history unless explicitly
requested.

## Completion report

At the end of a coding task, state concisely:

- what changed;
- what was verified;
- tests or benchmarks run and their results;
- remaining uncertainty or limitations.

Clearly distinguish verified facts from interpretation or recommendation.