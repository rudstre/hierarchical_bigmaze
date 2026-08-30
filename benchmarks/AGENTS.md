# AGENTS.md — benchmarks and profiling

Applies to `benchmarks/`. Follow the repository-root `AGENTS.md` as well.

Benchmarks are scientific evidence used to justify code changes. Validate the
measurement before explaining or acting on it.

## Define the claim

State what a benchmark measures and what conclusion it can support.

Distinguish among:

- an isolated kernel or microbenchmark;
- one forward likelihood evaluation;
- forward plus backward/autograd;
- optimizer-step throughput;
- a complete fit, including restarts and scheduling; and
- end-to-end analysis, including data loading and reusable setup.

Do not describe faster likelihood evaluation as faster fitting unless the
optimizer path was measured. Do not recommend a production change from a
single microbenchmark.

Benchmarks must call the production implementation for the operation being
claimed. Label synthetic, reduced, or reference calculations explicitly.

## Production relevance

Whenever practical, benchmark the production-relevant path using the real:

- dataset selection and trial count;
- workload sizes, shapes, and batching structure;
- dtype and device;
- active and frozen parameter set;
- controller, likelihood, and composition modes;
- forward, backward, optimizer, or evaluation path; and
- thread, process, and accelerator configuration.

Record deliberately reduced workloads and explain which production costs they
omit. Use them for diagnosis, not as sufficient evidence for changing a
default.

The current hierarchy benchmark is run with:

```bash
python benchmarks/benchmark_torch_hierarchy.py --repeats 5
python benchmarks/benchmark_torch_hierarchy.py --repeats 5 --profile
```

It measures exact full-batch Torch hierarchy likelihood forward and backward
passes without fitting. It depends on the configured external Doohan dataset.

## Correctness before timing

Before comparing performance, verify that baseline and candidate perform the
same scientific computation.

Compare, as applicable:

- scalar loss or total log likelihood;
- intermediate outputs relevant to the changed code;
- gradients for every active parameter;
- failure and impossible-event behavior; and
- included trials, goals, controller states, and operators.

Use tolerances justified by the production dtype and calculation. Report the
observed difference as well as the tolerance. Do not accept a speedup whose
numerical discrepancy has not been explained.

## Timing methodology

For every comparison:

1. Configure the runtime before warm-up.
2. Warm each configuration through the complete timed path.
3. Run enough steady-state repetitions to reveal variability.
4. Use identical timing boundaries and equivalent work.
5. Report raw observations or dispersion as well as a summary statistic.
6. Compare absolute time and relative speedup under the same conditions.

Use a fresh autograd graph and parameter state for each backward repetition.
Do not accidentally include gradient accumulation, graph retention, or
different materialization work in only one variant.

Separate one-time data loading, preparation, compilation, discovery, and cache
construction from steady-state runtime unless production pays those costs on
every operation. Report both when setup cost materially affects the conclusion.

After changing Torch thread counts, OpenMP/MKL settings, device state,
compilation, allocators, or caches, warm up again. Never interpret the first
operation after reconfiguration as steady state.

Synchronize accelerators before starting and stopping host-side timers. If a
future benchmark uses CUDA, use explicit synchronization or device-aware
timing rather than `perf_counter` alone.

## Fair comparisons

Run variants under comparable machine load and runtime state. Prefer the same
process when shared state is intentional; prefer fresh processes when testing
startup, compilation, allocator, peak-memory, or thread-configuration effects.

Interleave or randomize variant order when drift could bias the result. Do not
discard slow runs without a documented, externally observable reason.

Record CPU/device identity, Python and library versions, Torch intra-op and
inter-op thread counts, relevant environment variables, and whether the
worktree was dirty.

`resource.getrusage(...).ru_maxrss` is the maximum for the process lifetime.
Treat it as a process-level upper bound, not a per-stage memory measurement.
Use comparable fresh processes for peak-memory comparisons between variants.

## Conflicting measurements

If a microbenchmark and the production workload disagree, reproduce the
discrepancy and identify differences in code path, shapes, setup, backward
work, synchronization, or runtime state before changing production behavior.

A plausible performance story is a hypothesis. Do not claim a cause until the
measurement distinguishes it from credible alternatives.

If evidence remains ambiguous, report the ambiguity and leave the default
unchanged.

## Profiling

Profile the production-relevant workload before optimizing. Measure both
absolute time and fraction of end-to-end runtime, and estimate the maximum
possible overall speedup before rewriting a small component.

When production uses autograd, inspect forward and backward separately. Check
call counts, shapes, dispatches, and memory as well as aggregate operator time.

After optimizing a bottleneck, rerun the validated benchmark and re-profile
the full workload. Do not assume the original bottleneck remains dominant.

## Benchmark artifacts

Every retained CSV, JSON, profile, or comparison report must identify or be
traceable to:

- the generating script and exact command;
- source revision and relevant dirty-worktree state;
- data selection and configuration;
- seed or restart provenance;
- runtime and hardware settings; and
- metric definitions and units.

Keep raw observations when practical. Do not hand-edit generated numerical
results. If a compact CSV is derived from a detailed JSON result, generate
both from the same source and preserve the derivation.

Do not leave an artifact as evidence after its generating procedure is removed
or no longer reproducible; document it as historical or remove it when asked.

## Benchmark reports

A useful report states:

- the claim, workload, and exact code path;
- timing boundaries and excluded setup;
- runtime, device, thread, and version settings;
- warm-up and repetition procedure;
- baseline and candidate raw timings or variability;
- absolute timing, relative speedup, and end-to-end effect;
- output, loss, and gradient equivalence results;
- profiler evidence supporting any causal interpretation; and
- remaining uncertainty.

For a production optimization, report the end-to-end effect in addition to
any microbenchmark speedup.
