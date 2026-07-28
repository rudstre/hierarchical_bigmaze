# Model and implementation

## State and matrix conventions

`Maze` assigns row-major state IDs to free cells in any rectangular layout.
All transition matrices use:

```text
P[next_state, current_state] = P(next_state | current_state)
```

`LMDPEnvironment` owns the geometry-dependent physical passive matrix. It is
constructed once and reused by flat tasks, NMF goal ensembles, and every
hierarchy built from that environment.

## Flat first-exit tasks

For reward `r`, control cost `lambda`, and desirability `z`:

```math
q(s)=\exp(r(s)/\lambda)
```

The first-exit interior solution is:

```math
(I-\operatorname{diag}(q_i)P_{II}^{T})z_i
=\operatorname{diag}(q_i)P_{BI}^{T}z_b.
```

`environment.solve_flat(goal)` returns a `FlatSolution` containing the full
physical desirability and Equation 6 controlled policy:

```math
a^*(s'|s)=
\frac{P(s'|s)z(s')}{\sum_y P(y|s)z(y)}.
```

For a physical component disconnected from the goal, desirability is zero and
the high-level flat solution retains passive dynamics in otherwise undefined
zero-mass policy columns.

## Unified subgoal basis

`SubgoalBasis` always stores a state-by-subgoal profile matrix.

- `from_locations` creates one-hot columns for fixed point subgoals.
- `from_profiles` peak-normalizes distributed profiles and creates a separate,
  immutable execution view using the requested core gate.

For threshold `tau` and exponent `gamma`, the execution profile is:

```math
\widehat D_{sj} =
\left[\max\left(0,\frac{D_{sj}-\tau}{1-\tau}\right)\right]^\gamma.
```

No hierarchy code branches on a fixed maze size or subgoal count. All physical
and abstract shapes follow the supplied maze and number of profile columns.

## Goal-independent template and goal task

`environment.hierarchy(basis, parameters=...)` returns a
`HierarchyTemplate`. It exposes the task-independent passive subgoal graph and
caches `HierarchyTask` instances by goal.

For a goal-conditioned task, the goal is removed from the lower interior
partition. Profile access rows and the physical-goal boundary row form the
augmented first-exit dynamics. With:

```math
F=(I-\widetilde P_i^1)^{-1},
```

the first-hit probabilities and upper dynamics are:

```math
H=[\widetilde P_t^1;\widetilde P_g^1]F,
```

```math
P_{II}^2=\widetilde P_t^1F\widetilde P_t^{1T},
\qquad
P_{BI}^2=\widetilde P_g^1F\widetilde P_t^{1T}.
```

The reusable task basis stores boundary desirabilities `Q_b` and their solved
interior columns `Z_i`. A plan inpaints the upper controlled-minus-passive
signal, projects it through `Q_b`, clips negative task weights, and composes
the lower policy. When `include_goal_component_while_active=False`, the exact
goal column is disabled until an explicit upper termination.

## Unified rollout and online learning

`HierarchyTask.rollout` runs one event-recording state machine for point or
distributed profiles.

- Physical transitions advance the trajectory and physical clock.
- Lower subgoal accesses invoke the upper layer without moving physical time.
- A one-step refractory period prevents zero-time access loops.
- Upper termination permanently selects the goal-only policy.

With `goal_learning="online"`, the exact goal basis column is replaced by a
learned vector. Each nonterminal physical move performs the requested number
of Equation 5 updates:

```math
z_i \leftarrow q_i(P_{II}^{T}z_i+P_{BI}^{T}z_b).
```

The returned `Rollout` contains the physical trajectory, subgoal accesses,
task-weight history, complete event trace, Z-iteration count, and final learned
goal vector. Passing that final vector into the next rollout continues
learning across episodes.

## NMF discovery

Discovery parameters remain separate from execution parameters. A
`discover_soft_subgoals` call solves the selected flat goal family with the
shared environment, fits every requested rank exactly once, and returns an
`NMFStudy`. Each result factorizes:

```math
Z \approx DW,\qquad D,W\geq0,
```

using KL-NMF. Every `D` column is peak-normalized and its scale is absorbed
into the matching `W` row, preserving the reconstruction. `study.result(k)`
returns the already-fitted rank rather than running NMF again.
