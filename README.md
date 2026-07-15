# Two-Layer Multitask LMDP Maze Navigation

This project will reproduce the core ideas from Saxe, Earle, and Rosman
(2017), *Hierarchy Through Composition with Multitask LMDPs*, in a discrete
maze-navigation setting. The first objective is a transparent, exact
implementation of flat and two-layer linearly solvable Markov decision
processes (LMDPs). Once the exact mathematics is validated, the project will
reproduce the paper's qualitative flat-versus-hierarchical z-iteration
learning result in a four-room maze.

The implementation is intentionally staged. Exact flat solutions provide an
oracle for debugging task composition and the hierarchy. Learning experiments
come later, after the transition conventions, state abstraction, and
inter-layer communication are covered by tests.

## Current Implementation

The first physical-maze component is implemented in `andrew_mlmdp.maze`. Maze
geometry lives in separate text files and contains only walls and free cells.
Goals and subgoals will be assigned later by the MLMDP code, not by maze
construction.

```python
from andrew_mlmdp import (
    Maze,
    controlled_dynamics,
    desirability_grid,
    plot_controlled_dynamics,
    solve_desirability,
)

maze = Maze.from_file("mazes/four_rooms.txt")
goal = (10, 9)
z = solve_desirability(maze, goal)
z_grid = desirability_grid(maze, z)
controlled = controlled_dynamics(maze, z)
ax = plot_controlled_dynamics(maze, controlled, goal=goal)
```

The code deliberately uses direct loops and standard-library data structures
so coordinate conventions, state ordering, and obstacle handling remain easy
to inspect during research.

The current examples can also be run interactively in
`notebooks/flat_lmdp_examples.ipynb`:

```shell
uv sync --extra notebook
uv run jupyter lab notebooks/flat_lmdp_examples.ipynb
```

## Core Model

### First-exit LMDP

An LMDP is specified by a state space `S`, passive transition dynamics `P`, a
state reward `r`, and a control-cost temperature `lambda`. Control selects a
next-state distribution `a(.|s)` and receives

```math
r(s, a) = r(s) - \lambda\,\mathrm{KL}(a(\cdot|s)\,\|\,P(\cdot|s)).
```

This project uses first-exit tasks. States are partitioned into interior states
`I` and absorbing boundary states `B`. An episode ends after entering a
boundary state.

Define desirability and exponentiated reward as

```math
z(s) = \exp(V(s)/\lambda), \qquad q(s) = \exp(r(s)/\lambda).
```

The matrix convention throughout the project is

```text
P[next_state, current_state] = P(next_state | current_state)
```

so every column of `P` sums to one. With `M_i = diag(q_i)`, the interior
desirability is the solution of

```math
(I - M_i P_{II}^{T}) z_i = M_i P_{BI}^{T} z_b,
```

where `z_b = q_b`. The initial exact implementation uses a direct dense NumPy
solve so the equation remains visible in the code. The corresponding
synchronous z-iteration is

```math
z_i \leftarrow M_i P_{II}^{T} z_i + M_i P_{BI}^{T} z_b.
```

The optimal controlled transition distribution is available in closed form:

```math
a^*(s'|s) =
\frac{P(s'|s)z(s')}{\sum_y P(y|s)z(y)}.
```

All solver and policy code will preserve this matrix orientation explicitly.
Tests will reject transposed or non-stochastic transition matrices.

### Multitask composition

Component tasks share the same state space, passive dynamics, and interior
rewards, but differ in exponentiated boundary rewards. Their boundary vectors
form a task basis

```math
Q_b = [q_b^1\;q_b^2\;\cdots\;q_b^K].
```

Solving each component task gives the desirability basis

```math
Z_i = [z_i^1\;z_i^2\;\cdots\;z_i^K].
```

If a target boundary task is represented by `q_b = Q_b w`, linearity gives its
optimal desirability immediately as `z_i = Z_i w`. For approximate tasks, the
paper computes

```math
w = Q_b^\dagger q_b
```

and clips negative weights to zero. That pseudoinverse-and-clipping procedure
will be the paper-faithful default. A constrained least-squares implementation
may be added later as a diagnostic, but will not silently replace the default.

## Maze Adaptation

### Physical maze

The first benchmark is a hand-authored four-room grid stored in
`mazes/four_rooms.txt`. It contains only `#` walls and `.` free cells and
approximates the obstacle topology in the paper's rooms experiment rather than
claiming a pixel-exact reconstruction of an unpublished grid specification.

Physical states are traversable grid cells. The passive process samples
uniformly from five commands:

- north
- south
- east
- west
- stay

Moving into a wall or outside the grid returns the agent to its current cell.
Multiple invalid commands can therefore increase the passive self-transition
probability at corners and walls. Rollouts sample next physical states from the
controlled distribution rather than taking an `argmax`, preserving the LMDP's
stochastic policy semantics.

### Goal and subgoal roles

Goal and subgoal semantics belong to the future MLMDP model, not to `Maze` or
the geometry file. The MLMDP configuration will select two disjoint sets of
free physical states:

- **Candidate goals:** nodes from which a task selects exactly one current goal.
- **Subgoals:** reusable access points, normally placed at doors or useful
  junctions.

Layer 1 precomputes one component desirability for every candidate goal and
every subgoal. To keep physical subgoal and goal cells traversable, basis
targets are represented by abstract terminal copies attached to their physical
cells. Entering a subgoal copy invokes layer 2 without making the corresponding
physical cell permanently absorbing. A task enables all subgoal copies and its
selected goal copy; non-current goal copies are excluded from that task's
active abstract model.

The complete precomputed basis still has a fixed column order:

```text
[candidate goal copies..., subgoal copies...]
```

Restricting to one task selects columns and active abstract states; it does not
change node identifiers or mutate the stored basis.

### Abstract-target access dynamics

Layer 1 augments the physical passive dynamics with transitions from configured
target locations to their abstract copies. `P_b` contains access to candidate
goal copies and `P_t` contains access to subgoal copies. The default abstract
access probability is `alpha = 0.1`; inactive goal-copy rows are removed when a
task is selected. The physical, goal-boundary, and subgoal-access blocks are
stacked and each current-state column is renormalized:

```math
\widetilde P^1 = \mathcal N([P_i^1; P_b^1; P_t^1]).
```

Abstract-copy transitions do not consume physical time. A goal-copy transition
terminates the task. After a subgoal-copy transition, layer 2 supplies new
guidance and execution resumes from the associated physical cell.

## Two-Layer Architecture

For a task with selected goal `g`, the layer-2 state space is

```text
all configured subgoals + g
```

Subgoals are layer-2 interior states and `g` is its absorbing boundary. Other
candidate goals are not layer-2 states for this task.

### Deriving layer-2 passive dynamics

Layer 2 must inherit reachability from the physical maze rather than using a
fully connected or distance-only graph. Let

```math
F = (I - \widetilde P_i^1)^{-1}
```

be the layer-1 fundamental matrix. Following equations 8 and 9 of the paper,
the abstract transition blocks are derived from lower-layer first-passage
probabilities:

```math
P_{II}^2 = \widetilde P_t^1 F \widetilde P_t^{1T},
\qquad
P_{BI}^2 = \widetilde P_b^1 F \widetilde P_t^{1T}.
```

The implementation will compute these with linear solves rather than forming a
dense inverse. It will then select the current goal, remove inactive goal
copies, and renormalize each surviving layer-2 column. Zero-mass columns are a
configuration error, since they indicate an unreachable abstract state.

Layer-2 interior rewards use the same small negative step cost as the initial
paper replication. A later extension may accumulate expected layer-1 reward
along abstract transitions.

### Top-down reward inpainting

Solving the task-specific layer-2 LMDP yields `z2` and controlled dynamics
`a2`. At a current abstract state, layer 2 communicates its preference over
subgoals to layer 1 using the paper's controlled-minus-passive signal:

```math
r_t^1 = \beta\left(a_i^2(\cdot|s) - P_i^2(\cdot|s)\right),
```

where `beta = 1` is the initial reward-inpainting scale because the paper only
specifies proportionality. This reward is exponentiated, projected into the
active layer-1 task basis, and composed:

```math
q_t^1 = \exp(r_t^1/\lambda),
\qquad
w^1 = \max(0, (Q_b^1)^\dagger q_t^1),
\qquad
z^1 = Z^1 w^1.
```

The goal-task component remains available in the active basis so the hierarchy
can blend direct goal attraction with several subgoal policies. Every inpainted
reward vector, task-weight vector, and composed desirability can be recorded for
diagnostic plots.

### Starts outside layer 2

An episode may begin at any free physical cell rather than at a configured
subgoal. The planner will compute the first-hit distribution from that cell to
the active abstract states using the same layer-1 linear system used to derive
layer-2 dynamics. Reweighting this distribution by layer-2 desirability gives
the initial abstract preference and therefore the first layer-1 task blend. No
temporary start node is inserted into the persistent layer-2 model.

### Execution cycle

1. Select a candidate goal and construct its active layer-2 model.
2. Compute initial abstract guidance from the physical start state.
3. Compose the layer-1 desirability and controlled policy.
4. Sample physical transitions until the goal or a subgoal access copy is hit.
5. If a subgoal copy is hit, update the layer-2 state, recompute inpainted
   rewards and task weights, and resume from its physical cell.
6. If the selected goal copy is hit, terminate the episode.

A rollout has a configurable step cap and returns an explicit `reached_goal`,
`step_limit`, or `unreachable` status. Numerical failures and invalid
probability vectors are errors rather than implicit rollout termination.

## Implementation Roadmap

### Milestone 1: Flat exact LMDP

Implement maze parsing and indexing, passive dynamics, state partitions,
boundary rewards, exact dense desirability solving, synchronous z-iteration,
controlled policies, and seeded stochastic rollouts.

Acceptance criteria:

- transition columns are stochastic and encode only legal outcomes;
- exact solves have a small linear-system residual;
- policies are normalized wherever a valid continuation exists;
- z-iteration converges to the exact solution on line and small-grid fixtures;
- rollouts reach an absorbing target without crossing walls.

### Milestone 2: Multitask composition

Construct `Q_b` and `Z` for all goal and subgoal copies. Support exact task
composition and the paper's approximate pseudoinverse projection.

Acceptance criteria:

- a boundary task in the span of `Q_b` matches a direct exact solve;
- nonnegative component mixtures remain positive and produce normalized
  policies;
- task and basis ordering is deterministic and preserved through serialization.

### Milestone 3: Exact two-layer planner

Add subgoal access copies, first-passage abstraction, task-specific layer 2,
reward inpainting, task-weight projection, arbitrary-start initialization, and
hierarchical rollouts.

The exact flat solution is the debugging oracle. A hierarchy is not expected to
outperform a flat planner that has already solved the same task exactly. At this
stage it is evaluated on mathematical consistency, successful abstraction, and
correct execution.

Acceptance criteria:

- analytical layer-2 transitions agree with Monte Carlo first-hit estimates;
- unreachable subgoals are detected during model construction;
- subgoal access triggers layer 2 without advancing physical time;
- hierarchical rollouts reach the selected goal from multiple rooms;
- logged task weights show concurrent use of multiple component tasks in at
  least one symmetric or ambiguous route.

### Milestone 4: Paper-style learning curves

After exact validation, reproduce the qualitative experiment in Figure 3b of
the paper. Save the converged exact solution as an oracle, then reset a learner
copy of the selected goal desirability to its boundary initialization. Keep the
reusable subgoal basis and layer-2 model preinitialized. Compare:

- **Flat:** the evolving goal desirability alone controls the rollout.
- **Hierarchical:** the same evolving goal desirability is assisted by fixed
  subgoal components selected through layer 2.

One epoch is one synchronous full-state z-iteration update. Evaluation rollouts
do not update parameters. Conditions use identical starts and random-number
seeds, trajectories are capped at 500 physical steps, and results are aggregated
over 20 seeds as mean trajectory length with a 95% confidence interval.

This experiment concerns convergence under z-iteration. Sample-based or
temporal-difference z-learning is outside the initial replication and must be
reported separately if added.

Acceptance criteria:

- both conditions receive identical update and evaluation budgets;
- raw per-seed data and exact run configuration are saved;
- hierarchy-assisted rollouts show improved early performance without being
  presented as superior to the converged exact oracle;
- both conditions approach stable performance as desirability converges.

### Milestone 5: Research outputs

Generate reproducible figures for:

- maze topology, goals, subgoals, and sample starts;
- passive and controlled transition fields;
- flat, basis, and composed desirability heatmaps;
- the layer-2 transition graph;
- hierarchical sample paths;
- task weights and reward inpainting over time;
- flat versus hierarchy-assisted learning curves.

Figures and numerical results are generated artifacts and will not be committed
unless explicitly curated for a report.

## Planned Package Structure

```text
andrew_mlmdp/
|-- README.md
|-- pyproject.toml
|-- mazes/
|   `-- four_rooms.txt
|-- notebooks/
|   `-- flat_lmdp_examples.ipynb
|-- src/
|   `-- andrew_mlmdp/
|       |-- maze.py
|       |-- lmdp.py
|       |-- multitask.py
|       |-- hierarchy.py
|       |-- rollout.py
|       `-- plotting.py
|-- experiments/
|   |-- plot_flat_policy.py
|   |-- four_rooms_exact.py
|   `-- four_rooms_learning.py
`-- tests/
    |-- test_maze.py
    |-- test_lmdp.py
    |-- test_plotting.py
    |-- test_multitask.py
    `-- test_hierarchy.py
```

The initial runtime is Python 3.11 or newer with NumPy and Matplotlib. Tests use
pytest. Packaging and commands are managed through `pyproject.toml`.

## Planned Interfaces

The public data structures should be typed, immutable where practical, and
explicit about state ordering.

### `Maze`

Owns only grid geometry: walls, free cells, coordinate/index mappings, and
passive command outcomes. It loads `#` and `.` geometry from a text file and
has no knowledge of tasks, goals, subgoals, rewards, or hierarchy.

### Flat LMDP functions

`build_passive_dynamics` constructs the column-stochastic random walk directly
from a `Maze`. `solve_desirability` performs the exact one-goal first-exit
solve. `controlled_dynamics` applies the paper's closed-form next-state control
distribution, and `plot_controlled_dynamics` draws its directional probability
mass directly on the maze. These remain separate functions because they share
no mutable model state.

### `TaskBasis`

Owns ordered target IDs, `Q_b`, `Z_i`, and the shared LMDP definition. It
exposes exact linear composition and approximate target projection while
returning both weights and reconstruction error.

### `TwoLayerMLMDP`

Owns the layer-1 basis, subgoal access mapping, full derived abstract dynamics,
and task-specific layer-2 construction. It exposes initial guidance,
controlled-minus-passive reward inpainting, layer-1 composition, and updates
after subgoal access.

### Experiment records

Experiment configuration includes maze identity, rewards, `lambda`, `alpha`,
inpainting scale, epoch count, rollout cap, evaluation starts, and random seed.
Per-run records include trajectory lengths, termination status, convergence
residuals, task weights, and paths. Configuration and summary results will be
serializable to JSON; numerical arrays will use NumPy files where needed.

## Test Strategy

Unit tests will cover:

- row-major coordinate/index round trips;
- passive self-transitions at walls and corners;
- stochastic columns and valid transition-matrix shapes;
- exponentiated rewards and desirability positivity;
- exact-solve residuals and z-iteration convergence;
- policy normalization and boundary absorption;
- exact multitask composition and reconstruction error;
- deterministic basis ordering and task restriction;
- arbitrary-start first-hit distributions;
- unreachable and disconnected maze components.

Integration tests will cover:

- analytical versus Monte Carlo abstract transition probabilities;
- hierarchy access and return semantics;
- legal four-room hierarchical paths for multiple starts and goals;
- reproducibility under fixed seeds;
- equal learning budgets and evaluation schedules across experimental
  conditions.

## Initial Defaults and Boundaries

- Interior reward: a small negative step cost, initially `-0.1`.
- Goal reward: `1.0`.
- Control temperature: `lambda = 1.0`.
- Abstract target-access probability: `alpha = 0.1`.
- Reward-inpainting scale: `beta = 1.0`.
- Passive commands: north, south, east, west, and stay with equal command
  probability before invalid moves collapse into self-transitions.
- Candidate goals are selected only from the configured goal set.
- Candidate goal and subgoal sets are disjoint.
- Layer-1 basis desirabilities are precomputed for their union.
- The first benchmark has four rooms and six hand-selected subgoals.
- The paper's PDFs are research references and are not copied into this project.

These values will be configuration parameters even when the initial experiments
use the defaults. Any deviation used in a figure must be recorded with its raw
results.

## Reference

Andrew M. Saxe, Adam C. Earle, and Benjamin Rosman. "Hierarchy Through
Composition with Multitask LMDPs." *Proceedings of the 34th International
Conference on Machine Learning*, PMLR 70, 2017.

The accompanying supplementary material is particularly relevant for the
hierarchical execution pseudocode and the `alpha = 0.1` abstraction example.
