# Maze multitask LMDPs

Readable research code for flat and two-layer multitask linearly solvable
Markov decision processes (MLMDPs) in grid mazes. The implementation works for
arbitrary maze dimensions and numbers of subgoals; the included four-room maze
is an example, not an architectural constraint.

If you are new to this repository, read **MLMDP in 90 seconds** below and then
open the executable
[`notebooks/maze_lmdp_workflows.ipynb`](notebooks/maze_lmdp_workflows.ipynb).
For the complete mathematical and code-level walkthrough, see
[`docs/model.md`](docs/model.md).

## MLMDP in 90 seconds

An ordinary maze policy says where the agent should move next. This MLMDP
instead separates navigation into two coupled levels:

- **Layer 1, physical:** move between maze cells.
- **Layer 2, abstract:** decide which reusable subgoal region should organize
  the next stretch of physical behavior, or whether to finish at the goal.

The key LMDP trick is to solve for a positive **desirability** vector
`z = exp(V / lambda)`. Once `z` is known, the optimal controlled transition
probabilities are obtained by reweighting the passive random walk:

```text
controlled(next | current)
    proportional to passive(next | current) * desirability(next)
```

The hierarchical algorithm follows this pipeline:

```text
maze
  -> physical passive random walk P
  -> add subgoal-access and physical-goal boundary states
  -> compute first-hit probabilities for those boundaries
  -> turn first hits into a small passive LMDP over subgoals
  -> solve the abstract LMDP for the current goal
  -> convert the abstract policy into a desired boundary task
  -> express that task as weights over reusable Layer-1 solutions
  -> compose one physical policy and sample movement
```

More concretely:

1. `LMDPEnvironment` builds the maze's passive physical dynamics once.
2. `SubgoalBasis` represents each point or distributed subgoal as one profile
   column over physical states.
3. `HierarchyTemplate.for_goal` treats subgoal access and the physical goal as
   first-exit boundaries. A fundamental matrix converts the augmented physical
   dynamics into probabilities of hitting each boundary first.
4. Those first-hit probabilities define a small passive process whose states
   are the subgoals and whose sole boundary is the physical goal. Solving this
   upper LMDP produces the goal-directed abstract policy.
5. At the start and after every subgoal access, `HierarchyTask.plan` compares
   the controlled and passive abstract transitions. Reward inpainting turns
   that difference into a desired Layer-1 boundary desirability.
6. The desired task is projected onto a pre-solved basis of lower-level tasks.
   Non-negative basis weights compose a physical desirability and therefore a
   physical controlled policy.
7. A rollout samples that policy. Physical transitions consume time. Entering
   a subgoal boundary invokes Layer 2 without consuming physical time; upper
   termination switches permanently to the goal-only lower policy.

The hierarchy therefore does not choose a symbolic subgoal and run a separate
hand-written controller. At each planning event, its upper policy produces a
mixture of reusable lower desirability functions, and that mixture defines the
physical policy until the next abstract access.

## What is reused, and when?

| Lifetime | Computation | Main object |
| --- | --- | --- |
| Once per maze | Physical passive dynamics | `LMDPEnvironment` |
| Once per discovered/fixed basis | Subgoal profiles and optional access gate | `SubgoalBasis` |
| Once per goal, then cached | Lower first-exit dynamics, first-hit matrix, upper LMDP, and lower task basis | `HierarchyTask` |
| At rollout start and each abstract access | Reward inpainting, task weights, and composed physical policy | `LayerOnePlan` |
| After each physical move in online mode | One or more goal-column Z-iteration sweeps | `Rollout` state |

This separation is central to the multitask behavior: expensive reusable
structure is retained while goals and current abstract commands change.

## Code map

| Question | Start here |
| --- | --- |
| How is the passive maze random walk built? | [`build_passive_dynamics`](src/andrew_mlmdp/lmdp.py) |
| How is a flat first-exit LMDP solved? | [`solve_first_exit` and `LMDPEnvironment.solve_flat`](src/andrew_mlmdp/lmdp.py) |
| How are point and distributed subgoals represented? | [`SubgoalBasis`](src/andrew_mlmdp/hierarchy/core.py) |
| How is a goal-conditioned hierarchy constructed? | [`_build_hierarchy_task`](src/andrew_mlmdp/hierarchy/core.py) |
| How are the lower and upper passive dynamics derived? | [`_build_lower_dynamics_from_access` and `_build_upper_dynamics`](src/andrew_mlmdp/hierarchy/core.py) |
| How does an upper policy become a physical policy? | [`compute_hierarchy_plan`, `_plan_from_abstract_dynamics`, and `_compose_lower_policy`](src/andrew_mlmdp/hierarchy/core.py) |
| What exactly happens during a rollout? | [`_run_hierarchical_rollout`](src/andrew_mlmdp/hierarchy/rollout.py) |
| How are distributed subgoals discovered? | [`discover_soft_subgoals`](src/andrew_mlmdp/discovery.py) |

## Minimal end-to-end example

This uses the same goal and hierarchy tuning as the canonical notebook:

```python
from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    SubgoalBasis,
    hard_hierarchy_parameters,
)

maze = Maze.from_file("mazes/four_rooms.txt")
environment = LMDPEnvironment(maze)
goal = (1, 9)

# A flat LMDP solves directly for this physical goal.
flat = environment.solve_flat(goal)
flat_rollout = flat.rollout((3, 0), seed=0)

# The hierarchy reuses six subgoal task solutions.
subgoals = ((0, 0), (9, 2), (2, 3), (3, 7), (9, 7), (7, 9))
basis = SubgoalBasis.from_locations(maze, subgoals)
hierarchy = environment.hierarchy(
    basis,
    parameters=hard_hierarchy_parameters(upper_control_cost=0.65),
    include_goal_component_while_active=False,
)
task = hierarchy.for_goal(goal)

plan = task.plan((3, 2))
exact = task.rollout((3, 2), seed=0)
online = task.rollout(
    (3, 2),
    goal_learning="online",
    z_sweeps_per_step=1,
    seed=28,
)
```

Useful objects are deliberately inspectable:

- `task.lower_dynamics`: augmented physical first-exit process;
- `task.first_hit_probabilities`: boundary-first-hit probabilities from every
  physical interior state;
- `task.upper_dynamics` and `task.upper_controlled`: passive and controlled
  abstract processes;
- `task.task_basis`: reusable boundary tasks `Q_b` and their solved physical
  desirabilities `Z_i`;
- `plan.weights`: the current mixture of lower component tasks; and
- `exact.events`: the complete physical/abstract rollout trace.

## Flat LMDPs and passive-motion modes

By default, passive motion samples uniformly from north, south, east, west, and
stay. A blocked command becomes a self-transition. To sample uniformly only
from traversable cardinal neighbors:

```python
movement_only = LMDPEnvironment(
    maze,
    passive_mode="valid_neighbors",
)
```

A `FlatSolution` contains its desirability, controlled policy, rollout method,
and `movement_log_likelihood` method for scoring observed discrete movement
trajectories. Consecutive repeated observations are collapsed before scoring.

## Distributed subgoals discovered with NMF

Point subgoals are one-hot profiles. Distributed subgoals use the same
hierarchy engine but can be learned by factorizing a family of flat goal-task
desirabilities:

```python
from andrew_mlmdp import (
    NMFDiscoveryParameters,
    SubgoalBasis,
    discover_soft_subgoals,
    soft_hierarchy_parameters,
)

study = discover_soft_subgoals(
    environment,
    ranks=range(2, 13),
    parameters=NMFDiscoveryParameters(),
    seed=0,
)
rank_eight = study.result(8)  # returns the already-fitted result

soft_basis = SubgoalBasis.from_profiles(
    maze,
    rank_eight.profiles,
    core_threshold=0.8,
)
soft_hierarchy = environment.hierarchy(
    soft_basis,
    parameters=soft_hierarchy_parameters(8, upper_control_cost=0.18),
    include_goal_component_while_active=False,
)
soft_task = soft_hierarchy.for_goal(goal)
soft_rollout = soft_task.rollout((3, 2), seed=0)
```

NMF discovery and hierarchy execution have separate parameters. The original
peak-normalized NMF profiles and their gated access profiles are immutable.
Changing the goal builds or retrieves only a goal-conditioned hierarchy; it
does not rerun NMF or apply the gate again.

Set `lambda_smooth` to a positive value in `NMFDiscoveryParameters` to
penalize neighboring states with different profile values over the
passive-dynamics connectivity graph. The useful scale is specific to the
task ensemble because the optimization uses raw generalized KL. For a
regularized result, `objective_history` contains the initial raw
KL-plus-Laplacian objective followed by one value per iteration, while
`reconstruction_error` remains the normalized KL-only diagnostic. The
default zero strength retains the original scikit-learn solver and leaves
`objective_history` unset.

## Doohan edge-list mazes

The GridMaze data submodule defines mazes as labeled edges between towers.
`load_doohan_maze` keeps the 49 towers as physical states and uses the edge list
to restrict cardinal movement:

```python
from andrew_mlmdp import LMDPEnvironment, load_doohan_maze

definition = load_doohan_maze("maze_1")
environment = LMDPEnvironment(definition.maze)
start = definition.coordinate_for("A2")
goal = definition.coordinate_for("G7")

solution = environment.solve_flat(goal)
trajectory = solution.rollout(start, seed=0)
labels = [definition.label_for(coordinate) for coordinate in trajectory]
```

The exploratory workflow in
[`doohan_data_interaction/doohan_trial_lmdp.ipynb`](doohan_data_interaction/doohan_trial_lmdp.ipynb)
keeps its session, trial, NMF rank, regularization, and execution tuning as
notebook-local choices so they can be changed without redefining defaults.

By default the loader reads
`external/GridMaze-mFC-ephys-DATA/data/experiment_info/maze_configs.json`.
Pass `config_path` explicitly when the downloaded data lives elsewhere.

## Conventions

- Coordinates are `(row, column)` from the upper left.
- Matrices use `P[next_state, current_state]`, so probability columns sum to
  one.
- `profiles[state, subgoal]` describes subgoal membership; point subgoals are
  one-hot columns.
- `access_profiles` may be a core-gated copy of distributed `profiles` and
  determines where abstract access is possible.
- The final boundary row/column in a goal-conditioned hierarchy is always the
  physical goal; the preceding entries are subgoals.

## Install and validate

Python 3.11 or newer is required.

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,notebook]"
pytest -q
ruff check src tests notebooks/maze_lmdp_workflows.ipynb
vulture src/andrew_mlmdp --min-confidence 80
```

The test suite executes the canonical notebook in a clean kernel and exercises
the widget controller, so the full extras are required for complete
validation. See [`docs/four_rooms.md`](docs/four_rooms.md) for the deliberately
different library-default, notebook, and frozen-regression configurations.
