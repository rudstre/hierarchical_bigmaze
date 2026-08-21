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

1. `Environment` builds the maze's passive physical dynamics once.
2. `SubgoalBasis` represents each point or distributed subgoal as one profile
   column over physical states.
3. `Template.task` treats subgoal access and the physical goal as
   first-exit boundaries. A fundamental matrix converts the augmented physical
   dynamics into probabilities of hitting each boundary first.
4. Those first-hit probabilities define a small passive process whose states
   are the subgoals and whose sole boundary is the physical goal. Solving this
   upper LMDP produces the goal-directed abstract policy.
5. At the start and after every subgoal access, `Task.plan` compares
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
| Once per maze | Physical passive dynamics | `Environment` |
| Once per discovered/fixed basis | Subgoal profiles and optional access gate | `SubgoalBasis` |
| Once per goal, then cached | Lower first-exit dynamics, first-hit matrix, upper LMDP, and lower task basis | `Task` |
| At rollout start and each abstract access | Reward inpainting, task weights, and composed physical policy | `Plan` |
| After each physical move in online mode | One or more goal-column Z-iteration sweeps | `Rollout` state |

This separation is central to the multitask behavior: expensive reusable
structure is retained while goals and current abstract commands change.

## Code map

| Question | Start here |
| --- | --- |
| How is the passive maze random walk built? | [`passive_dynamics`](src/andrew_mlmdp/lmdp.py) |
| How is a flat first-exit LMDP solved? | [`solve_first_exit` and `Environment.solve`](src/andrew_mlmdp/lmdp.py) |
| How are point and distributed subgoals represented? | [`SubgoalBasis`](src/andrew_mlmdp/hierarchy/model.py) |
| How is a goal-conditioned hierarchy constructed? | [`_build_task`](src/andrew_mlmdp/hierarchy/model.py) |
| How are the lower and upper passive dynamics derived? | [`_lower_dynamics` and `_upper_dynamics`](src/andrew_mlmdp/hierarchy/model.py) |
| How does an upper policy become a physical policy? | [`compute_plan`, `_compose_plan`, and `_compose_policy`](src/andrew_mlmdp/hierarchy/model.py) |
| What exactly happens during a rollout? | [`_run_rollout`](src/andrew_mlmdp/hierarchy/rollout.py) |
| How are distributed subgoals discovered? | [`discover_subgoals`](src/andrew_mlmdp/discovery.py) |

## Minimal end-to-end example

This uses the same goal and hierarchy tuning as the canonical notebook:

```python
from andrew_mlmdp import (
    TaskLibrary,
    Environment,
    Maze,
    SubgoalBasis,
    point_parameters,
)

maze = Maze.from_file("mazes/four_rooms.txt")
environment = Environment(maze)
goal = (1, 9)

# A flat LMDP solves directly for this physical goal.
flat = environment.solve(goal)
flat_rollout = flat.rollout((3, 0), seed=0)

# The hierarchy reuses six subgoal task solutions.
subgoals = ((0, 0), (9, 2), (2, 3), (3, 7), (9, 7), (7, 9))
basis = SubgoalBasis.from_locations(maze, subgoals)
task_library = TaskLibrary.from_desirabilities(len(subgoals))
hierarchy = environment.hierarchy(
    basis,
    parameters=point_parameters(upper_control_cost=0.65),
    task_library=task_library,
)
task = hierarchy.task(goal)

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
- `task.first_hit`: boundary-first-hit probabilities from every
  physical interior state;
- `task.upper_dynamics` and `task.upper_controlled`: passive and controlled
  abstract processes;
- `task.task_basis`: reusable boundary tasks `Q_b` and their solved physical
  desirabilities `Z_i`;
- `hierarchy.task_library`: the fixed, full-rank Layer-1 task dictionary;
- `plan.weights`: the current mixture of lower component tasks; and
- `exact.events`: the complete physical/abstract rollout trace.

## Flat LMDPs and passive-motion modes

By default, passive motion samples uniformly from traversable cardinal
neighbors and has no self-transitions. The five-command model, which includes
`stay` and turns blocked commands into self-transitions, remains available
explicitly:

```python
command_sampling = Environment(
    maze,
    passive_mode="five_commands",
)
```

A `Solution` contains its desirability, controlled policy, rollout method,
and `log_likelihood` method for scoring observed discrete movement
trajectories. Consecutive repeated observations are collapsed before scoring.

For a dataset, keep trials separate so each trajectory starts with a fresh
controller state and uses its own goal. The dataset helpers retain per-trial
scores while summing their log-likelihoods:

```python
from andrew_mlmdp import Trial, score_flat_dataset

trials = [
    Trial("session-1", 1, (0, 2), ((0, 0), (0, 1), (0, 2))),
    Trial("session-1", 2, (0, 0), ((0, 2), (0, 1), (0, 0))),
]
dataset_score = score_flat_dataset(environment, trials)
print(dataset_score.total_log_likelihood)
print(dataset_score.mean_log_likelihood_per_transition)
```

The processed Doohan data can be assembled into the same trial representation
by session ID, subject, inclusive date range, or any intersection of those
selectors:

```python
from andrew_mlmdp import DoohanDataset

dataset = DoohanDataset.from_data_root(
    "external/GridMaze-mFC-ephys-DATA/data",
    subject_ids=["m2"],
    start_date="2022-06-23",
    end_date="2022-06-30",
    maze_name="maze_1",
)
flat_report = dataset.report(
    score_flat_dataset(environment, dataset.trials)
)
print(flat_report.summary_record())
```

The returned dataset retains typed session metadata, valid movement trials,
and explicit exclusions. Each call to `dataset.report(result)` produces trial,
session, and dataset summaries for that one model result. A selection must
resolve to exactly one maze; provide `maze_name` when a subject or date range
spans multiple maze configurations. Extracted trajectories stop at their first
entry into the trial goal, matching the likelihood models' absorbing-goal
assumption. Pandas is only required while loading the processed TSV files and
is available
through the `notebook` optional dependency.

## Distributed subgoals discovered with NMF

Point subgoals are one-hot profiles. Distributed subgoals use the same
hierarchy engine but can be learned by factorizing a family of flat goal-task
desirabilities:

```python
from andrew_mlmdp import (
    NMFConfig,
    SubgoalBasis,
    discover_subgoals,
    soft_parameters,
)

study = discover_subgoals(
    environment,
    ranks=range(2, 13),
    parameters=NMFConfig(),
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
    parameters=soft_parameters(8, upper_control_cost=0.18),
)
soft_task = soft_hierarchy.task(goal)
soft_rollout = soft_task.rollout((3, 2), seed=0)
```

NMF discovery and hierarchy execution have separate parameters. The original
peak-normalized NMF profiles and their gated access profiles are immutable.
Changing the goal builds or retrieves only a goal-conditioned hierarchy; it
does not rerun NMF or apply the gate again.

Set `lambda_smooth` to a positive value in `NMFConfig` to
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
from andrew_mlmdp import Environment, load_doohan_maze

definition = load_doohan_maze("maze_1")
environment = Environment(definition.maze)
start = definition.coordinate_for("A2")
goal = definition.coordinate_for("G7")

solution = environment.solve(goal)
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
