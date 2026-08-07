# Maze multitask LMDPs

Readable research code for flat and two-layer linearly solvable Markov
decision processes in grid mazes. The implementation is dimension-agnostic:
matrix sizes come from the supplied maze and subgoal basis rather than the
included four-room example.

The supported workflows are:

1. parse ASCII mazes or labeled tower graphs with explicit connections;
2. solve and sample a flat first-exit LMDP for any free start and goal;
3. construct a reusable point-subgoal hierarchy and solve new goals with
   exact or online Z-iteration;
4. inspect and plot task-independent passive subgoal dynamics; and
5. discover distributed subgoals with KL-NMF, core-gate their access profiles,
   and explore a single rollout using draggable start and goal markers.

The canonical, executable walkthrough is
[`notebooks/maze_lmdp_workflows.ipynb`](notebooks/maze_lmdp_workflows.ipynb).

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
validation.

## Core API

```python
from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    SubgoalBasis,
    hard_hierarchy_parameters,
)

maze = Maze.from_file("mazes/four_rooms.txt")
environment = LMDPEnvironment(maze)
goal = (10, 9)

flat = environment.solve_flat(goal)
flat_rollout = flat.rollout((3, 0), seed=0)

subgoals = ((0, 0), (9, 2), (2, 3), (3, 7), (9, 7), (7, 9))
basis = SubgoalBasis.from_locations(maze, subgoals)
hard_parameters = hard_hierarchy_parameters()
hierarchy = environment.hierarchy(
    basis,
    parameters=hard_parameters,
    include_goal_component_while_active=False,
)
task = hierarchy.for_goal(goal)

exact = task.rollout((3, 2), seed=0)
online = task.rollout(
    (3, 2),
    goal_learning="online",
    z_sweeps_per_step=1,
    seed=28,
)
```

`LMDPEnvironment` constructs the physical passive matrix once. By default it
samples uniformly from north, south, east, west, and stay commands. To instead
sample only from traversable cardinal neighbors, construct the environment as
follows:

```python
movement_only = LMDPEnvironment(
    maze,
    passive_mode="valid_neighbors",
)
```

`HierarchyTemplate.for_goal` caches goal-conditioned tasks, while each
`HierarchyTask` exposes its lower dynamics, first-hit probabilities, task
basis, upper dynamics, upper policy, plans, and recorded rollout events.
Point bases automatically use the validated `hard_hierarchy_parameters()`
defaults when no explicit parameters are supplied.

## Doohan edge-list mazes

The GridMaze data submodule defines its mazes as labeled edges between towers.
`load_doohan_maze` keeps the 49 towers as physical states and uses the edge
list to restrict which cardinal movements are allowed:

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

Plot the discrete maze, optionally labeling only the states of interest:

```python
from andrew_mlmdp import plotting

tower_labels = {
    coordinate: label
    for coordinate, label in definition.label_by_coordinate.items()
    if "-" not in label
}
plotting.plot_maze(definition.maze, labels=tower_labels)
```

By default the loader reads
`external/GridMaze-mFC-ephys-DATA/data/experiment_info/maze_configs.json`.
Pass `config_path` explicitly when the downloaded data lives elsewhere.

## NMF soft subgoals

```python
from andrew_mlmdp import (
    NMFDiscoveryParameters,
    SubgoalBasis,
    discover_soft_subgoals,
    plotting,
    soft_hierarchy_parameters,
)

study = discover_soft_subgoals(
    environment,
    ranks=range(2, 13),
    parameters=NMFDiscoveryParameters(),
    seed=0,
)
rank_eight = study.result(8)  # reuses the fit from the rank study

soft_basis = SubgoalBasis.from_profiles(
    maze,
    rank_eight.profiles,
    core_threshold=0.8,
)
soft_hierarchy = environment.hierarchy(
    soft_basis,
    parameters=soft_hierarchy_parameters(8),
    include_goal_component_while_active=False,
)
player = plotting.plot_interactive_soft_hierarchical_rollout(
    soft_hierarchy,
    start=(3, 2),
    goal=goal,
    seed=0,
)
```

The original peak-normalized NMF profiles and their gated execution profiles
are immutable. Changing the goal rebuilds only the goal-conditioned hierarchy;
it does not rerun NMF or apply the gate again.

## Conventions

- Coordinates are `(row, column)` from the upper left.
- Matrices use `P[next_state, current_state]`, so probability columns sum to
  one.
- Passive physical motion chooses north, south, east, west, or stay uniformly.
  Invalid moves become self-transitions.
- Point subgoals are one-hot profile columns. Point and distributed subgoals
  therefore share one hierarchy construction and rollout engine.

See [`docs/model.md`](docs/model.md) for the mathematical mapping and
[`docs/four_rooms.md`](docs/four_rooms.md) for the included regression
configuration.
