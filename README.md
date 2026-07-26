# Two-Layer Multitask LMDP Maze Navigation

Readable research code for the four-room demonstration in Saxe, Earle, and
Rosman (2017), *Hierarchy Through Composition with Multitask LMDPs*.

The repository implements:

- exact first-exit LMDP solutions;
- closed-form controlled dynamics;
- multitask desirability composition;
- KL-NMF discovery of distributed soft subtasks;
- a two-layer abstraction built from lower-layer first-hit probabilities;
- top-down reward inpainting and task blending;
- seeded flat and hierarchical rollouts; and
- the current four-room diagnostic figures.

The emphasis is transparency. Arrays remain inspectable, state ordering is
explicit, and the main functions follow the order of the paper's equations.
Additional paper experiments and learning curves are outside the current scope.

## Install and test

Python 3.11 or newer is required.

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,notebook]"
pytest -q
```

The equivalent `uv` setup is:

```shell
uv sync --extra test --extra notebook
uv run pytest -q
```

## Quick start

```python
from andrew_mlmdp import (
    Maze,
    ModelParameters,
    build_two_layer_model,
    compute_layer_one_plan,
    controlled_dynamics,
    sample_hierarchical_rollout,
    sample_rollout,
    solve_desirability,
)

maze = Maze.from_file("mazes/four_rooms.txt")
parameters = ModelParameters()
goal = (10, 9)

flat_desirability = solve_desirability(
    maze,
    goal,
    parameters=parameters,
)
flat_policy = controlled_dynamics(maze, flat_desirability)
flat_rollout = sample_rollout(
    maze,
    flat_policy,
    start=(0, 0),
    goal=goal,
    seed=7,
)

subgoals = ((0, 0), (9, 2), (2, 3), (3, 7), (9, 7), (7, 9))
model = build_two_layer_model(
    maze,
    subgoals,
    goal,
    parameters=parameters,
)
plan = compute_layer_one_plan(model, current=(1, 0))
hierarchical_rollout = sample_hierarchical_rollout(
    model,
    start=(1, 0),
    seed=28,
)
```

`model.lower_dynamics`, `model.task_basis`, `model.upper_dynamics`, and `plan`
expose the intermediate matrices used in the calculation.

Soft subtasks follow the same construction, replacing point access rows with
the paper's distributed `P_t = alpha * D.T` profiles:

```python
from andrew_mlmdp import (
    build_goal_task_ensemble,
    build_soft_two_layer_model,
    factorize_soft_subtasks,
    paper_hierarchy_parameters,
    sample_soft_hierarchical_rollout,
)

soft_parameters = paper_hierarchy_parameters()
ensemble = build_goal_task_ensemble(maze, parameters=soft_parameters)
discovery = factorize_soft_subtasks(ensemble, n_subtasks=4, seed=0)
soft_model = build_soft_two_layer_model(
    maze,
    discovery.profiles,
    goal,
    parameters=soft_parameters,
)
soft_rollout = sample_soft_hierarchical_rollout(
    soft_model,
    start=(1, 0),
    seed=3,
)
```

## Canonical parameters

`ModelParameters()` uses the project's established four-room regime:

| Parameter | Default |
| --- | ---: |
| Interior reward | `-0.1` |
| Goal reward | `1.0` |
| Lower control cost `lambda_1` | `0.15` |
| Upper control cost `lambda_2` | `0.3` |
| Subgoal-access mass `alpha` | `1.0` |
| Off-target basis reward | `-2.0` |
| Reward-inpainting scale `beta` | `10.0` |

These are project choices rather than values uniquely determined by the paper.
Construct another `ModelParameters` instance to study a different regime.

## Repository map

```text
src/andrew_mlmdp/
|-- maze.py       geometry and coordinate/state conversion
|-- lmdp.py       first-exit dynamics, Equations 4 and 6, flat solver
|-- discovery.py  goal-task ensembles and KL-NMF soft subtasks
|-- hierarchy.py  two-layer construction, composition, and rollout
`-- plotting.py   direct visualizations of policies and trajectories

experiments/      deterministic figure-producing scripts
mazes/            text-only maze geometry
notebooks/        an inspectable flat-to-hierarchical walkthrough
tests/            equation, simulation, and four-room regression checks
docs/             mathematical and experiment documentation
```

## Conventions

- Coordinates are `(row, column)`, with `(0, 0)` at the upper left.
- Physical states follow row-major `maze.free_cells` order.
- Every transition matrix uses
  `P[next_state, current_state] = P(next_state | current_state)`.
- Transition columns, rather than rows, sum to one.
- Lower boundary order is `[subgoals..., goal]`.
- Upper interior-state order is the caller-supplied subgoal order.
- A subgoal copy is an abstract boundary; its physical cell remains traversable.
- Abstract accesses consume no physical time.

See [docs/model.md](docs/model.md) for the derivation and code mapping, and
[docs/four_rooms.md](docs/four_rooms.md) for the exact experiment protocol.

## Figures and notebook

```shell
MPLBACKEND=Agg python experiments/plot_flat_policy.py
MPLBACKEND=Agg python experiments/plot_sample_rollout.py
MPLBACKEND=Agg python experiments/plot_passive_subgoal_graph.py
MPLBACKEND=Agg python experiments/plot_two_layer.py
jupyter lab notebooks/flat_lmdp_examples.ipynb
```

Figures are written to the ignored `output/` directory. The notebook is stored
without outputs so its displayed results always come from the current code.

## Reference

Andrew M. Saxe, Adam C. Earle, and Benjamin Rosman. "Hierarchy Through
Composition with Multitask LMDPs." *Proceedings of the 34th International
Conference on Machine Learning*, PMLR 70, 2017.
