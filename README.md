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
    sample_online_soft_hierarchical_rollout,
    sample_soft_hierarchical_rollout,
    soft_hierarchy_parameters,
)

number_of_subtasks = 8
soft_parameters = soft_hierarchy_parameters(k=number_of_subtasks)
ensemble = build_goal_task_ensemble(maze, parameters=soft_parameters)
discovery = factorize_soft_subtasks(
    ensemble,
    n_subtasks=number_of_subtasks,
    seed=0,
)
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

# Replace the exact goal solution with a zero-initialized solution that
# advances by one full Z-iteration after every nonterminal physical step.
online_soft_rollout = sample_online_soft_hierarchical_rollout(
    soft_model,
    start=(1, 0),
    seed=3,
)
```

`factorize_soft_subtasks` peak-normalizes every profile column and absorbs its
scale into the matching task-weight row. This leaves the NMF reconstruction
unchanged and makes `alpha` the maximum local passive access strength.
`build_soft_two_layer_model` then core-gates access at 25% of each profile
peak by default. Values below the threshold become exactly zero; values above
it are linearly rescaled to `[0, 1]`. Pass `core_threshold=None` for direct
paper access `P_t = alpha * D.T`, or set `core_exponent` above one to sharpen
the surviving core further.

## Reference parameters

`ModelParameters()` uses the sustained-hierarchy regime from the post-peak-
normalization rank-eight validation:

| Parameter | Default |
| --- | ---: |
| Interior reward | `-0.05` |
| Goal reward | `0.65` |
| Lower control cost `lambda_1` | `0.12` |
| Upper control cost `lambda_2` | `1.15` |
| Subgoal-access mass `alpha` | `0.08` |
| Off-target basis reward | `-1.3` |
| Reward-inpainting scale `beta` | `13.5` |

These are empirical project choices rather than values uniquely determined by
the paper. The validation used all 96 non-goal starts, eight rollout seeds,
and three NMF seeds (2,304 rollouts): 100% goal success, 20.44 mean physical
steps, and 38 steps at the 90th percentile. The hierarchy controlled 89.9% of
each rollout on average, made 88.0% normalized goalward progress while active,
and remained active through goal arrival in 71.4% of episodes.

The hierarchy materially changed the lower policy (mean total variation
`0.373`). Its core-gated accesses were region-selective: one profile supplied
99.8% of local access membership on average, and no access occurred below the
25% source-profile threshold. Signed Equation 10 commands assigned 70.3% of
their positive reward mass to the leading subtask on average, with an
effective positive command size of 1.89 subtasks. These signed reward and
access-selectivity measures are used instead of raw desirability-component
fractions, which are not meaningful activation measures after exponentiation.

Immediate handoff occurred in 5.3% of episodes and termination within five
physical steps in 4.3%. At the notebook demonstration start `(3, 2)`, none of
the 24 robust trials terminated within five steps and the median active phase
was 28.5 steps. The soft policy averaged 3.71 steps longer than its flat
solved-goal comparator; the preset deliberately favors sustained hierarchical
guidance over matching the speed of an already solved exact goal policy.

The earlier one-factor sensitivity count predates core gating and has not been
carried forward as a current claim. The reproducible sweep and its
signed-command selectivity and early-termination criteria live in
`experiments/sweep_soft_k8.py`.

`soft_hierarchy_parameters(k)` scales `alpha` and the upper control cost for
another NMF rank:

```python
parameters = soft_hierarchy_parameters(
    k=12,
    beta=10.0,  # optional explicit override
)
```

Only the rank-eight reference received this behavioral validation; the rank
scaling remains a heuristic. Every `ModelParameters` field is available as an
optional keyword override.
Use `paper_hierarchy_parameters()` when the paper-scale constants are desired
without the rank heuristic.

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
