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
    NMFDiscoveryParameters,
    build_goal_task_ensemble,
    build_soft_two_layer_model,
    factorize_soft_subtasks,
    plot_interactive_soft_hierarchical_rollout,
    sample_online_soft_hierarchical_rollout,
    sample_soft_hierarchical_rollout,
    soft_hierarchy_parameters,
)

number_of_subtasks = 8
discovery_parameters = NMFDiscoveryParameters()
execution_parameters = soft_hierarchy_parameters(
    k=number_of_subtasks,
    # lower_control_cost=0.11,  # sharpen execution without rediscovery
)
ensemble = build_goal_task_ensemble(
    maze,
    discovery_parameters=discovery_parameters,
)
discovery = factorize_soft_subtasks(
    ensemble,
    n_subtasks=number_of_subtasks,
    seed=0,
)
soft_model = build_soft_two_layer_model(
    maze,
    discovery.profiles,
    goal,
    parameters=execution_parameters,
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

For notebook inspection, prefer the paused frame player over serializing a
`FuncAnimation`. With `%matplotlib widget` active, it computes the rollout
once and redraws only when the slider or a step button changes:

```python
from IPython.display import display
import matplotlib.pyplot as plt

player = plot_interactive_soft_hierarchical_rollout(
    soft_model,
    start=(1, 0),
    seed=3,
)
display(player.controls)
plt.show()
```

The slider uses `continuous_update=False`, so dragging it does not queue
intermediate figure renders. The “Include exact goal component” checkbox
switches the heatmap between the full composition and the same weighted basis
with its final goal column removed; it does not change the sampled rollout.
`animate_soft_hierarchical_rollout` remains the export-oriented API for HTML,
GIF, or video output.

`factorize_soft_subtasks` peak-normalizes every profile column and absorbs its
scale into the matching task-weight row. This leaves the NMF reconstruction
unchanged and makes `alpha` the maximum local passive access strength.
`NMFDiscoveryParameters` is frozen into the ensemble and is independent of
the execution `ModelParameters`. Reuse the resulting `discovery.profiles`
when changing `lower_control_cost`; rebuilding the ensemble would intentionally
learn a different subtask library.
`build_soft_two_layer_model` then core-gates access at 80% of each profile
peak by default. Values below the threshold become exactly zero; values above
it are linearly rescaled to `[0, 1]`. Pass `core_threshold=None` for direct
paper access `P_t = alpha * D.T`, or set `core_exponent` above one to sharpen
the surviving core further.

## Reference parameters

`NMFDiscoveryParameters()` fixes the NMF task family at
`interior_reward=-0.4`, `goal_reward=6.5`, and `control_cost=1.2`.
`ModelParameters()` independently configures execution using the
sustained-hierarchy regime from the post-peak-normalization rank-eight
validation:

| Parameter | Default |
| --- | ---: |
| Interior reward | `-0.1` |
| Goal reward | `1.1` |
| Lower control cost `lambda_1` | `0.1` |
| Upper control cost `lambda_2` | `1.8` |
| Subgoal-access mass `alpha` | `0.2` |
| Off-target basis reward | `-0.7` |
| Reward-inpainting scale `beta` | `13.0` |

These are empirical project choices rather than values uniquely determined by
the paper. The 80% access core removes incidental abstract transitions at
profile fringes and doorway cells. `alpha=0.2` retains deliberate hierarchy
access after that narrowing, while `beta=13.0` keeps commands selective
without approaching the numerical span limit.

Verification used all 96 non-goal starts, 32 rollout seeds, and three frozen
NMF libraries (9,216 rollouts): 100% goal success, 13.45 mean steps, p90 24,
p95 27, p99 34, and maximum 49. The hierarchy controlled 94.2% of each
rollout and made 93.6% normalized goalward progress while active. Immediate
handoff occurred in 4.2% of episodes and termination within five steps in
4.4%. It averaged 2.05 continuing upper commands, and the mean command-policy
total variation from the goal-only policy was `0.499`.

A larger 36,864-rollout all-start stress test gave p99/p99.9 of 34/43 and a
maximum of 68. Because rollout actions are sampled from a stochastic policy,
the observed maximum is not a hard bound and grows with sample count. At
`(3, 2)`, 18,000 additional rollouts gave mean/p90/p95/p99 steps of
20.54/24/26/29, maximum 42, and 0.4% immediate handoff. The preset passed
every declared pathology criterion.

The sweep is now tiered. It first scans
`-discovery_interior_reward / discovery_control_cost` from `1e-3` to `10`
and measures KL reconstruction, NMF-seed stability, profile overlap, core
coverage, connectivity, and task-ensemble dynamic range. This is the only
discovery ratio that changes the ideal peak-normalized profile geometry:
scaling all discovery rewards and control cost together leaves `Z` unchanged,
while the shared goal reward multiplies every task column by one common
factor. The goal reward and control cost are therefore held at numerically
safe reporting values rather than treated as two extra identifiable search
dimensions.

After selecting the first stable profile-localization knee, the script
factorizes NMF seeds 0, 1, and 2 once and reuses those exact profile libraries.
Execution then uses a deliberately wide log-stratified search, an
evidence-driven factor-of-four neighborhood around broad-stage survivors, an
all-start robust check, and a separate 2,000-seed tail check at `(3, 2)`.
Refinement is allowed to cross an initial broad bound when a survivor lies
near that edge; much wider numerical safety limits prevent a boundary hit
from masquerading as an optimum. One-factor sensitivity is run around the
current default, the balanced recommendation, and the fastest clean finalist.
Run the complete funnel with:

```bash
.venv/bin/python experiments/sweep_soft_k8.py
```

The output directory contains `discovery.csv`, `broad.csv`, `focused.csv`,
`robust.csv`, `demonstration_tail.csv`, `sensitivity.csv`, `summary.json`, and
`report.md`.

`soft_hierarchy_parameters(k)` scales the execution `alpha` and upper control
cost for another NMF rank:

```python
parameters = soft_hierarchy_parameters(
    k=12,
    beta=13.0,  # optional explicit override
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
