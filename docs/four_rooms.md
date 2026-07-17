# Four-room demonstration

This document records the exact configuration used by the repository's
four-room figures and regression tests. It separates reproducible project
choices from facts explicitly specified by the paper.

## Geometry

The maze is stored in `mazes/four_rooms.txt` as an 11-by-11 grid containing 97
free cells and 24 walls. `#` is a wall and `.` is a free cell. Goals and
subgoals are not embedded in the geometry file.

The layout reproduces the obstacle topology of the paper's Figure 3 rooms
domain. It should be described as a faithful hand-authored reconstruction, not
as an unpublished original environment recovered from the authors.

## State locations

Coordinates are `(row, column)` from the upper left.

| Label | Role | Coordinate |
| --- | --- | ---: |
| A | subgoal | `(0, 0)` |
| B | subgoal | `(9, 2)` |
| C | subgoal | `(2, 3)` |
| D | subgoal | `(3, 7)` |
| E | subgoal | `(9, 7)` |
| F | subgoal | `(7, 9)` |
| goal | physical terminal boundary | `(10, 9)` |

The order `A, B, C, D, E, F` is used for upper states, task-basis rows and
columns, matrix labels, and task-weight histories.

The flat sample rollout starts at `(0, 0)`. The hierarchical sample rollout
starts at `(1, 0)`, immediately below A. Starts outside the upper state set use
the documented first-hit initialization in `docs/model.md`.

## Parameters

The canonical configuration is exactly `ModelParameters()`:

```python
ModelParameters(
    interior_reward=-0.1,
    goal_reward=1.0,
    lower_control_cost=0.15,
    upper_control_cost=0.3,
    alpha=1.0,
    off_target_reward=-2.0,
    beta=10.0,
)
```

The paper states small negative interior rewards, positive goal reward, and the
form of the abstraction and inpainting equations. The layer-dependent control
costs, off-target reward, and inpainting proportionality used here are project
choices. `alpha=1` is also the canonical value for this codebase even though
the paper and supplement discuss other examples.

## Passive movement

Before subgoal augmentation, north, south, east, west, and stay each have
probability `0.2`. Invalid movement commands become self-transitions. At a
subgoal cell, access mass `alpha` is added to its abstract copy and the complete
column is renormalized.

## Reproducible rollouts

The figure scripts use fixed seeds:

| Figure | Start | Seed | Canonical result |
| --- | ---: | ---: | --- |
| Flat sample rollout | `(0, 0)` | `7` | goal reached in 31 steps |
| Hierarchical rollout | `(1, 0)` | `28` | goal reached in 26 steps |

The hierarchical rollout accesses C, D, and F in that order. The full paths and
canonical matrices are stored in the readable regression fixture
`tests/data/four_rooms_regression.json`.

These individual trajectories are diagnostics, not estimates of expected
performance. A statistical comparison would require a separate, explicitly
specified experiment over many seeds.

## Generated figures

Run from the repository root:

```shell
MPLBACKEND=Agg python experiments/plot_flat_policy.py
MPLBACKEND=Agg python experiments/plot_sample_rollout.py
MPLBACKEND=Agg python experiments/plot_passive_subgoal_graph.py
MPLBACKEND=Agg python experiments/plot_two_layer.py
```

The scripts write these ignored files:

```text
output/four_rooms_controlled_policy.png
output/four_rooms_sample_rollout.png
output/four_rooms_passive_subgoal_graph.png
output/four_rooms_two_layer.png
```

Each script prints the active `ModelParameters` so a captured result records
its numerical regime. The notebook uses the same values and seeds but is kept
without stored outputs.

## Current scope

The repository demonstrates the exact flat solution, task-independent passive
subgoal graph, exact two-layer solution, reward inpainting, task weights, and
seeded rollouts. It does not currently implement the paper's learning curves,
ring-complexity experiment, or office/options comparison.
