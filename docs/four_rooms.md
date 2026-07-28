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

NMF discovery uses a separate frozen task family:

```python
NMFDiscoveryParameters(
    interior_reward=-0.4,
    goal_reward=6.5,
    control_cost=1.2,
)
```

The default execution configuration is the sustained-hierarchy rank-eight
regime selected after component-wise NMF peak normalization:

```python
ModelParameters(
    interior_reward=-0.1,
    goal_reward=1.1,
    lower_control_cost=0.1,
    upper_control_cost=1.8,
    alpha=0.2,
    off_target_reward=-0.7,
    beta=13.0,
)
```

Changing execution `lower_control_cost` reuses the profiles learned with the
discovery configuration above. The ensemble and NMF should be rebuilt only
when intentionally learning a new subtask library.

The paper states small negative interior rewards, positive goal reward, and the
form of the abstraction and inpainting equations. The layer-dependent control
costs, off-target reward, and inpainting proportionality used here are project
choices.

Execution uses an 80%-of-peak soft core. This excludes weak profile fringes,
including a doorway cell that previously caused an S5 command to be
interrupted by an unintended S7 access. The larger `alpha=0.2` compensates
for the narrower support so deliberate access remains frequent.

The rounded regime was checked over all 96 non-goal starts, 32 rollout seeds,
and three frozen NMF libraries (9,216 rollouts). It reached the goal in every
rollout, with mean/median/p90/p95/p99/maximum steps of
13.45/13/24/27/34/49. The hierarchy was active for 94.2% of each rollout,
made 93.6% normalized goalward progress while active, averaged 2.05
continuing upper commands, and had 4.2% immediate handoff and 4.4%
termination within five steps.

A 36,864-rollout all-start stress test gave p99/p99.9 of 34/43 and maximum
68. This maximum is an observed stochastic sample, not a guaranteed horizon.
At `(3, 2)`, 18,000 additional rollouts gave mean/p90/p95/p99/maximum steps
of 20.54/24/26/29/42 and 0.4% immediate handoff. The regime passed all
declared pathology criteria.

The current sweep implementation no longer assumes that focused neighborhood.
It first searches the identifiable discovery profile-shaping ratio over four
orders of magnitude, freezes the selected NMF libraries, searches execution
over deliberately broad ranges, and derives refinement neighborhoods from the
broad-stage survivors. It also reports per-start p90/p95 values and a separate
high-seed-count tail analysis for `(3, 2)`.

The earlier one-factor sensitivity count predates core gating and is not
reported as a current result.

Use `soft_hierarchy_parameters(k)` to apply the heuristic rank scaling or
`paper_hierarchy_parameters()` for the paper-scale preset. Ranks other than
eight have not received the same behavioral validation.

## Passive movement

Before subgoal augmentation, north, south, east, west, and stay each have
probability `0.2`. Invalid movement commands become self-transitions. At a
subgoal cell, access mass `alpha` is added to its abstract copy and the complete
column is renormalized.

## Reproducible rollouts

The figure scripts use fixed seeds:

| Figure | Start | Seed | Canonical result |
| --- | ---: | ---: | --- |
| Flat sample rollout | `(0, 0)` | `7` | generated from current defaults |
| Hierarchical rollout | `(1, 0)` | `28` | generated from current defaults |

The readable regression fixture `tests/data/four_rooms_regression.json`
retains the repository's historical fixed-subgoal regime explicitly. It no
longer relies on global defaults.

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
