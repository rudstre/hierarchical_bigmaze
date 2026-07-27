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

The default configuration is the sustained-hierarchy rank-eight regime selected
after component-wise NMF peak normalization:

```python
ModelParameters(
    interior_reward=-0.05,
    goal_reward=0.65,
    lower_control_cost=0.12,
    upper_control_cost=1.15,
    alpha=0.08,
    off_target_reward=-1.3,
    beta=13.5,
)
```

The paper states small negative interior rewards, positive goal reward, and the
form of the abstraction and inpainting equations. The layer-dependent control
costs, off-target reward, and inpainting proportionality used here are project
choices.

The post-normalization validation searched a broad range and a focused
neighborhood, then evaluated finalists over all 96 non-goal starts, eight
rollout seeds, and three NMF seeds. The rounded default reached the goal in all
2,304 rollouts. Mean/median/90th-percentile physical steps were
20.44/18/38; the flat solved-goal comparator averaged 16.73.

The hierarchy controlled 89.9% of each rollout on average and remained active
through goal arrival in 71.4% of episodes. It made 88.0% normalized goalward
progress while active, with positive progress in 98.7% of rollouts. Immediate
handoff occurred in 5.3% of episodes and termination within five physical
steps in 4.3%. At the notebook start `(3, 2)`, none of 24 robust trials
terminated within five steps; the median active phase was 28.5 steps.

Core-gated accesses were spatially selective: the dominant profile supplied
99.8% of local access membership on average, and no access occurred below the
25% source-profile threshold. Signed upper-layer reward commands put 70.3% of
their positive mass on the leading subtask and had an effective size of 1.89
positively rewarded subtasks. These replace raw desirability-component
fractions as selection criteria because positive basis coefficients do not
mean positive rewards in exponentiated reward space. Initial-policy total
variation from the goal-only policy was 0.373.

Numerically, 0.89% of projected weights were clipped, maximum relative
boundary projection error was `1.18e-7`, maximum command span was 4.69
decades, and the mean path-length range across NMF seeds was 0.85 steps. There
were no zero-policy, step-limit, or abstract-access-limit failures. The
hierarchy added 3.71 mean steps relative to an already solved flat goal policy;
the preset deliberately prioritizes sustained hierarchical guidance rather
than shorter solved-goal paths. Reproduce the screening and sensitivity
analysis with `experiments/sweep_soft_k8.py`.

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
