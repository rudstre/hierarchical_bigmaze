# Four-room regression example

The repository includes one 11-by-11 demonstration maze with 97 free cells
and 24 walls in `mazes/four_rooms.txt`. It is a regression fixture and
walkthrough example, not an architectural constraint.

Coordinates are `(row, column)`. The notebook uses these point subgoals:

| Label | Coordinate |
| --- | ---: |
| A | `(0, 0)` |
| B | `(9, 2)` |
| C | `(2, 3)` |
| D | `(3, 7)` |
| E | `(9, 7)` |
| F | `(7, 9)` |

The physical goal is `(10, 9)`. Tests retain the historical matrices, initial
plan, and seeded flat/hierarchical trajectories in
`tests/data/four_rooms_regression.json`.

The canonical notebook uses:

```python
ModelParameters(
    interior_reward=-0.1,
    goal_reward=1.1,
    lower_control_cost=0.1,
    upper_control_cost=0.85,
    alpha=0.2,
    off_target_reward=-1.0,
    beta=16.0,
)
```

NMF discovery uses its independent defaults:

```python
NMFDiscoveryParameters(
    interior_reward=-0.4,
    goal_reward=6.5,
    control_cost=1.2,
)
```

The soft example selects rank eight, applies an 80%-of-peak execution core,
excludes the exact goal component while the hierarchy is active, and restores
the goal-only task after upper termination.

Additional tests construct corridors, tall and wide rectangles, obstacle
layouts, different subgoal counts, and different NMF ranks to prevent this
fixture from determining implementation shapes.
