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

The canonical notebook uses these defaults for fixed one-hot subgoals:

```python
hard_hierarchy_parameters(
    interior_reward=-0.1,
    goal_reward=1.1,
    lower_control_cost=0.06,
    upper_control_cost=0.3,
    alpha=0.4,
    off_target_reward=-1.0,
    beta=16.0,
)
```

`LMDPEnvironment.hierarchy(point_basis)` selects these defaults
automatically. The same fallback applies to profile bases so an equivalent
one-hot profile remains identical to a point basis. Flat tasks retain
`ModelParameters()` defaults, and calibrated soft hierarchies pass their
independent `soft_hierarchy_parameters` explicitly.

The canonical fixed hierarchy sets
`include_goal_component_while_active=False`: active plans are composed only
from the fixed subgoal basis. The exact goal component is enabled only after
the upper layer terminates.

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
