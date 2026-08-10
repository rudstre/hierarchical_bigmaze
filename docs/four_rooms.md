# Four-room example configurations

The repository includes an 11-by-11 maze with 97 free cells and 24 walls in
`mazes/four_rooms.txt`. It is a demonstration and regression fixture, not an
architectural constraint.

Coordinates are `(row, column)` from the upper left. The shared point subgoals
are:

| Label | Coordinate |
| --- | ---: |
| A | `(0, 0)` |
| B | `(9, 2)` |
| C | `(2, 3)` |
| D | `(3, 7)` |
| E | `(9, 7)` |
| F | `(7, 9)` |

## Why the numbers differ across examples

There are three deliberate configurations. They serve different purposes and
should not be treated as one canonical parameter set.

| Configuration | Goal | Purpose |
| --- | ---: | --- |
| Library hard defaults | chosen by caller | General default for one-hot or equivalent profile bases |
| Canonical notebook | `(1, 9)` | Tuned visual and interactive walkthrough |
| Frozen regression fixture | `(10, 9)` | Preserve historical matrices and seeded trajectories |

The top-level README matches the canonical notebook. Tests that compare exact
historical arrays use the frozen regression fixture instead.

## Library hard defaults

Calling `LMDPEnvironment.hierarchy(basis)` without explicit parameters selects:

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

This fallback applies to point bases and equivalent one-hot profile bases so
both representations execute identically. `include_goal_component_while_active`
is independently `True` by default.

Flat tasks use `ModelParameters()` defaults rather than these hard-hierarchy
defaults.

## Canonical executable notebook

[`notebooks/maze_lmdp_workflows.ipynb`](../notebooks/maze_lmdp_workflows.ipynb)
uses goal `(1, 9)` and the hard defaults above with one explicit override:

```python
hard_parameters = hard_hierarchy_parameters(upper_control_cost=0.65)
```

The point-subgoal hierarchy also sets
`include_goal_component_while_active=False`. While the hierarchy is active,
plans are composed only from the six fixed subgoal tasks. Upper termination
then installs the goal-only task permanently.

The notebook's distributed example selects NMF rank eight, applies an
80%-of-peak execution core, and uses:

```python
soft_hierarchy_parameters(k=8, upper_control_cost=0.18)
```

It likewise excludes the exact goal component while the hierarchy is active.
These upper-cost overrides are example tuning, not package-wide defaults.

NMF discovery itself uses a separate flat-task ensemble and the independent
defaults:

```python
NMFDiscoveryParameters(
    interior_reward=-0.4,
    goal_reward=6.5,
    control_cost=1.2,
)
```

Changing hierarchy execution parameters cannot silently alter the already
discovered NMF profiles.

## Frozen regression fixture

`tests/test_four_rooms_regression.py` uses goal `(10, 9)` and the parameters
defined in `tests/conftest.py`:

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

This fixture retains historical passive and controlled matrices, initial plan
values, and seeded flat and hierarchical trajectories in
`tests/data/four_rooms_regression.json`. It intentionally does not track the
current notebook tuning or the library defaults.

Additional tests construct corridors, tall and wide rectangles, obstacle
layouts, different subgoal counts, and different NMF ranks so this fixture
cannot determine implementation shapes.
