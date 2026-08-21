"""Shared types and small utilities for plotting modules."""

from dataclasses import dataclass
from typing import Callable, cast

import matplotlib
import numpy as np
from matplotlib.colors import Colormap

from andrew_mlmdp.hierarchy import Plan
from andrew_mlmdp.maze import Coordinate


@dataclass(frozen=True)
class _RolloutFrame:
    """One drawable moment in a two-layer rollout."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: Plan | None
    active_subgoal: Coordinate | None
    requested_subgoal: Coordinate | None
    physical_steps: int
    abstract_accesses: int
    passive_access: float | None = None
    policy_access: float | None = None
    refractory: bool = False
    status: str | None = None
    goal_desirability: np.ndarray | None = None
    z_iterations: int = 0


@dataclass(frozen=True)
class _ProfileFrame:
    """One drawable physical or distributed-access event."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: Plan | None
    profile_subtask: int | None
    entered_subtask: int | None
    physical_steps: int
    abstract_accesses: int
    passive_access: float | None = None
    policy_access: float | None = None
    refractory: bool = False
    status: str | None = None



def _colormap(name: str, *, bad: str | None = None) -> Colormap:
    """Return a configured copy from Matplotlib's modern registry API."""

    registry = getattr(matplotlib, "colormaps")
    color_map = cast(Colormap, registry[name])
    if bad is None:
        return color_map
    with_extremes = cast(Callable[..., Colormap], color_map.with_extremes)
    return with_extremes(bad=bad)


_TRAJECTORY_ARROW_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _format_probability(value: float | None) -> str:
    return "n/a" if value is None else f"{value: .4f}"


def _event_title(event: str) -> str:
    titles = {
        "initial_plan": "initial request",
        "physical_step": "physical step",
        "subgoal_access": "new directions",
        "subtask_access": "new soft directions",
        "lower_access": "lower access",
        "upper_command": "upper command",
        "upper_termination": "upper termination",
        "terminal": "terminal",
    }
    return titles[event]


