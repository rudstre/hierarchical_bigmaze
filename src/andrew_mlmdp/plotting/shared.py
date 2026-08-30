"""Shared data structures and Plotly utilities for plotting modules."""

from dataclasses import dataclass

import numpy as np
from plotly.colors import get_colorscale, qualitative, sample_colorscale

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


_TRAJECTORY_ARROW_COLORS = tuple(qualitative.Plotly[:10])


def _colorscale(name: str) -> list[list[object]]:
    """Return a Plotly colorscale using familiar case-insensitive names."""

    aliases = {
        "viridis": "Viridis",
        "ylorrd": "YlOrRd",
        "coolwarm": "RdBu",
        "blues": "Blues",
    }
    return get_colorscale(aliases.get(name.lower(), name))


def _sample_color(name: str, value: float) -> str:
    """Sample a named Plotly colorscale at a clipped unit interval value."""

    clipped = min(1.0, max(0.0, float(value)))
    return str(sample_colorscale(_colorscale(name), [clipped])[0])


def _plotly_color(color: str) -> str:
    """Translate Matplotlib-style numeric gray strings to CSS colors."""

    try:
        gray = float(color)
    except (TypeError, ValueError):
        return color
    channel = round(255 * min(1.0, max(0.0, gray)))
    return f"rgb({channel},{channel},{channel})"


def _figure_size(figsize: tuple[float, float]) -> tuple[int, int]:
    """Convert the project's historical inch-based sizes to CSS pixels."""

    return round(100 * figsize[0]), round(100 * figsize[1])


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
