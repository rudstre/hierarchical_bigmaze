"""Hierarchical rollout state, events, and stochastic execution."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from andrew_mlmdp.hierarchy.model import (
    Plan,
    Task,
    _goal_only_plan,
    _goal_plan,
    _plan_from_weights,
    _validate_goal_desirability,
)
from andrew_mlmdp.lmdp import desirability_step
from andrew_mlmdp.maze import Coordinate


@dataclass(frozen=True)
class SubgoalAccess:
    """One lower-to-upper access in a unified hierarchical rollout."""

    index: int
    coordinate: Coordinate
    physical_steps: int
    terminated: bool


@dataclass(frozen=True)
class Rollout:
    """Unified exact/online and point/distributed hierarchy result."""

    trajectory: tuple[Coordinate, ...]
    accesses: tuple[SubgoalAccess, ...]
    weight_history: tuple[np.ndarray, ...]
    events: tuple["RolloutEvent", ...]
    physical_steps: int
    abstract_accesses: int
    reached_goal: bool
    status: str
    goal_learning: Literal["exact", "online"]
    goal_desirability_history: tuple[np.ndarray, ...] = ()
    z_iterations: int = 0

    @property
    def final_goal_desirability(self) -> np.ndarray | None:
        """Return the final learned vector, or ``None`` for exact execution."""

        if not self.goal_desirability_history:
            return None
        return self.goal_desirability_history[-1]


@dataclass(frozen=True)
class _UpperTransition:
    entered_state: int
    terminated: bool
    coordinate: Coordinate
    physical_steps: int


@dataclass(frozen=True)
class RolloutEvent:
    """Model-neutral event emitted by the shared rollout engine."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: Plan | None
    entered_state: int | None
    physical_steps: int
    abstract_accesses: int
    passive_access: float | None
    policy_access: float | None
    refractory: bool
    goal_desirability: np.ndarray | None = None
    z_iterations: int = 0
    status: str | None = None


@dataclass(frozen=True)
class _EngineResult:
    trajectory: list[Coordinate]
    upper_transitions: list[_UpperTransition]
    weight_history: list[np.ndarray]
    physical_steps: int
    reached_goal: bool
    status: str
    events: list[RolloutEvent]
    goal_desirability_history: list[np.ndarray] | None = None
    z_iterations: int = 0
def _rollout_from_engine(
    model: Task,
    result: _EngineResult,
    goal_learning: Literal["exact", "online"],
) -> Rollout:
    accesses = tuple(
        SubgoalAccess(
            index=transition.entered_state,
            coordinate=transition.coordinate,
            physical_steps=transition.physical_steps,
            terminated=transition.terminated,
        )
        for transition in result.upper_transitions
    )
    histories = (
        ()
        if result.goal_desirability_history is None
        else tuple(
            values.copy()
            for values in result.goal_desirability_history
        )
    )
    return Rollout(
        trajectory=tuple(result.trajectory),
        accesses=accesses,
        weight_history=tuple(
            values.copy() for values in result.weight_history
        ),
        events=tuple(result.events),
        physical_steps=result.physical_steps,
        abstract_accesses=len(accesses),
        reached_goal=result.reached_goal,
        status=result.status,
        goal_learning=goal_learning,
        goal_desirability_history=histories,
        z_iterations=result.z_iterations,
    )


def _rollout_columns(
    plans: tuple[Plan, ...],
    current_interiors: np.ndarray,
    n_interior: int,
    n_subtasks: int,
    *,
    suppress_access: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch the rollout engine's exact column normalization semantics."""

    current_indices = np.asarray(current_interiors, dtype=np.int64)
    if current_indices.shape != (len(plans),):
        raise ValueError("Each rollout plan requires one current interior state")
    probabilities = np.column_stack(
        [
            plan.lower_policy[:, current_index]
            for plan, current_index in zip(plans, current_indices)
        ]
    )
    if suppress_access:
        probabilities[
            n_interior : n_interior + n_subtasks,
            :,
        ] = 0.0
    probability_mass = probabilities.sum(axis=0)
    usable = (
        np.isfinite(probability_mass)
        & (probability_mass > 0.0)
        & ~np.any(probabilities < 0.0, axis=0)
    )
    probabilities[:, usable] /= probability_mass[usable][np.newaxis, :]
    probabilities[:, ~usable] = 0.0
    return probabilities, usable


def _rollout_column(
    plan: Plan,
    current_interior: int,
    n_interior: int,
    n_subtasks: int,
    *,
    suppress_access: bool,
) -> np.ndarray | None:
    """Apply the rollout engine's access suppression and normalization."""

    probabilities, usable = _rollout_columns(
        (plan,),
        np.asarray([current_interior]),
        n_interior,
        n_subtasks,
        suppress_access=suppress_access,
    )
    if not usable[0]:
        return None
    return probabilities[:, 0]


def _run_rollout(
    model: Task,
    start: Coordinate,
    *,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int | None = None,
) -> _EngineResult:
    """Execute fixed, soft, exact, online, and plotted rollouts identically."""

    model.maze.state_index(start)
    if max_steps < 0:
        raise ValueError("Maximum steps must be non-negative")
    if max_abstract_accesses < 0:
        raise ValueError("Maximum abstract accesses must be non-negative")

    online = z_sweeps_per_step is not None
    if online:
        if (
            isinstance(z_sweeps_per_step, (bool, np.bool_))
            or not isinstance(z_sweeps_per_step, (int, np.integer))
            or z_sweeps_per_step < 1
        ):
            raise ValueError("Z sweeps per step must be a positive integer")
        if initial_goal_desirability is None:
            goal_desirability = np.zeros(
                len(model.interior_states),
                dtype=np.float64,
            )
        else:
            goal_desirability = _validate_goal_desirability(
                model,
                initial_goal_desirability,
            ).copy()
        goal_history: list[np.ndarray] | None = [
            goal_desirability.copy()
        ]
        q_interior = np.exp(
            model.parameters.interior_reward.item()
            / model.parameters.lower_control_cost.item()
        )
        goal_boundary = np.zeros(
            model.lower_dynamics.n_boundary,
            dtype=np.float64,
        )
        goal_boundary[-1] = np.exp(
            model.parameters.goal_reward.item()
            / model.parameters.lower_control_cost.item()
        )
    else:
        if initial_goal_desirability is not None:
            raise ValueError(
                "Initial goal desirability requires online goal learning"
            )
        goal_desirability = None
        goal_history = None
        q_interior = None
        goal_boundary = None

    if start == model.goal:
        event = RolloutEvent(
            event="terminal",
            coordinate=start,
            trajectory=(start,),
            plan=None,
            entered_state=None,
            physical_steps=0,
            abstract_accesses=0,
            passive_access=None,
            policy_access=None,
            refractory=False,
            goal_desirability=(
                None
                if goal_desirability is None
                else goal_desirability.copy()
            ),
            status="reached_goal",
        )
        return _EngineResult(
            trajectory=[start],
            upper_transitions=[],
            weight_history=[],
            physical_steps=0,
            reached_goal=True,
            status="reached_goal",
            events=[event],
            goal_desirability_history=goal_history,
        )

    random_generator = np.random.default_rng(seed)
    trajectory = [start]
    upper_transitions: list[_UpperTransition] = []
    current = start
    current_plan = _plan_from_weights(
        model,
        current,
        beta=beta,
        goal_desirability=goal_desirability,
    )
    weight_history = [current_plan.weights.copy()]
    physical_steps = 0
    z_iterations = 0
    refractory = False
    hierarchy_disabled = False
    events = [
        RolloutEvent(
            event="initial_plan",
            coordinate=current,
            trajectory=tuple(trajectory),
            plan=current_plan,
            entered_state=None,
            physical_steps=0,
            abstract_accesses=0,
            passive_access=None,
            policy_access=None,
            refractory=False,
            goal_desirability=(
                None
                if goal_desirability is None
                else goal_desirability.copy()
            ),
        )
    ]

    def finish(status: str, reached_goal: bool = False) -> _EngineResult:
        if not (
            events
            and events[-1].event == "terminal"
            and events[-1].status == status
        ):
            events.append(
                RolloutEvent(
                    event="terminal",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    entered_state=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(upper_transitions),
                    passive_access=None,
                    policy_access=None,
                    refractory=refractory or hierarchy_disabled,
                    goal_desirability=(
                        None
                        if goal_desirability is None
                        else goal_desirability.copy()
                    ),
                    z_iterations=z_iterations,
                    status=status,
                )
            )
        return _EngineResult(
            trajectory=trajectory.copy(),
            upper_transitions=upper_transitions.copy(),
            weight_history=[weights.copy() for weights in weight_history],
            physical_steps=physical_steps,
            reached_goal=reached_goal,
            status=status,
            events=events.copy(),
            goal_desirability_history=(
                None
                if goal_history is None
                else [values.copy() for values in goal_history]
            ),
            z_iterations=z_iterations,
        )

    while physical_steps < max_steps:
        current_state = model.interior_index[current]
        n_interior = len(model.interior_states)
        n_subtasks = model.n_subtasks
        probabilities = _rollout_column(
            current_plan,
            current_state,
            n_interior,
            n_subtasks,
            suppress_access=refractory or hierarchy_disabled,
        )
        if probabilities is None:
            return finish("zero_policy")
        next_state = int(
            random_generator.choice(len(probabilities), p=probabilities)
        )

        if next_state < n_interior:
            physical_state = int(model.interior_states[next_state])
            current = model.maze.coordinate(physical_state)
            trajectory.append(current)
            physical_steps += 1
            refractory = False

            if online:
                assert goal_desirability is not None
                assert goal_boundary is not None
                assert q_interior is not None
                assert goal_history is not None
                assert z_sweeps_per_step is not None
                for _ in range(z_sweeps_per_step):
                    goal_desirability = desirability_step(
                        model.lower_dynamics,
                        goal_desirability,
                        goal_boundary,
                        q_interior,
                    )
                    z_iterations += 1
                goal_history.append(goal_desirability.copy())
                current_plan = _goal_plan(
                    model,
                    current_plan,
                    goal_desirability,
                )

            events.append(
                RolloutEvent(
                    event="physical_step",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    entered_state=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(upper_transitions),
                    passive_access=None,
                    policy_access=None,
                    refractory=hierarchy_disabled,
                    goal_desirability=(
                        None
                        if goal_desirability is None
                        else goal_desirability.copy()
                    ),
                    z_iterations=z_iterations,
                )
            )
            continue

        boundary_state = next_state - n_interior
        if boundary_state == n_subtasks:
            current = model.goal
            trajectory.append(current)
            physical_steps += 1
            events.append(
                RolloutEvent(
                    event="terminal",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    entered_state=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(upper_transitions),
                    passive_access=None,
                    policy_access=None,
                    refractory=hierarchy_disabled,
                    goal_desirability=(
                        None
                        if goal_desirability is None
                        else goal_desirability.copy()
                    ),
                    z_iterations=z_iterations,
                    status="reached_goal",
                )
            )
            return finish("reached_goal", reached_goal=True)

        if len(upper_transitions) >= max_abstract_accesses:
            return finish("abstract_access_limit")

        entered_state = boundary_state
        if model.basis.locations is not None:
            current = model.basis.locations[entered_state]
        passive_access = float(
            model.lower_dynamics.boundary_passive[
                entered_state,
                current_state,
            ]
        )
        controlled_access = float(
            current_plan.lower_policy[
                n_interior + entered_state,
                current_state,
            ]
        )
        next_access_count = len(upper_transitions) + 1
        events.append(
            RolloutEvent(
                event="lower_access",
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                entered_state=entered_state,
                physical_steps=physical_steps,
                abstract_accesses=next_access_count,
                passive_access=passive_access,
                policy_access=controlled_access,
                refractory=False,
                goal_desirability=(
                    None
                    if goal_desirability is None
                    else goal_desirability.copy()
                ),
                z_iterations=z_iterations,
            )
        )

        terminal_probability = float(
            model.upper_controlled[-1, entered_state]
        )
        terminated = random_generator.random() < terminal_probability
        transition = _UpperTransition(
            entered_state=entered_state,
            terminated=terminated,
            coordinate=current,
            physical_steps=physical_steps,
        )
        upper_transitions.append(transition)
        refractory = True

        if terminated:
            hierarchy_disabled = True
            current_plan = _goal_only_plan(
                model,
                current,
                goal_desirability=goal_desirability,
            )
            event_name = "upper_termination"
        else:
            current_plan = _plan_from_weights(
                model,
                current,
                upper_state=entered_state,
                beta=beta,
                goal_desirability=goal_desirability,
            )
            event_name = "upper_command"
        weight_history.append(current_plan.weights.copy())
        events.append(
            RolloutEvent(
                event=event_name,
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                entered_state=entered_state,
                physical_steps=physical_steps,
                abstract_accesses=len(upper_transitions),
                passive_access=passive_access,
                policy_access=controlled_access,
                refractory=True,
                goal_desirability=(
                    None
                    if goal_desirability is None
                    else goal_desirability.copy()
                ),
                z_iterations=z_iterations,
            )
        )

    return finish("step_limit")


