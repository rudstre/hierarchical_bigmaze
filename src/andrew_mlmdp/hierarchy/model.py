"""Two-layer multitask LMDPs for maze navigation.

The module follows the paper's construction in order: augment the physical
process with subgoal boundaries, derive first-hit dynamics, construct the task
basis, and solve the abstract layer. Intermediate arrays remain public so a
researcher can inspect every calculation directly.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from torch import Tensor, detach

from andrew_mlmdp.lmdp import (
    Dynamics,
    Environment,
    Parameters,
    _numpy,
    point_parameters,
)
from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.profiles import (
    ProfileNormalization,
    _normalize_profile_columns,
    _validate_profile_normalization,
)

if TYPE_CHECKING:
    from andrew_mlmdp.dataset import Trial
    from andrew_mlmdp.hierarchy.fitting import (
        FitResult,
        FitStep,
    )
    from andrew_mlmdp.hierarchy.prediction import MovementPredictions
    from andrew_mlmdp.hierarchy.rollout import Rollout


@dataclass(frozen=True)
class TaskLibrary:
    """Immutable boundary-desirability dictionary for Layer-1 composition."""

    boundary_desirability: np.ndarray
    target_value: float | None = None
    off_target_value: float | None = None
    goal_value: float | None = None

    def __post_init__(self) -> None:
        boundary = np.asarray(self.boundary_desirability, dtype=np.float64).copy()
        if boundary.ndim != 2 or boundary.shape[0] != boundary.shape[1]:
            raise ValueError("Layer-1 task library must be a square matrix")
        if boundary.shape[0] < 2:
            raise ValueError("Layer-1 task library requires at least two tasks")
        if np.any(boundary < 0.0) or not np.all(np.isfinite(boundary)):
            raise ValueError(
                "Layer-1 task library must be finite and non-negative"
            )
        singular_values = np.linalg.svd(boundary, compute_uv=False)
        effective_rank = int(np.linalg.matrix_rank(boundary))
        if effective_rank != boundary.shape[0]:
            raise ValueError("Layer-1 task library must have full rank")
        boundary.flags.writeable = False
        singular_values.flags.writeable = False
        object.__setattr__(self, "boundary_desirability", boundary)
        object.__setattr__(self, "_singular_values", singular_values)
        object.__setattr__(self, "_effective_rank", effective_rank)

    @classmethod
    def from_desirabilities(
        cls,
        n_subgoals: int,
        *,
        target_value: float = 1.0,
        off_target_value: float = 0.0,
        goal_value: float = 1.0,
    ) -> "TaskLibrary":
        """Build the standard block-diagonal multitask dictionary.

        The canonical defaults form an identity matrix. A positive
        ``off_target_value`` explicitly opts into finite desirability leakage
        between subgoal tasks.
        """

        if (
            isinstance(n_subgoals, (bool, np.bool_))
            or not isinstance(n_subgoals, (int, np.integer))
            or n_subgoals < 1
        ):
            raise ValueError("n_subgoals must be a positive integer")
        metadata = (
            target_value,
            off_target_value,
            goal_value,
        )
        if not np.all(np.isfinite(metadata)) or np.any(np.asarray(metadata) < 0.0):
            raise ValueError(
                "Task-library desirabilities must be finite and non-negative"
            )
        n_tasks = n_subgoals + 1
        boundary = np.zeros((n_tasks, n_tasks), dtype=np.float64)
        boundary[:-1, :-1] = off_target_value
        np.fill_diagonal(boundary[:-1, :-1], target_value)
        boundary[-1, -1] = goal_value
        return cls(
            boundary,
            target_value=float(target_value),
            off_target_value=float(off_target_value),
            goal_value=float(goal_value),
        )

    @classmethod
    def from_matrix(cls, boundary_desirability: np.ndarray) -> "TaskLibrary":
        """Snapshot an arbitrary fixed full-rank dictionary."""

        return cls(boundary_desirability)

    @property
    def singular_values(self) -> np.ndarray:
        return self._singular_values

    @property
    def effective_rank(self) -> int:
        return self._effective_rank

    @property
    def condition_number(self) -> float:
        return float(self._singular_values[0] / self._singular_values[-1])


@dataclass(frozen=True)
class ThresholdRange:
    """Goal-conditioned structural domain for a distributed-basis gate."""

    maximum: float
    limiting_pairs: tuple[tuple[Coordinate, int], ...]


@dataclass(frozen=True)
class SubgoalBasis:
    """A reusable point or distributed subgoal basis for any maze.

    ``profiles`` retains the caller's immutable normalized profiles.
    ``access_profiles`` contains the optional core-gated execution view.
    Point subgoals are represented by one-hot profile columns and therefore
    use the exact same hierarchy construction as distributed subtasks.
    """

    maze: Maze
    profiles: np.ndarray
    access_profiles: np.ndarray
    locations: tuple[Coordinate, ...] | None = None
    labels: tuple[str, ...] | None = None
    core_threshold: float | None = None
    core_exponent: float = 1.0
    profile_normalization: ProfileNormalization = "peak"

    def __post_init__(self) -> None:
        _validate_profile_normalization(self.profile_normalization)
        profiles = _validate_profiles(self.maze, self.profiles).copy()
        access = _validate_profiles(
            self.maze,
            self.access_profiles,
        ).copy()
        if access.shape != profiles.shape:
            raise ValueError(
                "Access profiles must have the same shape as profiles"
            )
        locations = None if self.locations is None else tuple(self.locations)
        if locations is not None:
            locations = _validate_subgoals(self.maze, locations)
            if len(locations) != profiles.shape[1]:
                raise ValueError(
                    "Point locations must match the number of profiles"
                )
        labels = self.labels
        if labels is not None:
            labels = tuple(str(label) for label in labels)
            if len(labels) != profiles.shape[1]:
                raise ValueError(
                    "Labels must match the number of subgoals"
                )
        profiles.flags.writeable = False
        access.flags.writeable = False
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "access_profiles", access)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_locations(
        cls,
        maze: Maze,
        locations: list[Coordinate] | tuple[Coordinate, ...],
        *,
        labels: list[str] | tuple[str, ...] | None = None,
    ) -> "SubgoalBasis":
        """Create a one-hot basis from arbitrary free-cell locations."""

        ordered = _validate_subgoals(maze, locations)
        profiles = np.zeros(
            (len(maze.free_cells), len(ordered)),
            dtype=np.float64,
        )
        for index, coordinate in enumerate(ordered):
            profiles[maze.state_index(coordinate), index] = 1.0
        return cls(
            maze=maze,
            profiles=profiles,
            access_profiles=profiles,
            locations=ordered,
            labels=None if labels is None else tuple(labels),
        )

    @classmethod
    def from_profiles(
        cls,
        maze: Maze,
        profiles: np.ndarray,
        *,
        core_threshold: float | Tensor | None = 0.8,
        core_exponent: float | Tensor = 1.0,
        labels: list[str] | tuple[str, ...] | None = None,
        profile_normalization: ProfileNormalization = "peak",
    ) -> "SubgoalBasis":
        """Create a normalized distributed basis and gate it once.

        Peak normalization is the default. Pass ``profile_normalization="l2"``
        to normalize both the stored and gated profiles to unit L2 norm.
        """

        threshold = (
            None
            if core_threshold is None
            else _detached_scalar(core_threshold)
        )
        exponent = _detached_scalar(core_exponent)
        supplied = _validate_profiles(maze, profiles)
        normalized, _ = _normalize_profile_columns(
            supplied,
            profile_normalization,
            empty_message="Every soft subtask profile must be nonempty",
        )
        access = _soft_core_profiles(
            normalized,
            threshold=threshold,
            exponent=exponent,
            profile_normalization=profile_normalization,
        )
        return cls(
            maze=maze,
            profiles=normalized,
            access_profiles=access,
            labels=None if labels is None else tuple(labels),
            core_threshold=threshold,
            core_exponent=exponent,
            profile_normalization=profile_normalization,
        )

    @property
    def n_subgoals(self) -> int:
        """Number of columns in the reusable subgoal basis."""

        return self.profiles.shape[1]

    @property
    def is_point_basis(self) -> bool:
        return self.locations is not None


class Template:
    """Goal-independent hierarchy configuration with per-goal task caching."""

    def __init__(
        self,
        *,
        environment: Environment,
        basis: SubgoalBasis,
        parameters: Parameters | None = None,
        task_library: TaskLibrary | None = None,
        composition_exponent: float = 1.0,
        composition_mode: Literal["power", "winner_take_all"] = "power",
    ) -> None:
        if basis.maze != environment.maze:
            raise ValueError(
                "Subgoal basis and environment must use the same maze"
            )
        if parameters is None:
            parameters = point_parameters()
        if task_library is None:
            task_library = TaskLibrary.from_desirabilities(
                basis.n_subgoals
            )
        if not isinstance(task_library, TaskLibrary):
            raise TypeError("task_library must be a TaskLibrary")
        expected_library_shape = (
            basis.n_subgoals + 1,
            basis.n_subgoals + 1,
        )
        if task_library.boundary_desirability.shape != expected_library_shape:
            raise ValueError(
                "Layer-1 task library must have shape "
                f"{expected_library_shape}, got "
                f"{task_library.boundary_desirability.shape}"
            )
        exponent = _detached_scalar(composition_exponent)
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("composition_exponent must be finite and positive")
        if composition_mode not in ("power", "winner_take_all"):
            raise ValueError(
                "composition_mode must be 'power' or 'winner_take_all'"
            )
        self.environment = environment
        self.basis = basis
        self.parameters = parameters
        self.task_library = task_library
        self.composition_exponent = exponent
        self.composition_mode = composition_mode
        self._task_cache: dict[Coordinate, Task] = {}
        self._passive_dynamics: np.ndarray | None = None

    @property
    def maze(self) -> Maze:
        return self.environment.maze

    def threshold_range(
        self,
        goals: Iterable[Coordinate] | None = None,
    ) -> ThresholdRange:
        """Return the strict threshold bound for the supplied physical goals.

        For every goal and subgoal component, at least one non-goal physical
        state must retain positive gated access. Therefore a threshold is
        structurally valid exactly when it is smaller than the minimum of the
        goal-conditioned component maxima.
        """

        if goals is None:
            goal_coordinates = tuple(self.maze.free_cells)
        else:
            goal_coordinates = tuple(dict.fromkeys(goals))
        if not goal_coordinates:
            raise ValueError("At least one physical goal is required")
        profiles = self.basis.profiles
        relative_profiles = profiles / profiles.max(axis=0, keepdims=True)
        candidates: list[tuple[float, Coordinate, int]] = []
        for goal in goal_coordinates:
            goal_state = self.maze.state_index(goal)
            keep = np.arange(len(profiles)) != goal_state
            if not np.any(keep):
                raise ValueError(
                    "A goal-conditioned hierarchy requires a non-goal state"
                )
            maxima = relative_profiles[keep].max(axis=0)
            candidates.extend(
                (float(value), goal, subgoal_index)
                for subgoal_index, value in enumerate(maxima)
            )
        maximum = min(value for value, _, _ in candidates)
        limiting_pairs = tuple(
            (goal, subgoal_index)
            for value, goal, subgoal_index in candidates
            if value == maximum
        )
        return ThresholdRange(maximum, limiting_pairs)

    def validate_threshold(
        self,
        threshold: float | Tensor,
        goals: Iterable[Coordinate],
    ) -> ThresholdRange:
        """Validate a public physical gate threshold for a goal set."""

        value = _detached_scalar(threshold)
        domain = self.threshold_range(goals)
        if not 0.0 <= value < domain.maximum:
            raise ValueError(
                "core_threshold must satisfy 0 <= threshold < "
                f"{domain.maximum:.17g} for the requested goals; limiting "
                f"(goal, subgoal) pairs are {domain.limiting_pairs}"
            )
        return domain

    @property
    def upper_passive(self) -> np.ndarray:
        """Return task-independent passive dynamics between basis states."""

        if self._passive_dynamics is None:
            from andrew_mlmdp.hierarchy.equations import (
                _template_upper_passive,
                parameter_values,
            )

            with torch.no_grad():
                upper = _template_upper_passive(
                    self,
                    parameter_values(self),
                )
            values = _public_array(upper)
            values.flags.writeable = False
            self._passive_dynamics = values
        upper_passive = self._passive_dynamics
        assert upper_passive is not None
        return upper_passive

    def task(self, goal: Coordinate) -> "Task":
        """Return a cached goal-conditioned hierarchy task."""

        self.maze.state_index(goal)
        if (
            self.basis.locations is not None
            and goal in self.basis.locations
        ):
            raise ValueError("The goal and point subgoals must be disjoint")
        if self.basis.core_threshold is not None:
            self.validate_threshold(
                self.basis.core_threshold,
                (goal,),
            )
        task = self._task_cache.get(goal)
        if task is None:
            task = _build_task(self, goal)
            self._task_cache[goal] = task
        return task

    def parameter_values(
        self,
        *,
        overrides: Mapping[str, "Tensor"] | None = None,
    ) -> dict[str, "Tensor"]:
        """Return strict physical tensors for the differentiable hierarchy."""

        from andrew_mlmdp.hierarchy.equations import (
            parameter_values,
        )

        return parameter_values(self, overrides=overrides)

    def log_likelihood(
        self,
        goal: Coordinate,
        trajectory: list[Coordinate] | tuple[Coordinate, ...],
        *,
        parameter_overrides: Mapping[str, "Tensor"] | None = None,
    ) -> "Tensor":
        """Score one trajectory through the fresh differentiable hierarchy."""

        from andrew_mlmdp.hierarchy.likelihood import (
            log_likelihood,
        )

        values = self.parameter_values(overrides=parameter_overrides)
        return log_likelihood(
            self,
            goal,
            trajectory,
            parameter_values=values,
        )

    def total_log_likelihood(
        self,
        trials: Iterable["Trial"],
        *,
        parameter_overrides: Mapping[str, "Tensor"] | None = None,
    ) -> "Tensor":
        """Sum independent trajectory scores in one differentiable graph."""

        from andrew_mlmdp.hierarchy.likelihood import (
            total_log_likelihood,
        )

        values = self.parameter_values(overrides=parameter_overrides)
        return total_log_likelihood(
            self,
            trials,
            parameter_values=values,
        )


    def fit(
        self,
        trials: Iterable["Trial"],
        *,
        names: Sequence[str],
        lr: float = 5e-2,
        max_steps: int = 1000,
        tolerance: float = 1e-8,
        scheduler_tolerance: float | None = None,
        convergence_tolerance: float | None = None,
        patience: int = 20,
        lr_decay: float = 0.3,
        lr_patience: int = 7,
        min_lr: float = 1e-5,
        callback: (
            Callable[["FitStep"], None] | None
        ) = None,
    ) -> "FitResult":
        """Fit private Torch parameters without mutating this template."""

        from andrew_mlmdp.hierarchy.fitting import (
            fit_parameters,
        )

        return fit_parameters(
            self,
            trials,
            names=names,
            lr=lr,
            max_steps=max_steps,
            tolerance=tolerance,
            scheduler_tolerance=scheduler_tolerance,
            convergence_tolerance=convergence_tolerance,
            patience=patience,
            lr_decay=lr_decay,
            lr_patience=lr_patience,
            min_lr=min_lr,
            callback=callback,
        )

@dataclass(frozen=True)
class Task:
    """Inspectable goal-conditioned task built from a reusable hierarchy."""

    template: Template
    goal: Coordinate
    interior_states: np.ndarray
    interior_index: dict[Coordinate, int]
    lower_dynamics: Dynamics
    first_hit: np.ndarray
    task_basis: "TaskBasis"
    upper_dynamics: Dynamics
    upper_desirability: np.ndarray
    upper_controlled: np.ndarray
    _tensor_model: object

    @property
    def maze(self) -> Maze:
        return self.template.maze

    @property
    def basis(self) -> SubgoalBasis:
        return self.template.basis

    @property
    def parameters(self) -> Parameters:
        return self.template.parameters

    @property
    def n_subtasks(self) -> int:
        """Number of reusable lower-layer subtasks."""

        return self.basis.n_subgoals

    @property
    def subtask_profiles(self) -> np.ndarray:
        return self.basis.access_profiles

    @property
    def subgoals(self) -> tuple[Coordinate, ...]:
        return () if self.basis.locations is None else self.basis.locations

    @property
    def subtask_access(self) -> np.ndarray:
        return self.lower_dynamics.boundary_passive[:-1]

    def plan(
        self,
        current: Coordinate,
        *,
        upper_state: int | None = None,
        beta: float | None = None,
        goal_desirability: np.ndarray | None = None,
    ) -> "Plan":
        """Compose the lower policy at a physical or entered upper state."""

        return compute_plan(
            self,
            current,
            upper_state=upper_state,
            beta=beta,
            goal_desirability=goal_desirability,
        )

    def log_likelihood(
        self,
        trajectory: list[Coordinate] | tuple[Coordinate, ...],
        *,
        beta: float | None = None,
    ) -> float:
        """Score physical movement after marginalizing hierarchy events.

        Consecutive repeated coordinates are collapsed, matching
        :meth:`Solution.log_likelihood`. Lower accesses and upper
        termination decisions are latent; their mutually exclusive routes are
        summed exactly rather than sampled.
        """

        overrides = None
        if beta is not None:
            overrides = {
                "beta": torch.as_tensor(
                    beta,
                    dtype=torch.float64,
                    device=self.parameters.beta.device,
                )
            }
        with torch.no_grad():
            score = self.template.log_likelihood(
                self.goal,
                trajectory,
                parameter_overrides=overrides,
            )
        return float(score.detach().cpu())

    def movement_predictions(
        self,
        trajectory: Sequence[Coordinate],
    ) -> "MovementPredictions":
        """Predict each departure before conditioning on that movement."""

        from andrew_mlmdp.hierarchy.prediction import movement_predictions

        return movement_predictions(self, trajectory)

    def rollout(
        self,
        start: Coordinate,
        *,
        goal_learning: Literal["exact", "online"] = "exact",
        initial_goal_desirability: np.ndarray | None = None,
        z_sweeps_per_step: int = 1,
        beta: float | None = None,
        max_steps: int = 500,
        max_abstract_accesses: int = 500,
        seed: int | None = None,
    ) -> "Rollout":
        """Run exact or online execution through one shared state machine."""

        if goal_learning not in {"exact", "online"}:
            raise ValueError("goal_learning must be 'exact' or 'online'")
        from andrew_mlmdp.hierarchy.rollout import (
            _rollout_from_engine,
            _run_rollout,
        )

        result = _run_rollout(
            self,
            start,
            beta=beta,
            max_steps=max_steps,
            max_abstract_accesses=max_abstract_accesses,
            seed=seed,
            initial_goal_desirability=initial_goal_desirability,
            z_sweeps_per_step=(
                z_sweeps_per_step if goal_learning == "online" else None
            ),
        )
        return _rollout_from_engine(self, result, goal_learning)


@dataclass(frozen=True)
class TaskBasis:
    """Boundary tasks and their solved interior desirabilities.

    Columns are component tasks. ``boundary_desirability`` is the paper's
    ``Q_b`` with shape ``(n_boundary, n_tasks)``; ``interior_desirability`` is
    ``Z_i`` with shape ``(n_interior, n_tasks)``.
    """

    boundary_desirability: np.ndarray
    interior_desirability: np.ndarray

    def __post_init__(self) -> None:
        boundary = np.asarray(self.boundary_desirability, dtype=np.float64)
        interior = np.asarray(self.interior_desirability, dtype=np.float64)
        if boundary.ndim != 2 or interior.ndim != 2:
            raise ValueError("Task-basis arrays must be matrices")
        if boundary.shape[1] != interior.shape[1]:
            raise ValueError("Task-basis matrices must have the same columns")
        object.__setattr__(self, "boundary_desirability", boundary)
        object.__setattr__(self, "interior_desirability", interior)


@dataclass(frozen=True)
class Plan:
    """Top-down task composition and lower policy at one physical location."""

    current: Coordinate
    upper_state: int | None
    upper_passive: np.ndarray
    upper_policy: np.ndarray
    rewards: np.ndarray
    target_boundary: np.ndarray
    raw_weights: np.ndarray
    clipped_weights: np.ndarray
    weights: np.ndarray
    boundary_desirability: np.ndarray
    desirability: np.ndarray
    lower_policy: np.ndarray


def _public_array(value: Tensor) -> np.ndarray:
    """Copy a detached tensor into a researcher-facing NumPy array."""

    return _numpy(value).copy()


def _build_task(
    template: Template,
    goal: Coordinate,
) -> Task:
    """Build one cached public task from the single Torch hierarchy."""

    from andrew_mlmdp.hierarchy.equations import (
        _build_hierarchy,
        parameter_values,
    )

    with torch.no_grad():
        tensor_model = _build_hierarchy(
            template,
            goal,
            parameter_values(template),
        )
    return Task(
        template=template,
        goal=goal,
        interior_states=np.asarray(
            tensor_model.interior_states,
            dtype=np.int64,
        ),
        interior_index=dict(tensor_model.interior_index),
        lower_dynamics=Dynamics(
            _public_array(tensor_model.lower_dynamics.interior_passive),
            _public_array(tensor_model.lower_dynamics.boundary_passive),
        ),
        first_hit=_public_array(tensor_model.first_hit),
        task_basis=TaskBasis(
            _public_array(tensor_model.task_basis.boundary_desirability),
            _public_array(tensor_model.task_basis.interior_desirability),
        ),
        upper_dynamics=Dynamics(
            _public_array(tensor_model.upper_dynamics.interior_passive),
            _public_array(tensor_model.upper_dynamics.boundary_passive),
        ),
        upper_desirability=_public_array(tensor_model.upper_desirability),
        upper_controlled=_public_array(tensor_model.upper_controlled),
        _tensor_model=tensor_model,
    )


def _tensor_goal_desirability(
    model: Task,
    values: np.ndarray | None,
) -> Tensor | None:
    if values is None:
        return None
    validated = _validate_goal_desirability(model, values)
    tensor_model = cast(object, model._tensor_model)
    return torch.as_tensor(
        validated,
        dtype=tensor_model.dtype,
        device=tensor_model.device,
    )


def _public_plan(
    current: Coordinate,
    upper_state: int | None,
    plan,
) -> Plan:
    return Plan(
        current=current,
        upper_state=upper_state,
        upper_passive=_public_array(plan.upper_passive),
        upper_policy=_public_array(plan.upper_policy),
        rewards=_public_array(plan.rewards),
        target_boundary=_public_array(plan.target_boundary),
        raw_weights=_public_array(plan.raw_weights),
        clipped_weights=_public_array(plan.clipped_weights),
        weights=_public_array(plan.weights),
        boundary_desirability=_public_array(plan.boundary_desirability),
        desirability=_public_array(plan.desirability),
        lower_policy=_public_array(plan.lower_policy),
    )


def compute_plan(
    model: Task,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None = None,
    goal_desirability: np.ndarray | None = None,
) -> Plan:
    """Compose the lower policy through the single Torch equation path."""

    from andrew_mlmdp.hierarchy.equations import _plan

    resolved_upper = (
        None
        if upper_state is None
        else _validated_upper_state(upper_state, model.n_subtasks)
    )
    with torch.no_grad():
        plan = _plan(
            model._tensor_model,
            current,
            upper_state=resolved_upper,
            beta=beta,
            goal_desirability=_tensor_goal_desirability(
                model,
                goal_desirability,
            ),
        )
    return _public_plan(current, resolved_upper, plan)


def _validated_upper_state(value: int, n_states: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < n_states
    ):
        raise ValueError("Upper state index is out of range")
    return int(value)


def _goal_only_plan(
    model: Task,
    current: Coordinate,
    *,
    goal_desirability: np.ndarray | None,
    tolerate_unreachable: bool = False,
) -> Plan:
    from andrew_mlmdp.hierarchy.equations import _goal_only_plan as tensor_plan

    with torch.no_grad():
        plan = tensor_plan(
            model._tensor_model,
            goal_desirability=_tensor_goal_desirability(
                model,
                goal_desirability,
            ),
            tolerate_unreachable=tolerate_unreachable,
        )
    return _public_plan(current, None, plan)


def _validate_goal_desirability(
    model: Task,
    values: np.ndarray,
) -> np.ndarray:
    goal_desirability = np.asarray(values, dtype=np.float64)
    expected_shape = (len(model.interior_states),)
    if goal_desirability.shape != expected_shape:
        raise ValueError(
            "Initial goal desirability must have shape "
            f"{expected_shape}, got {goal_desirability.shape}"
        )
    if (
        np.any(goal_desirability < 0.0)
        or not np.all(np.isfinite(goal_desirability))
    ):
        raise ValueError(
            "Initial goal desirability must be finite and non-negative"
        )
    return goal_desirability


def _goal_plan(
    model: Task,
    plan: Plan,
    goal_desirability: np.ndarray,
) -> Plan:
    from andrew_mlmdp.hierarchy.equations import (
        _compose_policy,
    )
    from andrew_mlmdp.hierarchy.equations import (
        _goal_only_plan as tensor_goal_only_plan,
    )

    learned = _tensor_goal_desirability(model, goal_desirability)
    assert learned is not None
    with torch.no_grad():
        if np.all(np.isneginf(plan.rewards[:-1])):
            updated = tensor_goal_only_plan(
                model._tensor_model,
                goal_desirability=learned,
            )
            physical = updated.desirability
            controlled = updated.lower_policy
        else:
            physical, controlled = _compose_policy(
                model._tensor_model,
                torch.as_tensor(
                    plan.weights,
                    dtype=learned.dtype,
                    device=learned.device,
                ),
                torch.as_tensor(
                    plan.boundary_desirability,
                    dtype=learned.dtype,
                    device=learned.device,
                ),
                goal_desirability=learned,
            )
    return replace(
        plan,
        desirability=_public_array(physical),
        lower_policy=_public_array(controlled),
    )


def _plan_from_weights(
    model: Task,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None,
    goal_desirability: np.ndarray | None,
) -> Plan:
    return compute_plan(
        model,
        current,
        upper_state=upper_state,
        beta=beta,
        goal_desirability=goal_desirability,
    )



def _validate_subgoals(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
) -> tuple[Coordinate, ...]:
    ordered_subgoals = tuple(subgoals)
    if not ordered_subgoals:
        raise ValueError("At least one subgoal is required")
    if len(set(ordered_subgoals)) != len(ordered_subgoals):
        raise ValueError("Subgoals must be unique")
    for subgoal in ordered_subgoals:
        maze.state_index(subgoal)
    return ordered_subgoals


def _validate_profiles(
    maze: Maze,
    profiles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float64)
    expected_rows = len(maze.free_cells)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(
            "Subtask profiles must have shape "
            f"({expected_rows}, n_subtasks)"
        )
    if not values.shape[1]:
        raise ValueError("At least one soft subtask is required")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("Subtask profiles must be finite and non-negative")
    if np.any(values.max(axis=0) <= 0.0):
        raise ValueError("Every soft subtask profile must be nonempty")
    return values


def _detached_scalar(value: float | Tensor) -> float:
    """Snapshot a scalar tensor for structural validation."""

    if isinstance(value, Tensor):
        return float(detach(cast(Tensor, value)))
    return value


def _soft_core_profiles(
    profiles: np.ndarray,
    *,
    threshold: float | None,
    exponent: float,
    profile_normalization: ProfileNormalization,
) -> np.ndarray:
    """Restrict soft access to the peak-relative core of each profile."""

    values = np.asarray(profiles, dtype=np.float64)
    if (
        not np.isfinite(exponent)
        or isinstance(exponent, (bool, np.bool_))
        or exponent <= 0.0
    ):
        raise ValueError("Core exponent must be finite and positive")
    if threshold is None:
        return values.copy()
    if (
        isinstance(threshold, (bool, np.bool_))
        or not np.isfinite(threshold)
        or not 0.0 <= threshold < 1.0
    ):
        raise ValueError(
            "Core threshold must be finite and in [0, 1), or None"
        )

    peaks = values.max(axis=0, keepdims=True)
    normalized = values / peaks
    core = np.maximum(
        0.0,
        (normalized - threshold) / (1.0 - threshold),
    )
    if exponent != 1.0:
        core = core**exponent
    normalized_core, _ = _normalize_profile_columns(
        core,
        profile_normalization,
        empty_message="Every soft subtask profile must be nonempty",
    )
    return normalized_core
