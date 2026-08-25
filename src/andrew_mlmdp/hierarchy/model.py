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
from torch import Tensor, detach

from andrew_mlmdp.lmdp import (
    Dynamics,
    Environment,
    Parameters,
    controlled_dynamics,
    point_parameters,
    solve_first_exit,
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
            access = self.parameters.alpha.item() * self.basis.access_profiles.T
            interior = self.environment.passive.copy()
            interior, access = _normalize_columns(interior, access)
            fundamental = _fundamental_matrix(interior)
            upper = access @ fundamental @ access.T
            column_sums = upper.sum(axis=0)
            if np.any(column_sums <= 0.0):
                raise ValueError(
                    "A subgoal has no reachable abstract target"
                )
            upper = upper / column_sums[np.newaxis, :]
            upper.flags.writeable = False
            self._passive_dynamics = upper
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

        from andrew_mlmdp.hierarchy.autodiff import (
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

        from andrew_mlmdp.hierarchy.autodiff import (
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

        from andrew_mlmdp.hierarchy.autodiff import (
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

        from andrew_mlmdp.hierarchy.likelihood import (
            _log_likelihood,
        )

        return _log_likelihood(
            self,
            trajectory,
            beta=beta,
        )

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


def _build_task(
    template: Template,
    goal: Coordinate,
) -> Task:
    """Build one goal task from a reusable point or distributed basis."""

    # Remove the absorbing goal, then express every subtask on that interior.
    interior_states, interior_by_coordinate = _interior_partition(
        template.maze,
        goal,
    )
    raw_access = (
        template.parameters.alpha.item()
        * template.basis.access_profiles[interior_states, :].T
    )
    if np.any(raw_access.max(axis=1) <= 0.0):
        raise ValueError(
            "Every subgoal must have positive access outside the goal"
        )
    # Layer 1 ends on either a subtask boundary copy or the physical goal.
    lower_dynamics = _lower_dynamics(
        template.maze,
        goal,
        interior_states,
        raw_access,
        physical_passive=template.environment.passive,
    )
    fundamental = _fundamental_matrix(lower_dynamics.interior_passive)
    first_hit = (
        lower_dynamics.boundary_passive @ fundamental
    )
    # First-hit probabilities become passive transitions for the abstract layer.
    upper_dynamics = _upper_dynamics(lower_dynamics, fundamental)
    # Solve the goal-conditioned abstract policy once; plans compose it below.
    upper_desirability, upper_controlled = _solve_upper(
        upper_dynamics,
        template.parameters,
    )
    return Task(
        template=template,
        goal=goal,
        interior_states=interior_states,
        interior_index=interior_by_coordinate,
        lower_dynamics=lower_dynamics,
        first_hit=first_hit,
        task_basis=_task_basis(
            lower_dynamics,
            template.parameters,
            template.task_library.boundary_desirability,
        ),
        upper_dynamics=upper_dynamics,
        upper_desirability=upper_desirability,
        upper_controlled=upper_controlled,
    )


def compute_plan(
    model: Task,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None = None,
    goal_desirability: np.ndarray | None = None,
) -> Plan:
    """Compose a lower plan for point or distributed subgoal bases."""

    model.maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")

    if upper_state is not None:
        abstract_state = _validated_upper_state(
            upper_state,
            model.n_subtasks,
        )
        upper_passive = model.upper_dynamics.passive[
            :, abstract_state
        ].copy()
        upper_policy = model.upper_controlled[
            :, abstract_state
        ].copy()
    elif (
        model.basis.locations is not None
        and current in model.basis.locations
    ):
        abstract_state = model.basis.locations.index(current)
        upper_passive = model.upper_dynamics.passive[
            :, abstract_state
        ].copy()
        upper_policy = model.upper_controlled[
            :, abstract_state
        ].copy()
    else:
        interior_state = model.interior_index[current]
        upper_passive = model.first_hit[
            :, interior_state
        ].copy()
        upper_policy = upper_passive * model.upper_desirability
        upper_policy /= upper_policy.sum()

    return _compose_plan(
        model,
        current,
        upper_passive,
        upper_policy,
        upper_state=upper_state,
        beta=beta,
        goal_desirability=goal_desirability,
    )


def _validated_upper_state(value: int, n_states: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < n_states
    ):
        raise ValueError("Upper state index is out of range")
    return int(value)


def _compose_plan(
    model: Task,
    current: Coordinate,
    upper_passive: np.ndarray,
    upper_policy: np.ndarray,
    *,
    upper_state: int | None,
    beta: float | None,
    goal_desirability: np.ndarray | None,
) -> Plan:
    """Apply reward inpainting and lower task composition."""

    inpainting_scale = model.parameters.beta.item() if beta is None else beta
    if not np.isfinite(inpainting_scale) or inpainting_scale <= 0.0:
        raise ValueError("Beta must be finite and positive")

    # Apply Equation 10 to every abstract outcome, including termination at
    # the physical goal. The fixed goal reward still defines the exact goal
    # basis task, while its active mixture coefficient is set by this target.
    rewards = inpainting_scale * (
        upper_policy - upper_passive
    )
    target_boundary = np.exp(
        rewards / model.parameters.lower_control_cost.item()
    )

    # Paper Equation 7, using its stated pseudoinverse-and-clipping
    # approximation when a target lies outside the non-negative cone of Q_b.
    # For the canonical identity library, raw_weights equals target_boundary
    # and clipping is a no-op.
    raw_weights = (
        np.linalg.pinv(model.task_basis.boundary_desirability)
        @ target_boundary
    )
    clipped_weights = np.maximum(0.0, raw_weights)
    weights = _shape_weights(
        clipped_weights,
        exponent=model.template.composition_exponent,
        mode=model.template.composition_mode,
    )
    reconstructed_boundary = (
        model.task_basis.boundary_desirability @ weights
    )
    desirability, lower_policy = _compose_policy(
        model,
        weights,
        reconstructed_boundary,
        goal_desirability=goal_desirability,
    )
    return Plan(
        current=current,
        upper_state=upper_state,
        upper_passive=upper_passive,
        upper_policy=upper_policy,
        rewards=rewards,
        target_boundary=target_boundary,
        raw_weights=raw_weights,
        clipped_weights=clipped_weights,
        weights=weights,
        boundary_desirability=reconstructed_boundary,
        desirability=desirability,
        lower_policy=lower_policy,
    )


def _goal_only_plan(
    model: Task,
    current: Coordinate,
    *,
    goal_desirability: np.ndarray | None,
    tolerate_unreachable: bool = False,
) -> Plan:
    """Construct the permanent physical-goal plan after upper termination."""

    n_boundaries = model.lower_dynamics.n_boundary
    weights = np.zeros(n_boundaries, dtype=np.float64)
    weights[-1] = 1.0
    inpainted = np.full(n_boundaries, -np.inf, dtype=np.float64)
    inpainted[-1] = model.parameters.goal_reward.item()
    target = np.zeros(n_boundaries, dtype=np.float64)
    target[-1] = np.exp(
        model.parameters.goal_reward.item() / model.parameters.lower_control_cost.item()
    )
    if goal_desirability is None:
        q_interior = np.exp(
            model.parameters.interior_reward.item()
            / model.parameters.lower_control_cost.item()
        )
        interior = solve_first_exit(model.lower_dynamics, target, q_interior)
    else:
        interior = _validate_goal_desirability(
            model,
            goal_desirability,
        )
    physical, controlled = _lower_policy(
        model,
        interior,
        target,
        tolerate_zero_columns=(
            goal_desirability is not None
            or tolerate_unreachable
        ),
    )
    return Plan(
        current=current,
        upper_state=None,
        upper_passive=np.zeros(n_boundaries, dtype=np.float64),
        upper_policy=np.zeros(n_boundaries, dtype=np.float64),
        rewards=inpainted,
        target_boundary=target,
        raw_weights=weights.copy(),
        clipped_weights=weights.copy(),
        weights=weights,
        boundary_desirability=target,
        desirability=physical,
        lower_policy=controlled,
    )


def _compose_policy(
    model: Task,
    weights: np.ndarray,
    reconstructed_boundary: np.ndarray,
    *,
    goal_desirability: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine fixed subtask solutions with an exact or learned goal column."""

    basis = model.task_basis.interior_desirability
    if goal_desirability is None:
        interior_desirability = basis @ weights
    else:
        learned_goal = _validate_goal_desirability(
            model,
            goal_desirability,
        )
        interior_desirability = (
            basis[:, :-1] @ weights[:-1]
            + learned_goal * weights[-1]
        )

    return _lower_policy(
        model,
        interior_desirability,
        reconstructed_boundary,
        tolerate_zero_columns=goal_desirability is not None,
    )


def _lower_policy(
    model: Task,
    interior_desirability: np.ndarray,
    boundary_desirability: np.ndarray,
    *,
    tolerate_zero_columns: bool,
) -> tuple[np.ndarray, np.ndarray]:
    desirability = np.empty(
        len(model.maze.free_cells),
        dtype=np.float64,
    )
    desirability[model.interior_states] = interior_desirability
    goal_state = model.maze.state_index(model.goal)
    desirability[goal_state] = boundary_desirability[-1]

    complete_desirability = np.concatenate(
        [interior_desirability, boundary_desirability]
    )
    if not tolerate_zero_columns:
        controlled = controlled_dynamics(
            model.lower_dynamics.passive,
            complete_desirability,
        )
    else:
        # Early online iterates may leave states with no usable desirability.
        # Keep those columns at zero so rollout code can report ``zero_policy``.
        unnormalized = (
            model.lower_dynamics.passive
            * complete_desirability[:, np.newaxis]
        )
        normalizers = unnormalized.sum(axis=0)
        controlled = np.zeros_like(unnormalized)
        usable = np.isfinite(normalizers) & (normalizers > 0.0)
        controlled[:, usable] = (
            unnormalized[:, usable] / normalizers[usable]
        )
    return desirability, controlled


def _shape_weights(
    clipped_weights: np.ndarray,
    *,
    exponent: float,
    mode: Literal["power", "winner_take_all"],
) -> np.ndarray:
    """Redistribute only subgoal weight mass; preserve the goal component."""

    weights = np.asarray(clipped_weights, dtype=np.float64)
    if mode == "power" and exponent == 1.0:
        return weights
    result = weights.copy()
    subgoal = weights[:-1]
    mass = float(subgoal.sum())
    if mass <= 0.0:
        return result
    if mode == "winner_take_all":
        maxima = subgoal == subgoal.max()
        result[:-1] = 0.0
        subgoal_result = result[:-1]
        subgoal_result[maxima] = mass / int(np.count_nonzero(maxima))
        return result
    positive = subgoal > 0.0
    powered = np.zeros_like(subgoal)
    powered[positive] = (subgoal[positive] / mass) ** exponent
    result[:-1] = mass * powered / powered.sum()
    return result


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
    if np.all(np.isneginf(plan.rewards[:-1])):
        learned_goal = _validate_goal_desirability(model, goal_desirability)
        physical, controlled = _lower_policy(
            model,
            learned_goal,
            plan.target_boundary,
            tolerate_zero_columns=True,
        )
    else:
        physical, controlled = _compose_policy(
            model,
            plan.weights,
            plan.boundary_desirability,
            goal_desirability=goal_desirability,
        )
    return replace(
        plan,
        desirability=physical,
        lower_policy=controlled,
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
    """Snapshot a scalar tensor for the current NumPy implementation."""

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


def _interior_partition(
    maze: Maze,
    goal: Coordinate,
) -> tuple[np.ndarray, dict[Coordinate, int]]:
    goal_state = maze.state_index(goal)
    interior_states = np.asarray(
        [
            state
            for state in range(len(maze.free_cells))
            if state != goal_state
        ],
        dtype=int,
    )
    coordinate_to_interior = {
        maze.coordinate(int(physical_state)): interior_state
        for interior_state, physical_state in enumerate(interior_states)
    }
    return interior_states, coordinate_to_interior


def _normalize_columns(
    interior_passive: np.ndarray,
    boundary_passive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    column_sums = np.vstack(
        [interior_passive, boundary_passive]
    ).sum(axis=0)
    if np.any(column_sums == 0.0):
        raise ValueError("Augmented passive dynamics contain an empty column")
    interior_passive /= column_sums[np.newaxis, :]
    boundary_passive /= column_sums[np.newaxis, :]
    return interior_passive, boundary_passive


def _lower_dynamics(
    maze: Maze,
    goal: Coordinate,
    interior_states: np.ndarray,
    subtask_access: np.ndarray,
    *,
    physical_passive: np.ndarray,
) -> Dynamics:
    """Augment physical dynamics with supplied abstract access rows."""

    access = np.asarray(subtask_access, dtype=np.float64).copy()
    expected_columns = len(interior_states)
    if access.ndim != 2 or access.shape[1] != expected_columns:
        raise ValueError(
            "Subtask access must have one column per interior state"
        )
    passive = np.asarray(physical_passive, dtype=np.float64)
    expected_passive_shape = (
        len(maze.free_cells),
        len(maze.free_cells),
    )
    if passive.shape != expected_passive_shape:
        raise ValueError(
            "Physical passive dynamics must have shape "
            f"{expected_passive_shape}"
        )
    interior_passive = passive[np.ix_(interior_states, interior_states)]
    goal_state = maze.state_index(goal)
    goal_passive = passive[goal_state, interior_states][np.newaxis, :]
    boundary_passive = np.vstack([access, goal_passive])
    interior_passive, boundary_passive = _normalize_columns(
        interior_passive,
        boundary_passive,
    )
    return Dynamics(
        interior_passive,
        boundary_passive,
    )


def _fundamental_matrix(interior_passive: np.ndarray) -> np.ndarray:
    identity = np.eye(interior_passive.shape[0])
    return np.linalg.solve(identity - interior_passive, identity)


def _upper_dynamics(
    lower: Dynamics,
    fundamental: np.ndarray,
) -> Dynamics:
    lower_subgoals = lower.boundary_passive[:-1]
    lower_goal = lower.boundary_passive[-1:]
    upper_interior = lower_subgoals @ fundamental @ lower_subgoals.T
    upper_boundary = lower_goal @ fundamental @ lower_subgoals.T
    upper_interior, upper_boundary = _normalize_columns(
        upper_interior,
        upper_boundary,
    )
    return Dynamics(upper_interior, upper_boundary)


def _solve_upper(
    dynamics: Dynamics,
    parameters: Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    q_interior = np.exp(
        parameters.interior_reward.item() / parameters.upper_control_cost.item()
    )
    goal_desirability = np.exp(
        parameters.goal_reward.item() / parameters.upper_control_cost.item()
    )
    interior_desirability = solve_first_exit(
        dynamics,
        np.asarray([goal_desirability]),
        q_interior,
    )
    desirability = np.concatenate(
        [interior_desirability, np.asarray([goal_desirability])]
    )
    controlled = controlled_dynamics(
        dynamics.passive,
        desirability,
    )
    return desirability, controlled


def _task_basis(
    lower: Dynamics,
    parameters: Parameters,
    boundary_desirability: np.ndarray,
) -> TaskBasis:
    n_targets = lower.n_boundary
    boundary_basis = np.asarray(boundary_desirability, dtype=np.float64)
    expected_shape = (n_targets, n_targets)
    if boundary_basis.shape != expected_shape:
        raise ValueError(
            f"Task-library boundary matrix must have shape {expected_shape}"
        )

    q_interior = np.exp(
        parameters.interior_reward.item() / parameters.lower_control_cost.item()
    )
    interior_basis = np.column_stack(
        [
            solve_first_exit(
                lower,
                boundary_basis[:, task],
                q_interior,
            )
            for task in range(n_targets)
        ]
    )
    return TaskBasis(boundary_basis, interior_basis)
