"""Discover distributed MLMDP subtasks with non-negative factorization."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF

from andrew_mlmdp.lmdp import (
    LMDPEnvironment,
    ModelParameters,
)
from andrew_mlmdp.maze import Coordinate, Maze


@dataclass(frozen=True)
class NMFDiscoveryParameters:
    """Task-family parameters used only to discover soft subtask profiles.

    These values define the fixed desirability ensemble supplied to NMF.
    They are intentionally separate from ``ModelParameters`` so execution
    tuning cannot silently rediscover a different hierarchy.
    """

    interior_reward: float = -0.4
    goal_reward: float = 6.5
    control_cost: float = 1.2

    def __post_init__(self) -> None:
        values = (
            self.interior_reward,
            self.goal_reward,
            self.control_cost,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("NMF discovery parameters must be finite")
        if self.interior_reward >= 0.0:
            raise ValueError("Discovery interior reward must be negative")
        if self.control_cost <= 0.0:
            raise ValueError("Discovery control cost must be positive")

@dataclass(frozen=True)
class GoalTaskEnsemble:
    """Flat goal tasks whose state-by-task desirabilities form ``Z``."""

    maze: Maze
    goals: tuple[Coordinate, ...]
    parameters: NMFDiscoveryParameters
    desirability: np.ndarray

    def __post_init__(self) -> None:
        goals = tuple(self.goals)
        values = np.array(self.desirability, dtype=np.float64, copy=True)
        expected_shape = (len(self.maze.free_cells), len(goals))
        if not goals:
            raise ValueError("A task ensemble must contain at least one goal")
        if values.shape != expected_shape:
            raise ValueError(
                "Task desirability must have shape "
                f"{expected_shape}, got {values.shape}"
            )
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Task desirability must be finite and non-negative"
            )
        if np.any(values.max(axis=0) <= 0.0):
            raise ValueError("Every task must have positive desirability")
        object.__setattr__(self, "goals", goals)
        values.flags.writeable = False
        object.__setattr__(self, "desirability", values)


@dataclass(frozen=True)
class SoftSubtaskDiscovery:
    """A KL-NMF decomposition ``Z ~= D W`` with inspectable factors."""

    ensemble: GoalTaskEnsemble
    profiles: np.ndarray
    task_weights: np.ndarray
    reconstruction: np.ndarray
    reconstruction_error: float
    n_iter: int
    converged: bool

    def __post_init__(self) -> None:
        profiles = np.array(self.profiles, dtype=np.float64, copy=True)
        weights = np.array(self.task_weights, dtype=np.float64, copy=True)
        reconstruction = np.array(
            self.reconstruction,
            dtype=np.float64,
            copy=True,
        )
        number_of_states, number_of_tasks = (
            self.ensemble.desirability.shape
        )
        if profiles.ndim != 2 or profiles.shape[0] != number_of_states:
            raise ValueError(
                "Soft profiles must have one row per physical state"
            )
        expected_weight_shape = (profiles.shape[1], number_of_tasks)
        if weights.shape != expected_weight_shape:
            raise ValueError(
                "Task weights must have shape "
                f"{expected_weight_shape}, got {weights.shape}"
            )
        if reconstruction.shape != (number_of_states, number_of_tasks):
            raise ValueError("Reconstruction must have the ensemble shape")
        if (
            np.any(profiles < 0.0)
            or np.any(weights < 0.0)
            or np.any(reconstruction < 0.0)
            or not np.all(np.isfinite(profiles))
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(reconstruction))
        ):
            raise ValueError("NMF factors must be finite and non-negative")
        if not np.isfinite(self.reconstruction_error):
            raise ValueError("Reconstruction error must be finite")
        profiles.flags.writeable = False
        weights.flags.writeable = False
        reconstruction.flags.writeable = False
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "task_weights", weights)
        object.__setattr__(self, "reconstruction", reconstruction)

    @property
    def number_of_subtasks(self) -> int:
        return self.profiles.shape[1]

@dataclass(frozen=True)
class NMFRankDiagnostics:
    """Normalized KL reconstruction errors for candidate NMF ranks."""

    ranks: np.ndarray
    reconstruction_errors: np.ndarray

    def __post_init__(self) -> None:
        ranks = np.asarray(self.ranks, dtype=int)
        errors = np.asarray(self.reconstruction_errors, dtype=np.float64)
        if ranks.ndim != 1 or not len(ranks):
            raise ValueError("Rank diagnostics require at least one rank")
        if errors.shape != ranks.shape:
            raise ValueError("Ranks and reconstruction errors must align")
        if np.any(ranks < 1):
            raise ValueError("NMF ranks must be positive")
        if np.any(errors < 0.0) or not np.all(np.isfinite(errors)):
            raise ValueError("Reconstruction errors must be finite")
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "reconstruction_errors", errors)


@dataclass(frozen=True)
class NMFStudy:
    """One goal ensemble and a cached factorization for every requested rank."""

    ensemble: GoalTaskEnsemble
    discoveries: dict[int, SoftSubtaskDiscovery]

    def __post_init__(self) -> None:
        if not self.discoveries:
            raise ValueError("An NMF study requires at least one rank")
        ordered = dict(sorted(self.discoveries.items()))
        for rank, discovery in ordered.items():
            if rank != discovery.number_of_subtasks:
                raise ValueError(
                    "Discovery rank does not match its profile count"
                )
            if discovery.ensemble is not self.ensemble:
                raise ValueError(
                    "All discoveries must use the study ensemble"
                )
        object.__setattr__(self, "discoveries", ordered)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(self.discoveries)

    @property
    def diagnostics(self) -> NMFRankDiagnostics:
        return NMFRankDiagnostics(
            ranks=np.asarray(self.ranks, dtype=int),
            reconstruction_errors=np.asarray(
                [
                    self.discoveries[rank].reconstruction_error
                    for rank in self.ranks
                ],
                dtype=np.float64,
            ),
        )

    def result(self, rank: int) -> SoftSubtaskDiscovery:
        """Return the already-fitted result for ``rank``."""

        try:
            return self.discoveries[rank]
        except KeyError as error:
            raise ValueError(
                f"Rank {rank} was not fitted; available ranks: {self.ranks}"
            ) from error


def discover_soft_subgoals(
    environment: LMDPEnvironment,
    *,
    ranks: list[int] | tuple[int, ...] | np.ndarray,
    parameters: NMFDiscoveryParameters = NMFDiscoveryParameters(),
    goals: list[Coordinate] | tuple[Coordinate, ...] | None = None,
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> NMFStudy:
    """Fit each requested rank once for an arbitrary maze task ensemble."""

    ordered_ranks = tuple(ranks)
    if not ordered_ranks:
        raise ValueError("Rank diagnostics require at least one rank")
    if len(set(ordered_ranks)) != len(ordered_ranks):
        raise ValueError("Diagnostic ranks must be unique")
    ordered_goals = tuple(
        environment.maze.free_cells if goals is None else goals
    )
    if not ordered_goals:
        raise ValueError("A task ensemble must contain at least one goal")
    if len(set(ordered_goals)) != len(ordered_goals):
        raise ValueError("Task goals must be unique")
    solver_parameters = ModelParameters(
        interior_reward=parameters.interior_reward,
        goal_reward=parameters.goal_reward,
        lower_control_cost=parameters.control_cost,
    )
    desirability = np.column_stack(
        [
            environment.solve_flat(
                goal,
                parameters=solver_parameters,
            ).desirability
            for goal in ordered_goals
        ]
    )
    ensemble = GoalTaskEnsemble(
        maze=environment.maze,
        goals=ordered_goals,
        parameters=parameters,
        desirability=desirability,
    )
    discoveries = {}
    for rank in ordered_ranks:
        result = _factorize_soft_subtasks(
            ensemble,
            rank,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        discoveries[int(rank)] = result
    return NMFStudy(
        ensemble=ensemble,
        discoveries=discoveries,
    )


def _factorize_soft_subtasks(
    ensemble: GoalTaskEnsemble,
    n_subtasks: int,
    *,
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> SoftSubtaskDiscovery:
    """Factor the ensemble with the paper's beta=1 (KL) NMF objective."""

    maximum_rank = min(ensemble.desirability.shape)
    _validate_nmf_options(n_subtasks, maximum_rank, max_iter, tolerance)
    # The discovered representation is a factorization of the actual task
    # desirabilities Z, not a column-normalized plotting view of Z.
    target = ensemble.desirability
    factorization = NMF(
        n_components=n_subtasks,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        tol=tolerance,
        random_state=seed,
    )
    raw_profiles = factorization.fit_transform(target)
    raw_weights = factorization.components_

    profiles, task_weights = _peak_normalize_nmf_factors(
        raw_profiles,
        raw_weights,
    )
    reconstruction = profiles @ task_weights

    return SoftSubtaskDiscovery(
        ensemble=ensemble,
        profiles=profiles,
        task_weights=task_weights,
        reconstruction=reconstruction,
        reconstruction_error=_normalized_kl_divergence(
            target,
            reconstruction,
        ),
        n_iter=int(factorization.n_iter_),
        converged=factorization.n_iter_ < max_iter,
    )


def _peak_normalize_nmf_factors(
    profiles: np.ndarray,
    task_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fix every NMF component gauge without changing its reconstruction.

    Each profile column is scaled to peak at one and its scale is absorbed by
    the corresponding task-weight row. This makes ``alpha`` in
    ``P_t = alpha D.T`` the maximum local passive access strength.
    """

    profile_values = np.asarray(profiles, dtype=np.float64)
    weight_values = np.asarray(task_weights, dtype=np.float64)
    if profile_values.ndim != 2 or weight_values.ndim != 2:
        raise ValueError("NMF factors must be matrices")
    expected_weight_rows = profile_values.shape[1]
    if weight_values.shape[0] != expected_weight_rows:
        raise ValueError(
            "NMF task weights must have one row per profile column"
        )
    if (
        np.any(profile_values < 0.0)
        or np.any(weight_values < 0.0)
        or not np.all(np.isfinite(profile_values))
        or not np.all(np.isfinite(weight_values))
    ):
        raise ValueError("NMF factors must be finite and non-negative")

    component_scales = profile_values.max(axis=0)
    if np.any(component_scales <= 0.0):
        raise ValueError("NMF produced an empty subtask profile")
    normalized_profiles = profile_values / component_scales[np.newaxis, :]
    normalized_weights = weight_values * component_scales[:, np.newaxis]
    return normalized_profiles, normalized_weights


def _validate_nmf_options(
    n_subtasks: int,
    maximum_rank: int,
    max_iter: int,
    tolerance: float,
) -> None:
    if (
        isinstance(n_subtasks, (bool, np.bool_))
        or not isinstance(n_subtasks, (int, np.integer))
        or not 1 <= n_subtasks <= maximum_rank
    ):
        raise ValueError(
            f"Number of subtasks must be between 1 and {maximum_rank}"
        )
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("Maximum NMF iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("NMF tolerance must be finite and positive")


def _normalized_kl_divergence(
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    safe_reconstruction = np.maximum(
        reconstruction,
        np.finfo(np.float64).tiny,
    )
    logarithmic_term = np.zeros_like(target)
    positive = target > 0.0
    logarithmic_term[positive] = target[positive] * np.log(
        target[positive] / safe_reconstruction[positive]
    )
    divergence = np.sum(
        logarithmic_term - target + safe_reconstruction
    )
    return float(divergence / target.sum())
