"""Discover distributed MLMDP subtasks with non-negative factorization."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF

from andrew_mlmdp.lmdp import ModelParameters, solve_desirability
from andrew_mlmdp.maze import Coordinate, Maze


@dataclass(frozen=True)
class GoalTaskEnsemble:
    """Flat goal tasks whose state-by-task desirabilities form ``Z``."""

    maze: Maze
    goals: tuple[Coordinate, ...]
    parameters: ModelParameters
    desirability: np.ndarray

    def __post_init__(self) -> None:
        goals = tuple(self.goals)
        values = np.asarray(self.desirability, dtype=np.float64)
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
        object.__setattr__(self, "desirability", values)

    @property
    def normalized_desirability(self) -> np.ndarray:
        """Return task columns scaled to a maximum of one."""

        return self.desirability / self.desirability.max(
            axis=0,
            keepdims=True,
        )


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
        profiles = np.asarray(self.profiles, dtype=np.float64)
        weights = np.asarray(self.task_weights, dtype=np.float64)
        reconstruction = np.asarray(self.reconstruction, dtype=np.float64)
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
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "task_weights", weights)
        object.__setattr__(self, "reconstruction", reconstruction)

    @property
    def number_of_subtasks(self) -> int:
        return self.profiles.shape[1]

    @property
    def display_profiles(self) -> np.ndarray:
        """Return independently peak-normalized profiles for visualization.

        The model-facing :attr:`profiles` retain the single global NMF gauge
        used to construct hierarchy-access dynamics.  Plotting must not change
        that scale, so this view is computed separately.
        """

        return self.profiles / self.profiles.max(axis=0, keepdims=True)


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


def build_goal_task_ensemble(
    maze: Maze,
    *,
    goals: list[Coordinate] | tuple[Coordinate, ...] | None = None,
    parameters: ModelParameters = ModelParameters(),
) -> GoalTaskEnsemble:
    """Solve one flat first-exit task for every requested physical goal."""

    ordered_goals = tuple(maze.free_cells if goals is None else goals)
    if not ordered_goals:
        raise ValueError("A task ensemble must contain at least one goal")
    if len(set(ordered_goals)) != len(ordered_goals):
        raise ValueError("Task goals must be unique")
    for goal in ordered_goals:
        maze.state_index(goal)

    desirability = np.column_stack(
        [
            solve_desirability(maze, goal, parameters=parameters)
            for goal in ordered_goals
        ]
    )
    return GoalTaskEnsemble(
        maze=maze,
        goals=ordered_goals,
        parameters=parameters,
        desirability=desirability,
    )


def factorize_soft_subtasks(
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

    if np.any(raw_profiles.max(axis=0) <= 0.0):
        raise ValueError("NMF produced an empty subtask profile")

    # NMF has a global gauge freedom D -> D/c, W -> cW.  Fix only that
    # global gauge so relative component strengths remain intact and the one
    # scalar alpha in P_t = alpha D^T has a consistent interpretation.
    gauge = float(np.median(raw_profiles.sum(axis=1)))
    if not np.isfinite(gauge) or gauge <= 0.0:
        raise ValueError("NMF produced a degenerate global profile gauge")
    profiles = raw_profiles / gauge
    task_weights = raw_weights * gauge
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


def evaluate_soft_subtask_ranks(
    ensemble: GoalTaskEnsemble,
    ranks: list[int] | tuple[int, ...] | np.ndarray,
    *,
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> NMFRankDiagnostics:
    """Fit candidate ranks and return their normalized KL errors."""

    ordered_ranks = tuple(ranks)
    if not ordered_ranks:
        raise ValueError("Rank diagnostics require at least one rank")
    if len(set(ordered_ranks)) != len(ordered_ranks):
        raise ValueError("Diagnostic ranks must be unique")
    discoveries = [
        factorize_soft_subtasks(
            ensemble,
            rank,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        for rank in ordered_ranks
    ]
    return NMFRankDiagnostics(
        ranks=np.asarray(ordered_ranks, dtype=int),
        reconstruction_errors=np.asarray(
            [result.reconstruction_error for result in discoveries]
        ),
    )


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
