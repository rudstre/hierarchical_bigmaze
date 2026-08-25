"""Discover distributed MLMDP subtasks with non-negative factorization."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd

from andrew_mlmdp.lmdp import (
    Environment,
    Parameters,
)
from andrew_mlmdp.maze import Coordinate, Maze


@dataclass(frozen=True)
class NMFConfig:
    """Task-family parameters used only to discover soft subtask profiles.

    These values define the fixed desirability ensemble and optional spatial
    penalty supplied to NMF. They are intentionally separate from
    ``Parameters`` so execution tuning cannot silently rediscover a
    different hierarchy.
    """

    interior_reward: float = -1.0
    goal_reward: float = 0.0
    control_cost: float = 3.0
    lambda_smooth: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.interior_reward,
            self.goal_reward,
            self.control_cost,
            self.lambda_smooth,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("NMF discovery parameters must be finite")
        if self.interior_reward >= 0.0:
            raise ValueError("Discovery interior reward must be negative")
        if self.control_cost <= 0.0:
            raise ValueError("Discovery control cost must be positive")
        if self.lambda_smooth < 0.0:
            raise ValueError(
                "Discovery smoothness strength must be non-negative"
            )


@dataclass(frozen=True)
class GoalTasks:
    """Flat goal tasks whose state-by-task desirabilities form ``Z``."""

    maze: Maze
    goals: tuple[Coordinate, ...]
    parameters: NMFConfig
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
class SubtaskDiscovery:
    """A KL-NMF decomposition ``Z ~= D W`` with inspectable factors."""

    ensemble: GoalTasks
    profiles: np.ndarray
    task_weights: np.ndarray
    reconstruction: np.ndarray
    reconstruction_error: float
    n_iter: int
    converged: bool
    objective_history: np.ndarray | None = None

    def __post_init__(self) -> None:
        profiles = np.array(self.profiles, dtype=np.float64, copy=True)
        weights = np.array(self.task_weights, dtype=np.float64, copy=True)
        reconstruction = np.array(
            self.reconstruction,
            dtype=np.float64,
            copy=True,
        )
        n_states, n_tasks = (
            self.ensemble.desirability.shape
        )
        if profiles.ndim != 2 or profiles.shape[0] != n_states:
            raise ValueError(
                "Soft profiles must have one row per physical state"
            )
        expected_weight_shape = (profiles.shape[1], n_tasks)
        if weights.shape != expected_weight_shape:
            raise ValueError(
                "Task weights must have shape "
                f"{expected_weight_shape}, got {weights.shape}"
            )
        if reconstruction.shape != (n_states, n_tasks):
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
        objective_history = self.objective_history
        if objective_history is not None:
            objective_history = np.array(
                objective_history,
                dtype=np.float64,
                copy=True,
            )
            if (
                objective_history.ndim != 1
                or not len(objective_history)
                or np.any(objective_history < 0.0)
                or not np.all(np.isfinite(objective_history))
            ):
                raise ValueError(
                    "NMF objective history must be a finite non-negative vector"
                )
            objective_history.flags.writeable = False
        profiles.flags.writeable = False
        weights.flags.writeable = False
        reconstruction.flags.writeable = False
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "task_weights", weights)
        object.__setattr__(self, "reconstruction", reconstruction)
        object.__setattr__(self, "objective_history", objective_history)

    @property
    def n_subtasks(self) -> int:
        """Number of discovered soft subtasks."""

        return self.profiles.shape[1]

@dataclass(frozen=True)
class RankDiagnostics:
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

    ensemble: GoalTasks
    discoveries: dict[int, SubtaskDiscovery]

    def __post_init__(self) -> None:
        if not self.discoveries:
            raise ValueError("An NMF study requires at least one rank")
        ordered = dict(sorted(self.discoveries.items()))
        for rank, discovery in ordered.items():
            if rank != discovery.n_subtasks:
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
    def diagnostics(self) -> RankDiagnostics:
        return RankDiagnostics(
            ranks=np.asarray(self.ranks, dtype=int),
            reconstruction_errors=np.asarray(
                [
                    self.discoveries[rank].reconstruction_error
                    for rank in self.ranks
                ],
                dtype=np.float64,
            ),
        )

    def result(self, rank: int) -> SubtaskDiscovery:
        """Return the already-fitted result for ``rank``."""

        try:
            return self.discoveries[rank]
        except KeyError as error:
            raise ValueError(
                f"Rank {rank} was not fitted; available ranks: {self.ranks}"
            ) from error


def discover_subgoals(
    environment: Environment,
    *,
    ranks: list[int] | tuple[int, ...] | np.ndarray,
    parameters: NMFConfig = NMFConfig(),
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
    solver_parameters = Parameters(
        interior_reward=parameters.interior_reward,
        goal_reward=parameters.goal_reward,
        lower_control_cost=parameters.control_cost,
    )
    desirability = np.column_stack(
        [
            environment.solve(
                goal,
                parameters=solver_parameters,
            ).desirability
            for goal in ordered_goals
        ]
    )
    ensemble = GoalTasks(
        maze=environment.maze,
        goals=ordered_goals,
        parameters=parameters,
        desirability=desirability,
    )
    adjacency = None
    if parameters.lambda_smooth > 0.0:
        adjacency = _graph_adjacency_from_passive(environment.passive)
    discoveries = {}
    for rank in ordered_ranks:
        if adjacency is None:
            result = _factorize_soft_subtasks(
                ensemble,
                rank,
                seed=seed,
                max_iter=max_iter,
                tolerance=tolerance,
            )
        else:
            result = _factorize_regularized_soft_subtasks(
                ensemble,
                rank,
                adjacency=adjacency,
                lambda_smooth=parameters.lambda_smooth,
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
    ensemble: GoalTasks,
    n_subtasks: int,
    *,
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> SubtaskDiscovery:
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

    profiles, task_weights = _unit_normalize_nmf_factors(
        raw_profiles,
        raw_weights,
    )
    reconstruction = profiles @ task_weights

    return SubtaskDiscovery(
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


def _unit_normalize_nmf_factors(
    profiles: np.ndarray,
    task_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fix every NMF component gauge without changing its reconstruction.

    Each profile column is scaled to Euclidean norm one and its scale is
    absorbed by the corresponding task-weight row.
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

    component_scales = np.linalg.norm(profile_values, axis=0)
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
    return _generalized_kl_divergence(target, reconstruction) / target.sum()


def _generalized_kl_divergence(
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
    return float(divergence)

def _factorize_regularized_soft_subtasks(
    ensemble: GoalTasks,
    n_subtasks: int,
    *,
    adjacency: np.ndarray,
    lambda_smooth: float,
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> SubtaskDiscovery:
    """Fit graph-regularized KL-NMF with non-negative MU updates.

    The graph multiplicative update is derived for the unconstrained objective.
    Unit-normalizing each profile after a sweep fixes the NMF scale ambiguity
    by additionally imposing ``||D[:, j]||_2 == 1``. The standard unconstrained
    MU descent guarantee therefore does not automatically apply to the tracked
    post-normalization objective; its monotonicity is tested empirically.
    """

    maximum_rank = min(ensemble.desirability.shape)
    _validate_nmf_options(n_subtasks, maximum_rank, max_iter, tolerance)
    if not np.isfinite(lambda_smooth) or lambda_smooth <= 0.0:
        raise ValueError(
            "Regularized NMF requires positive smoothness strength"
        )

    target = ensemble.desirability
    adjacency_values = _validated_graph_adjacency(
        adjacency,
        n_states=target.shape[0],
    )
    degree = adjacency_values.sum(axis=1)
    profiles, task_weights = _initialize_regularized_factors(
        target,
        n_subtasks,
        seed=seed,
    )
    profiles, task_weights = _unit_normalize_nmf_factors(
        profiles,
        task_weights,
    )

    objective_history = [
        _regularized_objective(
            target,
            profiles,
            task_weights,
            adjacency_values,
            degree,
            lambda_smooth,
        )
    ]
    epsilon = np.finfo(np.float64).eps
    converged = False
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        reconstruction = np.maximum(profiles @ task_weights, epsilon)
        ratio = target / reconstruction
        weight_numerator = profiles.T @ ratio
        weight_denominator = profiles.sum(axis=0)[:, np.newaxis]
        task_weights *= weight_numerator / np.maximum(
            weight_denominator,
            epsilon,
        )

        reconstruction = np.maximum(profiles @ task_weights, epsilon)
        ratio = target / reconstruction
        profile_numerator = (
            ratio @ task_weights.T
            + 2.0 * lambda_smooth * (adjacency_values @ profiles)
        )
        profile_denominator = (
            task_weights.sum(axis=1)[np.newaxis, :]
            + 2.0
            * lambda_smooth
            * degree[:, np.newaxis]
            * profiles
        )
        profiles *= profile_numerator / np.maximum(
            profile_denominator,
            epsilon,
        )
        profiles, task_weights = _unit_normalize_nmf_factors(
            profiles,
            task_weights,
        )

        objective = _regularized_objective(
            target,
            profiles,
            task_weights,
            adjacency_values,
            degree,
            lambda_smooth,
        )
        previous_objective = objective_history[-1]
        objective_history.append(objective)
        n_iter = iteration

        improvement = previous_objective - objective
        scale = max(abs(previous_objective), epsilon)
        numerical_tolerance = 10.0 * epsilon * scale
        if (
            improvement >= -numerical_tolerance
            and improvement / scale <= tolerance
        ):
            converged = iteration < max_iter
            break

    reconstruction = profiles @ task_weights
    return SubtaskDiscovery(
        ensemble=ensemble,
        profiles=profiles,
        task_weights=task_weights,
        reconstruction=reconstruction,
        reconstruction_error=_normalized_kl_divergence(
            target,
            reconstruction,
        ),
        n_iter=n_iter,
        converged=converged,
        objective_history=np.asarray(objective_history),
    )


def _graph_adjacency_from_passive(passive: np.ndarray) -> np.ndarray:
    """Return binary undirected connectivity in passive state order."""

    passive_values = np.asarray(passive, dtype=np.float64)
    if (
        passive_values.ndim != 2
        or passive_values.shape[0] != passive_values.shape[1]
        or np.any(passive_values < 0.0)
        or not np.all(np.isfinite(passive_values))
    ):
        raise ValueError(
            "Passive dynamics must be a finite non-negative square matrix"
        )
    adjacency = np.logical_or(
        passive_values > 0.0,
        passive_values.T > 0.0,
    )
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float64)


def _validated_graph_adjacency(
    adjacency: np.ndarray,
    *,
    n_states: int,
) -> np.ndarray:
    values = np.asarray(adjacency, dtype=np.float64)
    expected_shape = (n_states, n_states)
    if values.shape != expected_shape:
        raise ValueError(
            f"Graph adjacency must have shape {expected_shape}, "
            f"got {values.shape}"
        )
    if (
        np.any((values != 0.0) & (values != 1.0))
        or not np.array_equal(values, values.T)
        or np.any(np.diag(values) != 0.0)
    ):
        raise ValueError(
            "Graph adjacency must be symmetric, binary, and loop-free"
        )
    return values


def _initialize_regularized_factors(
    target: np.ndarray,
    n_subtasks: int,
    *,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an NNDSVDa initialization compatible with sklearn's choice."""

    left_vectors, singular_values, right_vectors = randomized_svd(
        target,
        n_components=n_subtasks,
        random_state=seed,
    )
    profiles = np.zeros_like(left_vectors)
    task_weights = np.zeros_like(right_vectors)

    leading_scale = np.sqrt(singular_values[0])
    profiles[:, 0] = leading_scale * np.abs(left_vectors[:, 0])
    task_weights[0, :] = leading_scale * np.abs(right_vectors[0, :])

    for component in range(1, n_subtasks):
        left = left_vectors[:, component]
        right = right_vectors[component, :]
        left_positive = np.maximum(left, 0.0)
        right_positive = np.maximum(right, 0.0)
        left_negative = np.maximum(-left, 0.0)
        right_negative = np.maximum(-right, 0.0)

        positive_norms = (
            np.linalg.norm(left_positive),
            np.linalg.norm(right_positive),
        )
        negative_norms = (
            np.linalg.norm(left_negative),
            np.linalg.norm(right_negative),
        )
        positive_product = positive_norms[0] * positive_norms[1]
        negative_product = negative_norms[0] * negative_norms[1]
        if positive_product > negative_product:
            selected_left = left_positive
            selected_right = right_positive
            selected_norms = positive_norms
            selected_product = positive_product
        else:
            selected_left = left_negative
            selected_right = right_negative
            selected_norms = negative_norms
            selected_product = negative_product

        if selected_product == 0.0:
            continue
        scale = np.sqrt(singular_values[component] * selected_product)
        profiles[:, component] = (
            scale * selected_left / selected_norms[0]
        )
        task_weights[component, :] = (
            scale * selected_right / selected_norms[1]
        )

    average = float(target.mean())
    profiles[profiles == 0.0] = average
    task_weights[task_weights == 0.0] = average
    return profiles, task_weights


def _regularized_objective(
    target: np.ndarray,
    profiles: np.ndarray,
    task_weights: np.ndarray,
    adjacency: np.ndarray,
    degree: np.ndarray,
    lambda_smooth: float,
) -> float:
    reconstruction = profiles @ task_weights
    laplacian_profiles = (
        degree[:, np.newaxis] * profiles - adjacency @ profiles
    )
    smoothness_penalty = float(np.sum(profiles * laplacian_profiles))
    return (
        _generalized_kl_divergence(target, reconstruction)
        + lambda_smooth * smoothness_penalty
    )
