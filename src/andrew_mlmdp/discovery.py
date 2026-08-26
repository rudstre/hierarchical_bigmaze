"""Discover distributed MLMDP subtasks with non-negative factorization."""

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from andrew_mlmdp.lmdp import (
    Environment,
    Parameters,
)
from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.profiles import (
    ProfileNormalization,
    _normalize_profile_columns,
    _validate_profile_normalization,
)


@dataclass(frozen=True)
class NMFConfig:
    """Task-family parameters used only to discover soft subtask profiles.

    These values define the fixed desirability ensemble and NMF scale gauge.
    Peak normalization is the default; ``"l2"`` selects a unit-L2 gauge. They
    are intentionally separate from ``Parameters`` so execution tuning cannot
    silently rediscover a different hierarchy.
    """

    interior_reward: float = -1.0
    goal_reward: float = 0.0
    control_cost: float = 3.0
    profile_normalization: ProfileNormalization = "peak"

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
        _validate_profile_normalization(self.profile_normalization)


@dataclass(frozen=True)
class NMFConnectivityConfig:
    """Connected-effective-support settings for stochastic NMF restarts."""

    support_mass: float = 0.95
    max_prune_refits: int = 3
    positive_fallback_attempts: int = 3
    restart_seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.support_mass)
            or not 0.0 < self.support_mass <= 1.0
        ):
            raise ValueError("Connectivity support mass must be in (0, 1]")
        if (
            isinstance(self.max_prune_refits, (bool, np.bool_))
            or not isinstance(self.max_prune_refits, (int, np.integer))
            or self.max_prune_refits < 1
        ):
            raise ValueError(
                "Maximum connectivity prune/refit rounds must be positive"
            )
        if (
            isinstance(self.positive_fallback_attempts, (bool, np.bool_))
            or not isinstance(
                self.positive_fallback_attempts,
                (int, np.integer),
            )
            or self.positive_fallback_attempts < 1
        ):
            raise ValueError(
                "Positive masked fallback attempts must be positive"
            )
        object.__setattr__(
            self,
            "positive_fallback_attempts",
            int(self.positive_fallback_attempts),
        )
        seeds = tuple(self.restart_seeds)
        if not seeds:
            raise ValueError("Connectivity requires at least one restart seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Connectivity restart seeds must be unique")
        for seed in seeds:
            if (
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or not 0 <= int(seed) <= np.iinfo(np.uint32).max
            ):
                raise ValueError(
                    "Connectivity restart seeds must be uint32 integers"
                )
        object.__setattr__(
            self,
            "restart_seeds",
            tuple(int(seed) for seed in seeds),
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
class NMFRestartResult:
    """Inspectable unconstrained and connectivity-constrained restart."""

    restart_id: int
    seed: int | None
    unconstrained_profiles: np.ndarray
    unconstrained_task_weights: np.ndarray
    unconstrained_kl: float
    connected_profiles: np.ndarray | None
    connected_task_weights: np.ndarray | None
    connected_kl: float | None
    forbidden_mask: np.ndarray
    discarded_mass_fractions: np.ndarray
    effective_support_sizes: np.ndarray | None
    effective_support_fractions: np.ndarray | None
    final_support_connected: np.ndarray
    prune_refit_rounds: int
    fit_iterations: tuple[int, ...]
    fit_converged: tuple[bool, ...]
    fully_forbidden_state_indices: np.ndarray
    positive_target_zero_reconstruction_counts: tuple[int, ...]
    positive_fallback_attempt_counts: tuple[int, ...]
    positive_fallback_success_counts: tuple[int, ...]
    feasible: bool
    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        profiles = np.array(
            self.unconstrained_profiles,
            dtype=np.float64,
            copy=True,
        )
        weights = np.array(
            self.unconstrained_task_weights,
            dtype=np.float64,
            copy=True,
        )
        if profiles.ndim != 2 or weights.ndim != 2:
            raise ValueError("Restart NMF factors must be matrices")
        n_states, n_components = profiles.shape
        if weights.shape[0] != n_components:
            raise ValueError("Restart NMF factors have incompatible shapes")
        if (
            np.any(profiles < 0.0)
            or np.any(weights < 0.0)
            or not np.all(np.isfinite(profiles))
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("Unconstrained restart factors must be finite")

        connected_profiles = self.connected_profiles
        connected_weights = self.connected_task_weights
        if (connected_profiles is None) != (connected_weights is None):
            raise ValueError("Connected D and W must both be present or absent")
        if connected_profiles is not None:
            connected_profiles = np.array(
                connected_profiles,
                dtype=np.float64,
                copy=True,
            )
            connected_weights = np.array(
                connected_weights,
                dtype=np.float64,
                copy=True,
            )
            if (
                connected_profiles.shape != profiles.shape
                or connected_weights.shape != weights.shape
                or np.any(connected_profiles < 0.0)
                or np.any(connected_weights < 0.0)
                or not np.all(np.isfinite(connected_profiles))
                or not np.all(np.isfinite(connected_weights))
            ):
                raise ValueError("Connected restart factors are invalid")

        forbidden = np.array(self.forbidden_mask, dtype=bool, copy=True)
        discarded = np.array(
            self.discarded_mass_fractions,
            dtype=np.float64,
            copy=True,
        )
        support_connected = np.array(
            self.final_support_connected,
            dtype=bool,
            copy=True,
        )
        if forbidden.shape != (n_states, n_components):
            raise ValueError("Forbidden mask must have the shape of D")
        if discarded.shape != (n_components,) or (
            np.any(discarded < 0.0)
            or np.any(discarded > 1.0)
            or not np.all(np.isfinite(discarded))
        ):
            raise ValueError("Discarded masses must be component fractions")
        if support_connected.shape != (n_components,):
            raise ValueError(
                "Connectivity flags must have one entry per component"
            )
        fully_forbidden = np.array(
            self.fully_forbidden_state_indices,
            dtype=int,
            copy=True,
        )
        if (
            fully_forbidden.ndim != 1
            or np.any(fully_forbidden < 0)
            or np.any(fully_forbidden >= n_states)
            or len(np.unique(fully_forbidden)) != len(fully_forbidden)
            or np.any(np.diff(fully_forbidden) <= 0)
        ):
            raise ValueError(
                "Fully forbidden state indices must be sorted and unique"
            )
        zero_counts = tuple(
            int(value)
            for value in self.positive_target_zero_reconstruction_counts
        )
        if any(value < 1 for value in zero_counts):
            raise ValueError(
                "Positive-target zero-reconstruction counts must be positive"
            )
        fallback_attempts = tuple(
            int(value) for value in self.positive_fallback_attempt_counts
        )
        fallback_successes = tuple(
            int(value) for value in self.positive_fallback_success_counts
        )
        if (
            len(fallback_attempts) != len(zero_counts)
            or len(fallback_successes) != len(zero_counts)
            or any(value < 1 for value in fallback_attempts)
            or any(
                not 0 <= successes <= attempts
                for successes, attempts in zip(
                    fallback_successes,
                    fallback_attempts,
                )
            )
        ):
            raise ValueError("Positive fallback diagnostics must align")

        effective_sizes = self.effective_support_sizes
        effective_fractions = self.effective_support_fractions
        if (effective_sizes is None) != (effective_fractions is None):
            raise ValueError("Effective support diagnostics must align")
        if effective_sizes is not None:
            effective_sizes = np.array(
                effective_sizes,
                dtype=np.float64,
                copy=True,
            )
            effective_fractions = np.array(
                effective_fractions,
                dtype=np.float64,
                copy=True,
            )
            if (
                effective_sizes.shape != (n_components,)
                or effective_fractions.shape != (n_components,)
                or np.any(effective_sizes <= 0.0)
                or not np.all(np.isfinite(effective_sizes))
                or not np.all(np.isfinite(effective_fractions))
            ):
                raise ValueError("Effective support diagnostics are invalid")

        iterations = tuple(int(value) for value in self.fit_iterations)
        converged = tuple(bool(value) for value in self.fit_converged)
        if not iterations or len(iterations) != len(converged):
            raise ValueError("Restart fit diagnostics must align")
        if any(value < 1 for value in iterations):
            raise ValueError("Restart iteration counts must be positive")
        if self.prune_refit_rounds > len(iterations) - 1:
            raise ValueError("Prune/refit count exceeds fit diagnostics")
        if np.isnan(self.unconstrained_kl) or (
            self.connected_kl is not None and np.isnan(self.connected_kl)
        ):
            raise ValueError("Restart KL diagnostics cannot be NaN")
        if self.eligible and (
            not self.feasible
            or connected_profiles is None
            or self.connected_kl is None
            or not np.isfinite(self.connected_kl)
            or not np.all(support_connected)
            or self.reason is not None
        ):
            raise ValueError("Eligible restart diagnostics are inconsistent")

        for array in (
            profiles,
            weights,
            forbidden,
            discarded,
            support_connected,
            fully_forbidden,
            connected_profiles,
            connected_weights,
            effective_sizes,
            effective_fractions,
        ):
            if array is not None:
                array.flags.writeable = False
        object.__setattr__(self, "unconstrained_profiles", profiles)
        object.__setattr__(self, "unconstrained_task_weights", weights)
        object.__setattr__(self, "connected_profiles", connected_profiles)
        object.__setattr__(self, "connected_task_weights", connected_weights)
        object.__setattr__(self, "forbidden_mask", forbidden)
        object.__setattr__(self, "discarded_mass_fractions", discarded)
        object.__setattr__(self, "final_support_connected", support_connected)
        object.__setattr__(
            self,
            "fully_forbidden_state_indices",
            fully_forbidden,
        )
        object.__setattr__(
            self,
            "positive_target_zero_reconstruction_counts",
            zero_counts,
        )
        object.__setattr__(
            self,
            "positive_fallback_attempt_counts",
            fallback_attempts,
        )
        object.__setattr__(
            self,
            "positive_fallback_success_counts",
            fallback_successes,
        )
        object.__setattr__(self, "effective_support_sizes", effective_sizes)
        object.__setattr__(self, "effective_support_fractions", effective_fractions)
        object.__setattr__(self, "fit_iterations", iterations)
        object.__setattr__(self, "fit_converged", converged)

    @property
    def fully_forbidden_state(self) -> bool:
        return len(self.fully_forbidden_state_indices) > 0

    @property
    def positive_masked_fallback_event_count(self) -> int:
        return len(self.positive_target_zero_reconstruction_counts)

    @property
    def positive_masked_fallback_count(self) -> int:
        return sum(self.positive_fallback_attempt_counts)

    @property
    def positive_fallback_success_count(self) -> int:
        return sum(self.positive_fallback_success_counts)

    @property
    def zero_locked_warm_start(self) -> bool:
        return self.positive_masked_fallback_event_count > 0

    @property
    def positive_fallback_failed(self) -> bool:
        return self.reason == "positive_fallback_failed"

    @property
    def positive_fallback_succeeded(self) -> bool:
        return (
            self.zero_locked_warm_start
            and not self.positive_fallback_failed
        )

    @property
    def delta_kl_connectivity(self) -> float | None:
        if self.connected_kl is None:
            return None
        return self.connected_kl - self.unconstrained_kl

    @property
    def relative_delta_kl_connectivity(self) -> float | None:
        delta = self.delta_kl_connectivity
        if delta is None or self.unconstrained_kl <= 0.0:
            return None
        return delta / self.unconstrained_kl


@dataclass(frozen=True)
class NMFRankResult:
    """All restart outcomes and the selected connected factorization."""

    rank: int
    restarts: tuple[NMFRestartResult, ...]
    selected_restart_id: int | None
    discovery: SubtaskDiscovery | None

    def __post_init__(self) -> None:
        restarts = tuple(self.restarts)
        if not restarts:
            raise ValueError("A rank result requires at least one restart")
        if tuple(result.restart_id for result in restarts) != tuple(
            range(len(restarts))
        ):
            raise ValueError("Restart identifiers must follow configured order")
        if self.discovery is not None and self.discovery.n_subtasks != self.rank:
            raise ValueError("Rank result and discovery rank do not match")
        if self.selected_restart_id is None:
            if self.discovery is not None:
                raise ValueError("Unselected rank cannot contain a discovery")
        elif (
            not 0 <= self.selected_restart_id < len(restarts)
            or not restarts[self.selected_restart_id].eligible
            or self.discovery is None
        ):
            raise ValueError("Selected restart is not eligible")
        object.__setattr__(self, "restarts", restarts)

    @property
    def best_unconstrained_restart_id(self) -> int | None:
        candidates = [
            result
            for result in self.restarts
            if result.fit_converged[0]
            and np.isfinite(result.unconstrained_kl)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda result: (result.unconstrained_kl, result.restart_id),
        ).restart_id

    @property
    def best_unconstrained_kl(self) -> float | None:
        restart_id = self.best_unconstrained_restart_id
        if restart_id is None:
            return None
        return self.restarts[restart_id].unconstrained_kl

    @property
    def best_connected_kl(self) -> float | None:
        if self.selected_restart_id is None:
            return None
        return self.restarts[self.selected_restart_id].connected_kl

    @property
    def delta_kl_connectivity(self) -> float | None:
        unconstrained = self.best_unconstrained_kl
        connected = self.best_connected_kl
        if unconstrained is None or connected is None:
            return None
        return connected - unconstrained


@dataclass(frozen=True)
class RankDiagnostics:
    """Normalized KL reconstruction errors for candidate NMF ranks."""

    ranks: np.ndarray
    reconstruction_errors: np.ndarray
    available: np.ndarray | None = None

    def __post_init__(self) -> None:
        ranks = np.array(self.ranks, dtype=int, copy=True)
        errors = np.array(
            self.reconstruction_errors,
            dtype=np.float64,
            copy=True,
        )
        available = (
            np.ones(ranks.shape, dtype=bool)
            if self.available is None
            else np.array(self.available, dtype=bool, copy=True)
        )
        if ranks.ndim != 1 or not len(ranks):
            raise ValueError("Rank diagnostics require at least one rank")
        if errors.shape != ranks.shape:
            raise ValueError("Ranks and reconstruction errors must align")
        if available.shape != ranks.shape:
            raise ValueError("Rank availability must align with ranks")
        if np.any(ranks < 1):
            raise ValueError("NMF ranks must be positive")
        if (
            np.any(errors[available] < 0.0)
            or not np.all(np.isfinite(errors[available]))
            or np.any(~np.isnan(errors[~available]))
        ):
            raise ValueError(
                "Available rank errors must be finite and unavailable errors NaN"
            )
        ranks.flags.writeable = False
        errors.flags.writeable = False
        available.flags.writeable = False
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "reconstruction_errors", errors)
        object.__setattr__(self, "available", available)


@dataclass(frozen=True)
class NMFStudy:
    """One goal ensemble and a connected restart result for every rank."""

    ensemble: GoalTasks
    rank_results: dict[int, NMFRankResult]

    def __post_init__(self) -> None:
        if not self.rank_results:
            raise ValueError("An NMF study requires at least one rank")
        ordered = dict(sorted(self.rank_results.items()))
        for rank, result in ordered.items():
            if rank != result.rank:
                raise ValueError(
                    "Rank-result key does not match its rank"
                )
            if (
                result.discovery is not None
                and result.discovery.ensemble is not self.ensemble
            ):
                raise ValueError(
                    "All discoveries must use the study ensemble"
                )
        object.__setattr__(self, "rank_results", ordered)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(self.rank_results)

    @property
    def discoveries(self) -> dict[int, SubtaskDiscovery]:
        """Selected discoveries for ranks with an eligible restart."""

        return {
            rank: result.discovery
            for rank, result in self.rank_results.items()
            if result.discovery is not None
        }

    @property
    def diagnostics(self) -> RankDiagnostics:
        available = np.asarray(
            [
                self.rank_results[rank].discovery is not None
                for rank in self.ranks
            ],
            dtype=bool,
        )
        return RankDiagnostics(
            ranks=np.asarray(self.ranks, dtype=int),
            reconstruction_errors=np.asarray(
                [
                    (
                        self.rank_results[rank].discovery.reconstruction_error
                        if self.rank_results[rank].discovery is not None
                        else np.nan
                    )
                    for rank in self.ranks
                ],
                dtype=np.float64,
            ),
            available=available,
        )

    def rank_result(self, rank: int) -> NMFRankResult:
        """Return all restart outcomes for the requested rank."""

        try:
            return self.rank_results[rank]
        except KeyError as error:
            raise ValueError(
                f"Rank {rank} was not fitted; available ranks: {self.ranks}"
            ) from error

    def result(self, rank: int) -> SubtaskDiscovery | None:
        """Return selected factors, or None when all restarts were excluded."""

        return self.rank_result(rank).discovery


def discover_subgoals(
    environment: Environment,
    *,
    ranks: list[int] | tuple[int, ...] | np.ndarray,
    parameters: NMFConfig = NMFConfig(),
    goals: list[Coordinate] | tuple[Coordinate, ...] | None = None,
    connectivity: NMFConnectivityConfig | None = NMFConnectivityConfig(),
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> NMFStudy:
    """Fit connected stochastic restarts for every requested NMF rank."""

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

    rank_results: dict[int, NMFRankResult] = {}
    if connectivity is None:
        for rank in ordered_ranks:
            discovery = _factorize_soft_subtasks(
                ensemble,
                rank,
                seed=seed,
                max_iter=max_iter,
                tolerance=tolerance,
            )
            rank_results[int(rank)] = _legacy_rank_result(discovery, seed)
    else:
        if not isinstance(connectivity, NMFConnectivityConfig):
            raise TypeError("connectivity must be NMFConnectivityConfig or None")
        connectivity = _connectivity_with_seed_shorthand(connectivity, seed)
        adjacency = _graph_adjacency_from_passive(environment.passive)
        for rank in ordered_ranks:
            rank_results[int(rank)] = _factorize_connected_soft_subtasks(
                ensemble,
                int(rank),
                adjacency=adjacency,
                connectivity=connectivity,
                max_iter=max_iter,
                tolerance=tolerance,
            )

    return NMFStudy(
        ensemble=ensemble,
        rank_results=rank_results,
    )


@dataclass(frozen=True)
class _NMFFit:
    profiles: np.ndarray
    task_weights: np.ndarray
    reconstruction: np.ndarray
    n_iter: int
    converged: bool


@dataclass(frozen=True)
class _MaskedNMFRefitResult:
    fit: _NMFFit | None
    reason: str | None
    fully_forbidden_state_indices: np.ndarray
    positive_target_zero_reconstruction_count: int
    used_positive_fallback: bool
    fit_iterations: tuple[int, ...]
    fit_converged: tuple[bool, ...]
    positive_fallback_attempt_count: int = 0
    positive_fallback_success_count: int = 0

    def __post_init__(self) -> None:
        indices = np.array(
            self.fully_forbidden_state_indices,
            dtype=int,
            copy=True,
        )
        if indices.ndim != 1:
            raise ValueError("Fully forbidden state indices must be a vector")
        if self.positive_target_zero_reconstruction_count < 0:
            raise ValueError("Zero-reconstruction count cannot be negative")
        if len(self.fit_iterations) != len(self.fit_converged):
            raise ValueError("Masked refit diagnostics must align")
        if (
            self.positive_fallback_attempt_count < 0
            or not 0
            <= self.positive_fallback_success_count
            <= self.positive_fallback_attempt_count
            or self.used_positive_fallback
            != (self.positive_fallback_attempt_count > 0)
        ):
            raise ValueError("Masked fallback diagnostics are invalid")
        indices.flags.writeable = False
        object.__setattr__(self, "fully_forbidden_state_indices", indices)


class _EmptyComponentError(RuntimeError):
    pass


def _connectivity_with_seed_shorthand(
    connectivity: NMFConnectivityConfig,
    seed: int | None,
) -> NMFConnectivityConfig:
    seeds = connectivity.restart_seeds
    if seeds == (0,):
        if seed in (None, 0):
            return connectivity
        return NMFConnectivityConfig(
            support_mass=connectivity.support_mass,
            max_prune_refits=connectivity.max_prune_refits,
            positive_fallback_attempts=(
                connectivity.positive_fallback_attempts
            ),
            restart_seeds=(seed,),
        )
    if seed not in (None, 0):
        raise ValueError(
            "seed cannot be combined with explicit connectivity restart seeds"
        )
    return connectivity


def _fit_nmf_factors(
    target: np.ndarray,
    n_subtasks: int,
    *,
    init: str,
    profile_normalization: ProfileNormalization,
    seed: int | None,
    max_iter: int,
    tolerance: float,
    initial_profiles: np.ndarray | None = None,
    initial_task_weights: np.ndarray | None = None,
    reemit_warnings: bool = False,
) -> _NMFFit:
    factorization = NMF(
        n_components=n_subtasks,
        init=init,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        tol=tolerance,
        random_state=seed,
    )
    fit_arguments = {}
    if init == "custom":
        if initial_profiles is None or initial_task_weights is None:
            raise ValueError("Custom NMF initialization requires D and W")
        fit_arguments = {
            "W": np.array(initial_profiles, dtype=np.float64, copy=True),
            "H": np.array(initial_task_weights, dtype=np.float64, copy=True),
        }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        raw_profiles = factorization.fit_transform(target, **fit_arguments)
        raw_weights = factorization.components_

    convergence_warnings = [
        item
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    for item in caught:
        if reemit_warnings or not issubclass(item.category, ConvergenceWarning):
            warnings.warn(item.message, item.category, stacklevel=3)

    if (
        not np.all(np.isfinite(raw_profiles))
        or not np.all(np.isfinite(raw_weights))
    ):
        raise ValueError("NMF produced non-finite factors")
    if np.any(raw_profiles.max(axis=0) <= 0.0):
        raise _EmptyComponentError("NMF produced an empty subtask profile")

    profiles, task_weights = _normalize_nmf_factors(
        raw_profiles,
        raw_weights,
        profile_normalization,
    )
    reconstruction = profiles @ task_weights
    return _NMFFit(
        profiles=profiles,
        task_weights=task_weights,
        reconstruction=reconstruction,
        n_iter=int(factorization.n_iter_),
        converged=not convergence_warnings,
    )


def _factorize_soft_subtasks(
    ensemble: GoalTasks,
    n_subtasks: int,
    *,
    seed: int | None = 0,
    max_iter: int = 2000,
    tolerance: float = 1e-5,
) -> SubtaskDiscovery:
    """Run a single seeded stochastic KL-NMF fit without connectivity."""

    maximum_rank = min(ensemble.desirability.shape)
    _validate_nmf_options(n_subtasks, maximum_rank, max_iter, tolerance)
    target = ensemble.desirability
    fit = _fit_nmf_factors(
        target,
        n_subtasks,
        init="random",
        profile_normalization=ensemble.parameters.profile_normalization,
        seed=seed,
        max_iter=max_iter,
        tolerance=tolerance,
        reemit_warnings=True,
    )
    return SubtaskDiscovery(
        ensemble=ensemble,
        profiles=fit.profiles,
        task_weights=fit.task_weights,
        reconstruction=fit.reconstruction,
        reconstruction_error=_normalized_kl_divergence(
            target,
            fit.reconstruction,
        ),
        n_iter=fit.n_iter,
        converged=fit.converged,
    )


def _legacy_rank_result(
    discovery: SubtaskDiscovery,
    seed: int | None,
) -> NMFRankResult:
    profiles = discovery.profiles
    weights = discovery.task_weights
    raw_kl = _generalized_kl_divergence(
        discovery.ensemble.desirability,
        discovery.reconstruction,
    )
    effective = _effective_support_sizes(profiles)
    restart = NMFRestartResult(
        restart_id=0,
        seed=seed,
        unconstrained_profiles=profiles,
        unconstrained_task_weights=weights,
        unconstrained_kl=raw_kl,
        connected_profiles=profiles,
        connected_task_weights=weights,
        connected_kl=raw_kl,
        forbidden_mask=np.zeros_like(profiles, dtype=bool),
        discarded_mass_fractions=np.zeros(profiles.shape[1]),
        effective_support_sizes=effective,
        effective_support_fractions=effective / profiles.shape[0],
        final_support_connected=np.ones(profiles.shape[1], dtype=bool),
        prune_refit_rounds=0,
        fit_iterations=(discovery.n_iter,),
        fit_converged=(discovery.converged,),
        fully_forbidden_state_indices=np.empty(0, dtype=int),
        positive_target_zero_reconstruction_counts=(),
        positive_fallback_attempt_counts=(),
        positive_fallback_success_counts=(),
        feasible=True,
        eligible=True,
    )
    return NMFRankResult(
        rank=discovery.n_subtasks,
        restarts=(restart,),
        selected_restart_id=0,
        discovery=discovery,
    )


def _factorize_connected_soft_subtasks(
    ensemble: GoalTasks,
    n_subtasks: int,
    *,
    adjacency: np.ndarray,
    connectivity: NMFConnectivityConfig,
    max_iter: int,
    tolerance: float,
) -> NMFRankResult:
    maximum_rank = min(ensemble.desirability.shape)
    _validate_nmf_options(n_subtasks, maximum_rank, max_iter, tolerance)
    adjacency_values = _validated_graph_adjacency(
        adjacency,
        n_states=ensemble.desirability.shape[0],
    ).astype(bool)
    restarts = tuple(
        _connectivity_restart(
            ensemble,
            n_subtasks,
            adjacency=adjacency_values,
            connectivity=connectivity,
            restart_id=restart_id,
            seed=seed,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        for restart_id, seed in enumerate(connectivity.restart_seeds)
    )
    eligible = [result for result in restarts if result.eligible]
    if not eligible:
        return NMFRankResult(
            rank=n_subtasks,
            restarts=restarts,
            selected_restart_id=None,
            discovery=None,
        )

    winner = min(
        eligible,
        key=lambda result: (result.connected_kl, result.restart_id),
    )
    assert winner.connected_profiles is not None
    assert winner.connected_task_weights is not None
    reconstruction = winner.connected_profiles @ winner.connected_task_weights
    discovery = SubtaskDiscovery(
        ensemble=ensemble,
        profiles=winner.connected_profiles,
        task_weights=winner.connected_task_weights,
        reconstruction=reconstruction,
        reconstruction_error=_normalized_kl_divergence(
            ensemble.desirability,
            reconstruction,
        ),
        n_iter=sum(winner.fit_iterations),
        converged=True,
    )
    return NMFRankResult(
        rank=n_subtasks,
        restarts=restarts,
        selected_restart_id=winner.restart_id,
        discovery=discovery,
    )


def _connectivity_restart(
    ensemble: GoalTasks,
    n_subtasks: int,
    *,
    adjacency: np.ndarray,
    connectivity: NMFConnectivityConfig,
    restart_id: int,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> NMFRestartResult:
    target = ensemble.desirability
    unconstrained = _fit_nmf_factors(
        target,
        n_subtasks,
        init="random",
        profile_normalization=ensemble.parameters.profile_normalization,
        seed=seed,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    unconstrained_kl = _strict_generalized_kl_divergence(
        target,
        unconstrained.reconstruction,
    )
    forbidden = np.zeros_like(unconstrained.profiles, dtype=bool)
    fit_iterations = [unconstrained.n_iter]
    fit_converged = [unconstrained.converged]
    fully_forbidden = np.empty(0, dtype=int)
    zero_reconstruction_counts: list[int] = []
    fallback_attempt_counts: list[int] = []
    fallback_success_counts: list[int] = []
    prune_refit_rounds = 0
    current = unconstrained
    current_respects_forbidden = True
    feasible = True
    reason = _strict_kl_issue(target, unconstrained.reconstruction)
    if reason is None and not unconstrained.converged:
        reason = "unconstrained_not_converged"
    if reason is not None and reason != "unconstrained_not_converged":
        feasible = False

    if reason is None:
        for prune_round in range(connectivity.max_prune_refits):
            expanded, additions, _ = _expand_forbidden_mask(
                current.profiles,
                adjacency,
                connectivity.support_mass,
                forbidden,
            )
            if not np.any(additions):
                break
            forbidden = expanded
            current_respects_forbidden = False
            outcome = _masked_nmf_refit(
                target,
                current.profiles,
                current.task_weights,
                forbidden,
                profile_normalization=(
                    ensemble.parameters.profile_normalization
                ),
                max_iter=max_iter,
                tolerance=tolerance,
                fallback_seeds=_derived_positive_fallback_seeds(
                    seed,
                    prune_round,
                    connectivity.positive_fallback_attempts,
                ),
            )
            fully_forbidden = outcome.fully_forbidden_state_indices
            if outcome.positive_target_zero_reconstruction_count:
                zero_reconstruction_counts.append(
                    outcome.positive_target_zero_reconstruction_count
                )
                fallback_attempt_counts.append(
                    outcome.positive_fallback_attempt_count
                )
                fallback_success_counts.append(
                    outcome.positive_fallback_success_count
                )
            fit_iterations.extend(outcome.fit_iterations)
            fit_converged.extend(outcome.fit_converged)
            if outcome.fit is None:
                feasible = (
                    outcome.reason
                    != "fully_forbidden_state"
                )
                reason = outcome.reason
                break
            current = outcome.fit
            current_respects_forbidden = True
            prune_refit_rounds += 1
            if outcome.reason is not None:
                reason = outcome.reason
                break

    final_connected = _component_support_connectivity(
        current.profiles,
        adjacency,
        connectivity.support_mass,
    )
    if reason is None and not np.all(final_connected):
        reason = "disconnected_after_max_rounds"

    connected_profiles = None
    connected_weights = None
    connected_kl = None
    if current_respects_forbidden:
        issue = _strict_kl_issue(target, current.reconstruction)
        if issue is None:
            final_kl = _strict_generalized_kl_divergence(
                target,
                current.reconstruction,
            )
            if np.isfinite(final_kl):
                connected_profiles = current.profiles
                connected_weights = current.task_weights
                connected_kl = final_kl
            elif reason is None:
                reason = "nonfinite_factors_or_kl"
        elif reason is None:
            reason = issue

    effective = _effective_support_sizes(current.profiles)
    discarded = np.sum(
        np.where(forbidden, unconstrained.profiles, 0.0),
        axis=0,
    ) / unconstrained.profiles.sum(axis=0)
    eligible = reason is None
    return NMFRestartResult(
        restart_id=restart_id,
        seed=seed,
        unconstrained_profiles=unconstrained.profiles,
        unconstrained_task_weights=unconstrained.task_weights,
        unconstrained_kl=unconstrained_kl,
        connected_profiles=connected_profiles,
        connected_task_weights=connected_weights,
        connected_kl=connected_kl,
        forbidden_mask=forbidden,
        discarded_mass_fractions=discarded,
        effective_support_sizes=effective,
        effective_support_fractions=effective / target.shape[0],
        final_support_connected=final_connected,
        prune_refit_rounds=prune_refit_rounds,
        fit_iterations=tuple(fit_iterations),
        fit_converged=tuple(fit_converged),
        fully_forbidden_state_indices=fully_forbidden,
        positive_target_zero_reconstruction_counts=tuple(
            zero_reconstruction_counts
        ),
        positive_fallback_attempt_counts=tuple(fallback_attempt_counts),
        positive_fallback_success_counts=tuple(fallback_success_counts),
        feasible=feasible,
        eligible=eligible,
        reason=reason,
    )


def _masked_nmf_refit(
    target: np.ndarray,
    profiles: np.ndarray,
    task_weights: np.ndarray,
    forbidden: np.ndarray,
    *,
    profile_normalization: ProfileNormalization,
    max_iter: int,
    tolerance: float,
    fallback_seeds: tuple[int, ...] = (0,),
) -> _MaskedNMFRefitResult:
    fully_forbidden = np.flatnonzero(
        (target > 0.0).any(axis=1) & np.all(forbidden, axis=1)
    )
    if len(fully_forbidden):
        return _MaskedNMFRefitResult(
            fit=None,
            reason="fully_forbidden_state",
            fully_forbidden_state_indices=fully_forbidden,
            positive_target_zero_reconstruction_count=0,
            used_positive_fallback=False,
            fit_iterations=(),
            fit_converged=(),
        )

    initial_profiles = np.array(profiles, dtype=np.float64, copy=True)
    initial_weights = np.array(task_weights, dtype=np.float64, copy=True)
    initial_profiles[forbidden] = 0.0
    initial_reconstruction = initial_profiles @ initial_weights
    issue = _strict_kl_issue(target, initial_reconstruction)
    attempted_fits: list[_NMFFit] = []
    zero_count = 0

    if issue is None:
        try:
            warm_fit = _fit_nmf_factors(
                target,
                profiles.shape[1],
                init="custom",
                profile_normalization=profile_normalization,
                seed=None,
                max_iter=max_iter,
                tolerance=tolerance,
                initial_profiles=initial_profiles,
                initial_task_weights=initial_weights,
            )
        except _EmptyComponentError:
            return _MaskedNMFRefitResult(
                fit=None,
                reason="empty_component",
                fully_forbidden_state_indices=fully_forbidden,
                positive_target_zero_reconstruction_count=0,
                used_positive_fallback=False,
                fit_iterations=(),
                fit_converged=(),
            )
        attempted_fits.append(warm_fit)
        if np.any(warm_fit.profiles[forbidden] != 0.0):
            raise RuntimeError(
                "Masked NMF changed a forbidden profile entry"
            )
        issue = _strict_kl_issue(target, warm_fit.reconstruction)
        if issue is None:
            reason = None if warm_fit.converged else "constrained_not_converged"
            return _masked_refit_result(
                warm_fit,
                reason,
                fully_forbidden,
                zero_count=0,
                used_positive_fallback=False,
                attempted_fits=attempted_fits,
            )
        if issue != "positive_target_zero_reconstruction":
            return _masked_refit_result(
                warm_fit,
                issue,
                fully_forbidden,
                zero_count=0,
                used_positive_fallback=False,
                attempted_fits=attempted_fits,
            )
        zero_count = _positive_target_zero_reconstruction_count(
            target,
            warm_fit.reconstruction,
        )
    elif issue == "positive_target_zero_reconstruction":
        zero_count = _positive_target_zero_reconstruction_count(
            target,
            initial_reconstruction,
        )
    else:
        return _masked_refit_result(
            None,
            issue,
            fully_forbidden,
            zero_count=0,
            used_positive_fallback=False,
            attempted_fits=attempted_fits,
        )

    fallback_source_profiles = (
        attempted_fits[-1].profiles
        if attempted_fits
        else initial_profiles
    )
    fallback_source_weights = (
        attempted_fits[-1].task_weights
        if attempted_fits
        else initial_weights
    )
    fallback_candidates: list[tuple[float, int, _NMFFit]] = []
    for attempt_index, fallback_seed in enumerate(fallback_seeds):
        fallback_profiles, fallback_weights = (
            _positive_masked_initialization(
                target,
                fallback_source_profiles,
                fallback_source_weights,
                forbidden,
                seed=fallback_seed,
            )
        )
        try:
            fallback_fit = _fit_nmf_factors(
                target,
                profiles.shape[1],
                init="custom",
                profile_normalization=profile_normalization,
                seed=None,
                max_iter=max_iter,
                tolerance=tolerance,
                initial_profiles=fallback_profiles,
                initial_task_weights=fallback_weights,
            )
        except _EmptyComponentError:
            continue
        attempted_fits.append(fallback_fit)
        if np.any(fallback_fit.profiles[forbidden] != 0.0):
            raise RuntimeError(
                "Masked NMF changed a forbidden profile entry"
            )
        issue = _strict_kl_issue(target, fallback_fit.reconstruction)
        if issue is not None or not fallback_fit.converged:
            continue
        fallback_kl = _strict_generalized_kl_divergence(
            target,
            fallback_fit.reconstruction,
        )
        fallback_candidates.append(
            (fallback_kl, attempt_index, fallback_fit)
        )

    attempt_count = len(fallback_seeds)
    success_count = len(fallback_candidates)
    if not fallback_candidates:
        return _masked_refit_result(
            None,
            "positive_fallback_failed",
            fully_forbidden,
            zero_count=zero_count,
            used_positive_fallback=True,
            attempted_fits=attempted_fits,
            fallback_attempt_count=attempt_count,
            fallback_success_count=0,
        )
    _, _, best_fallback = min(
        fallback_candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return _masked_refit_result(
        best_fallback,
        None,
        fully_forbidden,
        zero_count=zero_count,
        used_positive_fallback=True,
        attempted_fits=attempted_fits,
        fallback_attempt_count=attempt_count,
        fallback_success_count=success_count,
    )


def _masked_refit_result(
    fit: _NMFFit | None,
    reason: str | None,
    fully_forbidden: np.ndarray,
    *,
    zero_count: int,
    used_positive_fallback: bool,
    attempted_fits: list[_NMFFit],
    fallback_attempt_count: int = 0,
    fallback_success_count: int = 0,
) -> _MaskedNMFRefitResult:
    return _MaskedNMFRefitResult(
        fit=fit,
        reason=reason,
        fully_forbidden_state_indices=fully_forbidden,
        positive_target_zero_reconstruction_count=zero_count,
        used_positive_fallback=used_positive_fallback,
        fit_iterations=tuple(item.n_iter for item in attempted_fits),
        fit_converged=tuple(item.converged for item in attempted_fits),
        positive_fallback_attempt_count=fallback_attempt_count,
        positive_fallback_success_count=fallback_success_count,
    )


def _derived_positive_fallback_seeds(
    restart_seed: int,
    prune_round: int,
    attempts: int,
) -> tuple[int, ...]:
    return tuple(
        int(
            np.random.SeedSequence(
                [restart_seed, prune_round, attempt]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        for attempt in range(attempts)
    )


def _positive_masked_initialization(
    target: np.ndarray,
    profiles: np.ndarray,
    task_weights: np.ndarray,
    forbidden: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    profile_values = np.asarray(profiles, dtype=np.float64)
    weight_values = np.asarray(task_weights, dtype=np.float64)
    if profile_values.shape != forbidden.shape:
        raise ValueError("Fallback profiles and mask must have equal shapes")
    n_components = profile_values.shape[1]
    if weight_values.shape != (n_components, target.shape[1]):
        raise ValueError("Fallback NMF factors have incompatible shapes")

    target_scale = np.sqrt(float(np.mean(target)) / n_components)
    if not np.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("Positive masked initialization requires positive data")
    profile_scale = _typical_positive_value(profile_values, target_scale)
    weight_scale = _typical_positive_value(weight_values, target_scale)
    random = np.random.RandomState(seed)
    initial_profiles = np.zeros_like(profile_values)
    initial_weights = np.empty_like(weight_values)

    for component in range(n_components):
        allowed = ~forbidden[:, component]
        component_profile_scale = _typical_positive_value(
            profile_values[allowed, component],
            profile_scale,
        )
        initial_profiles[allowed, component] = (
            component_profile_scale
            * random.uniform(0.1, 1.0, size=np.count_nonzero(allowed))
        )
        component_weight_scale = _typical_positive_value(
            weight_values[component],
            weight_scale,
        )
        initial_weights[component] = (
            component_weight_scale
            * random.uniform(0.1, 1.0, size=target.shape[1])
        )

    if np.any(initial_profiles[~forbidden] <= 0.0):
        raise RuntimeError("Allowed fallback profile entries must be positive")
    if np.any(initial_profiles[forbidden] != 0.0):
        raise RuntimeError("Forbidden fallback entries must remain zero")
    if np.any(initial_weights <= 0.0):
        raise RuntimeError("Fallback task weights must be positive")
    if _strict_kl_issue(target, initial_profiles @ initial_weights) is not None:
        raise RuntimeError("Positive masked initialization is not KL-feasible")
    return initial_profiles, initial_weights


def _typical_positive_value(values: np.ndarray, fallback: float) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0.0]
    if not len(positive):
        return float(fallback)
    return float(np.median(positive))

def _positive_target_zero_reconstruction_count(
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> int:
    return int(
        np.count_nonzero((target > 0.0) & (reconstruction == 0.0))
    )

def _q_mass_support(
    column: np.ndarray,
    mass_fraction: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(column, dtype=np.float64)
    if (
        values.ndim != 1
        or np.any(values < 0.0)
        or not np.all(np.isfinite(values))
        or values.sum() <= 0.0
    ):
        raise ValueError("Mass support requires a positive non-negative vector")
    if not np.isfinite(mass_fraction) or not 0.0 < mass_fraction <= 1.0:
        raise ValueError("Mass fraction must be in (0, 1]")

    required_mass = mass_fraction * values.sum()
    levels, counts = np.unique(values, return_counts=True)
    cumulative = 0.0
    for cutoff, count in zip(levels[::-1], counts[::-1]):
        cumulative += float(cutoff) * int(count)
        if cumulative >= required_mass:
            return values >= cutoff, float(cutoff)
    raise RuntimeError("Failed to find a mass-support cutoff")


def _support_components(
    support: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, ...]:
    support_values = np.asarray(support, dtype=bool)
    adjacency_values = np.asarray(adjacency, dtype=bool)
    if adjacency_values.shape != (len(support_values), len(support_values)):
        raise ValueError("Support and adjacency shapes do not align")

    remaining = set(int(state) for state in np.flatnonzero(support_values))
    components: list[np.ndarray] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            state = stack.pop()
            component.append(state)
            neighbors = np.flatnonzero(
                adjacency_values[state] & support_values
            )
            for neighbor_value in neighbors:
                neighbor = int(neighbor_value)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=int))
    return tuple(sorted(components, key=lambda component: int(component[0])))


def _expand_forbidden_mask(
    profiles: np.ndarray,
    adjacency: np.ndarray,
    mass_fraction: float,
    forbidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    forbidden_values = np.asarray(forbidden, dtype=bool)
    if forbidden_values.shape != values.shape:
        raise ValueError("Forbidden mask must have the shape of D")
    additions = np.zeros_like(forbidden_values)
    connected = np.ones(values.shape[1], dtype=bool)

    for component_index in range(values.shape[1]):
        support, _ = _q_mass_support(
            values[:, component_index],
            mass_fraction,
        )
        components = _support_components(support, adjacency)
        if len(components) <= 1:
            continue
        connected[component_index] = False
        masses = np.asarray(
            [
                values[component, component_index].sum()
                for component in components
            ]
        )
        maximum = float(masses.max())
        tied = [
            component
            for component, mass in zip(components, masses)
            if np.isclose(mass, maximum, rtol=1e-12, atol=1e-15)
        ]
        retained = min(tied, key=lambda component: int(component.min()))
        for component in components:
            if component is not retained:
                additions[component, component_index] = True

    additions &= ~forbidden_values
    return forbidden_values | additions, additions, connected


def _component_support_connectivity(
    profiles: np.ndarray,
    adjacency: np.ndarray,
    mass_fraction: float,
) -> np.ndarray:
    connected = np.ones(profiles.shape[1], dtype=bool)
    for component_index in range(profiles.shape[1]):
        support, _ = _q_mass_support(
            profiles[:, component_index],
            mass_fraction,
        )
        connected[component_index] = (
            len(_support_components(support, adjacency)) <= 1
        )
    return connected


def _effective_support_sizes(profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float64)
    denominator = np.square(values).sum(axis=0)
    if np.any(denominator <= 0.0):
        raise _EmptyComponentError("Effective support is undefined")
    return np.square(values.sum(axis=0)) / denominator


def _strict_kl_issue(
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> str | None:
    target_values = np.asarray(target, dtype=np.float64)
    reconstruction_values = np.asarray(reconstruction, dtype=np.float64)
    if target_values.shape != reconstruction_values.shape:
        raise ValueError("KL target and reconstruction shapes must match")
    if (
        np.any(reconstruction_values < 0.0)
        or not np.all(np.isfinite(reconstruction_values))
    ):
        return "nonfinite_factors_or_kl"
    if np.any(
        (target_values > 0.0) & (reconstruction_values == 0.0)
    ):
        return "positive_target_zero_reconstruction"
    return None


def _strict_generalized_kl_divergence(
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    issue = _strict_kl_issue(target, reconstruction)
    if issue is not None:
        return np.inf
    target_values = np.asarray(target, dtype=np.float64)
    reconstruction_values = np.asarray(reconstruction, dtype=np.float64)
    logarithmic_term = np.zeros_like(target_values)
    positive = target_values > 0.0
    logarithmic_term[positive] = target_values[positive] * (
        np.log(target_values[positive])
        - np.log(reconstruction_values[positive])
    )
    return float(
        np.sum(logarithmic_term - target_values + reconstruction_values)
    )


def _normalize_nmf_factors(
    profiles: np.ndarray,
    task_weights: np.ndarray,
    profile_normalization: ProfileNormalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Fix every NMF component gauge without changing its reconstruction."""

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

    normalized_profiles, component_scales = _normalize_profile_columns(
        profile_values,
        profile_normalization,
        empty_message="NMF produced an empty subtask profile",
    )
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
