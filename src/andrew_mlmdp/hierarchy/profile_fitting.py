"""Direct marginal penalized-likelihood fitting of subgoal profiles.

The subgoal profile matrix ``D`` is learned from behavior together with the
behavioral parameters ``theta = (lower_control_cost, upper_control_cost, alpha,
beta)``. The latent controller sequence is marginalized exactly by the prepared
hierarchy likelihood. The optimizer variable is a non-negative raw matrix
``W``; the peak gauge ``D = W / max_s W`` is applied inside every forward pass,
so autodiff sees how each entry (including the peak) changes the whole column.

The minimized objective is

    J = -LL / N + l_c R_c(D) + l_d R_Dir(D) + l_o R_o(D),

where ``N`` is the number of likelihood factors (collapsed physical moves).
Dividing by ``N`` keeps the structural penalties at the same relative strength
for mice with different amounts of data, so this is penalized marginal
likelihood rather than MAP under a fixed prior.

After each Adam step ``W`` is projected onto ``W >= 0``, which preserves exact
zeros. A projected step that makes any column structurally infeasible for any
evaluation goal is rejected: parameters and Adam state are restored and the
learning rate is lowered. Rejections are optimizer events, never evidence that
the model is infeasible.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.discovery import _graph_adjacency_from_passive
from andrew_mlmdp.fitting import ParameterValues, _snapshot, validate_adam_config
from andrew_mlmdp.hierarchy.equations import NumericalError, parameter_values
from andrew_mlmdp.hierarchy.fitting import _plateau_scheduler, _RawParameters
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.model import SubgoalBasis
from andrew_mlmdp.maze import Coordinate, Maze

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy.model import Template


PENALTY_NAMES = ("compactness", "dirichlet", "overlap")
THETA_NAMES = ("lower_control_cost", "upper_control_cost", "alpha", "beta")
# Restarts ending for these reasons have no trustworthy best state.
FAILURE_REASONS = (
    "numerical_failure",
    "nonfinite_loss",
    "nonfinite_gradient",
    "rejection_limit",
)


@dataclass(frozen=True)
class ProfileGeometry:
    """Maze geometry used by the structural penalties."""

    distances: Tensor
    edges: Tensor


@dataclass(frozen=True)
class ProfileRejection:
    """One rejected projected step and why it was structurally invalid."""

    updates: int
    lr: float
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ProfileFitStep:
    """Diagnostics for one evaluated (W, theta) state."""

    evaluation: int
    updates: int
    lr: float
    objective: float
    log_likelihood_per_decision: float
    penalties: Mapping[str, float]
    best_objective: float | None
    theta_values: Mapping[str, float]
    theta_gradients: Mapping[str, float]
    profile_gradient_norm: float
    theta_gradient_norm: float
    raw_column_maxima: tuple[float, ...]
    peak_states: tuple[int, ...]
    goal_one_hot_margin: tuple[float, ...]
    # Validation LL per decision at this state, when it was evaluated.
    validation_ll_per_decision: float | None = None


@dataclass(frozen=True)
class ProfileFitResult:
    """Outcome, history, and provenance of one profile-fitting restart."""

    initial_raw_profiles: np.ndarray
    best_raw_profiles: np.ndarray | None
    last_raw_profiles: np.ndarray
    # The profiles the model uses: gated when access_threshold is set.
    best_profiles: np.ndarray | None
    # Peak-gauged profiles before the gate (equal to best_profiles without one);
    # strictly positive in log space, so usable to warm-start another fit.
    best_ungated_profiles: np.ndarray | None
    best_theta: ParameterValues | None
    # Training objective at the reported best state.
    best_objective: float | None
    history: tuple[ProfileFitStep, ...]
    rejections: tuple[ProfileRejection, ...]
    updates: int
    converged: bool
    reason: str
    provenance: Mapping[str, object]
    # "training": best state = lowest training objective. "validation": best
    # state = highest validation LL among the checked states.
    selection: str = "training"
    best_validation_ll_per_decision: float | None = None

    @property
    def objective_history(self) -> tuple[float, ...]:
        return tuple(step.objective for step in self.history)


def peak_gauge(raw_profiles: Tensor) -> Tensor:
    """Return ``D = W / max_s W`` column-wise (differentiable)."""

    return raw_profiles / raw_profiles.max(dim=0, keepdim=True).values


def log_peak_gauge(log_profiles: Tensor) -> Tensor:
    """Return ``D = exp(V - max_s V)`` column-wise (differentiable).

    Near ``D = 0`` the likelihood depends on ``log D``: the composed Layer-1
    policy can take a subgoal-access transition of passive probability
    ``alpha * D`` at a control cost of about ``lower_control_cost *
    log(1 / (alpha * D))``, which can undercut an exponentially discounted
    walk. Optimizing ``V = log D`` puts the parameter on that scale. ``D`` is
    then strictly positive; exact zeros are only the limit ``V -> -inf``.
    """

    return torch.exp(log_profiles - log_profiles.max(dim=0, keepdim=True).values)


def access_gate(profiles: Tensor, threshold: float | None) -> Tensor:
    """Fixed core gate ``max(0, D - tau) / (1 - tau)`` on peak-gauged ``D``.

    Entries below ``tau`` become exact zeros, so subgoal access there is
    impossible rather than merely improbable. Without it, strictly positive
    tails let the composed policy take subgoal access almost anywhere at a
    control cost of about ``lower_control_cost * log(1 / (alpha * D))``; when
    ``lower_control_cost`` is also free, that cost collapses and the joint fit
    degenerates. The peak stays at one. ``None`` leaves ``D`` unchanged.
    """

    if threshold is None:
        return profiles
    return torch.clamp(profiles - threshold, min=0.0) / (1.0 - threshold)


def profile_geometry(template: "Template") -> ProfileGeometry:
    """Diameter-normalized shortest paths and undirected passive edges."""

    maze = template.maze
    coordinates = maze.free_cells
    n_states = len(coordinates)
    distances = np.empty((n_states, n_states), dtype=np.float64)
    for source, coordinate in enumerate(coordinates):
        reached = maze.distances_from(coordinate)
        if len(reached) != n_states:
            raise ValueError("Profile compactness requires a connected maze")
        for target, other in enumerate(coordinates):
            distances[source, target] = reached[other]
    diameter = distances.max()
    if diameter <= 0.0:
        raise ValueError("Profile compactness requires at least two states")
    adjacency = _graph_adjacency_from_passive(template.environment.passive)
    edges = np.argwhere(np.triu(adjacency > 0.0, k=1))
    return ProfileGeometry(
        distances=torch.tensor(distances / diameter, dtype=torch.float64),
        edges=torch.tensor(edges, dtype=torch.long),
    )


def structural_penalties(
    profiles: Tensor,
    geometry: ProfileGeometry,
) -> dict[str, Tensor]:
    """Return compactness, graph-Dirichlet, and duplicate-subgoal penalties.

    ``profiles`` must be peak-gauged with no all-zero column.
    """

    n_subgoals = profiles.shape[1]
    distances = geometry.distances.to(profiles.device)
    edges = geometry.edges.to(profiles.device)

    mass = profiles / profiles.sum(dim=0, keepdim=True)
    compactness = (mass * (distances @ mass)).sum() / n_subgoals

    differences = profiles[edges[:, 0]] - profiles[edges[:, 1]]
    dirichlet = differences.pow(2).sum() / (n_subgoals * len(edges))

    if n_subgoals == 1:
        overlap = profiles.sum() * 0.0
    else:
        norms = torch.linalg.vector_norm(profiles, dim=0)
        cosine = (profiles.T @ profiles) / (norms[:, None] * norms[None, :])
        upper = torch.triu_indices(n_subgoals, n_subgoals, offset=1)
        overlap = (
            cosine[upper[0], upper[1]].pow(2).sum()
            * 2.0
            / (n_subgoals * (n_subgoals - 1))
        )
    return {
        "compactness": compactness,
        "dirichlet": dirichlet,
        "overlap": overlap,
    }


def profile_feasibility(
    raw_profiles: np.ndarray | Tensor,
    maze: Maze,
    goals: Iterable[Coordinate],
) -> tuple[str, ...]:
    """Return structural violations of a non-negative profile matrix.

    A column is usable for goal ``g`` only if it has positive mass on some
    state other than ``g``: the goal is removed from the Layer-1 interior, so a
    column supported only on ``g`` has no access row and its upper transition
    column is empty. In a connected maze this is exactly the condition that
    ``_build_hierarchy`` needs. The check is scale-free, so it can be applied
    to raw ``W`` before gauging.
    """

    if isinstance(raw_profiles, Tensor):
        raw_profiles = raw_profiles.detach().cpu().numpy()
    values = np.asarray(raw_profiles, dtype=np.float64)
    violations = []
    for column in range(values.shape[1]):
        profile = values[:, column]
        if not profile.max() > 0.0:
            violations.append(f"column {column} is all zero")
            continue
        for goal in goals:
            off_goal = profile.sum() - profile[maze.state_index(goal)]
            if not off_goal > 0.0:
                violations.append(
                    f"column {column} is supported only on goal {goal}"
                )
    return tuple(violations)


def goal_one_hot_margin(
    profiles: np.ndarray,
    maze: Maze,
    goals: Iterable[Coordinate],
) -> tuple[float, ...]:
    """Per column, the smallest off-goal mass over goals (peak-gauged D)."""

    goal_states = [maze.state_index(goal) for goal in goals]
    totals = profiles.sum(axis=0)
    return tuple(
        float(min(totals[column] - profiles[state, column] for state in goal_states))
        for column in range(profiles.shape[1])
    )


def random_localized_profiles(
    maze: Maze,
    n_subgoals: int,
    rng: np.random.Generator,
    *,
    shape: str = "exponential",
) -> np.ndarray:
    """Random smooth localized columns with peak one.

    ``shape="exponential"`` gives ``exp(-d / width)``; ``"gaussian"`` gives
    ``exp(-(d / width)**2 / 2)``, whose far tails are negligible, which matters
    because small positive access values act as cheap exits (see
    :func:`log_peak_gauge`). Centres are distinct random states and widths are
    uniform between one step and a quarter of the maze diameter. These are
    task-independent and are not derived from NMF.
    """

    if shape not in ("exponential", "gaussian"):
        raise ValueError("shape must be 'exponential' or 'gaussian'")

    coordinates = maze.free_cells
    n_states = len(coordinates)
    if not 1 <= n_subgoals <= n_states:
        raise ValueError("n_subgoals must be between 1 and the number of states")
    centres = rng.choice(n_states, size=n_subgoals, replace=False)
    profiles = np.empty((n_states, n_subgoals), dtype=np.float64)
    diameter = 0
    columns = []
    for centre in centres:
        reached = maze.distances_from(coordinates[centre])
        column = np.asarray([reached[other] for other in coordinates], dtype=np.float64)
        diameter = max(diameter, int(column.max()))
        columns.append(column)
    widths = rng.uniform(1.0, max(1.0, 0.25 * diameter), size=n_subgoals)
    for index, (column, width) in enumerate(zip(columns, widths, strict=True)):
        if shape == "exponential":
            profiles[:, index] = np.exp(-column / width)
        else:
            profiles[:, index] = np.exp(-0.5 * (column / width) ** 2)
    return profiles


def fit_profiles(
    template: "Template",
    trials: Iterable[Trial],
    *,
    initial_profiles: np.ndarray,
    evaluation_goals: Iterable[Coordinate],
    penalty_weights: Mapping[str, float],
    scheduler_tolerance: float,
    convergence_tolerance: float,
    theta_names: Sequence[str] = THETA_NAMES,
    lr: float = 5e-2,
    max_steps: int = 1000,
    patience: int = 20,
    lr_decay: float = 0.3,
    lr_patience: int = 7,
    min_lr: float = 1e-5,
    max_rejections: int = 20,
    parameterization: str = "linear",
    access_threshold: float | None = None,
    validation_trials: Iterable[Trial] | None = None,
    validation_interval: int = 10,
    validation_patience: int | None = None,
    freeze_profiles: bool = False,
    initial_profile_seed: int | None = None,
    initial_profile_description: str | None = None,
    callback: Callable[[ProfileFitStep], None] | None = None,
) -> ProfileFitResult:
    """Fit the raw profiles and ``theta`` by full-batch Adam on ``J``.

    ``parameterization="linear"`` optimizes ``W >= 0`` with ``D = W / max W``
    and projection after each step (exact zeros representable).
    ``parameterization="log"`` optimizes unconstrained ``V`` with
    ``D = exp(V - max V)`` (strictly positive ``D``, no projection); it needs
    strictly positive ``initial_profiles``.

    ``access_threshold`` (tau) applies :func:`access_gate` to the peak-gauged
    profiles before the likelihood, so the model uses exact-zero support. The
    structural penalties act on the ungated profiles, which keeps a gradient
    on sub-threshold entries so they can re-enter the support.

    ``freeze_profiles=True`` holds the profiles at ``initial_profiles`` (after
    the peak gauge and any gate) and fits only ``theta``; it is the theta step
    of a staged fit and needs at least one name in ``theta_names``.

    With ``validation_trials``, validation LL per decision is evaluated every
    ``validation_interval`` evaluations (no gradient flows through it), and the
    reported best state is the checked state with the highest validation LL
    rather than the lowest training objective. ``validation_patience`` stops
    the fit after that many consecutive checks without improvement. The
    validation trials then act as a selection set and must not also be used
    for final scoring.

    ``evaluation_goals`` must contain every goal the fitted model will be
    scored on, including held-out goals; feasibility is enforced for all of
    them. Tolerances are on the per-decision objective scale and must be
    chosen explicitly. The template's structural modes are frozen and recorded.
    """

    basis = template.basis
    if basis.locations is not None or basis.core_threshold is not None:
        raise ValueError("Profile fitting requires an ungated distributed basis")
    if template.composition_mode == "winner_take_all":
        raise ValueError("Hard winner-take-all composition cannot be fitted")
    if template.composition_exponent != 1.0:
        raise ValueError("Profile fitting fixes composition_exponent at 1.0")
    materialized = tuple(trials)
    if not materialized:
        raise ValueError("Profile fitting requires at least one trial")
    theta = tuple(theta_names)
    if len(set(theta)) != len(theta) or not set(theta) <= set(THETA_NAMES):
        raise ValueError(
            "theta_names must be distinct names from " + ", ".join(THETA_NAMES)
        )
    if set(penalty_weights) != set(PENALTY_NAMES):
        raise ValueError("penalty_weights must have keys " + ", ".join(PENALTY_NAMES))
    weights = {name: float(penalty_weights[name]) for name in PENALTY_NAMES}
    if not all(np.isfinite(value) and value >= 0.0 for value in weights.values()):
        raise ValueError("penalty_weights must be finite and non-negative")
    if isinstance(max_rejections, bool) or max_rejections < 0:
        raise ValueError("max_rejections must be a non-negative integer")
    if parameterization not in ("linear", "log"):
        raise ValueError("parameterization must be 'linear' or 'log'")
    log_space = parameterization == "log"
    if access_threshold is not None and not 0.0 <= access_threshold < 1.0:
        raise ValueError("access_threshold must be in [0, 1)")
    if isinstance(validation_interval, bool) or validation_interval < 1:
        raise ValueError("validation_interval must be a positive integer")
    if freeze_profiles and not theta:
        raise ValueError("freeze_profiles needs at least one theta name to fit")
    if validation_patience is not None and validation_patience < 1:
        raise ValueError("validation_patience must be a positive integer")
    config = validate_adam_config(
        lr=lr,
        max_steps=max_steps,
        tolerance=convergence_tolerance,
        scheduler_tolerance=scheduler_tolerance,
        convergence_tolerance=convergence_tolerance,
        patience=patience,
        lr_decay=lr_decay,
        lr_patience=lr_patience,
        min_lr=min_lr,
    )

    maze = template.maze
    training_goals = tuple(dict.fromkeys(trial.goal for trial in materialized))
    goals = tuple(dict.fromkeys(evaluation_goals))
    missing = set(training_goals) - set(goals)
    if missing:
        raise ValueError("evaluation_goals must include every training goal")
    initial = np.asarray(initial_profiles, dtype=np.float64)
    expected = (len(maze.free_cells), basis.n_subgoals)
    if initial.shape != expected:
        raise ValueError(f"initial_profiles must have shape {expected}")
    if np.any(initial < 0.0) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_profiles must be finite and non-negative")
    violations = profile_feasibility(initial, maze, goals)
    if violations:
        raise ValueError("Infeasible initial profiles: " + "; ".join(violations))
    if log_space and np.any(initial <= 0.0):
        raise ValueError(
            "Log parameterization needs strictly positive initial_profiles"
        )
    # Start with every column at peak one so lr has a meaning in D (or log D)
    # units.
    initial = initial / initial.max(axis=0, keepdims=True)
    gauge = log_peak_gauge if log_space else peak_gauge
    if access_threshold is not None:
        violations = profile_feasibility(
            access_gate(torch.tensor(initial), access_threshold), maze, goals
        )
        if violations:
            raise ValueError(
                "Infeasible initial profiles after the access gate: "
                + "; ".join(violations)
            )

    geometry = profile_geometry(template)
    prepared = prepare_batch(template, materialized)
    # Impossible trials (goal visited before the end) score -inf and are not
    # dropped; they contribute no prepared factors but their moves still count.
    n_decisions = _collapsed_move_count(materialized)
    possible = tuple(
        trial
        for trial, record in zip(materialized, prepared.trials, strict=True)
        if not record.impossible
    )
    prepared_factors = sum(len(trial.operator_indices) for trial in prepared.trials)
    if prepared_factors != _collapsed_move_count(possible):
        raise RuntimeError("Prepared factor count disagrees with collapsed moves")
    if n_decisions == 0:
        raise ValueError("Profile fitting requires at least one movement")

    validation = None if validation_trials is None else tuple(validation_trials)
    if validation is not None:
        validation_goals = {trial.goal for trial in validation}
        if not validation_goals <= set(goals):
            raise ValueError("evaluation_goals must include every validation goal")
        prepared_validation = prepare_batch(template, validation)
        n_validation = _collapsed_move_count(validation)
        if n_validation == 0:
            raise ValueError("validation_trials contain no movements")

    initial_values = {
        name: value.detach() for name, value in parameter_values(template).items()
    }
    raw_theta = _RawParameters(initial_values, theta, threshold_max=None)
    raw_initial = np.log(initial) if log_space else initial
    raw_profiles = nn.Parameter(
        torch.tensor(raw_initial, dtype=torch.float64),
        requires_grad=not freeze_profiles,
    )

    def feasibility_violations() -> tuple[str, ...]:
        # The check is scale-free, so linear W can be checked directly; log V
        # is mapped back to strictly positive D (always usable for every goal).
        with torch.no_grad():
            if not log_space:
                # An all-zero linear column has no peak to gauge by.
                violations = profile_feasibility(raw_profiles, maze, goals)
                if violations or access_threshold is None:
                    return violations
            checked = access_gate(gauge(raw_profiles), access_threshold)
        return profile_feasibility(checked, maze, goals)

    def make_optimizer(rate: float) -> torch.optim.Adam:
        fitted = [*raw_theta.parameters()]
        if not freeze_profiles:
            fitted.insert(0, raw_profiles)
        return torch.optim.Adam(fitted, lr=rate)

    def make_scheduler(optimizer):
        return _plateau_scheduler(
            optimizer,
            factor=config.lr_decay,
            patience=config.lr_patience,
            relative_threshold=config.scheduler_tolerance,
            min_lr=config.min_lr,
        )

    def clone_state() -> tuple[Tensor, dict[str, Tensor]]:
        return raw_profiles.detach().clone(), raw_theta.clone_raw_state()

    def restore_state(state: tuple[Tensor, dict[str, Tensor]]) -> None:
        with torch.no_grad():
            raw_profiles.copy_(state[0])
        raw_theta.restore_raw_state(state[1])

    optimizer = make_optimizer(config.lr)
    scheduler = make_scheduler(optimizer)
    history: list[ProfileFitStep] = []
    rejections: list[ProfileRejection] = []
    best_objective: float | None = None
    best_state = None
    stage_best: float | None = None
    stale_steps = 0
    at_final_rate = False
    updates = 0
    converged = False
    reason = "max_steps"
    best_validation: float | None = None
    best_validation_state = None
    best_validation_objective: float | None = None
    stale_checks = 0

    while True:
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        try:
            ungated = gauge(raw_profiles)
            profiles = access_gate(ungated, access_threshold)
            values = raw_theta.physical_values(initial_values)
            log_likelihood = total_prepared_log_likelihood(
                template,
                prepared,
                parameter_values=values,
                access_profiles=profiles,
            )
            penalties = structural_penalties(ungated, geometry)
        except NumericalError:
            reason = "numerical_failure"
            break
        objective = -log_likelihood / n_decisions + sum(
            weights[name] * penalties[name] for name in PENALTY_NAMES
        )
        objective_value = float(objective.detach())

        if not np.isfinite(objective_value):
            history.append(
                _step(history, updates, learning_rate_now, objective_value,
                      log_likelihood, n_decisions, penalties, best_objective,
                      values, theta, None, raw_profiles, profiles, maze, goals)
            )
            if callback is not None:
                callback(history[-1])
            reason = "nonfinite_loss"
            break

        if best_objective is None or objective_value < best_objective:
            best_objective = objective_value
            best_state = clone_state()

        validation_value = None
        if validation is not None and len(history) % validation_interval == 0:
            with torch.no_grad():
                validation_value = float(
                    total_prepared_log_likelihood(
                        template,
                        prepared_validation,
                        parameter_values={
                            name: value.detach() for name, value in values.items()
                        },
                        access_profiles=profiles.detach(),
                    )
                ) / n_validation
            if np.isfinite(validation_value) and (
                best_validation is None or validation_value > best_validation
            ):
                best_validation = validation_value
                best_validation_state = clone_state()
                best_validation_objective = objective_value
                stale_checks = 0
            else:
                stale_checks += 1

        try:
            objective.backward()
        except RuntimeError:
            reason = "numerical_failure"
            break
        history.append(
            _step(history, updates, learning_rate_now, objective_value,
                  log_likelihood, n_decisions, penalties, best_objective,
                  values, theta, raw_theta, raw_profiles, profiles, maze, goals,
                  validation_value)
        )
        if callback is not None:
            callback(history[-1])
        latest = history[-1]
        if validation_patience is not None and stale_checks >= validation_patience:
            reason = "validation_patience"
            break
        if not (
            np.isfinite(latest.profile_gradient_norm)
            and np.isfinite(latest.theta_gradient_norm)
        ):
            reason = "nonfinite_gradient"
            break

        # Convergence bookkeeping is identical to andrew_mlmdp.fitting.fit_adam.
        at_min_lr = learning_rate_now <= config.min_lr
        if not at_min_lr:
            stage_best = best_objective
            stale_steps = 0
            at_final_rate = False
        elif not at_final_rate:
            stage_best = best_objective
            stale_steps = 0
            at_final_rate = True
        elif updates > 0:
            meaningful = config.convergence_tolerance * max(1.0, abs(stage_best))
            if stage_best - best_objective > meaningful:
                stage_best = best_objective
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= config.patience:
                converged = True
                reason = "patience"
                break

        if updates >= config.max_steps:
            reason = "max_steps"
            break

        # Propose a projected step; on structural violation restore the exact
        # pre-step parameters and Adam state, lower lr, and retry with the
        # same gradients (they are untouched by the restore).
        rejected_out = False
        while True:
            state_before = clone_state()
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            optimizer.step()
            if not log_space:
                with torch.no_grad():
                    raw_profiles.clamp_(min=0.0)
            violations = feasibility_violations()
            if not violations:
                break
            step_lr = float(optimizer.param_groups[0]["lr"])
            restore_state(state_before)
            optimizer.load_state_dict(optimizer_before)
            rejections.append(ProfileRejection(updates, step_lr, violations))
            next_lr = step_lr * config.lr_decay
            if len(rejections) > max_rejections or next_lr < config.min_lr:
                rejected_out = True
                break
            for group in optimizer.param_groups:
                group["lr"] = next_lr
        if rejected_out:
            reason = "rejection_limit"
            break

        updates += 1
        rate_before_scheduler = float(optimizer.param_groups[0]["lr"])
        scheduler.step(objective_value)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        if next_learning_rate < rate_before_scheduler:
            assert best_state is not None
            restore_state(best_state)
            optimizer = make_optimizer(next_learning_rate)
            scheduler = make_scheduler(optimizer)
            stage_best = best_objective
            stale_steps = 0
            at_final_rate = False

    selection = "training"
    if best_validation_state is not None:
        # Report the validation-selected state instead of the training optimum.
        best_state = best_validation_state
        best_objective = best_validation_objective
        selection = "validation"
    last_raw = raw_profiles.detach().cpu().numpy().copy()
    best_raw = best_profiles = best_ungated = best_theta = None
    if best_state is not None:
        restore_state(best_state)
        best_raw = raw_profiles.detach().cpu().numpy().copy()
        with torch.no_grad():
            ungated_tensor = gauge(raw_profiles)
            best_ungated = ungated_tensor.cpu().numpy().copy()
            best_profiles = (
                access_gate(ungated_tensor, access_threshold).cpu().numpy().copy()
            )
        best_theta = _snapshot(raw_theta.physical_values(initial_values))
    provenance = {
        "n_subgoals": basis.n_subgoals,
        "penalty_weights": dict(weights),
        "theta_names": theta,
        "training_goals": training_goals,
        "evaluation_goals": goals,
        "n_trials": len(materialized),
        "n_decisions": n_decisions,
        "initial_profile_seed": initial_profile_seed,
        "initial_profile_description": initial_profile_description,
        "parameterization": parameterization,
        "access_threshold": access_threshold,
        "freeze_profiles": freeze_profiles,
        "validation": None
        if validation is None
        else {
            "n_trials": len(validation),
            "n_decisions": n_validation,
            "interval": validation_interval,
            "patience": validation_patience,
        },
        "goal_reward_mode": template.goal_reward_mode,
        "commitment_mode": template.commitment_mode,
        "commitment_radius": template.commitment_radius,
        "composition_exponent": float(template.composition_exponent),
        "adam": {
            "lr": config.lr,
            "max_steps": config.max_steps,
            "scheduler_tolerance": config.scheduler_tolerance,
            "convergence_tolerance": config.convergence_tolerance,
            "patience": config.patience,
            "lr_decay": config.lr_decay,
            "lr_patience": config.lr_patience,
            "min_lr": config.min_lr,
            "max_rejections": max_rejections,
        },
    }
    return ProfileFitResult(
        initial_raw_profiles=raw_initial,
        best_raw_profiles=best_raw,
        last_raw_profiles=last_raw,
        best_profiles=best_profiles,
        best_ungated_profiles=best_ungated,
        selection=selection,
        best_validation_ll_per_decision=best_validation,
        best_theta=best_theta,
        best_objective=best_objective,
        history=tuple(history),
        rejections=tuple(rejections),
        updates=updates,
        converged=converged,
        reason=reason,
        provenance=MappingProxyType(provenance),
    )


def fit_profile_restarts(
    template: "Template",
    trials: Iterable[Trial],
    *,
    seeds: Sequence[int],
    initial_shape: str = "exponential",
    **fit_options,
) -> tuple[ProfileFitResult, ...]:
    """Fit one restart per explicit seed from random localized profiles."""

    materialized = tuple(trials)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and distinct")
    results = []
    for seed in seeds:
        initial = random_localized_profiles(
            template.maze,
            template.basis.n_subgoals,
            np.random.default_rng(seed),
            shape=initial_shape,
        )
        results.append(
            fit_profiles(
                template,
                materialized,
                initial_profiles=initial,
                initial_profile_seed=int(seed),
                initial_profile_description=f"random localized {initial_shape}",
                **fit_options,
            )
        )
    return tuple(results)


def best_restart(results: Iterable[ProfileFitResult]) -> ProfileFitResult | None:
    """Return the eligible restart with the lowest training objective.

    Restarts that ended in a failure reason are ineligible but remain in the
    caller's result tuple. Returns ``None`` if none are eligible.
    """

    eligible = [
        result
        for result in results
        if result.reason not in FAILURE_REASONS and result.best_objective is not None
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda result: result.best_objective)


def fitted_basis(maze: Maze, result: ProfileFitResult) -> SubgoalBasis:
    """Ungated peak-normalized basis carrying the fitted profiles."""

    if result.best_profiles is None:
        raise ValueError("The restart has no best profiles")
    return SubgoalBasis.from_profiles(maze, result.best_profiles, core_threshold=None)


def _collapsed_move_count(trials: Sequence[Trial]) -> int:
    """Count consecutive distinct coordinates, as the likelihood factors do."""

    return sum(
        current != following
        for trial in trials
        for current, following in zip(trial.trajectory, trial.trajectory[1:])
    )


def _step(
    history, updates, lr, objective_value, log_likelihood, n_decisions,
    penalties, best_objective, values, theta, raw_theta, raw_profiles,
    profiles, maze, goals, validation_value=None,
) -> ProfileFitStep:
    theta_gradients = {}
    squared = 0.0
    for name in theta:
        gradient = None if raw_theta is None else raw_theta.raw[name].grad
        value = np.nan if gradient is None else float(gradient.detach())
        theta_gradients[name] = value
        squared += value**2
    profile_gradient = None if raw_theta is None else raw_profiles.grad
    if raw_theta is None:
        profile_norm = np.nan
    elif not raw_profiles.requires_grad:
        profile_norm = 0.0  # frozen profiles (theta-only step)
    elif profile_gradient is None:
        profile_norm = np.nan
    else:
        profile_norm = float(torch.linalg.vector_norm(profile_gradient.detach()))
    gauged = profiles.detach().cpu().numpy()
    raw = raw_profiles.detach().cpu().numpy()
    return ProfileFitStep(
        evaluation=len(history),
        updates=updates,
        lr=lr,
        objective=objective_value,
        log_likelihood_per_decision=float(log_likelihood.detach()) / n_decisions,
        penalties=MappingProxyType(
            {name: float(penalties[name].detach()) for name in PENALTY_NAMES}
        ),
        best_objective=best_objective,
        theta_values=MappingProxyType(
            {name: float(values[name].detach()) for name in theta}
        ),
        theta_gradients=MappingProxyType(theta_gradients),
        profile_gradient_norm=profile_norm,
        theta_gradient_norm=float(np.sqrt(squared)) if theta else 0.0,
        raw_column_maxima=tuple(float(value) for value in raw.max(axis=0)),
        peak_states=tuple(int(value) for value in raw.argmax(axis=0)),
        goal_one_hot_margin=goal_one_hot_margin(gauged, maze, goals),
        validation_ll_per_decision=validation_value,
    )
