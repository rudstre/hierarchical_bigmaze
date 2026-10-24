"""Parametric subgoal profiles and a discrete search over subgoal centres.

Free-form profile fitting (``profile_fitting.fit_profiles``) from random starts
gets trapped in local optima once theta is also free: which states are
subgoals is effectively a combinatorial choice, and a fitted bump cannot slide
continuously to a better location because intermediate profiles are worse
than either end. This module separates that choice from the continuous fit.

Each subgoal ``j`` is a bump around a centre state ``c_j`` with width ``w_j``,

    exponential:  D_j(s) = exp(-d(s, c_j) / w_j)
    gaussian:     D_j(s) = exp(-(d(s, c_j) / w_j) ** 2 / 2)

where ``d`` is the shortest-path distance in maze steps. Profiles are peak-1 by
construction; an optional fixed access gate (``access_gate``) gives exact-zero
support away from the centre. Widths are ``w = MIN_WIDTH + softplus(u)``: with
``MIN_WIDTH = 0.5`` the centre's neighbours keep D = exp(-2) > 0.05 under both
shapes, so with the default gate every column has mass on at least two states
and no arrangement is structurally infeasible.

The centres are searched discretely, k-medoids style:

* greedy addition: for K = 1, 2, ... every unused state is screened as the new
  centre (one likelihood evaluation with current widths and theta), the best
  ``screen_top`` candidates are refitted (widths and theta by Adam), and the
  best refit is kept;
* swaps: every (subgoal, new centre) move is screened the same way, the best
  candidates refitted, and the best move accepted while it improves training
  LL by more than ``tolerance``;
* local swaps: the same, but a centre may only move within ``radius`` maze
  steps, which costs about 4K screens instead of K times the state count;
* ``search``: variable-neighbourhood search alternating local swaps with
  single global swap moves until neither improves;
* ``seeded_centres``: k-means++-style seeding (Arthur & Vassilvitskii, 2007)
  in maze distance, weighted by state occupancy, as spread-out random starts.

Only training LL drives the search; validation and held-out data are scored
by the caller. Runner-up centres per subgoal (screened LL with the others
fixed) are reported as a measure of location uncertainty.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.hierarchy.equations import NumericalError
from andrew_mlmdp.hierarchy.fitting import _RawParameters
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.model import SubgoalBasis
from andrew_mlmdp.hierarchy.profile_fitting import (
    _collapsed_move_count,
    access_gate,
)
from andrew_mlmdp.lmdp import Environment, Parameters
from andrew_mlmdp.maze import Maze

SHAPES = ("exponential", "gaussian")
MIN_WIDTH = 0.5
BASE_PARAMETER_NAMES = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
)


def maze_distances(maze: Maze) -> np.ndarray:
    """Shortest-path distances in steps, ``[state, state]`` (connected maze)."""

    cells = maze.free_cells
    distances = np.empty((len(cells), len(cells)), dtype=np.float64)
    for source, cell in enumerate(cells):
        reached = maze.distances_from(cell)
        if len(reached) != len(cells):
            raise ValueError("Subgoal search requires a connected maze")
        distances[source] = [reached[other] for other in cells]
    return distances


def bump_profiles(
    distances: Tensor,
    centres: Sequence[int],
    widths: Tensor,
    shape: str,
) -> Tensor:
    """Peak-1 bump profiles ``D[state, subgoal]`` around the given centres."""

    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    scaled = distances[:, list(centres)] / widths
    if shape == "exponential":
        return torch.exp(-scaled)
    return torch.exp(-0.5 * scaled**2)


@dataclass(frozen=True)
class CentreFit:
    """Fitted widths and theta for one fixed arrangement of centres."""

    centres: tuple[int, ...]
    widths: tuple[float, ...]
    # All six base parameter values (physical), fitted or fixed.
    parameters: Mapping[str, float]
    train_ll_per_decision: float
    steps: int


@dataclass(frozen=True)
class SearchMove:
    """One accepted greedy addition or swap."""

    kind: str  # "add" or "swap"
    n_subgoals: int
    subgoal: int
    from_centre: int | None
    to_centre: int
    train_ll_per_decision: float
    # Screened candidates that were refitted: (subgoal, centre, refit LL).
    refitted: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class SearchResult:
    """Final arrangement for one K, the moves that led there, and runner-ups."""

    start: str
    fit: CentreFit
    moves: tuple[SearchMove, ...]
    # Per subgoal: best alternative centres as (centre, screened LL - fit LL).
    runner_ups: tuple[tuple[tuple[int, float], ...], ...]


class CentreSearch:
    """Discrete centre search with continuous widths and theta.

    ``fixed_parameters`` gives every base parameter value; those named in
    ``theta_names`` are starting values and are fitted, the rest stay fixed
    (for example ``alpha`` and the reward gauge). ``structure`` holds the
    template's goal-reward and commitment settings.
    """

    def __init__(
        self,
        environment: Environment,
        trials: Iterable[Trial],
        *,
        structure: Mapping[str, object],
        fixed_parameters: Mapping[str, float],
        theta_names: Sequence[str],
        shape: str = "exponential",
        access_threshold: float | None = 0.05,
        initial_width: float = 1.0,
        fit_steps: int = 150,
        fit_lr: float = 0.05,
        fit_patience: int = 20,
        screen_top: int = 5,
        tolerance: float = 1e-5,
    ) -> None:
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if set(fixed_parameters) != set(BASE_PARAMETER_NAMES):
            raise ValueError("fixed_parameters must give every base parameter")
        if not set(theta_names) <= set(BASE_PARAMETER_NAMES) - {
            "interior_reward",
            "goal_reward",
        }:
            raise ValueError("theta_names must be control costs, alpha, or beta")
        if initial_width <= MIN_WIDTH:
            raise ValueError(f"initial_width must exceed MIN_WIDTH={MIN_WIDTH}")
        self.environment = environment
        self.maze = environment.maze
        self.structure = dict(structure)
        self.fixed_parameters = {k: float(v) for k, v in fixed_parameters.items()}
        self.theta_names = tuple(theta_names)
        self.shape = shape
        self.access_threshold = access_threshold
        self.initial_width = float(initial_width)
        self.fit_steps = fit_steps
        self.fit_lr = fit_lr
        self.fit_patience = fit_patience
        self.screen_top = screen_top
        self.tolerance = tolerance
        self.n_states = len(self.maze.free_cells)
        self.distances = torch.tensor(maze_distances(self.maze), dtype=torch.float64)
        self._templates: dict[int, object] = {}
        self.trials = tuple(trials)
        # Prepared trial structure does not depend on K, only on goals,
        # trajectories, and the commitment radius.
        self.prepared = prepare_batch(self._template(1), self.trials)
        self.n_decisions = _collapsed_move_count(self.trials)
        if self.n_decisions == 0:
            raise ValueError("trials contain no movements")
        self.likelihood_evaluations = 0

    # -- model pieces ---------------------------------------------------------

    def _template(self, n_subgoals: int):
        if n_subgoals not in self._templates:
            basis = SubgoalBasis.from_profiles(
                self.maze, np.ones((self.n_states, n_subgoals)), core_threshold=None
            )
            self._templates[n_subgoals] = self.environment.hierarchy(
                basis,
                parameters=Parameters(**self.fixed_parameters),
                **self.structure,
            )
        return self._templates[n_subgoals]

    def profiles(self, centres: Sequence[int], widths: Tensor) -> Tensor:
        """Gated bump profiles used by the likelihood."""

        return access_gate(
            bump_profiles(self.distances, centres, widths, self.shape),
            self.access_threshold,
        )

    def _values(self, parameters: Mapping[str, float]) -> dict[str, Tensor]:
        return {
            name: torch.tensor(float(parameters[name]), dtype=torch.float64)
            for name in BASE_PARAMETER_NAMES
        }

    def log_likelihood(
        self,
        centres: Sequence[int],
        widths: Sequence[float],
        parameters: Mapping[str, float],
        *,
        trials: Sequence[Trial] | None = None,
    ) -> float:
        """LL per decision without gradients; -inf if the model cannot score it."""

        if trials is None:
            prepared, n_decisions = self.prepared, self.n_decisions
        else:
            prepared = prepare_batch(self._template(len(centres)), trials)
            n_decisions = _collapsed_move_count(trials)
        with torch.no_grad():
            try:
                value = total_prepared_log_likelihood(
                    self._template(len(centres)),
                    prepared,
                    parameter_values=self._values(parameters),
                    access_profiles=self.profiles(
                        centres, torch.tensor(list(widths), dtype=torch.float64)
                    ),
                )
            except (NumericalError, ValueError):
                return float("-inf")
        self.likelihood_evaluations += 1
        result = float(value) / n_decisions
        return result if np.isfinite(result) else float("-inf")

    # -- continuous fit for fixed centres -------------------------------------

    def fit(
        self,
        centres: Sequence[int],
        widths: Sequence[float],
        parameters: Mapping[str, float],
    ) -> CentreFit:
        """Fit widths and theta by full-batch Adam for fixed centres."""

        centres = tuple(int(c) for c in centres)
        template = self._template(len(centres))
        start_widths = torch.tensor(
            [max(float(w), MIN_WIDTH + 1e-3) for w in widths], dtype=torch.float64
        )
        shifted = start_widths - MIN_WIDTH
        raw_widths = nn.Parameter(shifted + torch.log(-torch.expm1(-shifted)))
        initial_values = self._values(parameters)
        raw_theta = _RawParameters(initial_values, self.theta_names, threshold_max=None)
        optimizer = torch.optim.Adam(
            [raw_widths, *raw_theta.parameters()], lr=self.fit_lr
        )

        def current():
            values = raw_theta.physical_values(initial_values)
            return MIN_WIDTH + functional.softplus(raw_widths), values

        best = None
        stale = 0
        steps = 0
        for steps in range(1, self.fit_steps + 1):
            optimizer.zero_grad()
            widths_now, values = current()
            try:
                log_likelihood = total_prepared_log_likelihood(
                    template,
                    self.prepared,
                    parameter_values=values,
                    access_profiles=self.profiles(centres, widths_now),
                )
            except (NumericalError, ValueError):
                break
            self.likelihood_evaluations += 1
            per_decision = float(log_likelihood.detach()) / self.n_decisions
            if not np.isfinite(per_decision):
                break
            if best is None or per_decision > best[0] + self.tolerance:
                stale = 0
            else:
                stale += 1
            if best is None or per_decision > best[0]:
                best = (
                    per_decision,
                    tuple(float(w) for w in widths_now.detach()),
                    {name: float(v.detach()) for name, v in values.items()},
                )
            if stale >= self.fit_patience:
                break
            (-log_likelihood / self.n_decisions).backward()
            gradients = [raw_widths.grad, *(p.grad for p in raw_theta.parameters())]
            if not all(g is None or bool(torch.isfinite(g).all()) for g in gradients):
                break
            optimizer.step()
        if best is None:
            return CentreFit(
                centres,
                tuple(float(w) for w in start_widths),
                dict(parameters),
                float("-inf"),
                steps,
            )
        return CentreFit(centres, best[1], best[2], best[0], steps)

    # -- discrete moves -------------------------------------------------------

    def _refit_best(
        self,
        candidates: list[tuple[float, int, tuple[int, ...], tuple[float, ...]]],
        parameters: Mapping[str, float],
    ) -> tuple[CentreFit | None, tuple[tuple[int, int, float], ...]]:
        """Refit the top screened candidates; return the best and a record."""

        candidates = sorted(candidates, key=lambda item: -item[0])
        best = None
        record = []
        for _, subgoal, centres, widths in candidates[: self.screen_top]:
            refit = self.fit(centres, widths, parameters)
            record.append((subgoal, centres[subgoal], refit.train_ll_per_decision))
            if best is None or refit.train_ll_per_decision > best.train_ll_per_decision:
                best = refit
        return best, tuple(record)

    def add(self, fit: CentreFit | None) -> tuple[CentreFit, SearchMove]:
        """Greedy addition of one subgoal to ``fit`` (``None`` starts at K=0)."""

        centres = () if fit is None else fit.centres
        widths = () if fit is None else fit.widths
        parameters = self.fixed_parameters if fit is None else fit.parameters
        subgoal = len(centres)
        candidates = []
        for state in range(self.n_states):
            if state in centres:
                continue
            new_centres = (*centres, state)
            new_widths = (*widths, self.initial_width)
            screened = self.log_likelihood(new_centres, new_widths, parameters)
            candidates.append((screened, subgoal, new_centres, new_widths))
        best, record = self._refit_best(candidates, parameters)
        assert best is not None
        move = SearchMove(
            "add",
            len(best.centres),
            subgoal,
            None,
            best.centres[subgoal],
            best.train_ll_per_decision,
            record,
        )
        return best, move

    def swap(
        self, fit: CentreFit, max_moves: int = 50, *, radius: int | None = None
    ) -> tuple[CentreFit, list]:
        """Accept best single-centre swaps while training LL improves.

        With ``radius``, a centre may only move to a state within ``radius``
        maze steps of its current position (local moves).
        """

        moves = []
        for _ in range(max_moves):
            candidates = []
            for subgoal in range(len(fit.centres)):
                for state in range(self.n_states):
                    if state in fit.centres:
                        continue
                    if (
                        radius is not None
                        and self.distances[fit.centres[subgoal], state] > radius
                    ):
                        continue
                    new_centres = list(fit.centres)
                    new_centres[subgoal] = state
                    screened = self.log_likelihood(
                        new_centres, fit.widths, fit.parameters
                    )
                    candidates.append(
                        (screened, subgoal, tuple(new_centres), fit.widths)
                    )
            best, record = self._refit_best(candidates, fit.parameters)
            if best is None or (
                best.train_ll_per_decision
                <= fit.train_ll_per_decision + self.tolerance
            ):
                break
            changed = next(
                j for j in range(len(fit.centres)) if best.centres[j] != fit.centres[j]
            )
            moves.append(
                SearchMove(
                    "swap",
                    len(best.centres),
                    changed,
                    fit.centres[changed],
                    best.centres[changed],
                    best.train_ll_per_decision,
                    record,
                )
            )
            fit = best
        return fit, moves

    def runner_ups(
        self, fit: CentreFit, n_alternatives: int = 3
    ) -> tuple[tuple[tuple[int, float], ...], ...]:
        """Best alternative centres per subgoal, screened with the others fixed."""

        result = []
        for subgoal in range(len(fit.centres)):
            scores = []
            for state in range(self.n_states):
                if state in fit.centres:
                    continue
                centres = list(fit.centres)
                centres[subgoal] = state
                scores.append(
                    (
                        state,
                        self.log_likelihood(centres, fit.widths, fit.parameters)
                        - fit.train_ll_per_decision,
                    )
                )
            scores.sort(key=lambda item: -item[1])
            result.append(tuple(scores[:n_alternatives]))
        return tuple(result)

    # -- full searches --------------------------------------------------------

    def greedy(
        self, max_subgoals: int, *, swaps: bool = True
    ) -> tuple[SearchResult, ...]:
        """Greedy addition with swap refinement after each K = 1..max_subgoals."""

        if not 1 <= max_subgoals < self.n_states:
            raise ValueError("max_subgoals out of range")
        results = []
        fit = None
        moves: list[SearchMove] = []
        for _ in range(max_subgoals):
            fit, added = self.add(fit)
            moves.append(added)
            if swaps:
                fit, swapped = self.swap(fit)
                moves.extend(swapped)
            results.append(
                SearchResult("greedy", fit, tuple(moves), self.runner_ups(fit))
            )
        return tuple(results)

    def from_random_centres(
        self, n_subgoals: int, rng: np.random.Generator
    ) -> SearchResult:
        """Swap search from a random arrangement (robustness start)."""

        centres = tuple(
            int(c) for c in rng.choice(self.n_states, size=n_subgoals, replace=False)
        )
        fit = self.fit(
            centres, (self.initial_width,) * n_subgoals, self.fixed_parameters
        )
        fit, moves = self.swap(fit)
        return SearchResult("random", fit, tuple(moves), self.runner_ups(fit))

    def search(
        self, fit: CentreFit, *, radius: int = 1, max_rounds: int = 20
    ) -> tuple[CentreFit, list]:
        """Variable-neighbourhood search: local swaps, then one global move.

        Local swaps (within ``radius`` steps) run until they stall; then one
        global swap move is tried. If it improves training LL the local phase
        resumes, otherwise the search stops.
        """

        moves = []
        for _ in range(max_rounds):
            fit, local = self.swap(fit, radius=radius)
            moves.extend(local)
            fit, jumped = self.swap(fit, max_moves=1)
            moves.extend(jumped)
            if not jumped:
                break
        return fit, moves

    def seeded_centres(
        self,
        n_subgoals: int,
        rng: np.random.Generator,
        weights: Sequence[float],
    ) -> tuple[int, ...]:
        """k-means++-style centres: occupancy times squared distance to chosen.

        The first centre is drawn proportional to ``weights[state]`` (e.g.
        training occupancy); each later one proportional to
        ``weights[state] * d(state, nearest chosen centre) ** 2``, so chosen
        states have zero probability and the centres spread over the maze.
        """

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.n_states,) or (weights < 0).any():
            raise ValueError("weights must be one non-negative value per state")
        if not 1 <= n_subgoals <= int((weights > 0).sum()):
            raise ValueError("n_subgoals out of range for the positive weights")
        distances = self.distances.numpy()
        centres: list[int] = []
        for _ in range(n_subgoals):
            if centres:
                nearest = distances[:, centres].min(axis=1)
                score = weights * nearest**2
            else:
                score = weights.copy()
            centres.append(int(rng.choice(self.n_states, p=score / score.sum())))
        return tuple(centres)

    def from_seeded_centres(
        self,
        n_subgoals: int,
        rng: np.random.Generator,
        weights: Sequence[float],
        *,
        radius: int = 1,
    ) -> SearchResult:
        """``search`` from k-means++-style seeded centres."""

        centres = self.seeded_centres(n_subgoals, rng, weights)
        fit = self.fit(
            centres, (self.initial_width,) * n_subgoals, self.fixed_parameters
        )
        fit, moves = self.search(fit, radius=radius)
        return SearchResult("seeded", fit, tuple(moves), self.runner_ups(fit))

    def ungated_profiles(self, fit: CentreFit) -> np.ndarray:
        """Strictly positive bump profiles (for warm-starting free-form fits)."""

        with torch.no_grad():
            return (
                bump_profiles(
                    self.distances,
                    fit.centres,
                    torch.tensor(fit.widths, dtype=torch.float64),
                    self.shape,
                )
                .numpy()
                .copy()
            )
