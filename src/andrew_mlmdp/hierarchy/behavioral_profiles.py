"""Behavior-derived subgoal profiles for one theta cell, and theta selection.

This is the canonical form of the procedure validated on simulated data
(``results/theta_grid/maze1_m2_seeds/theta_cells.sbatch``): with theta held
at one grid cell,

1. the discrete centre search (``subgoal_search.CentreSearch``) runs from a
   greedy start and from k-means++-seeded starts;
2. every searched arrangement warm-starts a free-form profile fit
   (``profile_fitting.fit_profiles``, log space, fixed access gate,
   structural penalties, theta held);
3. the refined start with the lowest training objective is the cell's D.

Theta itself is a hyperparameter: callers fit one cell per grid point and
choose among cells on held-out data (``select_theta_per_rank``). Nothing in
this module looks at validation data; training trials drive every fit.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.hierarchy.equations import NumericalError
from andrew_mlmdp.hierarchy.model import SubgoalBasis, Template
from andrew_mlmdp.hierarchy.profile_fitting import FAILURE_REASONS, fit_profiles
from andrew_mlmdp.hierarchy.subgoal_search import (
    BASE_PARAMETER_NAMES,
    CentreSearch,
    SearchResult,
)
from andrew_mlmdp.lmdp import Environment, Parameters

HELD_THETA_NAMES = ("lower_control_cost", "upper_control_cost", "beta")


@dataclass(frozen=True)
class BehavioralFitSettings:
    """Every choice of the per-cell procedure; defaults = theta_cells.sbatch."""

    # Frozen structural choices (see run_profile_recovery.STRUCTURE).
    goal_reward_mode: str = "deferred"
    commitment_mode: str = "both"
    commitment_radius: int | None = 2
    # Fixed gauge and access strength; theta = (lambda_1, lambda_2, beta).
    interior_reward: float = -1.0
    goal_reward: float = 0.0
    alpha: float = 1.0
    # Fixed core gate on D: exact-zero support below this peak fraction.
    access_threshold: float = 0.05
    # Centre search.
    shape: str = "gaussian"
    search_fit_steps: int = 60
    search_fit_patience: int = 10
    screen_top: int = 2
    local_radius: int = 1
    greedy_start: bool = True
    n_seeded_starts: int = 8
    # Free-form refinement.
    parameterization: str = "log"
    lr: float = 0.1
    max_steps: int = 1500
    tolerance: float = 1e-6
    lambda_compactness: float = 0.3
    lambda_dirichlet: float = 0.1
    lambda_overlap: float = 0.0
    # Log-space warm starts need strictly positive profiles; Gaussian bumps
    # underflow far from the centre. Initialization only.
    warm_floor: float = float(np.exp(-30.0))
    # Theta grids.
    coarse_lower: tuple[float, ...] = (0.3, 0.6, 1.2)
    coarse_upper: tuple[float, ...] = (0.6, 1.2, 2.4)
    coarse_beta: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    local_step: float = 2.0**0.5

    def __post_init__(self) -> None:
        for name in ("coarse_lower", "coarse_upper", "coarse_beta"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values or any(not (v > 0 and math.isfinite(v)) for v in values):
                raise ValueError(f"{name} must be non-empty positive values")
            object.__setattr__(self, name, values)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain distinct values")
            if len({f"{v:.6g}" for v in values}) != len(values):
                raise ValueError(f"{name} values collide at six significant figures")
        for name in (
            "search_fit_steps",
            "search_fit_patience",
            "screen_top",
            "local_radius",
            "max_steps",
            "n_seeded_starts",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"local_radius", "n_seeded_starts"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.greedy_start, bool):
            raise ValueError("greedy_start must be boolean")
        if self.shape not in {"gaussian", "exponential"}:
            raise ValueError("shape must be gaussian or exponential")
        if self.parameterization not in {"linear", "log"}:
            raise ValueError("parameterization must be linear or log")
        for name in ("alpha", "lr", "tolerance", "warm_floor", "local_step"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.access_threshold < 1 or self.warm_floor > 1:
            raise ValueError("access_threshold must be in [0, 1); warm_floor <= 1")
        for name in ("lambda_compactness", "lambda_dirichlet", "lambda_overlap"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not all(math.isfinite(v) for v in (self.interior_reward, self.goal_reward)):
            raise ValueError("rewards must be finite")
        if self.goal_reward_mode not in {"deferred", "inpainted", "fixed"}:
            raise ValueError("invalid goal_reward_mode")
        if self.commitment_mode not in {"both", "termination", "radius"}:
            raise ValueError("invalid commitment_mode")
        if self.commitment_mode == "termination":
            if self.commitment_radius is not None:
                raise ValueError("termination requires commitment_radius=None")
        elif (
            isinstance(self.commitment_radius, bool)
            or not isinstance(self.commitment_radius, int)
            or self.commitment_radius < 1
        ):
            raise ValueError("radius commitment requires a positive integer radius")
        if not self.greedy_start and self.n_seeded_starts < 1:
            raise ValueError("at least one centre-search start is required")
        if self.n_seeded_starts < 0:
            raise ValueError("n_seeded_starts must be non-negative")
        if not self.local_step > 1.0:
            raise ValueError("local_step must exceed 1")

    @property
    def structure(self) -> dict[str, object]:
        return {
            "goal_reward_mode": self.goal_reward_mode,
            "commitment_mode": self.commitment_mode,
            "commitment_radius": self.commitment_radius,
        }

    @property
    def penalty_weights(self) -> dict[str, float]:
        return {
            "compactness": self.lambda_compactness,
            "dirichlet": self.lambda_dirichlet,
            "overlap": self.lambda_overlap,
        }

    @property
    def start_names(self) -> tuple[str, ...]:
        greedy = ("greedy",) if self.greedy_start else ()
        return greedy + tuple(f"seed{i}" for i in range(self.n_seeded_starts))

    def as_dict(self) -> dict[str, object]:
        return {key: _plain(value) for key, value in asdict(self).items()}


def _plain(value):
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def theta_key(theta: Sequence[float]) -> str:
    """Stable cell name for (lambda_1, lambda_2, beta)."""

    lower, upper, beta = (float(v) for v in theta)
    return f"l1_{lower:.6g}_l2_{upper:.6g}_b_{beta:.6g}"


def coarse_theta_grid(settings: BehavioralFitSettings) -> tuple[tuple[float, ...], ...]:
    """All (lambda_1, lambda_2, beta) cells of the coarse grid."""

    return tuple(
        itertools.product(
            settings.coarse_lower, settings.coarse_upper, settings.coarse_beta
        )
    )


def local_theta_grid(
    centre: Sequence[float], step: float
) -> tuple[tuple[float, ...], ...]:
    """The 26 neighbours of ``centre`` on a 3x3x3 grid at factor ``step``.

    The centre itself is excluded: it has already been fitted. Values are
    rounded to six significant figures as in ``make_theta_cells.py`` so cell
    identities are reproducible from their names.
    """

    lower, upper, beta = (float(v) for v in centre)
    if not math.isfinite(step) or step <= 1:
        raise ValueError("local step must be finite and exceed one")
    if any(not math.isfinite(v) or v <= 0 for v in (lower, upper, beta)):
        raise ValueError("local centre must contain positive finite values")
    cells = []
    for f1, f2, fb in itertools.product((-1, 0, 1), repeat=3):
        if (f1, f2, fb) == (0, 0, 0):
            continue
        cells.append(
            tuple(
                float(f"{value * step**power:.6g}")
                for value, power in ((lower, f1), (upper, f2), (beta, fb))
            )
        )
    if len(set(cells)) != 26 or tuple(centre) in cells:
        raise ValueError("local step does not give 26 distinct neighbours")
    return tuple(cells)


def held_parameters(
    theta: Sequence[float], settings: BehavioralFitSettings
) -> dict[str, float]:
    """All six base parameter values for a theta cell (others fixed)."""

    lower, upper, beta = (float(v) for v in theta)
    values = {
        "interior_reward": float(settings.interior_reward),
        "goal_reward": float(settings.goal_reward),
        "lower_control_cost": lower,
        "upper_control_cost": upper,
        "alpha": float(settings.alpha),
        "beta": beta,
    }
    assert set(values) == set(BASE_PARAMETER_NAMES)
    return values


def template_for_profiles(
    environment: Environment,
    profiles: np.ndarray,
    parameters: Mapping[str, float],
    settings: BehavioralFitSettings,
) -> Template:
    """Ungated hierarchy carrying ``profiles`` as its distributed basis.

    With gated fitted profiles this is the model the fit scored: the fit
    passes the same gated matrix as ``access_profiles``.
    """

    basis = SubgoalBasis.from_profiles(
        environment.maze, np.asarray(profiles, dtype=np.float64), core_threshold=None
    )
    return environment.hierarchy(
        basis, parameters=Parameters(**dict(parameters)), **settings.structure
    )


def training_occupancy(environment: Environment, trials: Iterable[Trial]) -> np.ndarray:
    """Visits per state over training trajectories (k-means++ seeding weights)."""

    maze = environment.maze
    counts = np.zeros(len(maze.free_cells))
    for trial in trials:
        for coordinate in trial.trajectory:
            counts[maze.state_index(coordinate)] += 1
    return counts


@dataclass(frozen=True)
class CellFit:
    """The selected D for one (data, K, theta) cell and every start's record."""

    k: int
    theta: tuple[float, float, float]
    status: str  # "success" or "scientific_failure"
    selected_start: str | None
    # Gated D the model uses, and the positive pre-gate D; None on failure.
    profiles: np.ndarray | None
    ungated_profiles: np.ndarray | None
    parameters: Mapping[str, float]
    best_objective: float | None
    starts: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    seed: int | None = None


def _atomic_save(path: Path, write) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(temporary)
    os.replace(temporary, path)


def _save_start(directory: Path, name: str, record: dict, arrays: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    def write_arrays(path: Path) -> None:
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    # Arrays first: a JSON record is the completion marker.
    _atomic_save(directory / f"{name}.npz", write_arrays)
    _atomic_save(
        directory / f"{name}.json",
        lambda path: path.write_text(json.dumps(record, indent=2)),
    )


def _load_start(directory: Path, name: str) -> tuple[dict, dict] | None:
    record_path = directory / f"{name}.json"
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    with np.load(directory / f"{name}.npz") as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    return record, arrays


def _checkpoint_identity(
    directory, environment, trials, goals, k, theta, settings, seed
):
    """Refuse reuse if data, geometry, settings, or scientific source changed."""
    package = Path(__file__).resolve().parents[1]
    source_paths = [
        *sorted((package / "hierarchy").glob("*.py")),
        package / "lmdp.py",
        package / "maze.py",
        package / "dataset.py",
    ]
    identity = {
        "k": k,
        "theta": theta,
        "settings": settings.as_dict(),
        "seed": seed,
        "goals": goals,
        "trials": [asdict(t) for t in trials],
        "coordinates": environment.maze.free_cells,
        "passive_sha256": hashlib.sha256(environment.passive.tobytes()).hexdigest(),
        "source": {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
    }
    encoded = json.dumps(identity, sort_keys=True)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.is_file():
        if path.read_text() != encoded:
            raise ValueError(f"Incompatible behavioral start checkpoints: {directory}")
    else:
        if any(directory.glob("*.npz")) or any(directory.glob("*.json")):
            raise ValueError(f"Unidentified behavioral start checkpoints: {directory}")
        _atomic_save(path, lambda temporary: temporary.write_text(encoded))


def _search_start(
    search: CentreSearch,
    name: str,
    k: int,
    settings: BehavioralFitSettings,
    seed: int,
    weights: np.ndarray,
) -> SearchResult:
    if name == "greedy":
        # Greedy addition with local swaps after each addition, then
        # variable-neighbourhood search (run_centre_init.run_start).
        fit = None
        moves = []
        for _ in range(k):
            fit, added = search.add(fit)
            fit, local = search.swap(fit, radius=settings.local_radius)
            moves += [added, *local]
        fit, searched = search.search(fit, radius=settings.local_radius)
        return SearchResult("greedy", fit, tuple(moves + searched), ())
    index = int(name[len("seed") :])
    child = np.random.SeedSequence(seed).spawn(index + 1)[index]
    # CentreSearch.from_seeded_centres without the runner-up screen, which
    # does not affect the arrangement.
    centres = search.seeded_centres(k, np.random.default_rng(child), weights)
    fit = search.fit(centres, (search.initial_width,) * k, search.fixed_parameters)
    fit, moves = search.search(fit, radius=settings.local_radius)
    return SearchResult("seeded", fit, tuple(moves), ())


def fit_cell(
    environment: Environment,
    trials: Sequence[Trial],
    evaluation_goals: Iterable[tuple[int, int]],
    k: int,
    theta: Sequence[float],
    settings: BehavioralFitSettings,
    seed: int,
    *,
    checkpoint_dir: Path | None = None,
    log=None,
) -> CellFit:
    """Fit D for one theta cell on ``trials``; select by training objective.

    ``seed`` seeds the k-means++ starts (start ``seed<i>`` uses child ``i`` of
    ``SeedSequence(seed)``, as ``run_centre_init.py --search-seed``). Callers
    comparing theta cells on the same data should pass the same seed so the
    cells differ only in theta. ``evaluation_goals`` must contain every goal
    the fitted model will be scored on (structural feasibility only).

    With ``checkpoint_dir``, each finished start is written atomically and
    reused on a later call, so a preempted task resumes; the directory must
    belong to exactly this (data, K, theta, settings, seed) identity.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    trials = tuple(trials)
    theta = tuple(float(v) for v in theta)
    if len(theta) != 3 or any(not (v > 0 and math.isfinite(v)) for v in theta):
        raise ValueError("theta must be three positive finite values")
    goals = tuple(sorted({tuple(goal) for goal in evaluation_goals}))
    if not trials or not {t.goal for t in trials} <= set(goals):
        raise ValueError(
            "trials required; evaluation_goals must include training goals"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        _checkpoint_identity(
            checkpoint_dir, environment, trials, goals, k, theta, settings, seed
        )
    parameters = held_parameters(theta, settings)
    say = log or (lambda message: None)

    search = None

    def get_search() -> CentreSearch:
        nonlocal search
        if search is None:
            search = CentreSearch(
                environment,
                trials,
                structure=settings.structure,
                fixed_parameters=parameters,
                theta_names=(),
                shape=settings.shape,
                access_threshold=settings.access_threshold,
                fit_steps=settings.search_fit_steps,
                fit_patience=settings.search_fit_patience,
                screen_top=settings.screen_top,
            )
        return search

    weights = training_occupancy(environment, trials)
    n_states = len(environment.maze.free_cells)
    refine_template = template_for_profiles(
        environment, np.ones((n_states, k)), parameters, settings
    )
    starts: dict[str, dict] = {}
    start_arrays: dict[str, dict] = {}
    for name in settings.start_names:
        started = time.perf_counter()
        loaded = None if checkpoint_dir is None else _load_start(checkpoint_dir, name)
        if loaded is not None:
            starts[name], start_arrays[name] = loaded
            say(f"{name}: reused checkpoint")
            continue
        try:
            result = _search_start(get_search(), name, k, settings, seed, weights)
            searched = result.fit
            with torch.no_grad():
                ungated = get_search().ungated_profiles(searched)
            initial = np.maximum(ungated, settings.warm_floor)
            refined = fit_profiles(
                refine_template,
                trials,
                initial_profiles=initial,
                evaluation_goals=goals,
                penalty_weights=settings.penalty_weights,
                theta_names=(),
                lr=settings.lr,
                max_steps=settings.max_steps,
                scheduler_tolerance=settings.tolerance,
                convergence_tolerance=settings.tolerance,
                parameterization=settings.parameterization,
                access_threshold=settings.access_threshold,
                initial_profile_description=f"centre search start {name}",
            )
        except (NumericalError, ValueError) as error:
            # An unusable initialization or failed numerical solve invalidates
            # this start; keep its evidence and continue the prescribed starts.
            record = {
                "start": name,
                "eligible": False,
                "best_objective": None,
                "refine_reason": "initialization_failure",
                "failure": {"type": type(error).__name__, "message": str(error)},
                "seed": seed,
                "elapsed_seconds": time.perf_counter() - started,
            }
            starts[name], start_arrays[name] = record, {}
            if checkpoint_dir is not None:
                _save_start(checkpoint_dir, name, record, {})
            say(f"{name}: {type(error).__name__}: {error}")
            continue
        eligible = (
            refined.reason not in FAILURE_REASONS
            and refined.best_objective is not None
            and refined.best_profiles is not None
            and math.isfinite(refined.best_objective)
        )
        record = {
            "start": name,
            "seed": seed,
            "seed_spawn_key": (
                None if name == "greedy" else [int(name.removeprefix("seed"))]
            ),
            "search_centres": [
                list(environment.maze.coordinate(c)) for c in searched.centres
            ],
            "search_widths": list(searched.widths),
            "search_train_ll_per_decision": searched.train_ll_per_decision,
            "search_moves": len(result.moves),
            "refine_reason": refined.reason,
            "refine_converged": refined.converged,
            "refine_updates": refined.updates,
            "refine_rejections": len(refined.rejections),
            "rejections": [asdict(rejection) for rejection in refined.rejections],
            "refine_provenance": dict(refined.provenance),
            "elapsed_seconds": time.perf_counter() - started,
            "best_objective": refined.best_objective if eligible else None,
            "eligible": eligible,
        }
        arrays = {"initial": initial}
        if eligible:
            arrays["profiles"] = refined.best_profiles
            arrays["ungated"] = refined.best_ungated_profiles
        starts[name], start_arrays[name] = record, arrays
        if checkpoint_dir is not None:
            _save_start(checkpoint_dir, name, record, arrays)
        say(
            f"{name}: search LL {searched.train_ll_per_decision:.5f} "
            f"refine {refined.reason} J={refined.best_objective}"
        )

    # Training-objective selection among eligible starts (best_restart rule).
    eligible_names = [name for name in starts if starts[name]["eligible"]]
    if not eligible_names:
        return CellFit(
            k,
            theta,
            "scientific_failure",
            None,
            None,
            None,
            parameters,
            None,
            starts,
            seed,
        )
    chosen = min(eligible_names, key=lambda name: starts[name]["best_objective"])
    return CellFit(
        k,
        theta,
        "success",
        chosen,
        start_arrays[chosen]["profiles"],
        start_arrays[chosen]["ungated"],
        parameters,
        float(starts[chosen]["best_objective"]),
        starts,
        seed,
    )


def cell_template(
    environment: Environment, cell: CellFit, settings: BehavioralFitSettings
) -> Template:
    """The fitted model of a successful cell, for strict scoring and prediction."""

    if cell.status != "success" or cell.profiles is None:
        raise ValueError("Only a successful cell has a fitted model")
    return template_for_profiles(environment, cell.profiles, cell.parameters, settings)


def select_theta_per_rank(
    records: Iterable[Mapping[str, object]],
    *,
    ranks: Sequence[int],
    validation_session_ids: Sequence[object],
    cells: Mapping[int, Sequence[str]],
) -> dict[str, object]:
    """Choose theta per K by mean inner validation LL across sessions.

    ``records`` are cell artifacts with ``k``, ``validation_session_id``,
    ``theta_cell``, ``status`` and, on success, a finite
    ``validation_ll_per_transition``. ``cells[k]`` lists the expected cell
    names at each K. A cell is eligible only when it succeeded for every
    validation session; missing or operationally failed cells keep the whole
    selection pending. Returns per-K theta rows and, for each K with an
    eligible cell, one record per session of the chosen cell, in the form
    ``nested_rank_selection`` expects. Ranks without an eligible cell get
    scientific-failure records, so they are ineligible rather than dropped.
    """

    sessions = tuple(validation_session_ids)
    if not sessions or len(set(sessions)) != len(sessions):
        raise ValueError("validation_session_ids must be non-empty and distinct")
    if not ranks or len(set(ranks)) != len(ranks) or set(cells) != set(ranks):
        raise ValueError("ranks and cells must have the same distinct non-empty ranks")
    if any(not grid or len(set(grid)) != len(grid) for grid in cells.values()):
        raise ValueError("each rank requires distinct non-empty cells")
    indexed: dict[tuple[int, object, str], Mapping[str, object]] = {}
    for record in records:
        if record.get("status") not in {
            "success",
            "scientific_failure",
            "operational_failure",
        }:
            raise ValueError(f"Invalid cell status: {record.get('status')!r}")
        key = (int(record["k"]), record["validation_session_id"], record["theta_cell"])
        if key[0] not in cells or key[2] not in cells[key[0]] or key[1] not in sessions:
            raise ValueError(f"Unexpected cell identity: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate cell identity: {key}")
        indexed[key] = record

    pending = False
    rank_rows = []
    nested_records = []
    for rank in ranks:
        cell_rows = []
        for cell in cells[rank]:
            found = [indexed.get((rank, session, cell)) for session in sessions]
            statuses = [None if r is None else r.get("status") for r in found]
            cell_pending = any(s in (None, "operational_failure") for s in statuses)
            pending = pending or cell_pending
            eligible = all(s == "success" for s in statuses)
            values = []
            if eligible:
                for r in found:
                    value = float(r["validation_ll_per_transition"])
                    if not math.isfinite(value):
                        raise ValueError(
                            "Successful cells require finite validation LL"
                        )
                    values.append(value)
            cell_rows.append(
                {
                    "theta_cell": cell,
                    "theta": None if found[0] is None else found[0].get("theta"),
                    "state": "pending" if cell_pending else "terminal",
                    "eligible": eligible,
                    "scientific_failure_session_ids": [
                        session
                        for session, s in zip(sessions, statuses, strict=True)
                        if s == "scientific_failure"
                    ],
                    "validation_ll_per_transition_mean": (
                        float(np.mean(values)) if eligible else None
                    ),
                    "validation_ll_per_transition": values if eligible else None,
                    "validation_ll_per_transition_se": (
                        float(np.std(values, ddof=1) / math.sqrt(len(values)))
                        if eligible and len(values) > 1
                        else (0.0 if eligible else None)
                    ),
                }
            )
        eligible_rows = [row for row in cell_rows if row["eligible"]]
        # Ties break toward the earlier cell in the configured grid order.
        chosen = (
            max(
                eligible_rows,
                key=lambda row: row["validation_ll_per_transition_mean"],
            )
            if eligible_rows
            else None
        )
        rank_rows.append(
            {
                "k": rank,
                "chosen_theta_cell": None if chosen is None else chosen["theta_cell"],
                "chosen_theta": None if chosen is None else chosen["theta"],
                "cells": cell_rows,
            }
        )
        for index, session in enumerate(sessions):
            if chosen is None:
                nested_records.append(
                    {
                        "k": rank,
                        "validation_session_id": session,
                        "status": "scientific_failure",
                    }
                )
            else:
                nested_records.append(
                    {
                        "k": rank,
                        "validation_session_id": session,
                        "status": "success",
                        "validation_ll_per_transition": chosen[
                            "validation_ll_per_transition"
                        ][index],
                    }
                )
    return {
        "status": "pending" if pending else "complete",
        "rank_rows": rank_rows,
        # Pending selections must not be scored, so no records are offered.
        "nested_records": [] if pending else nested_records,
    }
