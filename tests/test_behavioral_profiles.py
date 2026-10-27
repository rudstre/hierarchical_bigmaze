import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy import simulate_trials
from andrew_mlmdp.hierarchy.behavioral_profiles import (
    BehavioralFitSettings,
    cell_template,
    coarse_theta_grid,
    fit_cell,
    held_parameters,
    local_theta_grid,
    select_theta_per_rank,
    theta_key,
    training_occupancy,
)
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.profile_fitting import _collapsed_move_count, access_gate
from andrew_mlmdp.hierarchy.subgoal_search import (
    CentreSearch,
    bump_profiles,
    maze_distances,
)
from andrew_mlmdp.nested_validation import nested_rank_selection

ROOT = Path(__file__).resolve().parents[1]
GRID = "\n".join(["....."] * 5)
GOALS = ((0, 0), (0, 4), (4, 0), (4, 4))
PLANTED = {
    "interior_reward": -1.0,
    "goal_reward": 0.0,
    "lower_control_cost": 0.5,
    "upper_control_cost": 1.0,
    "alpha": 1.0,
    "beta": 3.0,
}
# Tiny but complete procedure: one greedy and one seeded start, short fits.
TINY = BehavioralFitSettings(
    commitment_radius=1,
    search_fit_steps=10,
    search_fit_patience=3,
    screen_top=1,
    n_seeded_starts=1,
    max_steps=15,
    tolerance=1e-5,
)


def _make_theta_cells():
    spec = importlib.util.spec_from_file_location(
        "make_theta_cells", ROOT / "scripts" / "make_theta_cells.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def setting():
    maze = Maze.from_ascii(GRID)
    environment = Environment(maze)
    distances = torch.tensor(maze_distances(maze), dtype=torch.float64)
    centre = maze.state_index((2, 2))
    profiles = access_gate(
        bump_profiles(distances, [centre], torch.tensor([1.0]), "gaussian"), 0.05
    ).numpy()
    basis = SubgoalBasis.from_profiles(maze, profiles, core_threshold=None)
    planted = environment.hierarchy(
        basis, parameters=Parameters(**PLANTED), **TINY.structure
    )
    rng = np.random.default_rng(0)
    design = []
    for index in range(80):
        goal = GOALS[index % 4]
        candidates = [c for c, d in maze.distances_from(goal).items() if d >= 3]
        design.append(("s", candidates[rng.integers(len(candidates))], goal))
    trials = simulate_trials(planted, design, seed=1).trials
    return environment, trials


def test_coarse_grid_matches_experimental_cells():
    cells = _make_theta_cells()
    expected = tuple(itertools.product(cells.LOWER, cells.UPPER, cells.BETA))
    assert coarse_theta_grid(BehavioralFitSettings()) == expected
    assert len(expected) == 36


def test_default_starts_and_goal_collapse_treatment():
    # Chosen from the simulated theta grid (see BehavioralFitSettings).
    settings = BehavioralFitSettings()
    assert settings.start_names == ("greedy", "seed0", "seed1")
    assert settings.infeasible_step == "project"
    assert settings.as_dict()["projection_margin"] == 0.1


def test_local_grid_matches_experimental_cells_and_excludes_centre():
    cells = _make_theta_cells()
    centre = (0.6, 1.2, 2.0)
    local = local_theta_grid(centre, cells.STEP)
    expected = []
    for f1, f2, fb in itertools.product((-1, 0, 1), repeat=3):
        if (f1, f2, fb) == (0, 0, 0):
            continue
        theta = (0.6 * cells.STEP**f1, 1.2 * cells.STEP**f2, 2.0 * cells.STEP**fb)
        expected.append(tuple(float(f"{v:.6g}") for v in theta))
    assert local == tuple(expected)
    assert len(local) == 26 and centre not in local
    assert len({theta_key(theta) for theta in local}) == 26


def _cell_record(k, session, cell, value, status="success"):
    record = {
        "k": k,
        "validation_session_id": session,
        "theta_cell": cell,
        "theta": [float(v) for v in cell.split("_")[1::2]],
        "status": status,
    }
    if status == "success":
        record["validation_ll_per_transition"] = value
    return record


def test_theta_selection_chooses_best_mean_and_feeds_one_se_rule():
    a, b = theta_key((0.3, 0.6, 1.0)), theta_key((0.6, 0.6, 1.0))
    sessions = ("s1", "s2")
    records = [
        _cell_record(2, "s1", a, -0.50),
        _cell_record(2, "s2", a, -0.52),
        _cell_record(2, "s1", b, -0.40),
        _cell_record(2, "s2", b, -0.44),
        _cell_record(3, "s1", a, -0.39),
        _cell_record(3, "s2", a, -0.43),
        # b at K=3 failed for one session: ineligible, not dropped.
        _cell_record(3, "s1", b, -0.10),
        _cell_record(3, "s2", b, None, status="scientific_failure"),
    ]
    result = select_theta_per_rank(
        records,
        ranks=(2, 3),
        validation_session_ids=sessions,
        cells={2: (a, b), 3: (a, b)},
    )
    assert result["status"] == "complete"
    rows = {row["k"]: row for row in result["rank_rows"]}
    assert rows[2]["chosen_theta_cell"] == b
    assert rows[3]["chosen_theta_cell"] == a
    failed = next(c for c in rows[3]["cells"] if c["theta_cell"] == b)
    assert not failed["eligible"] and failed["scientific_failure_session_ids"] == ["s2"]
    nested = nested_rank_selection(
        result["nested_records"], ranks=(2, 3), validation_session_ids=sessions
    )
    # K=3 is best (-0.41) but K=2 (-0.42) is within one SE (0.02): choose 2.
    assert nested["selection"]["best_mean_k"] == 3
    assert nested["selection"]["selected_k"] == 2


def test_theta_selection_pending_when_a_cell_is_missing_or_operational():
    a = theta_key((0.3, 0.6, 1.0))
    records = [
        _cell_record(2, "s1", a, -0.5),
        _cell_record(2, "s2", a, None, status="operational_failure"),
    ]
    result = select_theta_per_rank(
        records, ranks=(2,), validation_session_ids=("s1", "s2"), cells={2: (a,)}
    )
    assert result["status"] == "pending"
    assert result["nested_records"] == []
    with pytest.raises(ValueError, match="Unexpected"):
        select_theta_per_rank(
            [_cell_record(4, "s1", a, -0.5)],
            ranks=(2,),
            validation_session_ids=("s1",),
            cells={2: (a,)},
        )


def test_rank_without_eligible_cell_is_ineligible_for_k_selection():
    a = theta_key((0.3, 0.6, 1.0))
    records = [
        _cell_record(2, "s1", a, -0.5),
        _cell_record(2, "s2", a, -0.5),
        _cell_record(3, "s1", a, None, status="scientific_failure"),
        _cell_record(3, "s2", a, -0.1),
    ]
    result = select_theta_per_rank(
        records,
        ranks=(2, 3),
        validation_session_ids=("s1", "s2"),
        cells={2: (a,), 3: (a,)},
    )
    nested = nested_rank_selection(
        result["nested_records"], ranks=(2, 3), validation_session_ids=("s1", "s2")
    )
    rows = {row["k"]: row for row in nested["rank_rows"]}
    assert not rows[3]["eligible"]
    assert nested["selection"]["selected_k"] == 2


@pytest.fixture(scope="module")
def tiny_cell(setting, tmp_path_factory):
    environment, trials = setting
    checkpoint = tmp_path_factory.mktemp("cell")
    cell = fit_cell(
        environment,
        trials,
        GOALS,
        2,
        (0.6, 1.2, 2.0),
        TINY,
        7,
        checkpoint_dir=checkpoint,
    )
    return cell, checkpoint


def test_fit_cell_selects_training_best_start_with_exact_zero_support(tiny_cell):
    cell, _ = tiny_cell
    assert cell.status == "success"
    assert set(cell.starts) == {"greedy", "seed0"}
    eligible = {n: r for n, r in cell.starts.items() if r["eligible"]}
    assert cell.selected_start == min(
        eligible, key=lambda n: eligible[n]["best_objective"]
    )
    assert cell.profiles.shape == (25, 2)
    assert np.allclose(cell.profiles.max(axis=0), 1.0)
    # The fixed gate gives exact zeros; the ungated warm-start D is positive.
    assert (cell.profiles == 0.0).any()
    assert (cell.ungated_profiles > 0.0).all()
    assert cell.parameters == held_parameters((0.6, 1.2, 2.0), TINY)


def test_fit_cell_resumes_from_checkpoints_identically(setting, tiny_cell):
    environment, trials = setting
    cell, checkpoint = tiny_cell
    messages = []
    resumed = fit_cell(
        environment,
        trials,
        GOALS,
        2,
        (0.6, 1.2, 2.0),
        TINY,
        7,
        checkpoint_dir=checkpoint,
        log=messages.append,
    )
    assert all("reused checkpoint" in message for message in messages)
    assert resumed.selected_start == cell.selected_start
    assert np.array_equal(resumed.profiles, cell.profiles)
    assert resumed.best_objective == cell.best_objective


def test_seeded_start_matches_centre_search_seeding(setting, tiny_cell):
    environment, trials = setting
    cell, _ = tiny_cell
    search = CentreSearch(
        environment,
        trials,
        structure=TINY.structure,
        fixed_parameters=held_parameters((0.6, 1.2, 2.0), TINY),
        theta_names=(),
        shape="gaussian",
        access_threshold=TINY.access_threshold,
        fit_steps=TINY.search_fit_steps,
        fit_patience=TINY.search_fit_patience,
        screen_top=TINY.screen_top,
    )
    child = np.random.SeedSequence(7).spawn(1)[0]
    reference = search.from_seeded_centres(
        2,
        np.random.default_rng(child),
        training_occupancy(environment, trials),
        radius=TINY.local_radius,
    )
    maze = environment.maze
    assert cell.starts["seed0"]["search_centres"] == [
        list(maze.coordinate(c)) for c in reference.fit.centres
    ]
    assert cell.starts["seed0"]["search_train_ll_per_decision"] == (
        reference.fit.train_ll_per_decision
    )


def test_cell_template_scores_exactly_what_the_fit_scored(setting, tiny_cell):
    environment, trials = setting
    cell, _ = tiny_cell
    template = cell_template(environment, cell, TINY)
    values = {
        name: torch.tensor(value, dtype=torch.float64)
        for name, value in cell.parameters.items()
    }
    fitted = total_prepared_log_likelihood(
        template,
        prepare_batch(template, trials),
        parameter_values=values,
        access_profiles=torch.tensor(cell.profiles, dtype=torch.float64),
    )
    from andrew_mlmdp.dataset import score_hierarchy_dataset

    strict = score_hierarchy_dataset(template, trials)
    assert strict.n_scored == len(trials) and not strict.exclusions
    assert strict.total_transitions == _collapsed_move_count(trials)
    assert strict.total_log_likelihood == pytest.approx(float(fitted), rel=1e-10)


def test_checkpoints_reject_changed_scientific_inputs(setting, tiny_cell):
    from dataclasses import replace

    environment, trials = setting
    _, directory = tiny_cell
    for theta, settings, seed, data in (
        ((0.3, 1.2, 2.0), TINY, 7, trials),
        ((0.6, 1.2, 2.0), replace(TINY, lr=0.05), 7, trials),
        ((0.6, 1.2, 2.0), TINY, 8, trials),
        ((0.6, 1.2, 2.0), TINY, 7, trials[:-1]),
    ):
        with pytest.raises(ValueError, match="Incompatible.*checkpoints"):
            fit_cell(
                environment,
                data,
                GOALS,
                2,
                theta,
                settings,
                seed,
                checkpoint_dir=directory,
            )


def test_refinement_matches_direct_experimental_warm_start(setting, tiny_cell):
    from andrew_mlmdp.hierarchy.behavioral_profiles import template_for_profiles
    from andrew_mlmdp.hierarchy.profile_fitting import fit_profiles

    environment, trials = setting
    cell, directory = tiny_cell
    with np.load(directory / "seed0.npz") as arrays:
        initial = arrays["initial"].copy()
        expected_profiles = arrays["profiles"].copy()
    # These are the fit_profiles arguments used by run_profile_recovery's
    # fixed-theta warm-start path, independently of fit_cell's start loop.
    template = template_for_profiles(
        environment, np.ones_like(initial), cell.parameters, TINY
    )
    reference = fit_profiles(
        template,
        trials,
        initial_profiles=initial,
        evaluation_goals=GOALS,
        penalty_weights=TINY.penalty_weights,
        theta_names=(),
        lr=TINY.lr,
        max_steps=TINY.max_steps,
        scheduler_tolerance=TINY.tolerance,
        convergence_tolerance=TINY.tolerance,
        parameterization="log",
        access_threshold=TINY.access_threshold,
    )
    assert reference.best_objective == cell.starts["seed0"]["best_objective"]
    assert np.array_equal(reference.best_profiles, expected_profiles)


def test_cell_predictions_factorize_the_strict_likelihood(setting, tiny_cell):
    environment, trials = setting
    cell, _ = tiny_cell
    template = cell_template(environment, cell, TINY)
    for trial in trials[:4]:
        task = template.task(trial.goal)
        predictions = task.movement_predictions(trial.trajectory)
        assert np.allclose(predictions.next_state_probabilities.sum(axis=1), 1.0)
        assert np.log(predictions.observed_probabilities).sum() == pytest.approx(
            task.log_likelihood(trial.trajectory), rel=1e-10
        )


def test_unusable_starts_are_recorded_and_do_not_abort_the_cell(setting, monkeypatch):
    from dataclasses import replace

    import andrew_mlmdp.hierarchy.behavioral_profiles as profiles

    def unusable(*args, **kwargs):
        raise ValueError("Infeasible initial profiles after the access gate")

    monkeypatch.setattr(profiles, "_search_start", unusable)
    environment, trials = setting
    cell = fit_cell(
        environment,
        trials,
        GOALS,
        2,
        (0.6, 1.2, 2.0),
        replace(TINY, n_seeded_starts=2),
        9,
    )
    assert cell.status == "scientific_failure" and cell.selected_start is None
    assert set(cell.starts) == {"greedy", "seed0", "seed1"}
    assert all(r["failure"]["type"] == "ValueError" for r in cell.starts.values())
