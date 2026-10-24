import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy import simulate_trials
from andrew_mlmdp.hierarchy.likelihood import total_prepared_log_likelihood
from andrew_mlmdp.hierarchy.profile_fitting import access_gate, profile_feasibility
from andrew_mlmdp.hierarchy.subgoal_search import (
    MIN_WIDTH,
    CentreSearch,
    bump_profiles,
    maze_distances,
)

GRID = "\n".join(["....."] * 5)
GOALS = ((0, 0), (0, 4), (4, 0), (4, 4))
PLANTED_CENTRE = (2, 2)
PLANTED_WIDTH = 1.0
PLANTED = {
    "interior_reward": -1.0,
    "goal_reward": 0.0,
    "lower_control_cost": 0.5,
    "upper_control_cost": 1.0,
    "alpha": 1.0,
    "beta": 3.0,
}
DEFAULT_START = {**PLANTED, "lower_control_cost": 1.0, "beta": 1.0}
THETA = ("lower_control_cost", "upper_control_cost", "beta")


@pytest.fixture(scope="module")
def setting():
    maze = Maze.from_ascii(GRID)
    environment = Environment(maze)
    distances = torch.tensor(maze_distances(maze), dtype=torch.float64)
    centre = maze.state_index(PLANTED_CENTRE)
    profiles = access_gate(
        bump_profiles(
            distances, [centre], torch.tensor([PLANTED_WIDTH]), "exponential"
        ),
        0.05,
    ).numpy()
    basis = SubgoalBasis.from_profiles(maze, profiles, core_threshold=None)
    planted = environment.hierarchy(basis, parameters=Parameters(**PLANTED))
    rng = np.random.default_rng(0)
    design = []
    for index in range(240):
        goal = GOALS[index % 4]
        candidates = [c for c, d in maze.distances_from(goal).items() if d >= 3]
        design.append(("s", candidates[rng.integers(len(candidates))], goal))
    trials = simulate_trials(planted, design, seed=1).trials

    def search(**overrides):
        options = dict(
            structure={},
            fixed_parameters=DEFAULT_START,
            theta_names=THETA,
            fit_steps=60,
            screen_top=3,
        )
        options.update(overrides)
        return CentreSearch(environment, trials, **options)

    return maze, centre, trials, search


def test_bump_profiles_peak_at_centres_and_decay_with_distance():
    maze = Maze.from_ascii(GRID)
    distances = torch.tensor(maze_distances(maze), dtype=torch.float64)
    centres = [maze.state_index((0, 0)), maze.state_index((2, 2))]
    widths = torch.tensor([1.0, 2.0], dtype=torch.float64)
    for shape in ("exponential", "gaussian"):
        profiles = bump_profiles(distances, centres, widths, shape)
        peaks = profiles[centres, [0, 1]]
        assert torch.allclose(peaks, torch.ones(2, dtype=torch.float64))
        for column, centre in enumerate(centres):
            order = torch.argsort(distances[centre])
            assert bool((torch.diff(profiles[order, column]) <= 1e-15).all())
    exponential = bump_profiles(distances, centres, widths, "exponential")
    gaussian = bump_profiles(distances, centres, widths, "gaussian")
    # exp(-d/w) and exp(-(d/w)^2/2) cross at d = 2w; beyond it the gaussian
    # tail is lighter (column 0 has w = 1, so d >= 3).
    far = distances[:, centres[0]] > 2 * widths[0]
    assert bool(far.any())
    assert bool((gaussian[far, 0] < exponential[far, 0]).all())


def test_minimum_width_keeps_goal_centred_subgoal_feasible():
    maze = Maze.from_ascii(GRID)
    distances = torch.tensor(maze_distances(maze), dtype=torch.float64)
    for shape in ("exponential", "gaussian"):
        gated = access_gate(
            bump_profiles(
                distances,
                [maze.state_index(GOALS[0])],
                torch.tensor([MIN_WIDTH], dtype=torch.float64),
                shape,
            ),
            0.05,
        )
        assert profile_feasibility(gated, maze, GOALS) == ()


def test_width_gradient_matches_finite_differences(setting):
    maze, centre, _, search = setting
    problem = search()
    values = problem._values(DEFAULT_START)

    def log_likelihood(width):
        return total_prepared_log_likelihood(
            problem._template(1),
            problem.prepared,
            parameter_values=values,
            access_profiles=problem.profiles([centre], width),
        )

    width = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    log_likelihood(width).backward()
    step = 1e-6
    with torch.no_grad():
        numeric = (
            float(log_likelihood(width.detach() + step))
            - float(log_likelihood(width.detach() - step))
        ) / (2 * step)
    assert width.grad.item() == pytest.approx(numeric, rel=1e-5, abs=1e-8)


def test_fit_with_true_centre_recovers_width_and_theta(setting):
    _, centre, _, search = setting
    problem = search(fit_steps=200)
    start = problem.log_likelihood([centre], [2.0], DEFAULT_START)
    fit = problem.fit([centre], [2.0], DEFAULT_START)
    assert fit.train_ll_per_decision > start
    assert 0.5 <= fit.widths[0] / PLANTED_WIDTH <= 2.0
    assert 0.5 <= fit.parameters["beta"] / PLANTED["beta"] <= 2.0
    assert fit.parameters["alpha"] == PLANTED["alpha"]  # not fitted


def test_greedy_search_finds_planted_centre(setting):
    maze, centre, _, search = setting
    (result,) = search().greedy(1)
    distances = maze.distances_from(PLANTED_CENTRE)
    found = maze.coordinate(result.fit.centres[0])
    assert distances[found] <= 1
    assert result.moves[0].kind == "add"
    assert len(result.runner_ups) == 1 and len(result.runner_ups[0]) == 3
    gaps = [gap for _, gap in result.runner_ups[0]]
    assert gaps == sorted(gaps, reverse=True)


def test_swap_moves_a_wrong_centre_to_the_planted_one(setting):
    maze, centre, _, search = setting
    problem = search()
    far = maze.state_index((0, 1))
    fit = problem.fit([far], [1.0], DEFAULT_START)
    moved, moves = problem.swap(fit)
    assert maze.distances_from(PLANTED_CENTRE)[maze.coordinate(moved.centres[0])] <= 1
    assert moves and moves[0].from_centre == far
    assert moved.train_ll_per_decision > fit.train_ll_per_decision


def test_random_start_is_deterministic(setting):
    _, _, _, search = setting
    first = search(fit_steps=20).from_random_centres(1, np.random.default_rng(5))
    second = search(fit_steps=20).from_random_centres(1, np.random.default_rng(5))
    assert first.fit == second.fit
    assert first.moves == second.moves


def test_seeded_centres_are_deterministic_distinct_and_spread(setting):
    maze, _, _, search = setting
    problem = search(fit_steps=5)
    weights = np.ones(problem.n_states)
    first = problem.seeded_centres(4, np.random.default_rng(3), weights)
    second = problem.seeded_centres(4, np.random.default_rng(3), weights)
    assert first == second
    assert len(set(first)) == 4
    # A zero-weight state is never chosen.
    weights[maze.state_index(PLANTED_CENTRE)] = 0.0
    for seed in range(20):
        chosen = problem.seeded_centres(3, np.random.default_rng(seed), weights)
        assert maze.state_index(PLANTED_CENTRE) not in chosen


def test_local_swap_only_moves_within_radius(setting):
    maze, centre, _, search = setting
    problem = search(theta_names=())
    near = maze.state_index((1, 2))  # one step from the planted centre
    fit = problem.fit([near], [1.0], PLANTED)
    moved, moves = problem.swap(fit, radius=1)
    assert moved.centres == (centre,)
    for move in moves:
        assert problem.distances[move.from_centre, move.to_centre] <= 1


def test_search_from_far_centre_reaches_planted_one(setting):
    maze, _, _, search = setting
    problem = search(theta_names=())
    fit = problem.fit([maze.state_index((0, 1))], [1.0], PLANTED)
    found, _ = problem.search(fit)
    assert maze.distances_from(PLANTED_CENTRE)[maze.coordinate(found.centres[0])] <= 1


def test_fit_with_fixed_theta_changes_only_widths(setting):
    _, centre, _, search = setting
    problem = search(theta_names=())
    fit = problem.fit([centre], [2.0], PLANTED)
    assert dict(fit.parameters) == pytest.approx(PLANTED)
    assert fit.widths[0] != pytest.approx(2.0)
