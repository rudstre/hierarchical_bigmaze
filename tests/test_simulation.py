import math
from collections import Counter

import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy import simulate_trials
from andrew_mlmdp.hierarchy.equations import parameter_values
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)

# K = 2 with exact zeros, already peak-normalized.
PROFILES = np.asarray(
    [
        [1.0, 0.0],
        [0.6, 0.0],
        [0.2, 0.0],
        [0.0, 0.3],
        [0.0, 1.0],
        [0.0, 0.4],
    ]
)
START = (0, 2)
GOAL = (0, 5)


def _parameters():
    return Parameters(
        interior_reward=-0.2,
        goal_reward=0.3,
        lower_control_cost=0.6,
        upper_control_cost=0.8,
        alpha=1.2,
        beta=0.9,
    )


def _template(profiles=PROFILES):
    maze = Maze.from_ascii("......")
    basis = SubgoalBasis.from_profiles(maze, profiles, core_threshold=None)
    return Environment(maze).hierarchy(basis, parameters=_parameters())


def _neighbours(maze, coordinate):
    outcomes = {
        maze.command_outcome(coordinate, command)
        for command in ("north", "south", "east", "west")
    }
    outcomes.discard(coordinate)
    return sorted(outcomes)


def _collapsed(trajectory):
    collapsed = [trajectory[0]]
    for coordinate in trajectory[1:]:
        if coordinate != collapsed[-1]:
            collapsed.append(coordinate)
    return tuple(collapsed)


def test_two_departure_prefix_frequencies_match_likelihood():
    template = _template()
    task = template.task(GOAL)
    n_trials = 4000
    simulated = simulate_trials(
        template,
        [("s", START, GOAL)] * n_trials,
        seed=11,
    )
    assert simulated.censored == ()
    counts = Counter(_collapsed(trial.trajectory)[:3] for trial in simulated.trials)

    maze = template.maze
    prefixes = [
        (START, first, second)
        for first in _neighbours(maze, START)
        for second in _neighbours(maze, first)
    ]
    exact = {
        prefix: math.exp(task.log_likelihood(list(prefix))) for prefix in prefixes
    }
    assert sum(exact.values()) == pytest.approx(1.0, abs=1e-10)
    assert set(counts) <= set(prefixes)
    for prefix in counts:
        assert exact[prefix] > 0.0
    for prefix, probability in exact.items():
        empirical = counts[prefix] / n_trials
        standard_error = math.sqrt(probability * (1.0 - probability) / n_trials)
        assert abs(empirical - probability) <= 4.0 * standard_error + 1e-3


def test_basis_likelihood_matches_fitting_override_likelihood():
    planted = _template()
    simulated = simulate_trials(planted, [("s", START, GOAL)] * 20, seed=3)
    other = _template(
        np.asarray(
            [
                [0.2, 1.0],
                [1.0, 0.5],
                [0.4, 0.5],
                [0.3, 0.2],
                [0.1, 0.9],
                [0.5, 0.1],
            ]
        )
    )
    task = planted.task(GOAL)
    expected = sum(
        task.log_likelihood(list(trial.trajectory)) for trial in simulated.trials
    )
    override = total_prepared_log_likelihood(
        other,
        prepare_batch(other, simulated.trials),
        parameter_values=parameter_values(other),
        access_profiles=torch.tensor(planted.basis.profiles, dtype=torch.float64),
    )
    assert float(override.detach()) == pytest.approx(expected, abs=1e-10)


def test_simulation_is_deterministic_in_seed():
    template = _template()
    design = [("s", START, GOAL)] * 30
    first = simulate_trials(template, design, seed=5)
    again = simulate_trials(template, design, seed=5)
    other = simulate_trials(template, design, seed=6)
    assert first == again
    assert [trial.trajectory for trial in first.trials] != [
        trial.trajectory for trial in other.trials
    ]


def test_design_bookkeeping_is_preserved():
    template = _template()
    design = [
        ("a", (0, 2), (0, 5)),
        ("b", (0, 1), (0, 0)),
        ("a", (0, 4), (0, 0)),
        ("b", (0, 3), (0, 5)),
    ]
    simulated = simulate_trials(template, design, seed=0)
    assert len(simulated.trials) + len(simulated.censored) == len(design)
    records = sorted(
        [(trial.session_id, trial.trial_id, trial.goal, trial.trajectory[0])
         for trial in simulated.trials]
        + [(record.session_id, record.trial_id, record.goal, record.start)
           for record in simulated.censored]
    )
    assert records == [
        ("a", 0, (0, 5), (0, 2)),
        ("a", 1, (0, 0), (0, 4)),
        ("b", 0, (0, 0), (0, 1)),
        ("b", 1, (0, 5), (0, 3)),
    ]
    for trial in simulated.trials:
        assert trial.trajectory[-1] == trial.goal


def test_step_limit_rollouts_are_censored_not_dropped():
    template = _template()
    simulated = simulate_trials(
        template,
        [("s", (0, 0), GOAL)] * 5,
        seed=2,
        max_steps=1,
    )
    assert simulated.trials == ()
    assert len(simulated.censored) == 5
    for record in simulated.censored:
        assert record.status == "step_limit"
        assert record.trajectory[0] == (0, 0)
        assert len(record.trajectory) == 2


def test_invalid_designs_raise():
    template = _template()
    with pytest.raises(ValueError, match="at least one"):
        simulate_trials(template, [], seed=0)
    with pytest.raises(ValueError, match="equals its goal"):
        simulate_trials(template, [("s", GOAL, GOAL)], seed=0)
