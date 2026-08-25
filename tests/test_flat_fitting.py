import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, Trial
from andrew_mlmdp.flat_autodiff import (
    parameter_values,
    prepare_batch,
    total_prepared_log_likelihood,
)


def _trials():
    return (
        Trial(
            "s",
            1,
            (0, 4),
            ((0, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        ),
        Trial(
            "s",
            2,
            (0, 0),
            ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0)),
        ),
    )


@pytest.mark.parametrize("cost", [0.3, 1.0, 3.0])
def test_torch_flat_likelihood_matches_numpy_oracle(cost):
    environment = Environment(Maze.from_ascii("....."))
    parameters = Parameters(lower_control_cost=cost)
    trials = _trials()
    values = parameter_values(parameters)
    prepared = prepare_batch(
        environment,
        trials,
        device=values["lower_control_cost"].device,
    )

    actual = total_prepared_log_likelihood(
        prepared,
        parameter_values=values,
    )
    expected = sum(
        environment.solve(
            trial.goal, parameters=parameters
        ).log_likelihood(trial.trajectory)
        for trial in trials
    )

    assert float(actual.detach()) == pytest.approx(expected, abs=1e-12)


def test_flat_control_cost_gradient_matches_central_difference():
    environment = Environment(Maze.from_ascii("....."))
    parameters = Parameters(lower_control_cost=1.4)
    initial = parameter_values(parameters)
    prepared = prepare_batch(
        environment,
        _trials(),
        device=initial["lower_control_cost"].device,
    )
    cost = initial["lower_control_cost"].detach().clone().requires_grad_()
    values = dict(initial)
    values["lower_control_cost"] = cost
    likelihood = total_prepared_log_likelihood(
        prepared,
        parameter_values=values,
    )
    likelihood.backward()

    step = 1e-5

    def evaluate(value):
        finite_values = dict(initial)
        finite_values["lower_control_cost"] = torch.tensor(
            value, dtype=torch.float64
        )
        return float(
            total_prepared_log_likelihood(
                prepared,
                parameter_values=finite_values,
            ).detach()
        )

    finite_difference = (evaluate(1.4 + step) - evaluate(1.4 - step)) / (
        2.0 * step
    )
    assert float(cost.grad) == pytest.approx(finite_difference, rel=1e-8)


def test_environment_fit_improves_likelihood_and_mutates_nothing():
    environment = Environment(Maze.from_ascii("....."))
    parameters = Parameters(lower_control_cost=3.0)
    before_parameters = {
        name: value.detach().clone()
        for name, value in parameters.named_parameters()
    }
    before_passive = environment.passive.copy()
    evaluations = []

    result = environment.fit(
        _trials(),
        parameters=parameters,
        max_steps=20,
        patience=50,
        callback=evaluations.append,
    )

    assert result.names == ("lower_control_cost",)
    assert set(result.initial_values) == {
        "interior_reward",
        "goal_reward",
        "lower_control_cost",
    }
    assert result.best_values is not None
    assert min(result.loss_history) < result.loss_history[0]
    assert tuple(evaluations) == result.history
    assert len(result.history) == result.updates + 1
    assert np.array_equal(environment.passive, before_passive)
    assert all(
        torch.equal(value, before_parameters[name])
        for name, value in parameters.named_parameters()
    )


def test_zero_step_and_zero_movement_have_explicit_behavior():
    environment = Environment(Maze.from_ascii("..."))
    trial = Trial("s", 1, (0, 2), ((0, 0), (0, 0)))

    result = environment.fit((trial,), max_steps=0)

    assert result.reason == "max_steps"
    assert result.updates == 0
    assert result.loss_history == pytest.approx((0.0,))
    assert result.gradient_norm_history == pytest.approx((0.0,))
    assert result.best_values is not None


@pytest.mark.parametrize(
    "trial",
    [
        Trial("s", 2, (0, 2), ((0, 2), (0, 1))),
        Trial("s", 3, (0, 2), ((0, 0), (0, 2))),
    ],
)
def test_impossible_flat_trajectory_terminates_on_nonfinite_loss(trial):
    result = Environment(Maze.from_ascii("...")).fit((trial,), max_steps=5)

    assert result.reason == "nonfinite_loss"
    assert result.updates == 0
    assert result.best_values is None
    assert np.isposinf(result.loss_history[0])


def test_flat_fitting_validates_dataset_and_optimizer_controls():
    environment = Environment(Maze.from_ascii("..."))
    trial = Trial("s", 1, (0, 2), ((0, 0), (0, 1), (0, 2)))

    with pytest.raises(ValueError, match="at least one trial"):
        environment.fit(())
    with pytest.raises(ValueError, match="non-negative"):
        environment.fit((trial,), max_steps=-1)
    with pytest.raises(ValueError, match="min_lr"):
        environment.fit((trial,), lr=0.01, min_lr=0.02)
