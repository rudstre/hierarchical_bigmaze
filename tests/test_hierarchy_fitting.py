import numpy as np
import pytest
import torch

from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    ModelParameters,
    MovementTrial,
    SubgoalBasis,
    fit_hierarchical_model_parameters,
    total_hierarchical_movement_log_likelihood_torch,
)
from andrew_mlmdp.hierarchy.fitting import DOMAIN_EPS, _physical_transform


def _template(*, threshold=0.2):
    maze = Maze.from_ascii("......")
    basis = SubgoalBasis.from_profiles(
        maze,
        np.asarray(
            [
                [1.0, 0.0],
                [0.85, 0.10],
                [0.60, 0.35],
                [0.35, 0.65],
                [0.10, 0.85],
                [0.0, 1.0],
            ]
        ),
        core_threshold=threshold,
        core_exponent=0.7,
    )
    parameters = ModelParameters(
        interior_reward=-0.15,
        goal_reward=0.35,
        lower_control_cost=0.55,
        upper_control_cost=0.9,
        alpha=0.8,
        off_target_reward=-0.3,
        beta=0.7,
        core_threshold=0.75,
        core_exponent=2.0,
    )
    return LMDPEnvironment(maze).hierarchy(basis, parameters=parameters)


def _trials():
    return (
        MovementTrial(
            "s",
            1,
            (0, 5),
            ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
        ),
        MovementTrial(
            "s",
            2,
            (0, 5),
            ((0, 2), (0, 3), (0, 4), (0, 5)),
        ),
    )


def _registered_values(template):
    return {
        name: parameter.detach().clone()
        for name, parameter in template.parameters.named_parameters()
    }


def test_fitting_improves_likelihood_restores_aligned_best_and_mutates_nothing():
    template = _template()
    before = _registered_values(template)
    assert template._task_cache == {}
    assert template._passive_dynamics is None

    result = fit_hierarchical_model_parameters(
        template,
        (trial for trial in _trials()),
        parameter_names=("alpha",),
        learning_rate=0.05,
        max_steps=12,
        patience=50,
    )

    assert result.termination_reason == "max_steps"
    assert not result.converged
    assert result.updates_completed == 12
    assert len(result.history) == 13
    assert min(result.loss_history) < result.loss_history[0]
    assert result.best_parameter_values is not None
    best_total = total_hierarchical_movement_log_likelihood_torch(
        template,
        _trials(),
        parameter_values=result.best_parameter_values,
    )
    assert -best_total == pytest.approx(min(result.loss_history), abs=1e-12)

    for evaluation in result.history:
        values = {
            name: torch.tensor(value, dtype=torch.float64)
            for name, value in evaluation.parameter_values.items()
        }
        aligned_total = total_hierarchical_movement_log_likelihood_torch(
            template,
            _trials(),
            parameter_values=values,
        )
        assert -aligned_total == pytest.approx(evaluation.loss, abs=1e-12)

    after = _registered_values(template)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert template._task_cache == {}
    assert template._passive_dynamics is None

    original_alpha = result.best_parameter_values["alpha"]
    original_alpha.add_(100.0)
    assert result.best_parameter_values["alpha"] < 100.0


def test_progress_callback_receives_every_evaluated_state():
    evaluations = []
    result = _template().fit_parameters(
        _trials(),
        parameter_names=("alpha",),
        max_steps=2,
        patience=20,
        progress_callback=evaluations.append,
    )

    assert tuple(evaluations) == result.history
    assert [evaluation.updates_completed for evaluation in evaluations] == [0, 1, 2]


def test_selected_gradients_are_finite_and_frozen_values_do_not_change():
    template = _template()
    result = fit_hierarchical_model_parameters(
        template,
        _trials(),
        parameter_names=("alpha", "core_threshold", "core_exponent"),
        max_steps=3,
        patience=20,
    )
    assert result.best_parameter_values is not None
    assert all(
        np.isfinite(gradient)
        for evaluation in result.history
        for gradient in evaluation.gradients.values()
    )
    initial = result.initial_parameter_values.as_floats()
    best = result.best_parameter_values.as_floats()
    for name in initial.keys() - set(result.parameter_names):
        assert best[name] == initial[name]


def test_zero_step_and_singleton_dataset_have_explicit_behavior():
    template = _template()
    zero_step = fit_hierarchical_model_parameters(
        template,
        _trials(),
        parameter_names=("alpha",),
        max_steps=0,
    )
    assert zero_step.updates_completed == 0
    assert len(zero_step.history) == 1
    assert zero_step.termination_reason == "max_steps"
    assert not zero_step.converged
    assert zero_step.best_parameter_values is not None
    assert zero_step.best_parameter_values.as_floats() == (
        zero_step.initial_parameter_values.as_floats()
    )

    singleton = MovementTrial("s", 3, (0, 5), ((0, 1), (0, 1), (0, 1)))
    result = fit_hierarchical_model_parameters(
        template,
        (singleton,),
        parameter_names=("alpha",),
        max_steps=1,
    )
    assert result.loss_history == pytest.approx((0.0, 0.0))
    assert all(
        evaluation.gradients["alpha"] == pytest.approx(0.0)
        for evaluation in result.history
    )


def test_impossible_trajectory_terminates_on_nonfinite_loss():
    template = _template()
    impossible = MovementTrial("s", 4, (0, 5), ((0, 0), (0, 4)))
    result = fit_hierarchical_model_parameters(
        template,
        (impossible,),
        parameter_names=("alpha",),
        max_steps=10,
    )
    assert result.termination_reason == "nonfinite_loss"
    assert not result.converged
    assert result.updates_completed == 0
    assert result.best_parameter_values is None
    assert np.isposinf(result.loss_history[0])
    assert np.isneginf(result.total_log_likelihood_history[0])


@pytest.mark.parametrize(
    "trials,names,max_steps,match",
    [
        ((), ("alpha",), 1, "at least one trial"),
        (_trials(), ("alpha", "alpha"), 1, "duplicates"),
        (_trials(), ("unknown",), 1, "Unknown"),
        (_trials(), ("alpha",), -1, "non-negative"),
    ],
)
def test_fitting_validates_edge_cases(trials, names, max_steps, match):
    with pytest.raises(ValueError, match=match):
        fit_hierarchical_model_parameters(
            _template(),
            trials,
            parameter_names=names,
            max_steps=max_steps,
        )


def test_inactive_and_boundary_gate_parameters_cannot_be_fitted():
    maze = Maze.from_ascii("....")
    ungated = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_profiles(
            maze,
            np.asarray([[1.0], [0.7], [0.3], [0.0]]),
            core_threshold=None,
        )
    )
    trial = MovementTrial("s", 1, (0, 3), ((0, 0), (0, 1), (0, 3)))
    with pytest.raises(ValueError, match="Inactive gate"):
        fit_hierarchical_model_parameters(
            ungated,
            (trial,),
            parameter_names=("core_exponent",),
        )

    with pytest.raises(ValueError, match="strictly inside"):
        fit_hierarchical_model_parameters(
            _template(threshold=0.0),
            _trials(),
            parameter_names=("core_threshold",),
            max_steps=0,
        )


def test_strict_domain_transforms_retain_float64_margins():
    raw = torch.tensor(-1000.0, dtype=torch.float64)
    positive = _physical_transform("alpha", raw)
    negative = _physical_transform("interior_reward", raw)
    threshold = _physical_transform("core_threshold", raw)
    assert positive >= DOMAIN_EPS
    assert positive > 0.0
    assert negative <= -DOMAIN_EPS
    assert negative < 0.0
    assert 0.0 < threshold < 1.0
