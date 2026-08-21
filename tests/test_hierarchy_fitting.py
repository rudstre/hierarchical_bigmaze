import numpy as np
import pytest
import torch

from andrew_mlmdp import (
    Environment,
    Maze,
    Parameters,
    SubgoalBasis,
    Trial,
    fit_parameters,
    total_log_likelihood,
)
from andrew_mlmdp.hierarchy.fitting import (
    DOMAIN_EPS,
    _core_threshold_interior_upper,
    _inverse_transform,
    _physical_transform,
)


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
    parameters = Parameters(
        interior_reward=-0.15,
        goal_reward=0.35,
        lower_control_cost=0.55,
        upper_control_cost=0.9,
        alpha=0.8,
        beta=0.7,
        core_threshold=0.75,
        core_exponent=2.0,
    )
    return Environment(maze).hierarchy(basis, parameters=parameters)


def _trials():
    return (
        Trial(
            "s",
            1,
            (0, 5),
            ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
        ),
        Trial(
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

    result = fit_parameters(
        template,
        (trial for trial in _trials()),
        names=("alpha",),
        lr=0.05,
        max_steps=12,
        patience=50,
    )

    assert result.reason == "max_steps"
    assert not result.converged
    assert result.updates == 12
    assert len(result.history) == 13
    assert min(result.loss_history) < result.loss_history[0]
    assert result.best_values is not None
    best_total = total_log_likelihood(
        template,
        _trials(),
        parameter_values=result.best_values,
    )
    assert -best_total == pytest.approx(min(result.loss_history), abs=1e-12)

    for evaluation in result.history:
        values = {
            name: torch.tensor(value, dtype=torch.float64)
            for name, value in evaluation.parameter_values.items()
        }
        aligned_total = total_log_likelihood(
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

    original_alpha = result.best_values["alpha"]
    original_alpha.add_(100.0)
    assert result.best_values["alpha"] < 100.0


def test_progress_callback_receives_every_evaluated_state():
    evaluations = []
    result = _template().fit(
        _trials(),
        names=("alpha",),
        max_steps=2,
        patience=20,
        callback=evaluations.append,
    )

    assert tuple(evaluations) == result.history
    assert [evaluation.updates for evaluation in evaluations] == [0, 1, 2]


def test_fitting_rejects_diagnostic_winner_take_all_composition():
    base = _template()
    template = base.environment.hierarchy(
        base.basis,
        parameters=base.parameters,
        task_library=base.task_library,
        composition_mode="winner_take_all",
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        fit_parameters(
            template,
            _trials(),
            names=("alpha",),
            max_steps=1,
        )


def test_fitting_fixes_composition_exponent_at_one():
    base = _template()
    assert base.composition_exponent == 1.0

    manually_sharpened = base.environment.hierarchy(
        base.basis,
        parameters=base.parameters,
        task_library=base.task_library,
        composition_exponent=1.5,
    )
    with pytest.raises(ValueError, match="fixes composition_exponent at 1.0"):
        fit_parameters(
            manually_sharpened,
            _trials(),
            names=("alpha",),
            max_steps=1,
        )

    with pytest.raises(ValueError, match="Unknown parameter names"):
        fit_parameters(
            base,
            _trials(),
            names=("composition_exponent",),
            max_steps=1,
        )


def test_selected_gradients_are_finite_and_frozen_values_do_not_change():
    template = _template()
    result = fit_parameters(
        template,
        _trials(),
        names=("alpha", "core_threshold", "core_exponent"),
        max_steps=3,
        patience=20,
    )
    assert result.best_values is not None
    assert all(
        np.isfinite(gradient)
        for evaluation in result.history
        for gradient in evaluation.gradients.values()
    )
    initial = result.initial_values.as_floats()
    best = result.best_values.as_floats()
    for name in initial.keys() - set(result.names):
        assert best[name] == initial[name]


def test_zero_step_and_singleton_dataset_have_explicit_behavior():
    template = _template()
    zero_step = fit_parameters(
        template,
        _trials(),
        names=("alpha",),
        max_steps=0,
    )
    assert zero_step.updates == 0
    assert len(zero_step.history) == 1
    assert zero_step.reason == "max_steps"
    assert not zero_step.converged
    assert zero_step.best_values is not None
    assert zero_step.best_values.as_floats() == (
        zero_step.initial_values.as_floats()
    )

    singleton = Trial("s", 3, (0, 5), ((0, 1), (0, 1), (0, 1)))
    result = fit_parameters(
        template,
        (singleton,),
        names=("alpha",),
        max_steps=1,
    )
    assert result.loss_history == pytest.approx((0.0, 0.0))
    assert all(
        evaluation.gradients["alpha"] == pytest.approx(0.0)
        for evaluation in result.history
    )


def test_impossible_trajectory_terminates_on_nonfinite_loss():
    template = _template()
    impossible = Trial("s", 4, (0, 5), ((0, 0), (0, 4)))
    result = fit_parameters(
        template,
        (impossible,),
        names=("alpha",),
        max_steps=10,
    )
    assert result.reason == "nonfinite_loss"
    assert not result.converged
    assert result.updates == 0
    assert result.best_values is None
    assert np.isposinf(result.loss_history[0])
    assert np.isneginf(result.log_likelihood_history[0])


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
        fit_parameters(
            _template(),
            trials,
            names=names,
            max_steps=max_steps,
        )


def test_inactive_and_boundary_gate_parameters_cannot_be_fitted():
    maze = Maze.from_ascii("....")
    ungated = Environment(maze).hierarchy(
        SubgoalBasis.from_profiles(
            maze,
            np.asarray([[1.0], [0.7], [0.3], [0.0]]),
            core_threshold=None,
        )
    )
    trial = Trial("s", 1, (0, 3), ((0, 0), (0, 1), (0, 3)))
    with pytest.raises(ValueError, match="Inactive gate"):
        fit_parameters(
            ungated,
            (trial,),
            names=("core_exponent",),
        )

    with pytest.raises(ValueError, match="strictly inside"):
        fit_parameters(
            _template(threshold=0.0),
            _trials(),
            names=("core_threshold",),
            max_steps=0,
        )

    with pytest.raises(ValueError, match="threshold <"):
        fit_parameters(
            _template(threshold=0.85),
            _trials(),
            names=("core_threshold",),
            max_steps=0,
        )


def test_reduce_on_plateau_lr_is_recorded_after_aligned_update():
    singleton = Trial("s", 5, (0, 5), ((0, 1),))
    result = _template().fit(
        (singleton,),
        names=("alpha",),
        lr=0.05,
        max_steps=6,
        patience=2,
        lr_decay=0.3,
        lr_patience=1,
        min_lr=1e-5,
    )

    assert result.updates == 6
    assert [evaluation.updates for evaluation in result.history] == list(
        range(7)
    )
    assert result.lr_history == pytest.approx(
        (0.05, 0.05, 0.05, 0.015, 0.015, 0.015, 0.0045)
    )
    assert result.loss_history == pytest.approx((0.0,) * 7)

    minimum = _template().fit(
        (singleton,),
        names=("alpha",),
        lr=0.05,
        max_steps=10,
        patience=3,
        lr_decay=0.3,
        lr_patience=1,
        min_lr=0.015,
    )
    assert minimum.reason == "patience"
    assert minimum.converged
    assert minimum.updates == 6
    assert minimum.lr_history == pytest.approx(
        (0.05, 0.05, 0.05, 0.015, 0.015, 0.015, 0.015)
    )


def test_lr_reduction_restores_best_raw_state_and_resets_adam(monkeypatch):
    import andrew_mlmdp.hierarchy.fitting as fitting

    evaluations = 0

    def flat_value_with_reversing_gradient(template, prepared, *, parameter_values):
        nonlocal evaluations
        del template, prepared
        alpha = parameter_values["alpha"]
        direction = 1.0 if evaluations < 3 else -1.0
        evaluations += 1
        return direction * (alpha - alpha.detach())

    monkeypatch.setattr(
        fitting,
        "total_prepared_log_likelihood",
        flat_value_with_reversing_gradient,
    )
    singleton = Trial("s", 6, (0, 5), ((0, 1),))
    result = _template().fit(
        (singleton,),
        names=("alpha",),
        lr=0.05,
        max_steps=4,
        patience=20,
        lr_decay=0.3,
        lr_patience=1,
        min_lr=1e-5,
    )

    assert result.lr_history == pytest.approx(
        (0.05, 0.05, 0.05, 0.015, 0.015)
    )
    initial = result.history[0].parameter_values["alpha"]
    restarted = result.history[3].parameter_values["alpha"]
    assert restarted == pytest.approx(initial, abs=1e-15)

    restarted_raw = _inverse_transform(
        "alpha", torch.tensor(restarted, dtype=torch.float64)
    )
    after_fresh_step_raw = _inverse_transform(
        "alpha",
        torch.tensor(
            result.history[4].parameter_values["alpha"], dtype=torch.float64
        ),
    )
    # The gradient reverses on entry to the new stage. Fresh Adam moments make
    # its first step follow that new gradient immediately.
    assert after_fresh_step_raw < restarted_raw
    assert restarted_raw - after_fresh_step_raw == pytest.approx(0.015, abs=1e-8)


def test_learning_rate_schedule_arguments_are_validated():
    common = {
        "template": _template(),
        "trials": _trials(),
        "names": ("alpha",),
        "max_steps": 0,
    }
    with pytest.raises(ValueError, match="lr_decay"):
        fit_parameters(**common, lr_decay=1.0)
    with pytest.raises(ValueError, match="lr_patience"):
        fit_parameters(**common, lr_patience=-1)
    with pytest.raises(ValueError, match="min_lr"):
        fit_parameters(
            **common, lr=0.01, min_lr=0.02
        )
    with pytest.raises(ValueError, match="scheduler_tolerance"):
        fit_parameters(
            **common, scheduler_tolerance=-1.0
        )
    with pytest.raises(ValueError, match="convergence_tolerance"):
        fit_parameters(
            **common, convergence_tolerance=np.inf
        )


def test_scheduler_and_convergence_thresholds_are_independent(monkeypatch):
    import andrew_mlmdp.hierarchy.fitting as fitting

    scheduler_thresholds = []
    original_scheduler = fitting._plateau_scheduler

    def recording_scheduler(optimizer, **kwargs):
        scheduler_thresholds.append(kwargs["relative_threshold"])
        return original_scheduler(optimizer, **kwargs)

    monkeypatch.setattr(fitting, "_plateau_scheduler", recording_scheduler)
    singleton = Trial("s", 7, (0, 5), ((0, 1),))
    result = _template().fit(
        (singleton,),
        names=("alpha",),
        lr=0.05,
        max_steps=4,
        tolerance=1e-8,
        scheduler_tolerance=1e-5,
        convergence_tolerance=2e-5,
        lr_patience=1,
        min_lr=1e-5,
    )

    assert result.reason == "max_steps"
    assert scheduler_thresholds == pytest.approx((1e-5, 1e-5))


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


def test_structural_threshold_transform_stays_one_ulp_inside_goal_domain():
    template = _template()
    goals = tuple(dict.fromkeys(trial.goal for trial in _trials()))
    domain = template.threshold_range(goals)
    reference = torch.tensor(0.0, dtype=torch.float64)
    interior_upper = _core_threshold_interior_upper(
        domain.maximum,
        reference=reference,
    )

    assert domain.maximum == pytest.approx(0.85)
    assert interior_upper == torch.nextafter(
        torch.tensor(domain.maximum, dtype=torch.float64),
        torch.tensor(-torch.inf, dtype=torch.float64),
    )
    assert interior_upper < domain.maximum

    profiles = template.basis.profiles
    for raw_value in (-1000.0, -10.0, 0.0, 10.0, 1000.0):
        threshold = _physical_transform(
            "core_threshold",
            torch.tensor(raw_value, dtype=torch.float64),
            threshold_max=domain.maximum,
        )
        assert 0.0 < threshold <= interior_upper < domain.maximum
        for goal in goals:
            goal_state = template.maze.state_index(goal)
            keep = np.arange(len(profiles)) != goal_state
            assert np.all(np.max(profiles[keep], axis=0) > float(threshold))

    for invalid in (float(interior_upper), domain.maximum, 0.9):
        with pytest.raises(ValueError, match="strictly inside"):
            _inverse_transform(
                "core_threshold",
                torch.tensor(invalid, dtype=torch.float64),
                threshold_max=domain.maximum,
            )
