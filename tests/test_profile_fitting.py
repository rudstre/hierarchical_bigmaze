import copy

import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis, Trial
from andrew_mlmdp.hierarchy.equations import parameter_values
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.profile_fitting import (
    access_gate,
    best_restart,
    fit_profile_restarts,
    fit_profiles,
    fitted_basis,
    log_peak_gauge,
    peak_gauge,
    profile_feasibility,
    profile_geometry,
    random_localized_profiles,
    structural_penalties,
)

PROFILES = np.asarray(
    [
        [1.0, 0.0],
        [0.85, 0.10],
        [0.60, 0.35],
        [0.35, 0.65],
        [0.10, 0.85],
        [0.0, 1.0],
    ]
)
GOAL = (0, 5)
PENALTIES = {"compactness": 0.05, "dirichlet": 0.05, "overlap": 0.05}
TOLERANCES = {"scheduler_tolerance": 1e-6, "convergence_tolerance": 1e-6}


def _parameters():
    return Parameters(
        interior_reward=-0.15,
        goal_reward=0.35,
        lower_control_cost=0.55,
        upper_control_cost=0.9,
        alpha=0.8,
        beta=0.7,
    )


def _template(threshold=None):
    maze = Maze.from_ascii("......")
    basis = SubgoalBasis.from_profiles(maze, PROFILES, core_threshold=threshold)
    return Environment(maze).hierarchy(basis, parameters=_parameters())


def _trials():
    return (
        Trial("s", 1, GOAL, ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        Trial("s", 2, GOAL, ((0, 2), (0, 2), (0, 3), (0, 4), (0, 5))),
        Trial("s", 3, GOAL, ((0, 3), (0, 2), (0, 3), (0, 4), (0, 5))),
    )


def _objective(template, prepared, raw, theta_values, geometry, n_decisions):
    profiles = peak_gauge(raw)
    log_likelihood = total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=theta_values,
        access_profiles=profiles,
    )
    penalties = structural_penalties(profiles, geometry)
    return -log_likelihood / n_decisions + sum(
        PENALTIES[name] * penalties[name] for name in PENALTIES
    )


def test_override_with_basis_profiles_matches_default_likelihood():
    template = _template()
    prepared = prepare_batch(template, _trials())
    values = parameter_values(template)
    default = total_prepared_log_likelihood(
        template, prepared, parameter_values=values
    )
    override = total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=values,
        access_profiles=torch.tensor(template.basis.profiles, dtype=torch.float64),
    )
    assert torch.equal(default, override)


def test_override_rejected_for_gated_basis():
    template = _template(threshold=0.2)
    prepared = prepare_batch(template, _trials())
    with pytest.raises(ValueError, match="ungated"):
        total_prepared_log_likelihood(
            template,
            prepared,
            parameter_values=parameter_values(template),
            access_profiles=torch.tensor(template.basis.profiles),
        )


def test_objective_is_invariant_to_raw_column_scale():
    template = _template()
    prepared = prepare_batch(template, _trials())
    geometry = profile_geometry(template)
    values = parameter_values(template)
    raw = torch.tensor(PROFILES + 0.05, dtype=torch.float64)
    scaled = raw * torch.tensor([3.7, 0.2], dtype=torch.float64)
    base = _objective(template, prepared, raw, values, geometry, 12)
    other = _objective(template, prepared, scaled, values, geometry, 12)
    assert torch.allclose(base, other, rtol=1e-12, atol=1e-12)


def test_gradient_through_gauge_matches_finite_differences():
    template = _template()
    prepared = prepare_batch(template, _trials())
    geometry = profile_geometry(template)
    rng = np.random.default_rng(3)
    # Strictly positive with a unique maximum in each column.
    start = PROFILES + 0.05 + 0.01 * rng.random(PROFILES.shape)
    raw = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    theta = {
        name: value.detach().clone().requires_grad_(name in ("alpha", "beta"))
        for name, value in parameter_values(template).items()
    }
    objective = _objective(template, prepared, raw, theta, geometry, 12)
    objective.backward()

    def evaluate(raw_value, theta_value):
        with torch.no_grad():
            return float(
                _objective(template, prepared, raw_value, theta_value, geometry, 12)
            )

    step = 1e-6
    peaks = raw.detach().argmax(dim=0).tolist()
    entries = [(peaks[0], 0), (peaks[1], 1), (2, 0), (3, 1), (1, 1)]
    for row, column in entries:
        plus = raw.detach().clone()
        minus = raw.detach().clone()
        plus[row, column] += step
        minus[row, column] -= step
        detached = {name: value.detach() for name, value in theta.items()}
        numeric = (evaluate(plus, detached) - evaluate(minus, detached)) / (2 * step)
        analytic = raw.grad[row, column].item()
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-8)
    for name in ("alpha", "beta"):
        plus = {key: value.detach().clone() for key, value in theta.items()}
        minus = {key: value.detach().clone() for key, value in theta.items()}
        plus[name] += step
        minus[name] -= step
        numeric = (evaluate(raw.detach(), plus) - evaluate(raw.detach(), minus)) / (
            2 * step
        )
        assert theta[name].grad.item() == pytest.approx(numeric, rel=1e-5, abs=1e-8)


def test_penalty_gradients_match_finite_differences():
    template = _template()
    geometry = profile_geometry(template)
    start = PROFILES + 0.05
    for name in ("compactness", "dirichlet", "overlap"):
        raw = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        structural_penalties(peak_gauge(raw), geometry)[name].backward()
        for row, column in ((0, 0), (5, 1), (2, 0), (3, 1)):
            plus = torch.tensor(start, dtype=torch.float64)
            minus = plus.clone()
            plus[row, column] += 1e-6
            minus[row, column] -= 1e-6
            numeric = (
                float(structural_penalties(peak_gauge(plus), geometry)[name])
                - float(structural_penalties(peak_gauge(minus), geometry)[name])
            ) / 2e-6
            assert raw.grad[row, column].item() == pytest.approx(
                numeric, rel=1e-5, abs=1e-9
            )


def test_zero_entry_receives_gradient_and_can_reenter():
    template = _template()
    prepared = prepare_batch(template, _trials())
    geometry = profile_geometry(template)
    raw = torch.tensor(PROFILES, dtype=torch.float64, requires_grad=True)
    assert raw[0, 1].item() == 0.0
    _objective(
        template, prepared, raw, parameter_values(template), geometry, 12
    ).backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[0, 1].item() != 0.0


def test_penalty_reference_values():
    template = _template()
    geometry = profile_geometry(template)
    one_hot = torch.zeros((6, 1), dtype=torch.float64)
    one_hot[2, 0] = 1.0
    assert structural_penalties(one_hot, geometry)["compactness"].item() == 0.0
    constant = torch.ones((6, 1), dtype=torch.float64)
    penalties = structural_penalties(constant, geometry)
    assert penalties["dirichlet"].item() == 0.0
    assert penalties["overlap"].item() == 0.0
    identical = torch.ones((6, 2), dtype=torch.float64)
    overlap = structural_penalties(identical, geometry)["overlap"].item()
    assert overlap == pytest.approx(1.0)
    disjoint = torch.zeros((6, 2), dtype=torch.float64)
    disjoint[:3, 0] = 1.0
    disjoint[3:, 1] = 1.0
    assert structural_penalties(disjoint, geometry)["overlap"].item() == 0.0
    # Distances are normalized by the diameter (5 steps in a 6-cell corridor).
    assert geometry.distances.max().item() == 1.0
    assert len(geometry.edges) == 5


def test_feasibility_flags_zero_column_and_heldout_goal_one_hot():
    maze = Maze.from_ascii("......")
    zero = PROFILES.copy()
    zero[:, 1] = 0.0
    assert profile_feasibility(zero, maze, [GOAL]) == ("column 1 is all zero",)
    one_hot = PROFILES.copy()
    one_hot[:, 0] = 0.0
    one_hot[1, 0] = 1.0
    # Feasible for the training goal, infeasible for a held-out goal at (0, 1).
    assert profile_feasibility(one_hot, maze, [GOAL]) == ()
    assert profile_feasibility(one_hot, maze, [GOAL, (0, 1)]) == (
        "column 0 is supported only on goal (0, 1)",
    )


def test_fit_rejects_initial_profiles_infeasible_for_evaluation_goal():
    template = _template()
    initial = PROFILES.copy()
    initial[:, 0] = 0.0
    initial[1, 0] = 1.0
    with pytest.raises(ValueError, match="Infeasible initial profiles"):
        fit_profiles(
            template,
            _trials(),
            initial_profiles=initial,
            evaluation_goals=[GOAL, (0, 1)],
            penalty_weights=PENALTIES,
            **TOLERANCES,
        )


def test_rejected_step_restores_parameters_and_adam_state(monkeypatch):
    import andrew_mlmdp.hierarchy.profile_fitting as module

    template = _template()
    snapshots = []
    real_feasibility = module.profile_feasibility
    calls = {"count": 0}

    def flaky_feasibility(raw, maze, goals):
        # The first call checks initial profiles; reject the next two proposals.
        calls["count"] += 1
        if calls["count"] in (2, 3):
            return ("forced violation",)
        return real_feasibility(raw, maze, goals)

    original_load = torch.optim.Adam.load_state_dict

    def recording_load(self, state):
        snapshots.append(copy.deepcopy(state))
        return original_load(self, state)

    monkeypatch.setattr(module, "profile_feasibility", flaky_feasibility)
    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", recording_load)
    result = fit_profiles(
        template,
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        max_steps=3,
        lr=0.1,
        **TOLERANCES,
    )
    assert [rejection.violations for rejection in result.rejections] == [
        ("forced violation",),
        ("forced violation",),
    ]
    assert [rejection.lr for rejection in result.rejections] == pytest.approx(
        [0.1, 0.03]
    )
    # Before the first accepted step, the restored Adam state is empty, so the
    # step counter and moments start from scratch after rejection.
    assert all(snapshot["state"] == {} for snapshot in snapshots)
    # The first accepted update used the twice-reduced learning rate.
    assert result.history[1].lr == pytest.approx(0.009)
    assert result.history[0].evaluation == 0
    # Evaluation 0 is unchanged by the rejected proposals.
    assert result.history[1].updates == 1


def test_rejection_restores_exact_raw_state(monkeypatch):
    import andrew_mlmdp.hierarchy.profile_fitting as module

    template = _template()
    real_feasibility = module.profile_feasibility
    calls = {"count": 0}

    def capture(raw, maze, goals):
        calls["count"] += 1
        if calls["count"] == 3:
            return ("forced violation",)
        return real_feasibility(raw, maze, goals)

    original_step = torch.optim.Adam.step
    states = []

    def recording_step(self, *args, **kwargs):
        states.append(
            {
                index: {
                    "value": param.detach().clone(),
                    **{
                        key: value.clone() if isinstance(value, torch.Tensor) else value
                        for key, value in self.state[param].items()
                    },
                }
                for index, param in enumerate(self.param_groups[0]["params"])
            }
        )
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(module, "profile_feasibility", capture)
    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    fit_profiles(
        template,
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        max_steps=2,
        **TOLERANCES,
    )
    # step 1 accepted (call 2), step 2 proposal rejected (call 3), retried.
    assert len(states) == 3
    before_rejected, before_retry = states[1], states[2]
    assert before_rejected.keys() == before_retry.keys()
    for key, state in before_rejected.items():
        for name, value in state.items():
            restored = torch.as_tensor(before_retry[key][name])
            assert torch.equal(torch.as_tensor(value), restored)


def test_rejection_limit_is_an_optimizer_failure(monkeypatch):
    import andrew_mlmdp.hierarchy.profile_fitting as module

    template = _template()
    real_feasibility = module.profile_feasibility
    calls = {"count": 0}

    def always_reject(raw, maze, goals):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_feasibility(raw, maze, goals)
        return ("forced violation",)

    monkeypatch.setattr(module, "profile_feasibility", always_reject)
    result = fit_profiles(
        template,
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        max_rejections=2,
        **TOLERANCES,
    )
    assert result.reason == "rejection_limit"
    assert len(result.rejections) == 3
    assert best_restart([result]) is None


def test_heldout_trajectory_through_zeroed_region_scores_finite():
    template = _template()
    profiles = np.zeros((6, 2))
    profiles[0, 0] = 1.0
    profiles[1, 0] = 0.5
    profiles[4, 1] = 1.0
    heldout = (Trial("h", 1, (0, 0), ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0))),)
    prepared = prepare_batch(template, heldout)
    log_likelihood = total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=parameter_values(template),
        access_profiles=torch.tensor(profiles, dtype=torch.float64),
    )
    assert torch.isfinite(log_likelihood)


def test_decision_count_matches_collapsed_moves():
    template = _template()
    result = fit_profiles(
        template,
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        max_steps=1,
        **TOLERANCES,
    )
    # 4 + 3 (one repeated coordinate collapsed) + 4 moves.
    assert result.provenance["n_decisions"] == 11


def test_short_fit_improves_objective_and_is_deterministic():
    template = _template()
    options = dict(
        seeds=(0, 1),
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        max_steps=25,
        **TOLERANCES,
    )
    first = fit_profile_restarts(template, _trials(), **options)
    second = fit_profile_restarts(template, _trials(), **options)
    for left, right in zip(first, second, strict=True):
        assert left.objective_history == right.objective_history
        assert np.array_equal(left.best_raw_profiles, right.best_raw_profiles)
        assert left.best_objective < left.objective_history[0]
        assert np.all(left.best_raw_profiles >= 0.0)
        assert np.allclose(left.best_profiles.max(axis=0), 1.0)
    chosen = best_restart(first)
    assert chosen.best_objective == min(result.best_objective for result in first)
    assert chosen.provenance["initial_profile_seed"] in (0, 1)
    basis = fitted_basis(template.maze, chosen)
    assert np.allclose(basis.profiles, chosen.best_profiles)


def test_random_localized_profiles_have_unit_peaks():
    maze = Maze.from_ascii("......")
    profiles = random_localized_profiles(maze, 3, np.random.default_rng(0))
    assert profiles.shape == (6, 3)
    assert np.allclose(profiles.max(axis=0), 1.0)
    assert np.all(profiles > 0.0)


def _log_objective(template, prepared, log_raw, values, n_decisions):
    log_likelihood = total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=values,
        access_profiles=log_peak_gauge(log_raw),
    )
    return -log_likelihood / n_decisions


def test_log_gauge_is_shift_invariant_with_unit_peaks():
    log_raw = torch.log(torch.tensor(PROFILES + 0.05, dtype=torch.float64))
    profiles = log_peak_gauge(log_raw)
    shifted = log_peak_gauge(log_raw + torch.tensor([3.0, -7.0], dtype=torch.float64))
    assert torch.allclose(profiles, shifted, rtol=1e-14, atol=0.0)
    peaks = profiles.max(dim=0).values
    assert torch.allclose(peaks, torch.ones(2, dtype=torch.float64))
    assert bool((profiles > 0.0).all())


def test_log_gradient_matches_finite_differences():
    template = _template()
    prepared = prepare_batch(template, _trials())
    values = parameter_values(template)
    rng = np.random.default_rng(4)
    start = np.log(PROFILES + 0.05 + 0.01 * rng.random(PROFILES.shape))
    log_raw = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    _log_objective(template, prepared, log_raw, values, 12).backward()
    peaks = log_raw.detach().argmax(dim=0).tolist()
    for row, column in [(peaks[0], 0), (peaks[1], 1), (2, 0), (4, 1)]:
        plus = log_raw.detach().clone()
        minus = log_raw.detach().clone()
        plus[row, column] += 1e-6
        minus[row, column] -= 1e-6
        with torch.no_grad():
            numeric = (
                float(_log_objective(template, prepared, plus, values, 12))
                - float(_log_objective(template, prepared, minus, values, 12))
            ) / 2e-6
        analytic = log_raw.grad[row, column].item()
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-9)


def test_log_parameterization_requires_positive_initial_profiles():
    with pytest.raises(ValueError, match="strictly positive"):
        fit_profiles(
            _template(),
            _trials(),
            initial_profiles=PROFILES,
            evaluation_goals=[GOAL],
            penalty_weights=PENALTIES,
            parameterization="log",
            **TOLERANCES,
        )


def test_log_fit_improves_objective_and_keeps_positive_unit_peak_profiles():
    template = _template()
    result = fit_profiles(
        template,
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        parameterization="log",
        max_steps=25,
        **TOLERANCES,
    )
    assert result.provenance["parameterization"] == "log"
    assert result.rejections == ()
    assert result.best_objective < result.objective_history[0]
    assert np.all(result.best_profiles > 0.0)
    assert np.allclose(result.best_profiles.max(axis=0), 1.0)
    # The raw state is log D: its column maxima are the gauge's zero point.
    gauged = np.exp(
        result.best_raw_profiles - result.best_raw_profiles.max(axis=0, keepdims=True)
    )
    assert np.allclose(gauged, result.best_profiles)


def test_gaussian_random_profiles_have_unit_peaks_and_lighter_tails():
    maze = Maze.from_ascii("......")
    gaussian = random_localized_profiles(
        maze, 2, np.random.default_rng(0), shape="gaussian"
    )
    exponential = random_localized_profiles(maze, 2, np.random.default_rng(0))
    assert np.allclose(gaussian.max(axis=0), 1.0)
    # Same centres and widths: the far tail is much lighter.
    assert np.all(gaussian.min(axis=0) < exponential.min(axis=0))


def test_access_gate_zeroes_entries_below_threshold_and_keeps_unit_peak():
    profiles = torch.tensor([[1.0, 0.04], [0.05, 1.0], [0.5, 0.2]], dtype=torch.float64)
    gated = access_gate(profiles, 0.05)
    assert gated[1, 0].item() == 0.0
    assert gated[0, 1].item() == 0.0
    assert torch.allclose(gated.max(dim=0).values, torch.ones(2, dtype=torch.float64))
    assert gated[2, 0].item() == pytest.approx((0.5 - 0.05) / 0.95)
    assert torch.equal(access_gate(profiles, None), profiles)


def test_gated_entries_get_gradient_only_through_penalties():
    template = _template()
    prepared = prepare_batch(template, _trials())
    geometry = profile_geometry(template)
    values = parameter_values(template)
    start = np.log(PROFILES + 0.01)
    # Entry (5, 0) has D ~ 0.01 / 1.01 < tau, so the gate removes it.
    below = (5, 0)

    def gradient(dirichlet_weight):
        log_raw = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        ungated = log_peak_gauge(log_raw)
        log_likelihood = total_prepared_log_likelihood(
            template,
            prepared,
            parameter_values=values,
            access_profiles=access_gate(ungated, 0.05),
        )
        penalty = structural_penalties(ungated, geometry)["dirichlet"]
        (-log_likelihood / 12 + dirichlet_weight * penalty).backward()
        return log_raw.grad[below].item()

    assert gradient(0.0) == 0.0
    assert gradient(1.0) != 0.0


def test_fit_with_access_threshold_reports_gated_and_ungated_profiles():
    template = _template()
    result = fit_profiles(
        template,
        _trials(),
        initial_profiles=PROFILES + 0.01,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        parameterization="log",
        access_threshold=0.05,
        max_steps=10,
        **TOLERANCES,
    )
    gated = result.best_profiles
    ungated = result.best_ungated_profiles
    assert result.provenance["access_threshold"] == 0.05
    assert np.all(ungated > 0.0)
    assert np.all(gated[ungated < 0.05] == 0.0)
    assert np.allclose(gated, np.maximum(ungated - 0.05, 0.0) / 0.95)
    assert np.allclose(gated.max(axis=0), 1.0)


def test_access_threshold_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="access_threshold"):
        fit_profiles(
            _template(),
            _trials(),
            initial_profiles=PROFILES + 0.01,
            evaluation_goals=[GOAL],
            penalty_weights=PENALTIES,
            parameterization="log",
            access_threshold=1.0,
            **TOLERANCES,
        )


def _validation_trials():
    return (
        Trial("v", 1, GOAL, ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        Trial("v", 2, GOAL, ((0, 1), (0, 2), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
    )


def test_validation_selection_reports_best_checked_state():
    result = fit_profiles(
        _template(),
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        parameterization="log",
        validation_trials=_validation_trials(),
        validation_interval=3,
        max_steps=20,
        **TOLERANCES,
    )
    checked = [
        (step.evaluation, step.validation_ll_per_decision, step.objective)
        for step in result.history
        if step.validation_ll_per_decision is not None
    ]
    assert [evaluation for evaluation, _, _ in checked] == list(
        range(0, len(result.history), 3)
    )
    best = max(checked, key=lambda item: item[1])
    assert result.selection == "validation"
    assert result.best_validation_ll_per_decision == best[1]
    assert result.best_objective == best[2]
    assert result.provenance["validation"]["interval"] == 3


def test_validation_patience_stops_after_stale_checks(monkeypatch):
    import andrew_mlmdp.hierarchy.profile_fitting as module

    real = module.total_prepared_log_likelihood
    calls = {"validation": 0}

    def fake(template, prepared, **kwargs):
        value = real(template, prepared, **kwargs)
        is_validation = len(prepared.trials) == 2 and not torch.is_grad_enabled()
        if is_validation:
            # Validation scores get strictly worse after the first check.
            calls["validation"] += 1
            return value - 10.0 * calls["validation"]
        return value

    monkeypatch.setattr(module, "total_prepared_log_likelihood", fake)
    result = fit_profiles(
        _template(),
        _trials(),
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        parameterization="log",
        validation_trials=_validation_trials(),
        validation_interval=1,
        validation_patience=2,
        max_steps=50,
        **TOLERANCES,
    )
    assert result.reason == "validation_patience"
    assert len(result.history) == 3
    assert result.best_objective == result.history[0].objective


def test_validation_arguments_are_checked():
    options = dict(
        initial_profiles=PROFILES + 0.05,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        parameterization="log",
        **TOLERANCES,
    )
    with pytest.raises(ValueError, match="validation_interval"):
        fit_profiles(
            _template(), _trials(), validation_trials=_validation_trials(),
            validation_interval=0, **options,
        )
    with pytest.raises(ValueError, match="validation goal"):
        fit_profiles(
            _template(), _trials(),
            validation_trials=(
                Trial("v", 1, (0, 0), ((0, 3), (0, 2), (0, 1), (0, 0))),
            ),
            **options,
        )


def test_freeze_profiles_fits_theta_only():
    initial = PROFILES + 0.05
    result = fit_profiles(
        _template(),
        _trials(),
        initial_profiles=initial,
        evaluation_goals=[GOAL],
        penalty_weights=PENALTIES,
        parameterization="log",
        theta_names=("lower_control_cost", "beta"),
        freeze_profiles=True,
        max_steps=15,
        **TOLERANCES,
    )
    gauged = initial / initial.max(axis=0, keepdims=True)
    assert np.allclose(result.best_profiles, gauged, rtol=1e-12, atol=0.0)
    first, best = result.history[0].theta_values, result.best_theta.as_floats()
    assert best["beta"] != first["beta"]
    assert result.best_objective < result.objective_history[0]
    assert all(step.profile_gradient_norm == 0.0 for step in result.history)
    with pytest.raises(ValueError, match="freeze_profiles"):
        fit_profiles(
            _template(), _trials(), initial_profiles=initial,
            evaluation_goals=[GOAL], penalty_weights=PENALTIES,
            parameterization="log", theta_names=(), freeze_profiles=True,
            **TOLERANCES,
        )
