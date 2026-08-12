import numpy as np
import pytest
import torch

from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    ModelParameters,
    MovementTrial,
    SubgoalBasis,
    hierarchical_movement_log_likelihood_torch,
    hierarchical_parameter_values,
    total_hierarchical_movement_log_likelihood_torch,
)
from andrew_mlmdp.hierarchy.core import _goal_only_plan as _numpy_goal_only_plan
from andrew_mlmdp.hierarchy.likelihood import (
    _hierarchical_physical_step_kernel as _numpy_step_kernel,
)
from andrew_mlmdp.hierarchy.torch_likelihood import (
    PINV_RCOND,
    _build_torch_hierarchy,
    _first_departure_forward_torch,
    _goal_only_plan,
    _physical_step_kernel,
    _plan,
)


def _parameters(**overrides):
    values = {
        "interior_reward": -0.15,
        "goal_reward": 0.35,
        "lower_control_cost": 0.55,
        "upper_control_cost": 0.9,
        "alpha": 0.8,
        "off_target_reward": -0.3,
        "beta": 0.7,
    }
    values.update(overrides)
    return ModelParameters(**values)


def _gated_template():
    maze = Maze.from_ascii("......")
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.85, 0.10],
            [0.60, 0.35],
            [0.35, 0.65],
            [0.10, 0.85],
            [0.0, 1.0],
        ]
    )
    basis = SubgoalBasis.from_profiles(
        maze,
        profiles,
        core_threshold=0.2,
        core_exponent=0.7,
    )
    # These legacy gate fields deliberately disagree with the basis.  The
    # differentiable path must follow the basis to match the NumPy oracle.
    parameters = _parameters(core_threshold=0.75, core_exponent=2.0)
    return LMDPEnvironment(maze).hierarchy(basis, parameters=parameters)


def _tensor_values(template, *, requires_grad=False):
    return {
        name: value.detach().clone().requires_grad_(requires_grad)
        for name, value in hierarchical_parameter_values(template).items()
    }


def _numpy_plans(task, start):
    return (
        task.plan(start),
        *(task.plan(start, upper_state=j) for j in range(task.number_of_subtasks)),
        _numpy_goal_only_plan(
            task,
            start,
            goal_interior_desirability=None,
        ),
    )


def _torch_plans(model, start):
    return (
        _plan(model, start),
        *(_plan(model, start, upper_state=j) for j in range(model.number_of_subtasks)),
        _goal_only_plan(model),
    )


@pytest.mark.parametrize("kind", ["point", "ungated", "gated"])
def test_torch_and_numpy_hierarchies_and_likelihoods_agree(kind):
    if kind == "gated":
        template = _gated_template()
        goal = (0, 5)
        trajectory = ((0, 1), (0, 2), (0, 3), (0, 5))
    else:
        maze = Maze.from_ascii("......")
        locations = ((0, 1), (0, 4))
        if kind == "point":
            basis = SubgoalBasis.from_locations(maze, locations)
        else:
            point = SubgoalBasis.from_locations(maze, locations)
            basis = SubgoalBasis.from_profiles(
                maze,
                point.profiles,
                core_threshold=None,
            )
        template = LMDPEnvironment(maze).hierarchy(
            basis,
            parameters=_parameters(),
        )
        goal = (0, 5)
        trajectory = ((0, 0), (0, 1), (0, 3), (0, 5))

    task = template.for_goal(goal)
    values = hierarchical_parameter_values(template)
    torch_model = _build_torch_hierarchy(template, goal, values)
    torch_plan = _plan(torch_model, trajectory[0])
    numpy_plan = task.plan(trajectory[0])

    assert torch_model.access_profiles.detach().numpy() == pytest.approx(
        task.subtask_profiles
    )
    assert torch_model.lower_dynamics.passive.detach().numpy() == pytest.approx(
        task.lower_dynamics.passive,
        abs=1e-12,
    )
    assert torch_model.first_hit_probabilities.detach().numpy() == pytest.approx(
        task.first_hit_probabilities,
        abs=1e-12,
    )
    assert torch_model.upper_dynamics.passive.detach().numpy() == pytest.approx(
        task.upper_dynamics.passive,
        abs=1e-12,
    )
    assert torch_model.upper_desirability.detach().numpy() == pytest.approx(
        task.upper_desirability,
        abs=1e-11,
    )
    assert torch_model.upper_controlled.detach().numpy() == pytest.approx(
        task.upper_controlled,
        abs=1e-12,
    )
    assert (
        torch_model.task_basis.boundary_desirability.detach().numpy()
        == pytest.approx(task.task_basis.boundary_desirability, abs=1e-12)
    )
    assert torch_model.task_basis.interior_desirability.detach().numpy() == (
        pytest.approx(task.task_basis.interior_desirability, abs=1e-11)
    )
    assert torch_plan.weights.detach().numpy() == pytest.approx(
        numpy_plan.weights,
        abs=1e-11,
    )
    assert torch_plan.layer_one_controlled.detach().numpy() == pytest.approx(
        numpy_plan.layer_one_controlled,
        abs=1e-11,
    )

    torch_kernel = _physical_step_kernel(
        torch_model,
        trajectory[0],
        _torch_plans(torch_model, trajectory[0]),
    )
    numpy_kernel = _numpy_step_kernel(
        task,
        trajectory[0],
        _numpy_plans(task, trajectory[0]),
    )
    assert torch_kernel.detach().numpy() == pytest.approx(numpy_kernel, abs=1e-11)

    torch_ll = hierarchical_movement_log_likelihood_torch(
        template,
        goal,
        trajectory,
        parameter_values=values,
    )
    assert torch_ll.dtype == torch.float64
    assert torch_ll == pytest.approx(
        task.movement_log_likelihood(trajectory), abs=1e-11
    )
    assert torch_ll <= 0.0


def test_probability_orientation_and_likelihood_invariants():
    template = _gated_template()
    goal = (0, 5)
    values = hierarchical_parameter_values(template)
    model = _build_torch_hierarchy(template, goal, values)
    plans = _torch_plans(model, (0, 1))

    for controlled in (
        model.upper_controlled,
        *(plan.layer_one_controlled for plan in plans),
    ):
        assert torch.all(torch.isfinite(controlled))
        assert torch.all(controlled >= -1e-14)
        assert torch.allclose(
            controlled.sum(dim=0),
            torch.ones(controlled.shape[1], dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )

    kernel = _physical_step_kernel(model, (0, 1), plans)
    assert torch.all(torch.isfinite(kernel))
    assert torch.all(kernel >= -1e-14)
    assert torch.allclose(
        kernel.sum(dim=(0, 1)),
        torch.ones(kernel.shape[2], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    forward = torch.nn.functional.one_hot(
        torch.tensor(0),
        num_classes=kernel.shape[1],
    ).to(torch.float64)
    departure = _first_departure_forward_torch(kernel, 1, 2, forward).sum()
    assert 0.0 <= departure <= 1.0

    assert template.torch_movement_log_likelihood(goal, [(0, 1)]) == 0.0
    repeated = [(0, 1), (0, 1), (0, 1)]
    assert template.torch_movement_log_likelihood(goal, repeated) == 0.0


def test_complete_mode_first_departure_uses_full_occupancy_system():
    self_kernel = torch.tensor(
        [
            [0.20, 0.00, 0.00],
            [0.30, 0.40, 0.00],
            [0.00, 0.20, 0.50],
        ],
        dtype=torch.float64,
    )
    exit_kernel = torch.diag(torch.tensor([0.50, 0.40, 0.50], dtype=torch.float64))
    kernel = torch.stack((self_kernel, exit_kernel), dim=0)
    forward = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    spectral_radius = torch.linalg.eigvals(self_kernel).abs().max()
    assert spectral_radius < 1.0
    expected_occupancy = torch.linalg.solve(
        torch.eye(3, dtype=torch.float64) - self_kernel,
        forward,
    )
    expected = exit_kernel @ expected_occupancy
    actual = _first_departure_forward_torch(kernel, 0, 1, forward)
    assert actual == pytest.approx(expected, abs=1e-14)
    assert actual[1] > 0.0
    assert actual[2] > 0.0

    geometric = torch.zeros(3, dtype=torch.float64)
    state = forward
    for _ in range(200):
        geometric = geometric + exit_kernel @ state
        state = self_kernel @ state
    assert actual == pytest.approx(geometric, abs=1e-14)


def test_pseudoinverse_cutoff_matches_numpy_near_rank_boundary():
    matrix = np.diag([1.0, 2.0e-15, 0.5e-15])
    numpy_result = np.linalg.pinv(matrix, rcond=PINV_RCOND)
    torch_result = torch.linalg.pinv(
        torch.tensor(matrix, dtype=torch.float64),
        rtol=PINV_RCOND,
    )
    assert torch_result.numpy() == pytest.approx(numpy_result, abs=0.0)
    assert np.count_nonzero(np.diag(numpy_result)) == 2


def test_masked_fractional_core_gate_has_finite_nonzero_gradients():
    template = _gated_template()
    goal = (0, 5)
    trajectory = ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    values = _tensor_values(template, requires_grad=True)
    model = _build_torch_hierarchy(template, goal, values)
    assert torch.count_nonzero(model.access_profiles == 0.0) >= 3

    total_log_likelihood = hierarchical_movement_log_likelihood_torch(
        template,
        goal,
        trajectory,
        parameter_values=values,
    )
    total_log_likelihood.backward()
    for value in values.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad)
    assert abs(values["core_threshold"].grad) > 1e-8
    assert abs(values["core_exponent"].grad) > 1e-8


def test_autograd_matches_central_finite_differences():
    template = _gated_template()
    goal = (0, 5)
    trajectory = ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    values = _tensor_values(template, requires_grad=True)
    log_likelihood = hierarchical_movement_log_likelihood_torch(
        template,
        goal,
        trajectory,
        parameter_values=values,
    )
    log_likelihood.backward()

    for name, value in values.items():
        step = 1e-6
        plus = {key: tensor.detach().clone() for key, tensor in values.items()}
        minus = {key: tensor.detach().clone() for key, tensor in values.items()}
        plus[name] = plus[name] + step
        minus[name] = minus[name] - step
        plus_ll = hierarchical_movement_log_likelihood_torch(
            template,
            goal,
            trajectory,
            parameter_values=plus,
        )
        minus_ll = hierarchical_movement_log_likelihood_torch(
            template,
            goal,
            trajectory,
            parameter_values=minus,
        )
        finite_difference = (plus_ll - minus_ll) / (2.0 * step)
        assert value.grad == pytest.approx(finite_difference, rel=2e-4, abs=2e-6)


def test_total_likelihood_materializes_sum_and_preserves_impossible_semantics():
    template = _gated_template()
    trials = (
        MovementTrial("s", 1, (0, 5), ((0, 1), (0, 2), (0, 5))),
        MovementTrial("s", 2, (0, 5), ((0, 2), (0, 3), (0, 5))),
    )
    values = hierarchical_parameter_values(template)
    expected = sum(
        template.for_goal(trial.goal).movement_log_likelihood(trial.trajectory)
        for trial in trials
    )
    actual = total_hierarchical_movement_log_likelihood_torch(
        template,
        (trial for trial in trials),
        parameter_values=values,
    )
    assert actual == pytest.approx(expected, abs=1e-11)

    impossible = MovementTrial("s", 3, (0, 5), ((0, 0), (0, 4)))
    result = total_hierarchical_movement_log_likelihood_torch(
        template,
        (impossible,),
        parameter_values=values,
    )
    assert torch.isneginf(result)


def test_parameter_mapping_is_strict_and_basis_owns_gate_defaults():
    template = _gated_template()
    values = hierarchical_parameter_values(template)
    assert values["core_threshold"] == pytest.approx(template.basis.core_threshold)
    assert values["core_exponent"] == pytest.approx(template.basis.core_exponent)

    missing = dict(values)
    del missing["alpha"]
    with pytest.raises(ValueError, match="missing alpha"):
        hierarchical_movement_log_likelihood_torch(
            template,
            (0, 5),
            ((0, 1),),
            parameter_values=missing,
        )
    wrong_dtype = dict(values)
    wrong_dtype["alpha"] = torch.tensor(0.8, dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        hierarchical_movement_log_likelihood_torch(
            template,
            (0, 5),
            ((0, 1),),
            parameter_values=wrong_dtype,
        )

    maze = Maze.from_ascii("....")
    ungated = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_profiles(
            maze,
            np.asarray([[1.0], [0.7], [0.3], [0.0]]),
            core_threshold=None,
        ),
        parameters=_parameters(),
    )
    with pytest.raises(ValueError, match="Inactive or unknown"):
        hierarchical_parameter_values(
            ungated,
            overrides={"core_exponent": torch.tensor(0.7, dtype=torch.float64)},
        )
