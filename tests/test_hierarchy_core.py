import numpy as np
import pytest
import torch

from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    ModelParameters,
    SubgoalBasis,
    hard_hierarchy_parameters,
    soft_hierarchy_parameters,
)
from andrew_mlmdp.hierarchy.core import _goal_only_plan


def _parameter_values(parameters: ModelParameters) -> dict[str, float]:
    return {
        name: parameter.item()
        for name, parameter in parameters.named_parameters()
    }


def test_model_parameters_are_trainable_float64_scalars():
    parameters = ModelParameters()
    expected = {
        "interior_reward": -0.1,
        "goal_reward": 1.1,
        "lower_control_cost": 0.1,
        "upper_control_cost": 0.25,
        "alpha": 0.2,
        "off_target_reward": -0.7,
        "beta": 16.0,
        "core_exponent": 1.0,
    }

    assert isinstance(parameters, torch.nn.Module)
    assert parameters.core_threshold is None
    assert _parameter_values(parameters) == pytest.approx(expected)
    assert set(parameters.state_dict()) == set(expected)
    assert "interior_reward=-0.1" in repr(parameters)
    assert "core_threshold=None" in repr(parameters)
    assert all(
        isinstance(parameter, torch.nn.Parameter)
        and parameter.shape == torch.Size([])
        and parameter.dtype == torch.float64
        and parameter.requires_grad
        for parameter in parameters.parameters()
    )
    trainable_names = set(dict(parameters.named_parameters()))
    assert not {
        "k",
        "passive_mode",
        "max_iter",
        "tolerance",
    } & trainable_names


def test_hierarchy_factories_preserve_core_defaults():
    hard = hard_hierarchy_parameters()
    soft = soft_hierarchy_parameters()
    ungated_soft = soft_hierarchy_parameters(core_threshold=None)

    assert hard.core_threshold is None
    assert ungated_soft.core_threshold is None
    assert "core_threshold" not in dict(hard.named_parameters())
    assert "core_threshold" not in dict(ungated_soft.named_parameters())
    assert isinstance(soft.core_threshold, torch.nn.Parameter)
    assert _parameter_values(soft) == pytest.approx(
        {
            "interior_reward": -0.1,
            "goal_reward": 1.1,
            "lower_control_cost": 0.1,
            "upper_control_cost": 0.25,
            "alpha": 0.2,
            "off_target_reward": -0.7,
            "beta": 16.0,
            "core_threshold": 0.8,
            "core_exponent": 1.0,
        }
    )


def test_point_basis_is_one_hot_and_validates_arbitrary_count():
    maze = Maze.from_ascii("....\n....")
    locations = ((0, 1), (1, 2), (1, 3))
    basis = SubgoalBasis.from_locations(maze, locations)

    assert basis.profiles.shape == (8, 3)
    assert basis.access_profiles == pytest.approx(basis.profiles)
    assert basis.profiles.sum(axis=0) == pytest.approx(np.ones(3))
    assert basis.locations == locations


def test_point_hierarchy_uses_swept_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))

    template = LMDPEnvironment(maze).hierarchy(basis)

    expected = ModelParameters(
        interior_reward=-0.1,
        goal_reward=1.1,
        lower_control_cost=0.06,
        upper_control_cost=0.3,
        alpha=0.4,
        off_target_reward=-1.0,
        beta=16.0,
    )
    assert _parameter_values(hard_hierarchy_parameters()) == pytest.approx(
        _parameter_values(expected)
    )
    assert _parameter_values(template.parameters) == pytest.approx(
        _parameter_values(expected)
    )


def test_profile_hierarchy_uses_same_default_as_point_hierarchy():
    maze = Maze.from_ascii(".....")
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    basis = SubgoalBasis.from_profiles(maze, profiles)

    template = LMDPEnvironment(maze).hierarchy(basis)

    assert _parameter_values(template.parameters) == pytest.approx(
        _parameter_values(hard_hierarchy_parameters())
    )


def test_explicit_point_hierarchy_parameters_override_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    supplied = ModelParameters(alpha=1.5)

    template = LMDPEnvironment(maze).hierarchy(
        basis,
        parameters=supplied,
    )

    assert template.parameters is supplied


@pytest.mark.parametrize(
    "layout,subgoals,goal",
    [
        ("......", ((0, 1), (0, 4)), (0, 5)),
        ("...\n...\n...", ((0, 0), (1, 2), (2, 0)), (2, 2)),
        ("....\n.##.\n....", ((0, 1), (2, 1)), (2, 3)),
    ],
)
def test_hierarchy_dimensions_derive_from_maze_and_basis(
    layout,
    subgoals,
    goal,
):
    maze = Maze.from_ascii(layout)
    environment = LMDPEnvironment(maze)
    basis = SubgoalBasis.from_locations(maze, subgoals)
    template = environment.hierarchy(basis)
    task = template.for_goal(goal)
    n = len(maze.free_cells) - 1
    k = len(subgoals)

    assert template.passive_dynamics.shape == (k, k)
    assert task.lower_dynamics.interior_passive.shape == (n, n)
    assert task.lower_dynamics.boundary_passive.shape == (k + 1, n)
    assert task.upper_dynamics.passive.shape == (k + 1, k)
    assert task.task_basis.interior_desirability.shape == (n, k + 1)


def test_template_caches_goal_tasks_and_reuses_environment_matrix():
    maze = Maze.from_ascii(".....")
    environment = LMDPEnvironment(maze)
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    )

    first = template.for_goal((0, 4))
    assert template.for_goal((0, 4)) is first
    assert template.for_goal((0, 2)) is not first
    assert first.template.environment.passive is environment.passive


def test_first_hit_and_upper_dynamics_are_stochastic():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 5))

    assert np.allclose(task.first_hit_probabilities.sum(axis=0), 1.0)
    assert np.allclose(task.upper_dynamics.passive.sum(axis=0), 1.0)
    assert np.allclose(task.upper_controlled.sum(axis=0), 1.0)


def test_plan_inpaints_goal_and_composes_exact_goal_column():
    maze = Maze.from_ascii(".....")
    environment = LMDPEnvironment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    task = environment.hierarchy(basis).for_goal((0, 4))
    plan = task.plan((0, 0))

    expected_rewards = task.parameters.beta.item() * (
        plan.controlled_abstract - plan.passive_abstract
    )
    expected_goal_weight = (
        plan.target_boundary_desirability[-1]
        / task.task_basis.boundary_desirability[-1, -1]
    )

    assert plan.inpainted_rewards == pytest.approx(expected_rewards)
    assert plan.weights[-1] == pytest.approx(expected_goal_weight)
    assert plan.weights[-1] > 0.0
    assert plan.reconstructed_boundary_desirability[-1] == pytest.approx(
        plan.target_boundary_desirability[-1]
    )
    assert plan.physical_desirability[task.interior_states] == pytest.approx(
        task.task_basis.interior_desirability @ plan.weights
    )
    assert plan.physical_desirability.shape == (5,)
    assert plan.layer_one_controlled.shape == (7, 4)
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)


def test_goal_only_plan_keeps_fixed_exact_goal_task():
    maze = Maze.from_ascii(".....")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 4))

    plan = _goal_only_plan(
        task,
        (0, 0),
        goal_interior_desirability=None,
    )

    expected_goal_desirability = np.exp(
        task.parameters.goal_reward.item()
        / task.parameters.lower_control_cost.item()
    )

    assert np.all(plan.weights[:-1] == 0.0)
    assert plan.weights[-1] == 1.0
    assert np.all(np.isneginf(plan.inpainted_rewards[:-1]))
    assert plan.inpainted_rewards[-1] == task.parameters.goal_reward.item()
    assert plan.target_boundary_desirability[-1] == pytest.approx(
        expected_goal_desirability
    )
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)


def test_online_goal_desirability_uses_same_inpainted_goal_weight():
    maze = Maze.from_ascii(".....")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 4))
    learned_goal = np.linspace(0.2, 0.8, len(task.interior_states))

    exact_plan = task.plan((0, 0))
    online_plan = task.plan((0, 0), goal_desirability=learned_goal)
    expected = (
        task.task_basis.interior_desirability[:, :-1]
        @ online_plan.weights[:-1]
        + learned_goal * online_plan.weights[-1]
    )

    assert online_plan.weights == pytest.approx(exact_plan.weights)
    assert online_plan.physical_desirability[task.interior_states] == (
        pytest.approx(expected)
    )


def test_core_gate_is_peak_relative_and_applied_once():
    maze = Maze.from_ascii("....")
    raw = np.asarray([[2.0], [1.6], [1.0], [0.0]])
    parameters = ModelParameters(core_threshold=0.5, core_exponent=2.0)
    basis = SubgoalBasis.from_profiles(
        maze,
        raw,
        core_threshold=parameters.core_threshold,
        core_exponent=parameters.core_exponent,
    )
    assert basis.profiles[:, 0] == pytest.approx([1.0, 0.8, 0.5, 0.0])
    assert basis.access_profiles[:, 0] == pytest.approx([1.0, 0.36, 0.0, 0.0])
    assert isinstance(basis.profiles, np.ndarray)
    assert isinstance(basis.access_profiles, np.ndarray)
    assert not basis.profiles.flags.writeable
    assert not basis.access_profiles.flags.writeable
    assert "profiles" not in dict(parameters.named_parameters())
    template = LMDPEnvironment(maze).hierarchy(basis)
    first = template.for_goal((0, 3))
    second = template.for_goal((0, 2))
    assert first.basis is second.basis is basis
    assert first.subtask_profiles is basis.access_profiles


def test_point_and_equivalent_profile_basis_match():
    maze = Maze.from_ascii("......")
    environment = LMDPEnvironment(maze)
    locations = ((0, 1), (0, 4))
    point = SubgoalBasis.from_locations(maze, locations)
    soft = SubgoalBasis.from_profiles(
        maze,
        point.profiles,
        core_threshold=None,
    )
    point_task = environment.hierarchy(point).for_goal((0, 5))
    soft_task = environment.hierarchy(soft).for_goal((0, 5))

    assert point_task.lower_dynamics.passive == pytest.approx(
        soft_task.lower_dynamics.passive
    )
    assert point_task.upper_dynamics.passive == pytest.approx(
        soft_task.upper_dynamics.passive
    )


