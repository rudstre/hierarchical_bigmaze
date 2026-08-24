import numpy as np
import pytest
import torch

from andrew_mlmdp import (
    Environment,
    Maze,
    Parameters,
    SubgoalBasis,
    TaskLibrary,
    fittable_parameters,
    point_parameters,
    required_parameters,
    soft_parameters,
)
from andrew_mlmdp.hierarchy.model import (
    _goal_only_plan,
    _shape_weights,
)


def _parameter_values(parameters: Parameters) -> dict[str, float]:
    return {
        name: parameter.item()
        for name, parameter in parameters.named_parameters()
    }


def test_model_parameters_are_trainable_float64_scalars():
    parameters = Parameters()
    expected = {
        "interior_reward": -1.0,
        "goal_reward": 0.0,
        "lower_control_cost": 1.0,
        "upper_control_cost": 2.5,
        "alpha": 0.2,
        "beta": 160.0,
        "core_exponent": 1.0,
    }

    assert isinstance(parameters, torch.nn.Module)
    assert parameters.core_threshold is None
    assert _parameter_values(parameters) == pytest.approx(expected)
    assert set(parameters.state_dict()) == set(expected)
    assert "interior_reward=-1" in repr(parameters)
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
    hard = point_parameters()
    soft = soft_parameters()
    ungated_soft = soft_parameters(core_threshold=None)

    assert hard.core_threshold is None
    assert ungated_soft.core_threshold is None
    assert "core_threshold" not in dict(hard.named_parameters())
    assert "core_threshold" not in dict(ungated_soft.named_parameters())
    assert isinstance(soft.core_threshold, torch.nn.Parameter)
    assert _parameter_values(soft) == pytest.approx(
        {
            "interior_reward": -1.0,
            "goal_reward": 0.0,
            "lower_control_cost": 1.0,
            "upper_control_cost": 2.5,
            "alpha": 0.2,
            "beta": 160.0,
            "core_threshold": 0.8,
            "core_exponent": 1.0,
        }
    )


def test_fittable_parameters_exclude_rewards_and_follow_basis_gate():
    maze = Maze.from_ascii(".....")
    environment = Environment(maze)
    point = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    )
    gated = environment.hierarchy(
        SubgoalBasis.from_profiles(
            maze,
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.5, 0.5],
                    [0.2, 0.8],
                    [0.0, 1.0],
                ]
            ),
            core_threshold=0.4,
        )
    )

    complete = (
        "interior_reward",
        "goal_reward",
        "lower_control_cost",
        "upper_control_cost",
        "alpha",
        "beta",
    )
    fittable = (
        "lower_control_cost",
        "upper_control_cost",
        "alpha",
        "beta",
    )
    assert required_parameters(point) == complete
    assert fittable_parameters(point) == fittable
    assert required_parameters(gated) == complete + (
        "core_threshold",
        "core_exponent",
    )
    assert fittable_parameters(gated) == fittable + (
        "core_threshold",
        "core_exponent",
    )


def test_canonical_gauge_preserves_old_hierarchical_policy_and_likelihood():
    maze = Maze.from_ascii("......")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 4)))
    old = environment.hierarchy(
        basis,
        parameters=Parameters(
            interior_reward=-0.1,
            goal_reward=1.1,
            lower_control_cost=0.1,
            upper_control_cost=0.25,
            alpha=0.2,
            beta=16.0,
        ),
    ).task((0, 5))
    canonical = environment.hierarchy(
        basis,
        parameters=Parameters(),
    ).task((0, 5))

    assert canonical.upper_controlled == pytest.approx(old.upper_controlled)
    for upper_state in (None, 0, 1):
        old_plan = old.plan((0, 2), upper_state=upper_state)
        canonical_plan = canonical.plan((0, 2), upper_state=upper_state)
        assert (
            canonical_plan.rewards
            / canonical.parameters.lower_control_cost.item()
        ) == pytest.approx(
            old_plan.rewards / old.parameters.lower_control_cost.item()
        )
        assert canonical_plan.weights == pytest.approx(old_plan.weights)
        assert canonical_plan.lower_policy == pytest.approx(
            old_plan.lower_policy
        )

    trajectory = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    assert canonical.log_likelihood(trajectory) == pytest.approx(
        old.log_likelihood(trajectory),
        abs=2e-11,
    )


def test_goal_reward_alone_does_not_change_hierarchical_behavior():
    maze = Maze.from_ascii("......")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 4)))
    low = environment.hierarchy(
        basis,
        parameters=Parameters(goal_reward=-2.0),
    ).task((0, 5))
    high = environment.hierarchy(
        basis,
        parameters=Parameters(goal_reward=3.0),
    ).task((0, 5))

    assert high.upper_controlled == pytest.approx(low.upper_controlled)
    for upper_state in (None, 0, 1):
        low_plan = low.plan((0, 2), upper_state=upper_state)
        high_plan = high.plan((0, 2), upper_state=upper_state)
        assert high_plan.weights == pytest.approx(low_plan.weights)
        assert high_plan.lower_policy == pytest.approx(low_plan.lower_policy)

    low_goal = _goal_only_plan(low, (0, 2), goal_desirability=None)
    high_goal = _goal_only_plan(high, (0, 2), goal_desirability=None)
    assert high_goal.lower_policy == pytest.approx(low_goal.lower_policy)

    trajectory = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    assert high.log_likelihood(trajectory) == pytest.approx(
        low.log_likelihood(trajectory),
        abs=2e-11,
    )


def test_canonical_layer_one_task_library_is_immutable_full_rank_and_normalized():
    library = TaskLibrary.from_desirabilities(8)
    boundary = library.boundary_desirability

    assert boundary.shape == (9, 9)
    assert np.diag(boundary) == pytest.approx(np.ones(9))
    assert boundary[0, 1] == pytest.approx(np.exp(-18.0))
    assert np.all(boundary[:-1, -1] == 0.0)
    assert np.all(boundary[-1, :-1] == 0.0)
    assert library.target_value == 1.0
    assert library.off_target_value == pytest.approx(np.exp(-18.0))
    assert library.goal_value == 1.0
    assert library.effective_rank == 9
    assert library.singular_values[:2] == pytest.approx(
        [1.00000010661, 1.0],
        abs=5e-13,
    )
    assert library.singular_values[2:] == pytest.approx(
        np.full(7, 0.999999984770),
        abs=5e-13,
    )
    assert library.condition_number == pytest.approx(1.00000012184, abs=5e-13)
    with pytest.raises(ValueError, match="read-only"):
        boundary[0, 0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        library.singular_values[0] = 2.0


def test_arbitrary_task_library_snapshots_matrix_and_has_no_canonical_metadata():
    source = np.asarray([[2.0, 0.1, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 3.0]])
    library = TaskLibrary.from_matrix(source)
    source[0, 0] = 99.0

    assert library.boundary_desirability[0, 0] == 2.0
    assert library.target_value is None
    assert library.off_target_value is None
    assert library.goal_value is None
    assert library.effective_rank == 3
    assert np.isfinite(library.condition_number)


@pytest.mark.parametrize(
    "matrix,match",
    [
        (np.ones(3), "square"),
        (np.eye(1), "at least two"),
        (np.asarray([[1.0, -0.1], [0.0, 1.0]]), "non-negative"),
        (np.asarray([[1.0, np.nan], [0.0, 1.0]]), "finite"),
        (np.ones((2, 2)), "full rank"),
    ],
)
def test_task_library_rejects_invalid_matrices(matrix, match):
    with pytest.raises(ValueError, match=match):
        TaskLibrary.from_matrix(matrix)


def test_task_library_and_composition_configuration_validate_against_basis():
    maze = Maze.from_ascii(".....")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))

    with pytest.raises(ValueError, match="must have shape"):
        environment.hierarchy(
            basis,
            task_library=TaskLibrary.from_matrix(np.eye(4)),
        )
    with pytest.raises(TypeError, match="TaskLibrary"):
        environment.hierarchy(basis, task_library=np.eye(3))
    with pytest.raises(ValueError, match="finite and positive"):
        environment.hierarchy(basis, composition_exponent=0.0)
    with pytest.raises(ValueError, match="composition_mode"):
        environment.hierarchy(basis, composition_mode="invalid")


def test_threshold_range_reports_all_limiting_goal_subgoal_pairs():
    maze = Maze.from_ascii("....")
    profiles = np.asarray(
        [
            [1.0, 0.3],
            [0.4, 1.0],
            [0.4, 0.4],
            [0.2, 0.4],
        ]
    )
    template = Environment(maze).hierarchy(
        SubgoalBasis.from_profiles(
            maze,
            profiles,
            core_threshold=0.39,
        )
    )

    domain = template.threshold_range(((0, 0), (0, 1)))
    assert domain.maximum == 0.4
    assert domain.limiting_pairs == (((0, 0), 0), ((0, 1), 1))
    assert template.validate_threshold(
        0.39,
        ((0, 0), (0, 1)),
    ) == domain
    with pytest.raises(ValueError, match="threshold < 0.4"):
        template.validate_threshold(
            0.4,
            ((0, 0), (0, 1)),
        )


def test_goal_task_construction_rejects_threshold_that_eliminates_final_support():
    maze = Maze.from_ascii("....")
    template = Environment(maze).hierarchy(
        SubgoalBasis.from_profiles(
            maze,
            np.asarray([[1.0], [0.4], [0.2], [0.0]]),
            core_threshold=0.4,
        )
    )

    with pytest.raises(ValueError, match="limiting"):
        template.task((0, 0))


def test_fixed_task_library_is_invariant_to_behavioral_parameters():
    maze = Maze.from_ascii(".....")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    library = TaskLibrary.from_desirabilities(
        2,
        target_value=1.2,
        off_target_value=0.1,
        goal_value=0.8,
    )
    first = environment.hierarchy(
        basis,
        parameters=Parameters(goal_reward=0.2, lower_control_cost=0.4),
        task_library=library,
    ).task((0, 4))
    second = environment.hierarchy(
        basis,
        parameters=Parameters(goal_reward=2.0, lower_control_cost=1.3),
        task_library=library,
    ).task((0, 4))

    assert first.task_basis.boundary_desirability == pytest.approx(
        library.boundary_desirability
    )
    assert second.task_basis.boundary_desirability == pytest.approx(
        library.boundary_desirability
    )
    assert first.template.task_library is second.template.task_library


def test_normalized_canonical_library_preserves_global_scale_cancellation():
    maze = Maze.from_ascii("......")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 4)))
    parameters = Parameters()
    normalized = TaskLibrary.from_desirabilities(2)
    scale = np.exp(11.0)
    scaled = TaskLibrary.from_matrix(
        scale * normalized.boundary_desirability
    )
    normalized_task = environment.hierarchy(
        basis,
        parameters=parameters,
        task_library=normalized,
    ).task((0, 5))
    scaled_task = environment.hierarchy(
        basis,
        parameters=parameters,
        task_library=scaled,
    ).task((0, 5))

    for upper_state in (None, 0, 1):
        normalized_plan = normalized_task.plan((0, 0), upper_state=upper_state)
        scaled_plan = scaled_task.plan((0, 0), upper_state=upper_state)
        assert scale * scaled_plan.weights == pytest.approx(
            normalized_plan.weights,
            rel=2e-11,
            abs=2e-11,
        )
        assert scaled_plan.desirability == pytest.approx(
            normalized_plan.desirability,
            rel=2e-11,
            abs=2e-11,
        )
        assert scaled_plan.lower_policy == pytest.approx(
            normalized_plan.lower_policy,
            rel=2e-11,
            abs=2e-11,
        )

    trajectory = ((0, 0), (0, 1), (0, 2), (0, 4), (0, 5))
    assert scaled_task.log_likelihood(trajectory) == pytest.approx(
        normalized_task.log_likelihood(trajectory),
        abs=2e-11,
    )


def test_goal_only_behavior_is_independent_of_fixed_goal_library_column():
    maze = Maze.from_ascii(".....")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    parameters = Parameters(goal_reward=0.4, lower_control_cost=0.3)
    first = environment.hierarchy(
        basis,
        parameters=parameters,
        task_library=TaskLibrary.from_desirabilities(
            2,
            goal_value=0.5,
        ),
    ).task((0, 4))
    second = environment.hierarchy(
        basis,
        parameters=parameters,
        task_library=TaskLibrary.from_desirabilities(
            2,
            goal_value=4.0,
        ),
    ).task((0, 4))

    exact_first = _goal_only_plan(
        first,
        (0, 0),
        goal_desirability=None,
    )
    exact_second = _goal_only_plan(
        second,
        (0, 0),
        goal_desirability=None,
    )
    learned = np.linspace(0.1, 0.8, len(first.interior_states))
    online_first = _goal_only_plan(
        first,
        (0, 0),
        goal_desirability=learned,
    )
    online_second = _goal_only_plan(
        second,
        (0, 0),
        goal_desirability=learned,
    )

    assert exact_first.desirability == pytest.approx(
        exact_second.desirability
    )
    assert exact_first.lower_policy == pytest.approx(
        exact_second.lower_policy
    )
    assert online_first.desirability == pytest.approx(
        online_second.desirability
    )
    assert online_first.lower_policy == pytest.approx(
        online_second.lower_policy
    )


@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 8.0])
def test_power_composition_preserves_zeros_subgoal_mass_and_goal_weight(exponent):
    weights = np.asarray([0.0, 1.0, 3.0, 0.0, 2.5])
    sharpened = _shape_weights(weights, exponent=exponent, mode="power")

    assert sharpened[:-1].sum() == pytest.approx(weights[:-1].sum())
    assert sharpened[-1] == weights[-1]
    assert np.array_equal(sharpened[[0, 3]], np.zeros(2))
    if exponent == 1.0:
        assert sharpened is weights
        assert np.array_equal(sharpened, weights)


def test_power_composition_changes_only_subgoal_concentration():
    weights = np.asarray([1.0, 3.0, 0.0, 4.0])
    diffuse = _shape_weights(weights, exponent=0.5, mode="power")
    sharp = _shape_weights(weights, exponent=4.0, mode="power")

    diffuse_p = diffuse[:-1] / diffuse[:-1].sum()
    original_p = weights[:-1] / weights[:-1].sum()
    sharp_p = sharp[:-1] / sharp[:-1].sum()
    assert np.sum(diffuse_p**2) < np.sum(original_p**2) < np.sum(sharp_p**2)
    assert diffuse[-1] == sharp[-1] == weights[-1]


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([1.0, 4.0, 2.0, 7.0], [0.0, 7.0, 0.0, 7.0]),
        ([4.0, 4.0, 2.0, 7.0], [5.0, 5.0, 0.0, 7.0]),
        ([0.0, 0.0, 0.0, 7.0], [0.0, 0.0, 0.0, 7.0]),
    ],
)
def test_winner_take_all_preserves_mass_goal_and_exact_ties(weights, expected):
    actual = _shape_weights(
        np.asarray(weights),
        exponent=1.0,
        mode="winner_take_all",
    )
    assert actual == pytest.approx(expected)


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

    template = Environment(maze).hierarchy(basis)

    expected = Parameters(
        interior_reward=-1.0,
        goal_reward=0.0,
        lower_control_cost=0.6,
        upper_control_cost=3.0,
        alpha=0.4,
        beta=160.0,
    )
    assert _parameter_values(point_parameters()) == pytest.approx(
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

    template = Environment(maze).hierarchy(basis)

    assert _parameter_values(template.parameters) == pytest.approx(
        _parameter_values(point_parameters())
    )


def test_explicit_point_hierarchy_parameters_override_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    supplied = Parameters(alpha=1.5)

    template = Environment(maze).hierarchy(
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
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, subgoals)
    template = environment.hierarchy(basis)
    task = template.task(goal)
    n = len(maze.free_cells) - 1
    k = len(subgoals)

    assert template.upper_passive.shape == (k, k)
    assert task.lower_dynamics.interior_passive.shape == (n, n)
    assert task.lower_dynamics.boundary_passive.shape == (k + 1, n)
    assert task.upper_dynamics.passive.shape == (k + 1, k)
    assert task.task_basis.interior_desirability.shape == (n, k + 1)


def test_template_caches_goal_tasks_and_reuses_environment_matrix():
    maze = Maze.from_ascii(".....")
    environment = Environment(maze)
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    )

    first = template.task((0, 4))
    assert template.task((0, 4)) is first
    assert template.task((0, 2)) is not first
    assert first.template.environment.passive is environment.passive


def test_first_hit_and_upper_dynamics_are_stochastic():
    maze = Maze.from_ascii("......")
    task = Environment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).task((0, 5))

    assert np.allclose(task.first_hit.sum(axis=0), 1.0)
    assert np.allclose(task.upper_dynamics.passive.sum(axis=0), 1.0)
    assert np.allclose(task.upper_controlled.sum(axis=0), 1.0)


def test_plan_inpaints_goal_and_composes_exact_goal_column():
    maze = Maze.from_ascii(".....")
    environment = Environment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    task = environment.hierarchy(basis).task((0, 4))
    plan = task.plan((0, 0))

    expected_rewards = task.parameters.beta.item() * (
        plan.upper_policy - plan.upper_passive
    )
    expected_goal_weight = (
        plan.target_boundary[-1]
        / task.task_basis.boundary_desirability[-1, -1]
    )

    assert plan.rewards == pytest.approx(expected_rewards)
    assert plan.weights[-1] == pytest.approx(expected_goal_weight)
    assert plan.weights[-1] > 0.0
    assert plan.boundary_desirability[-1] == pytest.approx(
        plan.target_boundary[-1]
    )
    assert plan.desirability[task.interior_states] == pytest.approx(
        task.task_basis.interior_desirability @ plan.weights
    )
    assert plan.desirability.shape == (5,)
    assert plan.lower_policy.shape == (7, 4)
    assert np.allclose(plan.lower_policy.sum(axis=0), 1.0)


def test_goal_only_plan_keeps_fixed_exact_goal_task():
    maze = Maze.from_ascii(".....")
    task = Environment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).task((0, 4))

    plan = _goal_only_plan(
        task,
        (0, 0),
        goal_desirability=None,
    )

    expected_goal_desirability = np.exp(
        task.parameters.goal_reward.item()
        / task.parameters.lower_control_cost.item()
    )

    assert np.all(plan.weights[:-1] == 0.0)
    assert plan.weights[-1] == 1.0
    assert np.all(np.isneginf(plan.rewards[:-1]))
    assert plan.rewards[-1] == task.parameters.goal_reward.item()
    assert plan.target_boundary[-1] == pytest.approx(
        expected_goal_desirability
    )
    assert np.allclose(plan.lower_policy.sum(axis=0), 1.0)


def test_online_goal_desirability_uses_same_inpainted_goal_weight():
    maze = Maze.from_ascii(".....")
    task = Environment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).task((0, 4))
    learned_goal = np.linspace(0.2, 0.8, len(task.interior_states))

    exact_plan = task.plan((0, 0))
    online_plan = task.plan((0, 0), goal_desirability=learned_goal)
    expected = (
        task.task_basis.interior_desirability[:, :-1]
        @ online_plan.weights[:-1]
        + learned_goal * online_plan.weights[-1]
    )

    assert online_plan.weights == pytest.approx(exact_plan.weights)
    assert online_plan.desirability[task.interior_states] == (
        pytest.approx(expected)
    )


def test_core_gate_is_peak_relative_and_applied_once():
    maze = Maze.from_ascii("....")
    raw = np.asarray([[2.0], [1.6], [1.0], [0.0]])
    parameters = Parameters(core_threshold=0.5, core_exponent=2.0)
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
    template = Environment(maze).hierarchy(basis)
    first = template.task((0, 3))
    second = template.task((0, 2))
    assert first.basis is second.basis is basis
    assert first.subtask_profiles is basis.access_profiles


def test_point_and_equivalent_profile_basis_match():
    maze = Maze.from_ascii("......")
    environment = Environment(maze)
    locations = ((0, 1), (0, 4))
    point = SubgoalBasis.from_locations(maze, locations)
    soft = SubgoalBasis.from_profiles(
        maze,
        point.profiles,
        core_threshold=None,
    )
    point_task = environment.hierarchy(point).task((0, 5))
    soft_task = environment.hierarchy(soft).task((0, 5))

    assert point_task.lower_dynamics.passive == pytest.approx(
        soft_task.lower_dynamics.passive
    )
    assert point_task.upper_dynamics.passive == pytest.approx(
        soft_task.upper_dynamics.passive
    )
