from __future__ import annotations

import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis, Trial
from andrew_mlmdp.hierarchy.equations import NumericalError, fittable_parameters
from andrew_mlmdp.hierarchy.likelihood import (
    PreparedBatch,
    _batched_trajectory_recursion,
    _PreparedTrajectoryStep,
    _PreparedTrial,
    _reference_prepared_log_likelihoods,
    _reference_trajectory_recursion,
    prepare_batch,
    prepared_log_likelihoods,
)

RTOL = 1e-10
ATOL = 1e-11


def _template():
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
        core_threshold=0.2,
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


def _values(template):
    fitted = set(fittable_parameters(template))
    return {
        name: value.detach().clone().requires_grad_(name in fitted)
        for name, value in template.parameter_values().items()
    }


def _direct_batch(
    operator_rows: tuple[tuple[int, ...], ...],
    *,
    impossible: tuple[bool, ...] | None = None,
) -> PreparedBatch:
    device = torch.device("cpu")
    if impossible is None:
        impossible = (False,) * len(operator_rows)
    trials = tuple(
        _PreparedTrial(
            torch.tensor(row, dtype=torch.long),
            trial_impossible,
        )
        for row, trial_impossible in zip(operator_rows, impossible, strict=True)
    )
    steps = []
    for depth in range(max((len(row) for row in operator_rows), default=0)):
        active = tuple(
            index
            for index, (row, trial_impossible) in enumerate(
                zip(operator_rows, impossible, strict=True)
            )
            if not trial_impossible and depth < len(row)
        )
        steps.append(
            _PreparedTrajectoryStep(
                trial_indices=torch.tensor(active, dtype=torch.long),
                operator_indices=torch.tensor(
                    tuple(operator_rows[index][depth] for index in active),
                    dtype=torch.long,
                ),
            )
        )
    empty = torch.empty(0, dtype=torch.long, device=device)
    return PreparedBatch(
        goals=(),
        trials=trials,
        trajectory_steps=tuple(steps),
        closure_shared_indices=empty,
        closure_x_states=empty,
        operator_closure_indices=empty,
        operator_y_states=empty,
        n_shared=0,
        n_closures=0,
        n_operators=0,
    )


def _recursion(function, operators, prepared):
    return function(
        operators,
        prepared,
        zero=torch.zeros((), dtype=torch.float64),
        negative_infinity=torch.full((), -torch.inf, dtype=torch.float64),
        n_modes=operators.shape[-1],
        device=torch.device("cpu"),
    )


def test_failed_trial_skips_malformed_later_operator_and_preserves_other_trials():
    prepared = _direct_batch(((0, 99), (1,)))
    operators = torch.stack(
        (
            torch.zeros((2, 2), dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )
    )

    reference = _recursion(_reference_trajectory_recursion, operators, prepared)
    batched = _recursion(_batched_trajectory_recursion, operators, prepared)

    assert torch.isneginf(reference[0])
    assert torch.isneginf(batched[0])
    assert reference[1] == 0.0
    assert batched[1] == 0.0


@pytest.mark.parametrize(
    "function",
    [_reference_trajectory_recursion, _batched_trajectory_recursion],
)
def test_reachable_malformed_later_operator_is_not_suppressed(function):
    prepared = _direct_batch(((0, 99),))
    operators = torch.eye(2, dtype=torch.float64).unsqueeze(0)

    with pytest.raises(IndexError):
        _recursion(function, operators, prepared)


@pytest.mark.parametrize(
    "operator",
    [
        torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[torch.inf, 0.0], [0.0, 1.0]], dtype=torch.float64),
    ],
)
@pytest.mark.parametrize(
    "function",
    [_reference_trajectory_recursion, _batched_trajectory_recursion],
)
def test_reachable_invalid_recurrence_values_retain_numerical_checks(
    function,
    operator,
):
    prepared = _direct_batch(((0,),))

    with pytest.raises(NumericalError):
        _recursion(function, operator.unsqueeze(0), prepared)


def test_batched_values_and_six_gradients_match_reference():
    template = _template()
    trials = (
        Trial("s", 1, (0, 5), ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        Trial("s", 2, (0, 5), ((0, 2), (0, 3), (0, 4), (0, 5))),
        Trial("s", 3, (0, 5), ((0, 1),)),
        Trial(
            "s",
            4,
            (0, 5),
            ((0, 1), (0, 1), (0, 2), (0, 2), (0, 3)),
        ),
        Trial("s", 5, (0, 5), ((0, 5), (0, 4))),
        Trial("s", 6, (0, 5), ((0, 0), (0, 2))),
        Trial("s", 7, (0, 0), ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0))),
    )
    prepared = prepare_batch(template, trials)
    reference_values = _values(template)
    batched_values = _values(template)

    reference = _reference_prepared_log_likelihoods(
        template,
        prepared,
        parameter_values=reference_values,
    )
    batched = prepared_log_likelihoods(
        template,
        prepared,
        parameter_values=batched_values,
    )

    assert torch.equal(torch.isfinite(reference), torch.isfinite(batched))
    assert torch.equal(torch.isneginf(reference), torch.isneginf(batched))
    finite = torch.isfinite(reference)
    torch.testing.assert_close(reference[finite], batched[finite], rtol=RTOL, atol=ATOL)

    reference[finite].sum().backward()
    batched[finite].sum().backward()
    for name in fittable_parameters(template):
        reference_gradient = reference_values[name].grad
        batched_gradient = batched_values[name].grad
        assert reference_gradient is not None
        assert batched_gradient is not None
        torch.testing.assert_close(
            reference_gradient,
            batched_gradient,
            rtol=RTOL,
            atol=ATOL,
        )
