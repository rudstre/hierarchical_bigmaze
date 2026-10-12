import nbformat
import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis, Trial
from andrew_mlmdp.hierarchy.equations import fittable_parameters, parameter_values
from andrew_mlmdp.hierarchy.fitting import _RawParameters
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.minibatch_experiment import (
    PilotFitError,
    epoch_order,
    fit_minibatch_pilot,
)


def _template():
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
        maze, profiles, core_threshold=0.2, core_exponent=0.7
    )
    return Environment(maze).hierarchy(
        basis,
        parameters=Parameters(
            interior_reward=-1.0,
            goal_reward=0.0,
            lower_control_cost=0.55,
            upper_control_cost=0.9,
            alpha=0.8,
            beta=0.7,
        ),
    )


def _trials():
    return (
        Trial("train", 1, (0, 5), ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        Trial("train", 2, (0, 5), ((0, 2), (0, 3), (0, 4), (0, 5))),
    )


def _validation():
    return (Trial("validation", 1, (0, 5), ((0, 3), (0, 4), (0, 5))),)


def test_full_batch_matches_existing_fit_for_one_update():
    template = _template()
    pilot = fit_minibatch_pilot(
        template,
        _trials(),
        _validation(),
        names=("alpha",),
        batch_size=len(_trials()),
        shuffle_seed=None,
        epochs=1,
        learning_rates=((0, 0.05),),
        checkpoint_epochs=(0, 1),
    )
    reference = template.fit(_trials(), names=("alpha",), lr=0.05, max_steps=1)

    assert pilot.updates == 1
    assert pilot.order_sha256 is None
    assert pilot.checkpoints[0].training_total_log_likelihood == pytest.approx(
        reference.history[0].log_likelihood, abs=1e-12
    )
    assert pilot.checkpoints[1].training_total_log_likelihood == pytest.approx(
        reference.history[1].log_likelihood, abs=1e-12
    )
    for name, value in reference.history[1].parameter_values.items():
        assert pilot.final_values[name] == pytest.approx(value, abs=1e-12)
    assert all(parameter.grad is None for parameter in template.parameters.parameters())


def test_epoch_order_is_complete_and_reproducible():
    first = np.random.default_rng(17)
    repeat = np.random.default_rng(17)
    orders = [epoch_order(7, 3, first) for _ in range(3)]
    replay = [epoch_order(7, 3, repeat) for _ in range(3)]

    for order, same in zip(orders, replay, strict=True):
        assert np.array_equal(order, same)
        assert sorted(order.tolist()) == list(range(7))
        batches = [order[index : index + 3] for index in range(0, 7, 3)]
        assert [len(batch) for batch in batches] == [3, 3, 1]


def test_weighted_minibatch_gradients_average_to_full_gradient():
    template = _template()
    reverse_goal = Trial("train", 3, (0, 0), ((0, 4), (0, 3), (0, 2), (0, 1), (0, 0)))
    trials = (*_trials(), reverse_goal)
    initial = {
        name: value.detach() for name, value in parameter_values(template).items()
    }
    names = fittable_parameters(template)
    threshold_max = template.validate_threshold(
        initial["core_threshold"], {trial.goal for trial in trials}
    ).maximum
    raw = _RawParameters(initial, names, threshold_max=threshold_max)

    def gradient(trials, scale):
        raw.zero_grad(set_to_none=True)
        prepared = prepare_batch(template, trials)
        values = raw.physical_values(initial)
        loss = -scale * total_prepared_log_likelihood(
            template, prepared, parameter_values=values
        )
        loss.backward()
        return {name: raw.raw[name].grad.detach().clone() for name in names}

    full = gradient(trials, 1.0)
    singletons = [gradient((trial,), len(trials)) for trial in trials]
    for name in names:
        singleton_average = torch.stack(
            [gradients[name] for gradients in singletons]
        ).mean()
        torch.testing.assert_close(
            full[name], singleton_average, rtol=1e-10, atol=1e-11
        )


def test_minibatch_fit_replays_seed_and_uses_all_trials():
    kwargs = dict(
        names=("alpha",),
        batch_size=1,
        shuffle_seed=7,
        epochs=2,
        learning_rates=((0, 0.05),),
        checkpoint_epochs=(0, 1, 2),
    )
    first = fit_minibatch_pilot(_template(), _trials(), _validation(), **kwargs)
    repeat = fit_minibatch_pilot(_template(), _trials(), _validation(), **kwargs)

    assert first.updates == 4
    assert first.order_sha256 == repeat.order_sha256
    assert first.final_values == repeat.final_values
    assert first.checkpoints == repeat.checkpoints


def test_progress_callback_reports_every_completed_pass():
    progress = []
    fit_minibatch_pilot(
        _template(),
        _trials(),
        _validation(),
        names=("alpha",),
        batch_size=1,
        shuffle_seed=7,
        epochs=2,
        learning_rates=((0, 0.05),),
        checkpoint_epochs=(0, 2),
        progress_callback=progress.append,
    )

    assert [event.passes for event in progress] == [0, 1, 2]
    assert [event.updates for event in progress] == [0, 2, 4]
    assert progress[0].checkpoint is not None
    assert progress[1].checkpoint is None
    assert progress[2].checkpoint is not None
    assert all(event.total_passes == 2 for event in progress)
    assert all(event.elapsed_seconds >= 0.0 for event in progress)


def test_impossible_training_trial_is_reported_without_filtering():
    impossible = Trial("train", 3, (0, 5), ((0, 5), (0, 4)))
    with pytest.raises(PilotFitError, match="Nonfinite training score after 0 passes"):
        fit_minibatch_pilot(
            _template(),
            (_trials()[0], impossible),
            _validation(),
            names=("alpha",),
            batch_size=1,
            shuffle_seed=1,
            epochs=1,
            learning_rates=((0, 0.05),),
            checkpoint_epochs=(0, 1),
        )


def test_pilot_notebook_compiles_and_fit_cell_is_unexecuted():
    notebook = nbformat.read(
        "doohan_data_interaction/minibatch_adam_pilot.ipynb", as_version=4
    )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for cell in code_cells:
        compile(cell.source, f"pilot notebook cell {cell.id}", "exec")
    fit_cell = next(cell for cell in code_cells if cell.id == "pilot-run-fits")
    assert fit_cell.execution_count is None
    assert fit_cell.outputs == []
