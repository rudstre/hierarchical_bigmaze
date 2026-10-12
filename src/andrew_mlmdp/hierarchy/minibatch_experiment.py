"""Private, fixed-schedule minibatch Adam pilot for hierarchical likelihoods.

This module is deliberately separate from the production fit API. It compares
batch sizes at the same number of training-data passes without using finite
validation scores to choose an optimizer state or a learning-rate change.
An unscorable required split fails the run explicitly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.hierarchy.equations import (
    NumericalError,
    fittable_parameters,
    parameter_values,
    required_parameters,
)
from andrew_mlmdp.hierarchy.fitting import _RawParameters
from andrew_mlmdp.hierarchy.likelihood import (
    PreparedBatch,
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.model import Template

PILOT_EPOCHS = 40
PILOT_LEARNING_RATES = ((0, 0.05), (20, 0.015), (35, 0.0045))
PILOT_CHECKPOINTS = (0, 5, 10, 20, 30, 35, 40)


class PilotFitError(RuntimeError):
    """A required trial or optimization step could not be scored finitely."""


@dataclass(frozen=True)
class PilotCheckpoint:
    passes: int
    updates: int
    learning_rate: float
    training_total_log_likelihood: float
    validation_total_log_likelihood: float
    training_log_likelihood_per_transition: float
    validation_log_likelihood_per_transition: float
    parameter_values: dict[str, float]


@dataclass(frozen=True)
class PilotFitResult:
    batch_size: int
    shuffle_seed: int | None
    order_sha256: str | None
    updates: int
    checkpoints: tuple[PilotCheckpoint, ...]
    final_values: dict[str, float]


@dataclass(frozen=True)
class PilotProgress:
    """Progress after one complete pass through the training trials."""

    passes: int
    total_passes: int
    updates: int
    learning_rate: float
    elapsed_seconds: float
    checkpoint: PilotCheckpoint | None


def epoch_order(n_trials: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return one permutation, or the original order for a full batch."""

    if n_trials < 1 or batch_size < 1:
        raise ValueError("n_trials and batch_size must be positive")
    return rng.permutation(n_trials) if batch_size < n_trials else np.arange(n_trials)


def fit_minibatch_pilot(
    template: Template,
    training_trials: Sequence[Trial],
    validation_trials: Sequence[Trial],
    *,
    names: Sequence[str],
    batch_size: int,
    shuffle_seed: int | None,
    epochs: int = PILOT_EPOCHS,
    learning_rates: tuple[tuple[int, float], ...] = PILOT_LEARNING_RATES,
    checkpoint_epochs: tuple[int, ...] = PILOT_CHECKPOINTS,
    progress_callback: Callable[[PilotProgress], None] | None = None,
) -> PilotFitResult:
    """Fit one final iterate and record exact full-set checkpoint scores.

    Each batch optimizes ``-N / b * sum(log p(trial))``. A shuffled epoch uses
    every training trial exactly once, including its potentially smaller last
    batch. Adam state persists for the whole run. Finite validation scores are
    reported without changing updates, learning rates, or final-iterate selection.
    A nonfinite required score fails the run rather than dropping trials.
    """

    training = tuple(training_trials)
    validation = tuple(validation_trials)
    selected = tuple(names)
    if not training or not validation:
        raise ValueError("Training and validation each require at least one trial")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("names must be nonempty and unique")
    inactive = set(selected) - set(fittable_parameters(template))
    if inactive:
        raise ValueError("Parameters cannot be fitted: " + ", ".join(sorted(inactive)))
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if batch_size < len(training) and (
        isinstance(shuffle_seed, bool)
        or not isinstance(shuffle_seed, int)
        or shuffle_seed < 0
    ):
        raise ValueError("A minibatch fit requires a non-negative integer seed")
    if (
        not learning_rates
        or learning_rates[0][0] != 0
        or any(
            not 0 <= start < epochs or not np.isfinite(rate) or rate <= 0.0
            for start, rate in learning_rates
        )
        or any(
            first[0] >= second[0]
            for first, second in zip(learning_rates, learning_rates[1:])
        )
    ):
        raise ValueError("learning_rates must start at epoch 0 and increase by epoch")
    if (
        not checkpoint_epochs
        or checkpoint_epochs[0] != 0
        or checkpoint_epochs[-1] != epochs
        or any(not 0 <= epoch <= epochs for epoch in checkpoint_epochs)
        or any(a >= b for a, b in zip(checkpoint_epochs, checkpoint_epochs[1:]))
    ):
        raise ValueError("checkpoints must increase from 0 through epochs")
    training_keys = {(trial.session_id, trial.trial_id) for trial in training}
    validation_keys = {(trial.session_id, trial.trial_id) for trial in validation}
    if len(training_keys) != len(training) or len(validation_keys) != len(validation):
        raise ValueError("Trial session/id keys must be unique within each split")
    if training_keys & validation_keys:
        raise ValueError("Training and validation trials must be disjoint")

    initial_values = {
        name: value.detach() for name, value in parameter_values(template).items()
    }
    threshold_max = None
    if "core_threshold" in required_parameters(template):
        goals = tuple(dict.fromkeys(trial.goal for trial in training))
        threshold_max = template.validate_threshold(
            initial_values["core_threshold"], goals
        ).maximum
    raw_parameters = _RawParameters(
        initial_values, selected, threshold_max=threshold_max
    )
    prepared_training = prepare_batch(template, training)
    prepared_validation = prepare_batch(template, validation)
    training_transitions = _transition_count(training)
    validation_transitions = _transition_count(validation)
    if training_transitions == 0 or validation_transitions == 0:
        raise ValueError("Both splits require at least one movement transition")

    optimizer = torch.optim.Adam(raw_parameters.parameters(), lr=learning_rates[0][1])
    generator = np.random.default_rng(0 if shuffle_seed is None else shuffle_seed)
    order_digest = hashlib.sha256()
    updates = 0
    checkpoints: list[PilotCheckpoint] = []
    started = perf_counter()

    def record(passes: int, rate: float) -> None:
        values = raw_parameters.physical_values(initial_values)
        with torch.no_grad():
            training_score = _score(
                template, prepared_training, values, split="training", passes=passes
            )
            validation_score = _score(
                template, prepared_validation, values, split="validation", passes=passes
            )
        checkpoints.append(
            PilotCheckpoint(
                passes=passes,
                updates=updates,
                learning_rate=rate,
                training_total_log_likelihood=training_score,
                validation_total_log_likelihood=validation_score,
                training_log_likelihood_per_transition=(
                    training_score / training_transitions
                ),
                validation_log_likelihood_per_transition=(
                    validation_score / validation_transitions
                ),
                parameter_values={
                    name: float(value.detach().cpu()) for name, value in values.items()
                },
            )
        )

    record(0, learning_rates[0][1])
    if progress_callback is not None:
        progress_callback(
            PilotProgress(
                passes=0,
                total_passes=epochs,
                updates=updates,
                learning_rate=learning_rates[0][1],
                elapsed_seconds=perf_counter() - started,
                checkpoint=checkpoints[-1],
            )
        )
    rate = learning_rates[0][1]
    for epoch in range(epochs):
        rate = next(
            candidate_rate
            for start, candidate_rate in reversed(learning_rates)
            if start <= epoch
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        order = epoch_order(len(training), batch_size, generator)
        if batch_size < len(training):
            order_digest.update(order.astype(np.int64, copy=False).tobytes())
        for offset in range(0, len(training), batch_size):
            indices = order[offset : offset + batch_size]
            batch = tuple(training[int(index)] for index in indices)
            prepared_batch = prepare_batch(template, batch)
            optimizer.zero_grad(set_to_none=True)
            values = raw_parameters.physical_values(initial_values)
            batch_number = offset // batch_size + 1
            try:
                log_likelihood = total_prepared_log_likelihood(
                    template, prepared_batch, parameter_values=values
                )
            except (NumericalError, torch.linalg.LinAlgError) as error:
                raise PilotFitError(
                    f"Numerical failure at epoch {epoch + 1}, batch {batch_number}"
                ) from error
            loss = -(len(training) / len(batch)) * log_likelihood
            if not bool(torch.isfinite(loss)):
                raise PilotFitError(
                    f"Nonfinite batch loss at epoch {epoch + 1}, batch {batch_number}"
                )
            try:
                loss.backward()
            except RuntimeError as error:
                raise PilotFitError(
                    f"Backward failure at epoch {epoch + 1}, batch {batch_number}"
                ) from error
            if any(
                raw_parameters.raw[name].grad is None
                or not bool(torch.isfinite(raw_parameters.raw[name].grad))
                for name in selected
            ):
                raise PilotFitError(
                    f"Nonfinite batch gradient at epoch {epoch + 1}, "
                    f"batch {batch_number}"
                )
            optimizer.step()
            updates += 1
        checkpoint = None
        if epoch + 1 in checkpoint_epochs:
            record(epoch + 1, rate)
            checkpoint = checkpoints[-1]
        if progress_callback is not None:
            progress_callback(
                PilotProgress(
                    passes=epoch + 1,
                    total_passes=epochs,
                    updates=updates,
                    learning_rate=rate,
                    elapsed_seconds=perf_counter() - started,
                    checkpoint=checkpoint,
                )
            )

    final_values = dict(checkpoints[-1].parameter_values)
    return PilotFitResult(
        batch_size=batch_size,
        shuffle_seed=shuffle_seed if batch_size < len(training) else None,
        order_sha256=(order_digest.hexdigest() if batch_size < len(training) else None),
        updates=updates,
        checkpoints=tuple(checkpoints),
        final_values=final_values,
    )


def _score(
    template: Template,
    prepared: PreparedBatch,
    values: Mapping[str, Tensor],
    *,
    split: str,
    passes: int,
) -> float:
    try:
        likelihood = total_prepared_log_likelihood(
            template, prepared, parameter_values=values
        )
    except (NumericalError, torch.linalg.LinAlgError) as error:
        raise PilotFitError(
            f"Numerical failure on {split} after {passes} passes"
        ) from error
    if not bool(torch.isfinite(likelihood)):
        raise PilotFitError(f"Nonfinite {split} score after {passes} passes")
    return float(likelihood.detach().cpu())


def _transition_count(trials: Sequence[Trial]) -> int:
    return sum(
        current != following
        for trial in trials
        for current, following in zip(trial.trajectory, trial.trajectory[1:])
    )
