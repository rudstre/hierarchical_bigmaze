"""Reusable fitted-MLMDP helpers for research workflows."""

from __future__ import annotations

from andrew_mlmdp.validation import (
    RankValidationError,
    _fit_result_payload,
    _fitted_template,
    _initial_template,
    _strict_score,
)


def fit_explicit_split(
    dataset,
    profiles,
    config,
    k,
    training_trials,
    validation_trials=None,
):
    """Fit one rank on explicit trials and optionally score validation trials."""
    from andrew_mlmdp.lmdp import Environment

    environment = Environment(dataset.definition.maze)
    initial, threshold_range, initial_values = _initial_template(
        environment,
        profiles,
        config,
        k,
        {trial.goal for trial in training_trials},
    )
    initial_score = _strict_score(initial, training_trials)
    adam = config.adam
    fit_result = initial.fit(
        training_trials,
        names=adam.fitted_names,
        lr=adam.learning_rate,
        max_steps=adam.max_steps,
        tolerance=adam.convergence_tolerance,
        convergence_tolerance=adam.convergence_tolerance,
        scheduler_tolerance=adam.scheduler_tolerance,
        patience=adam.patience,
        lr_decay=adam.lr_decay,
        lr_patience=adam.lr_patience,
        min_lr=adam.min_lr,
    )
    if fit_result.best_values is None:
        raise RankValidationError(
            f"ADAM found no finite parameter state ({fit_result.reason})"
        )
    if not fit_result.converged:
        raise RankValidationError(f"ADAM did not converge ({fit_result.reason})")
    best_values = dict(fit_result.best_values.as_floats())
    fitted_template = _fitted_template(
        environment,
        profiles,
        config,
        k,
        best_values,
        initial,
    )
    return {
        "optimizer": {
            "initial_values": initial_values,
            "threshold_domain": {
                "maximum": threshold_range.maximum,
                "limiting_pairs": [
                    {"goal": list(goal), "subgoal": subgoal}
                    for goal, subgoal in threshold_range.limiting_pairs
                ],
            },
            "fit_result": _fit_result_payload(
                fit_result,
                threshold_cap=threshold_range.maximum,
            ),
        },
        "training": {
            "initial": initial_score,
            "fitted": _strict_score(fitted_template, training_trials),
        },
        "validation": (
            None
            if validation_trials is None
            else _strict_score(fitted_template, validation_trials)
        ),
        "_template": fitted_template,
    }


def trials_for_sessions(dataset, session_ids):
    """Return all valid trials from exactly the supplied typed sessions."""
    selected = set(session_ids)
    trials = tuple(trial for trial in dataset.trials if trial.session_id in selected)
    if not trials:
        raise ValueError(f"No valid trials for sessions {sorted(selected)}")
    return trials


def qin_maze_id(maze_name):
    """Map the supported Doohan maze names to Qin's integer identities."""
    try:
        return {"maze_1": 1, "maze_2": 2}[maze_name]
    except KeyError as error:
        raise ValueError(f"Unsupported Qin maze {maze_name!r}") from error
