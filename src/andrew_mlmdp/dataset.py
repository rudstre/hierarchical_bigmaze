"""Dataset-level aggregation for discrete movement likelihoods."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from andrew_mlmdp.maze import Coordinate

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy import Template
    from andrew_mlmdp.lmdp import Environment, Parameters


@dataclass(frozen=True)
class Trial:
    """One independently scored movement trajectory and its goal."""

    session_id: str
    trial_id: int
    goal: Coordinate
    trajectory: tuple[Coordinate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))


@dataclass(frozen=True)
class TrialScore:
    """Likelihood and movement count for one included trial."""

    session_id: str
    trial_id: int
    goal: Coordinate
    n_transitions: int
    log_likelihood: float


@dataclass(frozen=True)
class ExcludedTrial:
    """One trial that could not be scored by a requested model."""

    session_id: str
    trial_id: int
    goal: Coordinate
    reason: str


@dataclass(frozen=True)
class DatasetScore:
    """Auditable per-trial scores and their dataset-level aggregate."""

    model: Literal["flat", "hierarchical"]
    trial_likelihoods: tuple[TrialScore, ...]
    exclusions: tuple[ExcludedTrial, ...]

    @property
    def n_scored(self) -> int:
        return len(self.trial_likelihoods)

    @property
    def n_excluded(self) -> int:
        return len(self.exclusions)

    @property
    def total_transitions(self) -> int:
        return sum(
            trial.n_transitions for trial in self.trial_likelihoods
        )

    @property
    def total_log_likelihood(self) -> float:
        return float(
            sum(trial.log_likelihood for trial in self.trial_likelihoods)
        )

    @property
    def mean_log_likelihood_per_transition(self) -> float | None:
        if self.total_transitions == 0:
            return None
        return self.total_log_likelihood / self.total_transitions


def score_flat_dataset(
    environment: "Environment",
    trials: Iterable[Trial],
    *,
    parameters: "Parameters | None" = None,
) -> DatasetScore:
    """Score independent trials together through the prepared flat engine."""

    from andrew_mlmdp.flat_likelihood import trial_log_likelihoods
    from andrew_mlmdp.lmdp import Parameters

    materialized = tuple(trials)
    valid, exclusions = _validated_trials(
        materialized,
        validate=lambda trial: _validate_trial(environment.maze, trial),
    )
    if parameters is None:
        parameters = Parameters()
    scores = trial_log_likelihoods(
        environment,
        valid,
        parameters=parameters,
    )
    return _dataset_score(
        "flat",
        valid,
        scores.detach().cpu().tolist(),
        exclusions,
    )


def score_hierarchy_dataset(
    template: "Template",
    trials: Iterable[Trial],
    *,
    beta: float | None = None,
) -> DatasetScore:
    """Score independent trials together through the prepared hierarchy."""

    from andrew_mlmdp.hierarchy.likelihood import trial_log_likelihoods

    materialized = tuple(trials)

    def validate(trial: Trial) -> None:
        _validate_trial(template.maze, trial)
        if (
            template.basis.locations is not None
            and trial.goal in template.basis.locations
        ):
            raise ValueError("The goal and point subgoals must be disjoint")
        if template.basis.core_threshold is not None:
            template.validate_threshold(
                template.basis.core_threshold,
                (trial.goal,),
            )

    valid, exclusions = _validated_trials(materialized, validate=validate)
    overrides = None
    if beta is not None:
        overrides = {"beta": template.parameters.beta.new_tensor(beta)}
    values = template.parameter_values(overrides=overrides)
    scores = trial_log_likelihoods(
        template,
        valid,
        parameter_values=values,
    )
    return _dataset_score(
        "hierarchical",
        valid,
        scores.detach().cpu().tolist(),
        exclusions,
    )


def _validated_trials(
    trials: tuple[Trial, ...],
    *,
    validate: Callable[[Trial], None],
) -> tuple[tuple[Trial, ...], tuple[ExcludedTrial, ...]]:
    valid = []
    exclusions = []
    for trial in trials:
        try:
            validate(trial)
        except ValueError as error:
            exclusions.append(
                ExcludedTrial(
                    session_id=trial.session_id,
                    trial_id=trial.trial_id,
                    goal=trial.goal,
                    reason=str(error),
                )
            )
        else:
            valid.append(trial)
    return tuple(valid), tuple(exclusions)


def _validate_trial(maze, trial: Trial) -> None:
    if not trial.trajectory:
        raise ValueError("Trajectory must contain at least one coordinate")
    maze.state_index(trial.goal)
    for coordinate in trial.trajectory:
        maze.state_index(coordinate)


def _dataset_score(
    model: Literal["flat", "hierarchical"],
    trials: tuple[Trial, ...],
    values: list[float],
    exclusions: tuple[ExcludedTrial, ...],
) -> DatasetScore:
    likelihoods = tuple(
        TrialScore(
            session_id=trial.session_id,
            trial_id=trial.trial_id,
            goal=trial.goal,
            n_transitions=_movement_count(trial.trajectory),
            log_likelihood=float(value),
        )
        for trial, value in zip(trials, values, strict=True)
    )
    return DatasetScore(
        model=model,
        trial_likelihoods=likelihoods,
        exclusions=exclusions,
    )



def _movement_count(
    trajectory: tuple[Coordinate, ...],
) -> int:
    return sum(
        current != following
        for current, following in zip(trajectory, trajectory[1:])
    )
