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
    """Score independent trials under flat policies cached by trial goal."""

    solutions = {}

    def score(trial: Trial) -> float:
        solution = solutions.get(trial.goal)
        if solution is None:
            solution = environment.solve(
                trial.goal,
                parameters=parameters,
            )
            solutions[trial.goal] = solution
        return solution.log_likelihood(trial.trajectory)

    return _score_movement_dataset("flat", trials, score)


def score_hierarchy_dataset(
    template: "Template",
    trials: Iterable[Trial],
    *,
    beta: float | None = None,
) -> DatasetScore:
    """Score independent trials with hierarchy tasks cached by trial goal."""

    def score(trial: Trial) -> float:
        task = template.task(trial.goal)
        return task.log_likelihood(trial.trajectory, beta=beta)

    return _score_movement_dataset("hierarchical", trials, score)


def _score_movement_dataset(
    model: Literal["flat", "hierarchical"],
    trials: Iterable[Trial],
    score: Callable[[Trial], float],
) -> DatasetScore:
    likelihoods = []
    exclusions = []
    for trial in trials:
        try:
            log_likelihood = score(trial)
        except ValueError as error:
            exclusions.append(
                ExcludedTrial(
                    session_id=trial.session_id,
                    trial_id=trial.trial_id,
                    goal=trial.goal,
                    reason=str(error),
                )
            )
            continue

        likelihoods.append(
            TrialScore(
                session_id=trial.session_id,
                trial_id=trial.trial_id,
                goal=trial.goal,
                n_transitions=_movement_count(
                    trial.trajectory
                ),
                log_likelihood=float(log_likelihood),
            )
        )

    return DatasetScore(
        model=model,
        trial_likelihoods=tuple(likelihoods),
        exclusions=tuple(exclusions),
    )


def _movement_count(
    trajectory: tuple[Coordinate, ...],
) -> int:
    return sum(
        current != following
        for current, following in zip(trajectory, trajectory[1:])
    )
