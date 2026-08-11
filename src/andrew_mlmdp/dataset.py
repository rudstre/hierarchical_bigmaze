"""Dataset-level aggregation for discrete movement likelihoods."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from andrew_mlmdp.maze import Coordinate

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy import HierarchyTemplate
    from andrew_mlmdp.lmdp import LMDPEnvironment, ModelParameters


@dataclass(frozen=True)
class MovementTrial:
    """One independently scored movement trajectory and its goal."""

    session_id: str
    trial_id: int
    goal: Coordinate
    trajectory: tuple[Coordinate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))


@dataclass(frozen=True)
class MovementTrialLikelihood:
    """Likelihood and movement count for one included trial."""

    session_id: str
    trial_id: int
    goal: Coordinate
    number_of_transitions: int
    log_likelihood: float


@dataclass(frozen=True)
class MovementTrialExclusion:
    """One trial that could not be scored by a requested model."""

    session_id: str
    trial_id: int
    goal: Coordinate
    reason: str


@dataclass(frozen=True)
class MovementDatasetLikelihood:
    """Auditable per-trial scores and their dataset-level aggregate."""

    model: Literal["flat", "hierarchical"]
    trial_likelihoods: tuple[MovementTrialLikelihood, ...]
    exclusions: tuple[MovementTrialExclusion, ...]

    @property
    def number_of_scored_trials(self) -> int:
        return len(self.trial_likelihoods)

    @property
    def number_of_excluded_trials(self) -> int:
        return len(self.exclusions)

    @property
    def total_transitions(self) -> int:
        return sum(
            trial.number_of_transitions for trial in self.trial_likelihoods
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


def score_flat_movement_dataset(
    environment: "LMDPEnvironment",
    trials: Iterable[MovementTrial],
    *,
    parameters: "ModelParameters | None" = None,
) -> MovementDatasetLikelihood:
    """Score independent trials under flat policies cached by trial goal."""

    solutions = {}

    def score(trial: MovementTrial) -> float:
        solution = solutions.get(trial.goal)
        if solution is None:
            solution = environment.solve_flat(
                trial.goal,
                parameters=parameters,
            )
            solutions[trial.goal] = solution
        return solution.movement_log_likelihood(trial.trajectory)

    return _score_movement_dataset("flat", trials, score)


def score_hierarchical_movement_dataset(
    template: "HierarchyTemplate",
    trials: Iterable[MovementTrial],
    *,
    beta: float | None = None,
) -> MovementDatasetLikelihood:
    """Score independent trials with hierarchy tasks cached by trial goal."""

    def score(trial: MovementTrial) -> float:
        task = template.for_goal(trial.goal)
        return task.movement_log_likelihood(trial.trajectory, beta=beta)

    return _score_movement_dataset("hierarchical", trials, score)


def _score_movement_dataset(
    model: Literal["flat", "hierarchical"],
    trials: Iterable[MovementTrial],
    score: Callable[[MovementTrial], float],
) -> MovementDatasetLikelihood:
    likelihoods = []
    exclusions = []
    for trial in trials:
        try:
            log_likelihood = score(trial)
        except ValueError as error:
            exclusions.append(
                MovementTrialExclusion(
                    session_id=trial.session_id,
                    trial_id=trial.trial_id,
                    goal=trial.goal,
                    reason=str(error),
                )
            )
            continue

        likelihoods.append(
            MovementTrialLikelihood(
                session_id=trial.session_id,
                trial_id=trial.trial_id,
                goal=trial.goal,
                number_of_transitions=_number_of_movement_transitions(
                    trial.trajectory
                ),
                log_likelihood=float(log_likelihood),
            )
        )

    return MovementDatasetLikelihood(
        model=model,
        trial_likelihoods=tuple(likelihoods),
        exclusions=tuple(exclusions),
    )


def _number_of_movement_transitions(
    trajectory: tuple[Coordinate, ...],
) -> int:
    return sum(
        current != following
        for current, following in zip(trajectory, trajectory[1:])
    )
