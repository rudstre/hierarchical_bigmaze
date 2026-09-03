"""Notebook-first research tools for maze-based multitask LMDPs.

Every public name below is resolved lazily (PEP 562): ``import andrew_mlmdp``
itself stays cheap, and only touching a specific name (e.g.
``andrew_mlmdp.Environment``) imports the submodule that defines it -- which
for several names pulls in torch/sklearn. Lightweight callers (config
loading, artifact classification) that only need a handful of these names
never pay for the rest.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "Coordinate",
    "DatasetScore",
    "DoohanDataset",
    "Dynamics",
    "Environment",
    "ExcludedTrial",
    "Exclusion",
    "FitResult",
    "FitStep",
    "GoalTasks",
    "LabeledMaze",
    "Maze",
    "MovementPredictions",
    "NMFConfig",
    "NMFConnectivityConfig",
    "NMFRankResult",
    "NMFRestartResult",
    "NMFStudy",
    "NumericalError",
    "PairEntropy",
    "ParameterValues",
    "Parameters",
    "PassiveMode",
    "Plan",
    "ProfileNormalization",
    "RankDiagnostics",
    "Rollout",
    "RolloutEvent",
    "ScoreReport",
    "SessionRecord",
    "Solution",
    "SubgoalAccess",
    "SubgoalBasis",
    "SubtaskDiscovery",
    "Task",
    "TaskBasis",
    "TaskLibrary",
    "Template",
    "ThresholdRange",
    "Trial",
    "TrialScore",
    "controlled_dynamics",
    "doohan_to_canonical_decisions",
    "hierarchy_to_canonical_action_predictions",
    "desirability_grid",
    "desirability_step",
    "discover_subgoals",
    "fit_parameters",
    "fittable_parameters",
    "load_doohan_maze",
    "log_likelihood",
    "maze_from_labeled_edges",
    "parameter_values",
    "plotting",
    "point_parameters",
    "required_parameters",
    "score_flat_dataset",
    "score_hierarchy_dataset",
    "soft_parameters",
    "solve_first_exit",
    "total_log_likelihood",
]

_LAZY_SUBMODULES = {"plotting": "andrew_mlmdp.plotting"}

_LAZY_ATTRS = (
    {
        name: "andrew_mlmdp.dataset"
        for name in (
            "DatasetScore",
            "ExcludedTrial",
            "Trial",
            "TrialScore",
            "score_flat_dataset",
            "score_hierarchy_dataset",
        )
    }
    | {
        name: "andrew_mlmdp.discovery"
        for name in (
            "GoalTasks",
            "NMFRankResult",
            "NMFRestartResult",
            "NMFStudy",
            "RankDiagnostics",
            "SubtaskDiscovery",
            "discover_subgoals",
        )
    }
    | {
        name: "andrew_mlmdp.discovery_config"
        for name in ("NMFConfig", "NMFConnectivityConfig")
    }
    | {
        name: "andrew_mlmdp.doohan_canonical"
        for name in (
            "doohan_to_canonical_decisions",
            "hierarchy_to_canonical_action_predictions",
        )
    }
    | {
        name: "andrew_mlmdp.doohan_dataset"
        for name in ("DoohanDataset", "Exclusion", "ScoreReport", "SessionRecord")
    }
    | {
        name: "andrew_mlmdp.hierarchy"
        for name in (
            "FitResult",
            "FitStep",
            "MovementPredictions",
            "NumericalError",
            "ParameterValues",
            "Plan",
            "Rollout",
            "RolloutEvent",
            "SubgoalAccess",
            "SubgoalBasis",
            "Task",
            "TaskBasis",
            "TaskLibrary",
            "Template",
            "ThresholdRange",
            "fit_parameters",
            "fittable_parameters",
            "log_likelihood",
            "parameter_values",
            "required_parameters",
            "total_log_likelihood",
        )
    }
    | {
        name: "andrew_mlmdp.labeled_maze"
        for name in ("LabeledMaze", "load_doohan_maze", "maze_from_labeled_edges")
    }
    | {
        name: "andrew_mlmdp.lmdp"
        for name in (
            "Dynamics",
            "Environment",
            "PairEntropy",
            "Parameters",
            "PassiveMode",
            "Solution",
            "controlled_dynamics",
            "desirability_grid",
            "desirability_step",
            "point_parameters",
            "soft_parameters",
            "solve_first_exit",
        )
    }
    | {name: "andrew_mlmdp.maze" for name in ("Coordinate", "Maze")}
    | {
        "ProfileNormalization": "andrew_mlmdp.profiles",
    }
)


def __getattr__(name: str) -> Any:
    submodule_path = _LAZY_SUBMODULES.get(name)
    if submodule_path is not None:
        value: Any = importlib.import_module(submodule_path)
    else:
        module_path = _LAZY_ATTRS.get(name)
        if module_path is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
