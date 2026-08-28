import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from andrew_mlmdp import (  # noqa: E402
    Environment,
    Maze,
    SubgoalBasis,
    Trial,
    soft_parameters,
)
from doohan_data_interaction.fit_workflow import (  # noqa: E402
    RestartDefaults,
    fit_hierarchy_restarts,
)


def _template():
    maze = Maze.from_ascii("....")
    basis = SubgoalBasis.from_profiles(
        maze,
        np.asarray(
            [
                [1.0, 0.1],
                [0.8, 0.3],
                [0.3, 0.8],
                [0.1, 1.0],
            ]
        ),
        core_threshold=0.2,
        core_exponent=0.7,
    )
    return Environment(maze).hierarchy(
        basis,
        parameters=soft_parameters(2, alpha=0.6),
    )


def _trials():
    return (Trial("session", 1, (0, 3), ((0, 0), (0, 1), (0, 2), (0, 3))),)


def test_notebook_fit_workflow_exposes_defaults_and_reuses_cache(tmp_path):
    defaults = {
        "lr": 0.05,
        "max_steps": 0,
        "tolerance": 1e-4,
        "scheduler_tolerance": 3e-4,
        "convergence_tolerance": 1e-4,
        "patience": 20,
        "lr_decay": 0.3,
        "lr_patience": 7,
        "min_lr": 1e-3,
    }
    first = fit_hierarchy_restarts(
        _template(),
        _trials(),
        names=("alpha", "core_threshold", "core_exponent"),
        initial_values={"alpha": 0.75},
        optimizer_defaults=defaults,
        restart_defaults=RestartDefaults(count=1, seed=123, log_scale=0.45),
        cache_dir=tmp_path,
        progress=False,
    )
    second = fit_hierarchy_restarts(
        _template(),
        _trials(),
        names=("alpha", "core_threshold", "core_exponent"),
        initial_values={"alpha": 0.75},
        optimizer_defaults=defaults,
        restart_defaults=RestartDefaults(count=1, seed=123, log_scale=0.45),
        cache_dir=tmp_path,
        progress=False,
    )

    assert not first.loaded_from_cache
    assert second.loaded_from_cache
    assert first.cache_path == second.cache_path
    assert first.cache_path is not None and first.cache_path.is_file()
    assert first.result.history == second.result.history
    assert first.result.initial_values.as_floats()["alpha"] == pytest.approx(0.75)
    assert first.best_values == second.best_values
    assert first.summary()["fitted_trials"] == 1
    assert {row["parameter"] for row in first.parameter_rows()} == {
        "alpha",
        "core_threshold",
        "core_exponent",
    }


def test_notebook_fit_workflow_rejects_unknown_defaults():
    with pytest.raises(ValueError, match="Unknown optimizer defaults"):
        fit_hierarchy_restarts(
            _template(),
            _trials(),
            names=("alpha",),
            optimizer_defaults={"lr": 0.05, "max_steps": 0, "typo": 1},
            progress=False,
        )


@pytest.mark.parametrize("count", [0, -1])
def test_restart_defaults_require_positive_count(count):
    with pytest.raises(ValueError, match="restart count"):
        RestartDefaults(count=count)
