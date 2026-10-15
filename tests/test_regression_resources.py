from dataclasses import replace
from pathlib import Path

import pytest

import andrew_mlmdp.regression_workflow as workflow
from andrew_mlmdp.regression_resources import resolve_resource_plan
from andrew_mlmdp.validation import AdamValidationConfig

ROOT = Path(__file__).parents[1]


def _config(*, route_family="pca", routes=5, epochs=500, method="blocked_trial_kfold"):
    settings = (
        {"n_components": routes}
        if route_family == "pca"
        else {"n_routes": routes, "epochs": epochs}
    )
    return workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", ("m2",), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        subgoal_selection=workflow.SubgoalSelectionConfig("session_cv", (4, 6)),
        regression_cv=(
            workflow.RegressionCVConfig(method=method, n_subdivisions=5)
            if method == "blocked_trial_learning_curve"
            else workflow.RegressionCVConfig(method=method, n_splits=5)
        ),
        predictors=workflow.PredictorConfig(
            route_family=route_family,
            **({"pca": settings} if route_family == "pca" else {"hmm": settings}),
        ),
        project_root=ROOT,
    )


def _slurm(**updates):
    result = {
        "partition": "cpu",
        "resource_profile": "auto",
        "stage_overrides": {},
        "max_concurrent": 48,
        "resource_bands": [
            {"rank_range": [4, 4], "memory": "2G", "time": "01:00:00"},
            {"rank_range": [5, 6], "memory": "4G", "time": "04:00:00"},
        ],
    }
    result.update(updates)
    return result


def test_pca_and_learning_curve_profiles_use_stage_specific_resources():
    ordinary = resolve_resource_plan(_config(), _slurm())
    learning = resolve_resource_plan(
        _config(method="blocked_trial_learning_curve"), _slurm()
    )

    assert ordinary["profile"] == "auto-cpu-v1"
    assert ordinary["stages"]["predictor-bundle"]["cpus_per_task"] == 1
    assert ordinary["stages"]["feature-generation"]["time"] == "01:00:00"
    assert ordinary["stages"]["selected-refit"]["memory"] == "4G"
    assert ordinary["candidate_bands"][0]["memory"] == "2G"
    assert learning["stages"]["regression"]["time"] == "04:00:00"


def test_calibrated_fixed_and_automatic_hmm_profiles_request_eight_cpus():
    fixed = resolve_resource_plan(_config(route_family="hmm", routes=10), _slurm())
    automatic = resolve_resource_plan(
        replace(
            _config(route_family="hmm", routes=10),
            predictors=workflow.PredictorConfig(
                route_family="hmm", hmm={"n_routes": "auto", "epochs": 500}
            ),
        ),
        _slurm(),
    )

    assert fixed["calibrated"] and fixed["launchable"]
    assert fixed["route_count_bound"] == 10
    assert fixed["stages"]["predictor-bundle"] == {
        "partition": "cpu",
        "account": None,
        "cpus_per_task": 8,
        "memory": "16G",
        "time": "04:00:00",
        "max_concurrent": 8,
    }
    assert automatic["route_count_bound"] == 7
    assert automatic["launchable"]


def test_general_backfill_profile_uses_available_cpu_capacity():
    plan = resolve_resource_plan(
        _config(route_family="hmm", routes=10),
        _slurm(partition="cpu_backfill", max_concurrent=256),
    )

    assert plan["stages"]["predictor-bundle"]["max_concurrent"] == 96
    assert plan["stages"]["feature-generation"]["max_concurrent"] == 96
    assert plan["stages"]["regression"]["max_concurrent"] == 256


def test_uncalibrated_hmm_requires_complete_explicit_stage_overrides():
    config = _config(route_family="hmm", routes=11, epochs=501)
    blocked = resolve_resource_plan(config, _slurm())
    overrides = {
        stage: {
            "cpus_per_task": 12,
            "memory": "32G",
            "time": "08:00:00",
            "max_concurrent": 4,
        }
        for stage in ("predictor-bundle", "feature-generation")
    }
    explicit = resolve_resource_plan(
        config, _slurm(stage_overrides=overrides)
    )

    assert not blocked["calibrated"]
    assert not blocked["launchable"]
    assert "predictor-bundle" in blocked["launch_block_reason"]
    assert explicit["launchable"]
    assert explicit["stages"]["feature-generation"]["memory"] == "32G"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"unknown": {"memory": "4G"}}, "Unknown.*stages"),
        ({"regression": {"cpus_per_task": 0}}, "positive integer"),
        ({"regression": {"memory": "lots"}}, "SLURM memory"),
        ({"regression": {"time": "tomorrow"}}, "SLURM duration"),
        ({"regression": {"gpus": 1}}, "Unknown.*fields"),
    ],
)
def test_invalid_stage_overrides_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        resolve_resource_plan(_config(), _slurm(stage_overrides=overrides))


@pytest.mark.parametrize(
    ("routes", "epochs", "calibrated", "time"),
    [
        (10, 500, True, "04:00:00"),
        (6, 2000, True, "08:00:00"),
        (7, 2000, False, "08:00:00"),
        (10, 501, False, "08:00:00"),
    ],
)
def test_hmm_calibration_envelopes_and_predictor_time(routes, epochs, calibrated, time):
    plan = resolve_resource_plan(
        _config(route_family="hmm", routes=routes, epochs=epochs), _slurm()
    )

    assert plan["calibrated"] is calibrated
    assert plan["stages"]["predictor-bundle"]["time"] == time


def test_hmm_resources_use_fitting_defaults_when_unspecified():
    config = _config(route_family="hmm")
    config = replace(
        config, predictors=workflow.PredictorConfig(route_family="hmm", hmm={})
    )
    plan = resolve_resource_plan(config, _slurm())

    assert plan["route_count_bound"] == 6
    assert plan["hmm_epochs"] == 2000
    assert plan["calibrated"]
