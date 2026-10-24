import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import andrew_mlmdp.doohan_canonical as canonical
import andrew_mlmdp.regression_behavioral as behavioral
import andrew_mlmdp.regression_execution as execution
import andrew_mlmdp.regression_reporting as reporting
import andrew_mlmdp.regression_workflow as workflow
from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy import simulate_trials
from andrew_mlmdp.hierarchy.behavioral_profiles import (
    CellFit,
    held_parameters,
    local_theta_grid,
    theta_key,
)
from andrew_mlmdp.hierarchy.profile_fitting import access_gate
from andrew_mlmdp.hierarchy.subgoal_search import bump_profiles, maze_distances
from andrew_mlmdp.validation import AdamValidationConfig

GRID = "\n".join(["....."] * 5)
GOALS = ((0, 0), (0, 4), (4, 0), (4, 4))
SESSIONS = ("1", "2", "3")
# Tiny but complete per-cell procedure and a two-cell coarse grid.
TINY = {
    "commitment_radius": 1,
    "search_fit_steps": 8,
    "search_fit_patience": 3,
    "screen_top": 1,
    "n_seeded_starts": 1,
    "max_steps": 10,
    "tolerance": 1e-5,
    "coarse_lower": [0.6],
    "coarse_upper": [1.2],
    "coarse_beta": [1.0, 3.0],
}


def _gated_bumps(maze, centres):
    distances = torch.tensor(maze_distances(maze), dtype=torch.float64)
    states = [maze.state_index(c) for c in centres]
    ungated = bump_profiles(
        distances, states, torch.ones(len(states), dtype=torch.float64), "gaussian"
    )
    return access_gate(ungated, 0.05).numpy(), ungated.numpy()


@pytest.fixture(scope="module")
def simulated():
    maze = Maze.from_ascii(GRID)
    environment = Environment(maze)
    profiles, _ = _gated_bumps(maze, [(2, 2)])
    basis = SubgoalBasis.from_profiles(maze, profiles, core_threshold=None)
    planted = environment.hierarchy(
        basis,
        parameters=Parameters(
            interior_reward=-1.0,
            goal_reward=0.0,
            lower_control_cost=0.5,
            upper_control_cost=1.0,
            alpha=1.0,
            beta=3.0,
        ),
        goal_reward_mode="deferred",
        commitment_mode="both",
        commitment_radius=1,
    )
    rng = np.random.default_rng(0)
    design = []
    for session in SESSIONS:
        for index in range(16):
            goal = GOALS[index % 4]
            candidates = [c for c, d in maze.distances_from(goal).items() if d >= 3]
            design.append((session, candidates[rng.integers(len(candidates))], goal))
    trials = simulate_trials(planted, design, seed=3).trials
    rows = []
    for session_order, session in enumerate(SESSIONS):
        for trial_id in range(3):
            for decision_order in range(2):
                rows.append(
                    {
                        "subject_id": 10,
                        "session_id": session,
                        "session_order": session_order,
                        "trial_id": trial_id,
                        "trial_order": trial_id,
                        "decision_order": decision_order,
                        "timestamp": decision_order,
                        "maze_id": 1,
                        "pos_idx": 1,
                        "reward_idx": 2,
                        "action_class": decision_order,
                        "trial_phase": "navigation",
                        "reward_cos_angle": 1.0,
                        "reward_sin_angle": 0.0,
                    }
                )
    return maze, trials, pd.DataFrame(rows)


def _config(sources, theta_search="coarse"):
    return workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", (10,), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        heldout_sessions="last",
        subgoal_selection=workflow.SubgoalSelectionConfig("session_cv", (2, 3)),
        regression_cv=workflow.RegressionCVConfig(n_splits=2),
        predictors=workflow.PredictorConfig(pca={"n_components": 1}),
        mlmdp=workflow.MLMDPConfig(sources, theta_search, TINY),
        project_root=Path(__file__).parents[1],
    )


def _patch_data(monkeypatch, maze, trials, table):
    def load(config, partition, session_ids):
        selected = set(session_ids)
        return SimpleNamespace(
            trials=tuple(t for t in trials if t.session_id in selected),
            definition=SimpleNamespace(maze=maze),
        )

    def predictions(dataset, template, *, session_ids):
        heldout = table.loc[table["session_id"].isin(session_ids)]
        frame = heldout.loc[
            :, ["subject_id", "session_id", "trial_id", "decision_order"]
        ].copy()
        for action in range(4):
            frame[f"action_{action}"] = 0.25
        return frame

    monkeypatch.setattr(behavioral, "_load_dataset", load)
    monkeypatch.setattr(
        canonical, "hierarchy_to_canonical_action_predictions", predictions
    )


def _patch_downstream(monkeypatch, config, table, counts):
    """Synthetic NMF candidates, route/habit, Qin features and regression."""

    def candidate(
        config, output_dir, partition, *, k, validation_session_id, force=False
    ):
        path = workflow._candidate_path(
            Path(output_dir), partition, k, validation_session_id
        )
        if path.is_file() and not force:
            return workflow._read_json(path)
        counts["nmf"] += 1
        compatibility = workflow.candidate_compatibility(
            config,
            partition,
            k,
            validation_session_id,
            canonical_data_signature=workflow._candidate_input_signatures(
                config, Path(output_dir), k
            )[0],
            discovery_digest=None,
        )
        heldout = table.loc[table["session_id"] == partition.heldout_session_id]
        predictions = heldout.loc[
            :, ("subject_id", "session_id", "trial_id", "decision_order")
        ].copy()
        for action in range(4):
            predictions[f"action_{action}"] = float(action)
        record = {
            "schema_version": workflow.REGRESSION_WORKFLOW_SCHEMA_VERSION,
            "artifact_type": "predictor_candidate",
            "status": "success",
            "compatibility": compatibility,
            "k": k,
            "validation_session_id": validation_session_id,
            "validation_ll_per_transition": -1.0 - 0.01 * k,
            "training": {"fitted": {"total_log_likelihood": -2.0}},
            "discovery": {"digest": "synthetic-discovery"},
            "heldout_predictions": predictions.to_dict("records"),
            "prediction_columns": list(predictions.columns),
        }
        record["artifact_digest"] = workflow._payload_digest(record)
        workflow._atomic_write_json(path, record)
        return record

    def route_training(config, partition):
        training = table.loc[table["session_id"].isin(partition.training_session_ids)]
        return table, training.reset_index(drop=True)

    def fit_route_habit(config, partition, rows, route_settings):
        n_components = route_settings["n_components"]
        return {
            "route": {
                "family": "pca",
                "settings": dict(route_settings),
                "components": [[0.0] * n_components] * 196,
                "explained_variance": [1.0] + [0.0] * (n_components - 1),
                "seed": None,
            },
            "habit_counts": [0.0] * 196,
            "n_training_decisions": len(rows),
            "training_data_digest": "synthetic-training",
            "torch_version": torch.__version__,
        }

    class Processor:
        def __init__(self, *args, regressors, **kwargs):
            counts["regressors"] = list(regressors)

        def process_fold(self, fold, rows, *, external_regressors, **kwargs):
            counts["external"] = sorted(external_regressors)
            n_predictors = len(config.predictors.names)
            design = torch.zeros((len(rows), 4, n_predictors + 2))
            design[..., 0] = 1.0
            for index in range(n_predictors):
                design[..., index + 1] = torch.arange(4) + index
            actions = torch.tensor(rows["action_class"].to_numpy())
            return {"X_t": design, "Y_t": torch.nn.functional.one_hot(actions, 4)}

    class Finder:
        def get_unique_predictability(self, *args):
            n_predictors = args[0].shape[-1] - 1
            n_models = n_predictors + 2
            return {
                "neg_log_likelihoods": torch.linspace(1.2, 1.0, n_models),
                "accuracies": torch.full((n_models,), 0.5),
                "coefs": torch.zeros((n_models, n_predictors + 1)),
                "optimizer_iterations": torch.ones(n_models, dtype=torch.int64),
            }

    monkeypatch.setattr(workflow, "run_candidate_fit", candidate)
    monkeypatch.setattr(execution, "_route_training_table", route_training)
    monkeypatch.setattr(execution, "_fit_route_and_habit", fit_route_habit)
    monkeypatch.setattr(
        execution,
        "_reconstruct_route",
        lambda *args: {"pcs": None, "hmm_model": None, "n_components": 0},
    )
    monkeypatch.setattr(
        execution,
        "_qin_feature_imports",
        lambda root: {"optimal": lambda maze: None, "processor": Processor},
    )
    monkeypatch.setattr(
        execution,
        "_qin_regression_imports",
        lambda root: {"finder": lambda *args: Finder()},
    )
    write_report = reporting.write_regression_report

    def without_plots(*args, **kwargs):
        kwargs["plot"] = False
        return write_report(*args, **kwargs)

    monkeypatch.setattr(reporting, "write_regression_report", without_plots)
    return candidate


def test_config_names_signature_and_method_restrictions():
    legacy = workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", (10,), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
    )
    assert "mlmdp" not in legacy.normalized_payload()
    assert legacy.predictors.names[2] == "hierarchical_mlmdp" and legacy.nmf
    assert "mlmdp_predictors" not in legacy.normalized_payload()["predictors"]

    both = _config(("nmf", "behavioral"))
    # Canonical source order regardless of the configured list order.
    assert both.mlmdp.profile_sources == ("behavioral", "nmf")
    assert both.predictors.names[2:4] == ("behavioral_mlmdp", "hierarchical_mlmdp")
    payload = both.normalized_payload()
    restored = workflow.regression_config_from_payload(
        json.loads(json.dumps(payload)), Path(__file__).parents[1]
    )
    assert restored.signature == both.signature
    assert restored.mlmdp.settings == both.mlmdp.settings

    with pytest.raises(ValueError, match="session_cv or fixed"):
        workflow.RegressionWorkflowConfig(
            dataset=workflow.RegressionDatasetConfig("unused", (10,), "maze_1"),
            discovery_config="unused.json",
            adam=AdamValidationConfig(),
            subgoal_selection=workflow.SubgoalSelectionConfig("training_ll", (2, 3)),
            mlmdp=workflow.MLMDPConfig(("behavioral",)),
        )
    with pytest.raises(ValueError, match="Unknown mlmdp.behavioral"):
        workflow.MLMDPConfig(("behavioral",), behavioral={"no_such_setting": 1})
    with pytest.raises(ValueError, match="profile_sources"):
        workflow.MLMDPConfig(())


@pytest.mark.parametrize("sources", [("behavioral",), ("behavioral", "nmf")])
def test_behavioral_workflow_selects_theta_and_k_and_enters_regression(
    sources, simulated, monkeypatch, tmp_path
):
    maze, trials, table = simulated
    config = _config(sources)
    counts = {"nmf": 0, "cells": 0}
    _patch_data(monkeypatch, maze, trials, table)
    candidate = _patch_downstream(monkeypatch, config, table, counts)
    fit_cell = behavioral.fit_cell

    def counted(*args, **kwargs):
        counts["cells"] += 1
        return fit_cell(*args, **kwargs)

    monkeypatch.setattr(behavioral, "fit_cell", counted)
    result = workflow.run_local_workflow(
        config, tmp_path, table, fit_candidates=candidate
    )
    assert result["summary"]["status"] == "complete"
    # 2 ranks x 2 inner sessions x 2 coarse cells, plus one refit.
    assert counts["cells"] == 9
    partition_dir = next((tmp_path / "partitions").iterdir())
    selection = workflow._read_json(partition_dir / "behavioral" / "selection.json")
    assert selection["status"] == "selected"
    rows = {row["k"]: row for row in selection["theta_rows"]}
    for k, row in rows.items():
        means = {
            cell["theta_cell"]: cell["validation_ll_per_transition_mean"]
            for cell in row["cells"]
        }
        assert row["chosen_theta_cell"] == max(means, key=means.get)
    k_star = selection["selection"]["selected_k"]
    assert selection["selection"]["theta_cell"] == rows[k_star]["chosen_theta_cell"]

    refit = workflow._read_json(partition_dir / "behavioral" / "refit.json")
    assert refit["status"] == "success" and refit["k"] == k_star
    assert refit["compatibility"]["role"] == "refit"
    assert refit["compatibility"]["training_session_ids"] == ["1", "2"]
    assert refit["theta"] == selection["selection"]["theta"]

    bundle = workflow._read_json(partition_dir / "predictor.json")
    expected = [
        workflow.MLMDP_PREDICTORS[source] for source in config.mlmdp.profile_sources
    ]
    assert sorted(bundle["mlmdp_candidates"]) == sorted(expected)
    assert counts["external"] == sorted(expected)
    assert counts["regressors"] == list(config.predictors.names)
    assert (counts["nmf"] > 0) == ("nmf" in sources)
    features = workflow._read_json(partition_dir / "features.json")
    assert features["predictor_names"] == list(config.predictors.names)
    route_only = workflow._read_json(partition_dir / "regression_route_only.json")
    mlmdp_only = workflow._read_json(partition_dir / "regression_mlmdp_only.json")
    assert not set(route_only["predictor_names"]) & workflow.MLMDP_PREDICTOR_NAMES
    assert set(expected) <= set(mlmdp_only["predictor_names"])

    # Rerunning reuses every cell, the selection, and the refit.
    workflow.run_local_workflow(config, tmp_path, table, fit_candidates=candidate)
    assert counts["cells"] == 9


def test_local_grid_refines_theta_at_selected_rank_only(
    simulated, monkeypatch, tmp_path
):
    maze, trials, table = simulated
    config = _config(("behavioral",), "coarse_and_local")
    settings = config.mlmdp.settings
    _patch_data(monkeypatch, maze, trials, table)
    fitted = []

    def fixed_d_cell(environment, trials, goals, k, theta, settings, seed, **kw):
        # Fixed D: cells differ only in theta, so each is scored exactly by
        # the real strict likelihood without the cost of fitting.
        fitted.append((k, tuple(theta)))
        centres = [(1, 1), (3, 3), (1, 3)][:k]
        profiles, ungated = _gated_bumps(maze, centres)
        return CellFit(
            k,
            tuple(theta),
            "success",
            "fixed",
            profiles,
            ungated,
            held_parameters(theta, settings),
            0.0,
            {},
            seed,
        )

    monkeypatch.setattr(behavioral, "fit_cell", fixed_d_cell)
    root = tmp_path
    workflow.write_manifest(config, root, table)
    (partition,), _ = workflow.build_predictor_partitions(table, config)
    for k, session, cell in behavioral.cell_tasks(config, partition):
        behavioral.run_behavioral_cell(
            config,
            root,
            partition,
            k=k,
            validation_session_id=session,
            theta_cell=cell,
        )
    # Local cells are blocked until the coarse selection exists.
    assert (
        behavioral.run_behavioral_local_cell(
            config, root, partition, validation_session_id="1", offset_index=0
        )["status"]
        == "blocked"
    )
    selection = behavioral.aggregate_behavioral_selection(config, root, partition)
    k_star = selection["selection"]["selected_k"]
    centre = tuple(selection["selection"]["theta"])
    tasks = behavioral.local_cell_tasks(config, partition)
    assert len(tasks) == 2 * 26
    for session, index in tasks:
        behavioral.run_behavioral_local_cell(
            config, root, partition, validation_session_id=session, offset_index=index
        )
    assert {k for k, _ in fitted[-len(tasks) :]} == {k_star}
    final = behavioral.aggregate_behavioral_final_selection(config, root, partition)
    assert final["status"] == "selected"
    assert final["selection"]["selected_k"] == k_star
    assert final["selection"]["coarse_theta_cell"] == theta_key(centre)
    (row,) = final["theta_rows"]
    assert len(row["cells"]) == 27
    means = {
        cell["theta_cell"]: cell["validation_ll_per_transition_mean"]
        for cell in row["cells"]
    }
    assert final["selection"]["theta_cell"] == max(means, key=means.get)
    allowed = {theta_key(centre)} | {
        theta_key(t) for t in local_theta_grid(centre, settings.local_step)
    }
    assert set(means) == allowed


def test_legacy_normalization_preserves_exact_historical_signature():
    # Recorded from HEAD before the behavioral change, with a portable root.
    legacy = workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", (10,), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        project_root=Path("/project"),
    )
    assert legacy.signature == (
        "a8a1a544593fcf752efe48805e9eae163cbd54dac4978b64ab5eb8feb4f9d5f2"
    )


def test_fixed_behavioral_rank_still_has_inner_theta_folds(simulated):
    from dataclasses import replace

    config = replace(
        _config(("behavioral",)),
        subgoal_selection=workflow.SubgoalSelectionConfig("fixed", (2, 2)),
    )
    _, _, table = simulated
    (partition,), _ = workflow.build_predictor_partitions(table, config)
    tasks = behavioral.cell_tasks(config, partition)
    assert {k for k, _, _ in tasks} == {2}
    assert {s for _, s, _ in tasks} == set(partition.training_session_ids)
    for k, session, _ in tasks:
        seeds = {
            behavioral.behavioral_compatibility(
                config,
                Path("/tmp/no-behavioral-run"),
                partition,
                role="coarse",
                k=k,
                theta=theta,
                validation_session_id=session,
            )["seed"]
            for theta in behavioral.coarse_cells(config).values()
        }
        assert len(seeds) == 1


def test_behavioral_selection_figures_include_rank_uncertainty_and_local_cells():
    cells = [
        {
            "theta_cell": "centre",
            "theta": [0.6, 1.2, 2.0],
            "eligible": True,
            "validation_ll_per_transition_mean": -0.5,
        }
    ]
    selection = {
        "selection": {"selected_k": 2, "theta": cells[0]["theta"]},
        "rank_rows": [
            {
                "k": 2,
                "validation_ll_per_transition_mean": -0.5,
                "validation_ll_per_transition_se": 0.03,
            }
        ],
        "theta_rows": [{"k": 2, "cells": cells}],
    }
    local = {**cells[0], "theta_cell": "local", "theta": [0.6, 1.2, 3.0]}
    final = {
        "selection": {"selected_k": 2, "theta": local["theta"]},
        "theta_rows": [{"k": 2, "cells": [*cells, local]}],
    }
    curve = reporting.make_behavioral_rank_figure(selection)
    assert list(curve.data[0].error_y.array) == [0.03]
    assert curve.layout.shapes[0].x0 == 2
    landscape = reporting.make_behavioral_theta_figure(selection, final)
    assert list(landscape.data[0].z) == [2.0, 3.0]
    assert list(landscape.data[1].z) == [3.0]


@pytest.mark.parametrize(
    "stage,dependency",
    [
        ("behavioral", "src/andrew_mlmdp/hierarchy/prediction.py"),
        (
            "feature",
            "external/qin_route_model/fixed_maze_analysis/src/regressionhelper/constants.py",
        ),
        (
            "regression",
            "external/qin_route_model/fixed_maze_analysis/src/regressionhelper/constants.py",
        ),
    ],
)
def test_cache_source_covers_predictions_and_external_registration(
    stage, dependency, tmp_path
):
    from dataclasses import replace

    config = _config(("behavioral",))
    recorded = workflow._stage_source(config, stage)
    assert dependency in {record["path"] for record in recorded["files"]}
    for record in recorded["files"]:
        destination = tmp_path / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("original scientific source\n")
    fixture_config = replace(config, project_root=tmp_path)
    before = workflow._stage_source(fixture_config, stage)
    (tmp_path / dependency).write_text("changed scientific source\n")
    after = workflow._stage_source(fixture_config, stage)
    assert before["sha256"] != after["sha256"]
