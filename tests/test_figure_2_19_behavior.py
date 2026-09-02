import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from doohan_data_interaction import (  # noqa: E402
    reproduce_figure_2_19_behavior as figure_2_19,
)

N_PREDICTORS = len(figure_2_19.FIGURE_REGRESSORS)


def _result(records):
    return {
        "format": "canonical_fold_results_v1",
        "maze_id": 1,
        "regressors": list(figure_2_19.FIGURE_REGRESSORS),
        "random_seed": 0,
        "pca_configuration": {"alpha": 0.1, "n_components": 3},
        "data_signature": "synthetic-signature",
        "fold_results": records,
    }


def _complete_fold(subject, fold_index, validation_session, delta):
    full = 0.20
    # models: [drop intercept, drop predictor 0..n-1, full]; predictor j loses
    # delta + 0.01 * j held-out mean NLL relative to the full model.
    losses = torch.tensor(
        [0.9] + [full + delta + 0.01 * j for j in range(N_PREDICTORS)] + [full]
    )
    return {
        "subject_id": subject,
        "fold_index": fold_index,
        "training_session_ids": (f"{subject}/train",),
        "validation_session_id": validation_session,
        "route_training_session_ids": (f"{subject}/route",),
        "status": "complete",
        "reason": None,
        "n_training_decisions": 20,
        "n_validation_decisions": 18,
        "neg_log_likelihoods": losses,
        "accuracies": torch.linspace(0.1, 0.9, N_PREDICTORS + 2),
    }


def test_build_fold_table_indexes_reduced_models_and_keeps_failures():
    complete = _complete_fold("m2", 0, "m2/s1", 0.03)
    failed = {
        "subject_id": "m2",
        "fold_index": 1,
        "training_session_ids": ("m2/s1",),
        "validation_session_id": "m2/s2",
        "route_training_session_ids": ("m2/s0",),
        "status": "failed",
        "reason": "optimizer failed",
    }
    order = {("m2", "m2/s1"): 2, ("m2", "m2/s2"): 3}

    table = figure_2_19.build_fold_table(_result([complete, failed]), order)

    assert len(table) == 2 * N_PREDICTORS
    first = figure_2_19.FIGURE_REGRESSORS[0]
    last = figure_2_19.FIGURE_REGRESSORS[-1]
    complete_rows = table[table.status == "complete"]
    assert complete_rows.loc[complete_rows.predictor == first, "delta_mean_nll"].iloc[
        0
    ] == pytest.approx(0.03)
    assert complete_rows.loc[complete_rows.predictor == last, "delta_mean_nll"].iloc[
        0
    ] == pytest.approx(0.03 + 0.01 * (N_PREDICTORS - 1))
    assert complete_rows.validation_session_order.eq(2).all()
    assert (table.status == "failed").sum() == N_PREDICTORS
    assert table.loc[table.status == "failed", "delta_mean_nll"].isna().all()


def test_build_fold_table_rejects_unexpected_loss_shape():
    fold = _complete_fold("m2", 0, "m2/s1", 0.03)
    fold["neg_log_likelihoods"] = torch.zeros(3)
    with pytest.raises(ValueError, match="expected"):
        figure_2_19.build_fold_table(_result([fold]), {})


def test_summary_averages_within_animal_then_across_animals():
    records = [
        _complete_fold("A", 0, "A/s1", 0.10),
        _complete_fold("A", 1, "A/s2", 0.30),
        _complete_fold("B", 0, "B/s1", 0.60),
        {
            "subject_id": "C",
            "fold_index": None,
            "status": "unavailable",
            "reason": "fewer than two sessions",
        },
    ]
    order = {("A", "A/s1"): 2, ("A", "A/s2"): 3, ("B", "B/s1"): 2}
    table = figure_2_19.build_fold_table(_result(records), order)

    per_animal = figure_2_19.animal_means(table)
    group = figure_2_19.summarise_predictability(table, ["A", "B", "C"])

    predictor = figure_2_19.FIGURE_REGRESSORS[0]
    animal_a = per_animal[
        (per_animal.subject_id == "A") & (per_animal.predictor == predictor)
    ].iloc[0]
    row = group[group.predictor == predictor].iloc[0]
    assert animal_a.mean_delta_mean_nll == pytest.approx(0.20)
    assert row.mean_delta_mean_nll == pytest.approx((0.20 + 0.60) / 2)
    assert row.n_animals == 2
    assert row.n_animals_selected == 3


def test_write_outputs_publishes_grid_summary_provenance_and_figure(tmp_path):
    result = _result([_complete_fold("A", 0, "A/s1", 0.03)])
    table = figure_2_19.build_fold_table(result, {("A", "A/s1"): 2})
    group = figure_2_19.summarise_predictability(table, ["A"])

    paths = figure_2_19.write_outputs(
        tmp_path,
        regression_result=result,
        fold_table=table,
        group_summary=group,
        subject_ids=["A"],
        maze_id=1,
        provenance={"tag": "test"},
    )

    assert set(paths) == set(figure_2_19.OUTPUT_FILES)
    assert all(path.is_file() and path.stat().st_size for path in paths.values())
    folds = pd.read_csv(paths["folds"])
    assert len(folds) == N_PREDICTORS
    assert folds.delta_mean_nll.notna().all()
    provenance = json.loads(paths["provenance"].read_text())
    assert provenance["tag"] == "test"
    assert provenance["fold_status_counts"]["complete"] == N_PREDICTORS
    saved = torch.load(paths["regression"], weights_only=False)
    assert saved["data_signature"] == "synthetic-signature"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        figure_2_19.write_outputs(
            tmp_path,
            regression_result=result,
            fold_table=table,
            group_summary=group,
            subject_ids=["A"],
            maze_id=1,
            provenance={},
        )


def test_write_outputs_publishes_nothing_when_the_figure_fails(tmp_path, monkeypatch):
    result = _result([_complete_fold("A", 0, "A/s1", 0.03)])
    table = figure_2_19.build_fold_table(result, {("A", "A/s1"): 2})
    group = figure_2_19.summarise_predictability(table, ["A"])

    def fail(*_args, **_kwargs):
        raise RuntimeError("plot failed deliberately")

    monkeypatch.setattr(figure_2_19, "make_figure", fail)
    output_dir = tmp_path / "outputs"

    with pytest.raises(RuntimeError, match="plot failed deliberately"):
        figure_2_19.write_outputs(
            output_dir,
            regression_result=result,
            fold_table=table,
            group_summary=group,
            subject_ids=["A"],
            maze_id=1,
            provenance={},
        )
    assert not output_dir.exists()


def test_cli_parses_repeated_subject_ids_and_options():
    args = figure_2_19.build_arg_parser().parse_args(
        [
            "--data-root",
            "/data",
            "--output-dir",
            "/out",
            "--subject-id",
            "mouse/opaque-A",
            "--subject-id",
            "cohort:B",
            "--maze-name",
            "maze_2",
            "--random-seed",
            "7",
            "--fold-scheme",
            "leave_one_out",
        ]
    )

    assert args.subject_ids == ["mouse/opaque-A", "cohort:B"]
    assert args.maze_name == "maze_2"
    assert args.random_seed == 7
    assert args.fold_scheme == "leave_one_out"
    assert args.overwrite is False


def test_make_figure_runs_for_one_and_several_animals():
    records = [
        _complete_fold("A", 0, "A/s1", 0.10),
        _complete_fold("A", 1, "A/s2", 0.30),
        _complete_fold("B", 0, "B/s1", 0.60),
    ]
    order = {("A", "A/s1"): 2, ("A", "A/s2"): 3, ("B", "B/s1"): 2}
    table = figure_2_19.build_fold_table(_result(records), order)

    for subjects in (["A"], ["A", "B"]):
        group = figure_2_19.summarise_predictability(table, subjects)
        figure = figure_2_19.make_figure(table, group, subject_ids=subjects, maze_id=1)
        assert len(figure.axes) == 2
        figure_2_19.plt.close(figure)


def test_figure_2_20_selects_hmm_routes_and_distinct_outputs():
    result = _result([_complete_fold("A", 0, "A/s1", 0.03)])
    result["regressors"] = list(figure_2_19.FIGURE_2_20_REGRESSORS)
    result["figure_number"] = "2.20"
    table = figure_2_19.build_fold_table(result, {("A", "A/s1"): 2})
    group = figure_2_19.summarise_predictability(table, ["A"])

    assert list(table["predictor"].drop_duplicates()) == list(
        figure_2_19.FIGURE_2_20_REGRESSORS
    )
    assert "hmm_route" in set(group.predictor)
    assert "pca_route" not in set(group.predictor)
    assert figure_2_19.output_files("2.20")["png"] == ("figure_2_20_behavior.png")
    figure = figure_2_19.make_figure(
        table, group, subject_ids=["A"], maze_id=1, figure_number="2.20"
    )
    assert "2.20" in figure._suptitle.get_text()
    figure_2_19.plt.close(figure)


def test_cli_accepts_figure_2_20_hmm_fit_options():
    args = figure_2_19.build_arg_parser().parse_args(
        [
            "--data-root",
            "/data",
            "--output-dir",
            "/out",
            "--figure-number",
            "2.20",
            "--hmm-n-routes",
            "5",
            "--hmm-epochs",
            "12",
        ]
    )

    assert args.figure_number == "2.20"
    assert args.hmm_n_routes == 5
    assert args.hmm_epochs == 12
