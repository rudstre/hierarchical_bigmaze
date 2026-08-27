import json

import numpy as np
import pytest
from test_rank_validation import _config, _context, _successful_shard

import andrew_mlmdp.validation_aggregation as aggregation


def test_aggregate_writes_auditable_diagnostic_plots(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    context = _context(config)
    monkeypatch.setattr(aggregation, "_load_problem_context", lambda _: context)
    shards = tmp_path / "shards"
    shards.mkdir()
    successful = _successful_shard(config, context.compatibility, 2, -4.0)
    downstream_failure = _successful_shard(
        config,
        context.compatibility,
        3,
        -5.0,
    )
    downstream_failure.update(
        {
            "status": "failure",
            "stage": "fit_adam",
            "failure": {"type": "RuntimeError", "message": "test failure"},
        }
    )
    (shards / "k_02.json").write_text(json.dumps(successful))
    (shards / "k_03.json").write_text(json.dumps(downstream_failure))

    output = tmp_path / "aggregate"
    result = aggregation.aggregate_rank_results(config, shards, output)
    rows = result["summary_rows"]

    held_out = aggregation._numeric_series(
        rows,
        "validation_ll_per_transition",
    )
    fitted_training = aggregation._numeric_series(
        rows,
        "training_fitted_ll_per_transition",
    )
    normalized_kl = aggregation._numeric_series(
        rows,
        "nmf_reconstruction_error",
        require_success=False,
    )
    fitted_alpha = aggregation._numeric_series(rows, "best_alpha")
    fitted_threshold = aggregation._numeric_series(
        rows,
        "best_core_threshold_fraction",
    )

    assert held_out[0] == pytest.approx(-2.0)
    assert fitted_training[0] == pytest.approx(-1.5)
    assert normalized_kl[0] == pytest.approx(0.2)
    assert normalized_kl[1] == pytest.approx(0.2)
    assert rows[1]["nmf_selected_restart"] == 11
    assert rows[1]["nmf_selected_seed"] == 11
    assert rows[1]["nmf_reconstruction_error"] == pytest.approx(0.2)
    assert fitted_alpha[0] == pytest.approx(0.8)
    assert fitted_threshold[0] == pytest.approx(0.7 / 0.9)
    assert np.isnan(fitted_alpha[1])
    assert np.isnan(fitted_threshold[1])

    expected = (
        "held_out_log_likelihood_vs_k",
        "selected_nmf_normalized_kl_vs_k",
        "fitted_parameters_vs_k",
    )
    for stem in expected:
        assert (output / f"{stem}.png").is_file()
        assert (output / f"{stem}.svg").is_file()

    parameter_svg = (output / "fitted_parameters_vs_k.svg").read_text()
    assert "Lower control cost" in parameter_svg
    assert "Upper control cost" in parameter_svg
    assert "Alpha" in parameter_svg
    assert "Beta" in parameter_svg
    assert "Core threshold / structural cap" in parameter_svg
    assert "Core exponent" in parameter_svg


def test_numeric_series_turns_missing_nonfinite_and_failed_values_into_gaps():
    rows = [
        {"status": "success", "value": 1.0},
        {"status": "success", "value": float("inf")},
        {"status": "success", "value": None},
        {"status": "failure", "value": 2.0},
    ]

    successful = aggregation._numeric_series(rows, "value")
    all_statuses = aggregation._numeric_series(
        rows,
        "value",
        require_success=False,
    )

    np.testing.assert_allclose(successful[:1], [1.0])
    assert np.isnan(successful[1:]).all()
    np.testing.assert_allclose(all_statuses[[0, 3]], [1.0, 2.0])
    assert np.isnan(all_statuses[1:3]).all()
