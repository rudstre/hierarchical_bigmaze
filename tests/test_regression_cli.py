from pathlib import Path

from andrew_mlmdp.regression_cli import _format_result, parser


def test_cli_defaults_to_concise_output_and_keeps_explicit_json_mode(tmp_path):
    concise = parser().parse_args(
        ["status", "--config", "config.json", "--output-dir", str(tmp_path)]
    )
    machine = parser().parse_args(
        [
            "status",
            "--config",
            "config.json",
            "--output-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert concise.json is False
    assert machine.json is True


def test_cli_report_summary_lists_paths_without_nested_payload(tmp_path):
    result = {
        "status": "incomplete",
        "result": tmp_path / "report" / "results.json",
        "provenance": tmp_path / "report" / "provenance.json",
        "plots": {"html": tmp_path / "report" / "figure.html"},
        "accuracy_plots": {"html": tmp_path / "report" / "accuracy.html"},
        "records": [{"large": list(range(100))}],
    }

    rendered = _format_result("plotting", result, tmp_path)

    assert rendered.splitlines() == [
        "plotting: incomplete",
        f"Results: {tmp_path / 'report' / 'results.json'}",
        f"Provenance: {tmp_path / 'report' / 'provenance.json'}",
        f"Figures: {tmp_path.resolve() / 'report'}",
    ]
    assert "large" not in rendered


def test_cli_status_summary_counts_partition_states(tmp_path):
    rendered = _format_result(
        "status",
        {
            "partitions": [
                {
                    "selection": "selected",
                    "predictor_bundle": "success",
                    "features": "success",
                    "regression": "success",
                    "policy_overlap": "success",
                },
                {
                    "selection": "missing",
                    "predictor_bundle": "missing",
                    "features": "missing",
                    "regression": "missing",
                    "policy_overlap": "missing",
                },
            ],
            "unavailable_partitions": {"m3": "too few trials"},
        },
        Path(tmp_path),
    )

    assert "Workflow status: 2 partitions" in rendered
    assert "regression: 1 missing, 1 success" in rendered
    assert "policy_overlap: 1 missing, 1 success" in rendered
    assert "Unavailable partitions: 1" in rendered
