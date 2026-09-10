from pathlib import Path

import plotly.graph_objects as go

from andrew_mlmdp.regression_reporting import (
    make_regression_figure,
    write_plotly_outputs,
)


def _summary(*, complete=True):
    return {
        "status": "complete" if complete else "incomplete",
        "subjects": [
            {
                "subject_id": "m2",
                "predictor": "vector",
                "n_sessions": 2,
                "unique_predictability": 0.2,
            },
            {
                "subject_id": "m3",
                "predictor": "vector",
                "n_sessions": 1,
                "unique_predictability": 0.4,
            },
        ],
        "sessions": [
            {
                "subject_id": "m2",
                "heldout_session_id": "s1",
                "heldout_session_order": 0,
                "predictor": "vector",
                "n_decisions": 20,
                "unique_predictability": 0.1,
            },
            {
                "subject_id": "m3",
                "heldout_session_id": "s2",
                "heldout_session_order": 0,
                "predictor": "vector",
                "n_decisions": 12,
                "unique_predictability": 0.4,
            },
        ],
        "partial_group": {
            "rows": [
                {
                    "predictor": "vector",
                    "mean_unique_predictability": 0.3,
                    "sem_across_subjects": 0.1,
                    "n_subjects": 2,
                }
            ]
        },
        "missing_partitions": [] if complete else ["missing"],
        "failed_partitions": [],
    }


def test_plotly_report_has_points_intervals_zero_line_and_units():
    figure = make_regression_figure(_summary(), ("vector",))

    assert isinstance(figure, go.Figure)
    assert any(trace.name == "Subjects" for trace in figure.data)
    mean = next(trace for trace in figure.data if trace.name == "Mean +/- SEM")
    assert list(mean.error_y.array) == [0.1]
    assert figure.layout.yaxis.title.text.endswith("(nats/decision)")
    assert any(shape.y0 == 0 and shape.y1 == 0 for shape in figure.layout.shapes)
    assert all(trace.type == "scatter" for trace in figure.data)


def test_plotly_report_displays_incomplete_grid():
    figure = make_regression_figure(_summary(complete=False), ("vector",))

    assert any(
        "Incomplete result" in annotation.text
        for annotation in figure.layout.annotations
    )


def test_plotly_writer_requests_self_contained_html_and_all_static_formats(tmp_path):
    calls = []

    class Figure:
        def write_html(self, path, **kwargs):
            calls.append(("html", Path(path).suffix, kwargs))

        def write_image(self, path, **kwargs):
            calls.append(("image", Path(path).suffix, kwargs))

    paths = write_plotly_outputs(Figure(), tmp_path, "regression_pca_test")

    assert calls[0][0] == "html"
    assert calls[0][2]["include_plotlyjs"] is True
    assert {suffix for kind, suffix, _ in calls if kind == "image"} == {
        ".png",
        ".svg",
        ".pdf",
    }
    assert set(paths) == {"html", "png", "svg", "pdf"}
