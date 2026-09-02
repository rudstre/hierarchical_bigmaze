import pytest

from andrew_mlmdp.nested_validation import nested_rank_selection


def _success(k, session, score):
    return {
        "k": k,
        "validation_session_id": session,
        "status": "success",
        "validation_ll_per_transition": score,
    }


def test_nested_selection_uses_unweighted_session_mean_and_one_se():
    records = [
        _success(2, "a", -1.2),
        _success(2, "b", -1.0),
        _success(3, "a", -1.2),
        _success(3, "b", -0.8),
    ]

    result = nested_rank_selection(
        records,
        ranks=(2, 3),
        validation_session_ids=("a", "b"),
    )

    assert result["status"] == "selected"
    assert result["rank_rows"][0]["validation_ll_per_transition_mean"] == pytest.approx(
        -1.1
    )
    assert result["selection"]["best_mean_k"] == 3
    assert result["selection"]["selected_k"] == 2


def test_scientific_failure_excludes_rank_without_blocking_fold():
    records = [
        _success(2, "a", -2.0),
        _success(2, "b", -2.0),
        _success(3, "a", -1.0),
        {
            "k": 3,
            "validation_session_id": "b",
            "status": "scientific_failure",
        },
    ]

    result = nested_rank_selection(
        records,
        ranks=(2, 3),
        validation_session_ids=("a", "b"),
    )

    assert result["status"] == "selected"
    assert result["selection"]["selected_k"] == 2
    assert result["rank_rows"][1]["ineligibility_reason"] == (
        "scientific_inner_fit_failure"
    )


@pytest.mark.parametrize("record", [None, "operational"])
def test_missing_or_operational_fit_keeps_selection_pending(record):
    records = [
        _success(2, "a", -1.0),
        _success(2, "b", -1.0),
        _success(3, "a", -1.0),
    ]
    if record == "operational":
        records.append(
            {
                "k": 3,
                "validation_session_id": "b",
                "status": "operational_failure",
            }
        )

    result = nested_rank_selection(
        records,
        ranks=(2, 3),
        validation_session_ids=("a", "b"),
    )

    assert result["status"] == "pending"
    assert result["selection"]["selected_k"] is None
