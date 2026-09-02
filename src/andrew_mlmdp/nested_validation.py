"""Selection rules for MLMDP ranks nested inside regression folds.

This module contains only the scientific aggregation rule. Expensive fitting and
artifact orchestration remain in the Doohan workflow, while the rule itself is
tested independently of production-only optimizer guards.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def nested_rank_selection(
    records: Iterable[Mapping[str, object]],
    *,
    ranks: Sequence[int],
    validation_session_ids: Sequence[str],
) -> dict[str, object]:
    """Apply terminal-state checks and the one-standard-error rank rule.

    Each successful record represents one inner LOSO session and must contain a
    finite validation_ll_per_transition. Scientific failures make only their
    rank ineligible. Missing or operationally failed records keep selection
    pending, because those jobs must be retried rather than interpreted as model
    evidence. Rank means and standard errors are unweighted across sessions.
    """

    expected_ranks = tuple(int(rank) for rank in ranks)
    expected_sessions = tuple(str(value) for value in validation_session_ids)
    if not expected_ranks or len(set(expected_ranks)) != len(expected_ranks):
        raise ValueError("ranks must be non-empty and unique")
    if not expected_sessions or len(set(expected_sessions)) != len(expected_sessions):
        raise ValueError("validation_session_ids must be non-empty and unique")

    expected_keys = {
        (rank, session_id)
        for rank in expected_ranks
        for session_id in expected_sessions
    }
    indexed: dict[tuple[int, str], Mapping[str, object]] = {}
    for record in records:
        rank = int(record["k"])
        session_id = str(record["validation_session_id"])
        key = (rank, session_id)
        if key not in expected_keys:
            raise ValueError(f"Unexpected inner-fit identity: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate inner-fit identity: {key}")
        status = record.get("status")
        if status not in {"success", "scientific_failure", "operational_failure"}:
            raise ValueError(f"Invalid inner-fit status for {key}: {status!r}")
        indexed[key] = record

    rows: list[dict[str, object]] = []
    selection_pending = False
    for rank in expected_ranks:
        rank_records = [
            indexed.get((rank, session_id)) for session_id in expected_sessions
        ]
        missing = [
            session_id
            for session_id, record in zip(expected_sessions, rank_records, strict=True)
            if record is None
        ]
        operational = [
            str(record["validation_session_id"])
            for record in rank_records
            if record is not None and record["status"] == "operational_failure"
        ]
        scientific = [
            str(record["validation_session_id"])
            for record in rank_records
            if record is not None and record["status"] == "scientific_failure"
        ]
        successful_values: list[float] = []
        for record in rank_records:
            if record is None or record["status"] != "success":
                continue
            value = float(record["validation_ll_per_transition"])
            if not math.isfinite(value):
                raise ValueError(
                    "Successful inner fits require finite held-out log likelihood"
                )
            successful_values.append(value)

        pending = bool(missing or operational)
        selection_pending = selection_pending or pending
        eligible = (
            not pending
            and not scientific
            and len(successful_values) == len(expected_sessions)
        )
        mean = float(np.mean(successful_values)) if eligible else None
        standard_error = (
            float(np.std(successful_values, ddof=1) / math.sqrt(len(successful_values)))
            if eligible and len(successful_values) > 1
            else (0.0 if eligible else None)
        )
        if pending:
            reason = "missing_or_operational_inner_fit"
        elif scientific:
            reason = "scientific_inner_fit_failure"
        elif not eligible:
            reason = "incomplete_inner_fit_grid"
        else:
            reason = None
        rows.append(
            {
                "k": rank,
                "state": "pending" if pending else "terminal",
                "eligible": eligible,
                "ineligibility_reason": reason,
                "missing_validation_session_ids": missing,
                "operational_failure_session_ids": operational,
                "scientific_failure_session_ids": scientific,
                "successful_inner_fits": len(successful_values),
                "validation_ll_per_transition_mean": mean,
                "validation_ll_per_transition_se": standard_error,
            }
        )

    if selection_pending:
        selected = _empty_selection()
        selected["status"] = "pending"
    else:
        selected = _one_standard_error_selection(rows)
        selected["status"] = (
            "selected" if selected["selected_k"] is not None else "unavailable"
        )
    return {
        "status": selected["status"],
        "expected_inner_fit_count": len(expected_keys),
        "observed_inner_fit_count": len(indexed),
        "rank_rows": rows,
        "selection": selected,
    }


def _one_standard_error_selection(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if row.get("eligible") and _finite(row.get("validation_ll_per_transition_mean"))
    ]
    if not eligible:
        return _empty_selection()
    best = min(
        eligible,
        key=lambda row: (
            -float(row["validation_ll_per_transition_mean"]),
            int(row["k"]),
        ),
    )
    best_mean = float(best["validation_ll_per_transition_mean"])
    best_se = float(best["validation_ll_per_transition_se"])
    threshold = best_mean - best_se
    selected = min(
        (
            row
            for row in eligible
            if float(row["validation_ll_per_transition_mean"]) >= threshold
        ),
        key=lambda row: int(row["k"]),
    )
    return {
        "selected_k": int(selected["k"]),
        "best_mean_k": int(best["k"]),
        "best_mean": best_mean,
        "best_mean_standard_error": best_se,
        "threshold": threshold,
    }


def _empty_selection() -> dict[str, object]:
    return {
        "selected_k": None,
        "best_mean_k": None,
        "best_mean": None,
        "best_mean_standard_error": None,
        "threshold": None,
    }


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
