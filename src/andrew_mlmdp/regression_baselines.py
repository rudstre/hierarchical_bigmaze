"""Geometry-only likelihood and aggregation for regression comparisons."""

import numpy as np


def uniform_action_log_likelihood(allowed_actions, observed_actions):
    """Return float64 LL per decision for uniform allowed-action probabilities.

    ``allowed_actions[decision, action]`` is boolean. Forbidden observations
    have exactly zero probability and therefore negative-infinite likelihood.
    """
    allowed = np.asarray(allowed_actions)
    actions = np.asarray(observed_actions)
    if allowed.ndim != 2 or allowed.dtype != np.bool_ or allowed.shape[1] == 0:
        raise ValueError("Allowed actions must be a boolean [decision, action] matrix")
    if (
        actions.shape != (len(allowed),)
        or not np.issubdtype(actions.dtype, np.integer)
        or np.any(actions < 0)
        or np.any(actions >= allowed.shape[1])
    ):
        raise ValueError("Observed actions must be aligned, in-range integer indices")
    counts = allowed.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every decision must have at least one allowed action")
    likelihood = -np.log(counts.astype(np.float64))
    likelihood[~allowed[np.arange(len(actions)), actions]] = -np.inf
    return likelihood


def aggregate_uniform_baseline(sessions):
    """Average decision LL within sessions, then sessions and subjects equally.

    Each session row supplies subject identity, decision count and total LL.
    Cohort completeness is checked by the caller against its manifest.
    """
    session_rows = []
    identities = set()
    for session in sessions:
        identity = (session["subject_id"], session["heldout_session_id"])
        if identity in identities:
            raise ValueError("Duplicate uniform-baseline session")
        identities.add(identity)
        count = session["n_decisions"]
        total = session["total_log_likelihood"]
        if count <= 0 or np.isnan(total) or total > 0:
            raise ValueError("Invalid uniform-baseline count or likelihood")
        session_rows.append({**session, "mean_log_likelihood": total / count})
    subjects = []
    for subject in dict.fromkeys(row["subject_id"] for row in session_rows):
        rows = [row for row in session_rows if row["subject_id"] == subject]
        subjects.append(
            {
                "subject_id": subject,
                "n_sessions": len(rows),
                "n_decisions": sum(row["n_decisions"] for row in rows),
                "mean_log_likelihood": float(
                    np.mean([row["mean_log_likelihood"] for row in rows])
                ),
            }
        )
    return {
        "sessions": session_rows,
        "subjects": subjects,
        "n_sessions": len(session_rows),
        "n_subjects": len(subjects),
        "n_decisions": sum(row["n_decisions"] for row in session_rows),
        "mean_log_likelihood": (
            float(np.mean([row["mean_log_likelihood"] for row in subjects]))
            if subjects
            else None
        ),
    }
