"""Shared normalization modes for distributed subgoal profiles."""

from typing import Literal

import numpy as np

ProfileNormalization = Literal["peak", "l2"]


def _validate_profile_normalization(
    profile_normalization: ProfileNormalization,
) -> None:
    if profile_normalization not in {"peak", "l2"}:
        raise ValueError("profile_normalization must be either 'peak' or 'l2'")


def _normalize_profile_columns(
    profiles: np.ndarray,
    profile_normalization: ProfileNormalization,
    *,
    empty_message: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize columns and return both normalized values and their scales."""

    _validate_profile_normalization(profile_normalization)
    values = np.asarray(profiles, dtype=np.float64)
    if profile_normalization == "peak":
        scales = values.max(axis=0)
    else:
        scales = np.linalg.norm(values, axis=0)
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError(empty_message)
    return values / scales[np.newaxis, :], scales
