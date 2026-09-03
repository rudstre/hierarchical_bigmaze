"""Pure NMF-discovery configuration dataclasses.

Split out of ``discovery.py`` so that validating/loading a configuration
(which constructs these to check field ranges) never has to import sklearn
or torch -- only the functions that actually run NMF discovery do that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from andrew_mlmdp.profiles import ProfileNormalization, _validate_profile_normalization


@dataclass(frozen=True)
class NMFConfig:
    """Task-family parameters used only to discover soft subtask profiles.

    These values define the fixed desirability ensemble and NMF scale gauge.
    Peak normalization is the default; ``"l2"`` selects a unit-L2 gauge. They
    are intentionally separate from ``Parameters`` so execution tuning cannot
    silently rediscover a different hierarchy.
    """

    interior_reward: float = -1.0
    goal_reward: float = 0.0
    control_cost: float = 3.0
    profile_normalization: ProfileNormalization = "peak"

    def __post_init__(self) -> None:
        values = (
            self.interior_reward,
            self.goal_reward,
            self.control_cost,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("NMF discovery parameters must be finite")
        if self.interior_reward >= 0.0:
            raise ValueError("Discovery interior reward must be negative")
        if self.control_cost <= 0.0:
            raise ValueError("Discovery control cost must be positive")
        _validate_profile_normalization(self.profile_normalization)


@dataclass(frozen=True)
class NMFConnectivityConfig:
    """Connected-effective-support settings for stochastic NMF restarts."""

    support_mass: float = 0.95
    max_prune_refits: int = 3
    positive_fallback_attempts: int = 3
    restart_seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not np.isfinite(self.support_mass) or not 0.0 < self.support_mass <= 1.0:
            raise ValueError("Connectivity support mass must be in (0, 1]")
        if (
            isinstance(self.max_prune_refits, (bool, np.bool_))
            or not isinstance(self.max_prune_refits, (int, np.integer))
            or self.max_prune_refits < 1
        ):
            raise ValueError("Maximum connectivity prune/refit rounds must be positive")
        if (
            isinstance(self.positive_fallback_attempts, (bool, np.bool_))
            or not isinstance(
                self.positive_fallback_attempts,
                (int, np.integer),
            )
            or self.positive_fallback_attempts < 1
        ):
            raise ValueError("Positive masked fallback attempts must be positive")
        object.__setattr__(
            self,
            "positive_fallback_attempts",
            int(self.positive_fallback_attempts),
        )
        seeds = tuple(self.restart_seeds)
        if not seeds:
            raise ValueError("Connectivity requires at least one restart seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Connectivity restart seeds must be unique")
        for seed in seeds:
            if (
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or not 0 <= int(seed) <= np.iinfo(np.uint32).max
            ):
                raise ValueError("Connectivity restart seeds must be uint32 integers")
        object.__setattr__(
            self,
            "restart_seeds",
            tuple(int(seed) for seed in seeds),
        )
