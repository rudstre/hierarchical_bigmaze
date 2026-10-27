"""CPU resource plans for the held-out regression workflow."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Mapping

RESOURCE_PROFILE_VERSION = 3
RESOURCE_PROFILE_NAME = "auto-cpu-v3"

# Measured HMM route fits (8 CPUs, cpu_backfill, 36 partitions each): worst
# seconds per epoch (elapsed_seconds / epochs) and peak MaxRSS from sacct.
#   full_hmm2:      10 routes x  500 epochs, 11.19 s/epoch, 8.07 GB
#   full_hmm_xiao2:  6 routes x 2000 epochs,  4.41 s/epoch, 3.57 GB
# Every epoch does the same work, so time is linear in epochs. Route
# dependence enters through the (routes + 1) route-state dimension: the
# per-epoch time exponent below is fitted to the two worst cases, and memory
# follows the expected (routes + 1)^2 (measured ratio 2.26 vs 2.47 expected).
# Beyond 10 routes the time exponent is the dense-solve bound 3 rather than an
# extrapolated fit, and estimates are reported as uncalibrated.
_HMM_REFERENCE_ROUTES = 10
_HMM_REFERENCE_SECONDS_PER_EPOCH = 11.19
_HMM_REFERENCE_MEMORY_GB = 8.07
_HMM_TIME_EXPONENT = math.log(11.19 / 4.41) / math.log(11 / 7)
_HMM_EXTRAPOLATION_TIME_EXPONENT = 3.0
_HMM_MEASURED_MAX_ROUTES = 10
_HMM_HEADROOM = 2.0
# HMM stages run on 2 CPUs: the route fit scales poorly with threads. On one
# node (results/thread_bench: a route-data-curve condition at 200 epochs),
# whole-task wall time was 1027 s / 665 s / 583 s / 437 s at 1 / 2 / 4 / 8
# threads, so 2 CPUs give the most work per core. The per-epoch times above
# were measured at 8 CPUs; estimates scale them by the 2-thread slowdown.
_HMM_CPUS = 2
_HMM_CPU_SLOWDOWN = 665.4 / 436.8
# Smallest memory request; the measured peaks (3.57-8.07 GB) are well above it.
_HMM_MIN_MEMORY_GB = 4


def _slurm_hours(hours):
    whole = max(1, math.ceil(hours))
    days, hours = divmod(whole, 24)
    return f"{days}-{hours:02d}:00:00" if days else f"{hours:02d}:00:00"


def hmm_resource_estimate(n_routes, epochs):
    """Predictor-bundle and feature-generation estimates from measured fits."""
    scale = (n_routes + 1) / (_HMM_REFERENCE_ROUTES + 1)
    exponent = (
        _HMM_TIME_EXPONENT
        if n_routes <= _HMM_REFERENCE_ROUTES or scale <= 1
        else _HMM_EXTRAPOLATION_TIME_EXPONENT
    )
    # Seconds per epoch at the 8 CPUs the reference fits were measured on.
    seconds_per_epoch = _HMM_REFERENCE_SECONDS_PER_EPOCH * scale**exponent
    fit_hours = (
        _HMM_HEADROOM * _HMM_CPU_SLOWDOWN * seconds_per_epoch * epochs / 3600
    )
    memory_gb = _HMM_HEADROOM * _HMM_REFERENCE_MEMORY_GB * scale**2
    memory = f"{max(_HMM_MIN_MEMORY_GB, math.ceil(memory_gb))}G"
    return {
        "calibrated": n_routes <= _HMM_MEASURED_MAX_ROUTES,
        "seconds_per_epoch": seconds_per_epoch,
        "predictor_bundle": {"memory": memory, "time": _slurm_hours(fit_hours)},
        # Feature generation rebuilds the model and runs no epochs; its cost
        # is scaled from the 1 h request by the per-epoch route scaling.
        "feature_generation": {
            "memory": memory,
            "time": _slurm_hours(max(1.0, _HMM_CPU_SLOWDOWN * scale**exponent)),
        },
    }


STAGES = (
    "discovery",
    "candidate",
    "selection",
    "selected-refit",
    "behavioral-cell",
    "behavioral-selection",
    "behavioral-local-cell",
    "behavioral-local-selection",
    "behavioral-refit",
    "predictor-bundle",
    "feature-generation",
    "regression",
    "policy-overlap",
    "lattice-precompute",
    "aggregation",
    "plotting",
)

_OVERRIDE_FIELDS = {"cpus_per_task", "memory", "time", "max_concurrent"}
_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?[KMGTP]?[Bb]?$")
_TIME_PATTERN = re.compile(r"^(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _memory(value, name):
    if not isinstance(value, str) or not _MEMORY_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a SLURM memory value such as '16G'")
    return value


def _time(value, name):
    if not isinstance(value, str) or not _TIME_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a SLURM duration such as '04:00:00'")
    return value


def resource_bands(raw_slurm: Mapping[str, object], rank_range):
    """Validate and return the configured inclusive rank resource bands."""
    raw_bands = raw_slurm.get("resource_bands")
    if not isinstance(raw_bands, list) or not raw_bands:
        raise ValueError("slurm.resource_bands must be a non-empty list")
    configured = set(range(rank_range[0], rank_range[1] + 1))
    covered = {}
    result = []
    for index, raw_band in enumerate(raw_bands):
        if not isinstance(raw_band, Mapping) or set(raw_band) != {
            "rank_range",
            "memory",
            "time",
        }:
            raise ValueError("Each resource band needs rank_range, memory, and time")
        bounds = raw_band["rank_range"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in bounds
            )
            or bounds[0] > bounds[1]
        ):
            raise ValueError("Resource-band rank_range must be [lower, higher]")
        ranks = configured & set(range(bounds[0], bounds[1] + 1))
        for rank in ranks:
            if rank in covered:
                raise ValueError(f"Resource-band overlap at rank {rank}")
            covered[rank] = index
        result.append(
            {
                "rank_range": list(bounds),
                "memory": _memory(
                    raw_band["memory"], f"slurm.resource_bands[{index}].memory"
                ),
                "time": _time(raw_band["time"], f"slurm.resource_bands[{index}].time"),
            }
        )
    missing = sorted(configured - set(covered))
    if missing:
        raise ValueError(f"Resource-band gap for ranks {missing}")
    return [
        band
        for band in result
        if configured & set(range(band["rank_range"][0], band["rank_range"][1] + 1))
    ]


def _resource(partition, account, cpus, memory, time, concurrency):
    return {
        "partition": partition,
        "account": account,
        "cpus_per_task": cpus,
        "memory": memory,
        "time": time,
        "max_concurrent": concurrency,
    }


def _validated_overrides(raw_slurm):
    overrides = raw_slurm.get("stage_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("slurm.stage_overrides must be an object")
    unknown_stages = sorted(set(overrides) - set(STAGES))
    if unknown_stages:
        raise ValueError(f"Unknown slurm.stage_overrides stages: {unknown_stages}")
    result = {}
    for stage, values in overrides.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"slurm.stage_overrides.{stage} must be an object")
        unknown_fields = sorted(set(values) - _OVERRIDE_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"Unknown slurm.stage_overrides.{stage} fields: {unknown_fields}"
            )
        checked = {}
        for name, value in values.items():
            qualified = f"slurm.stage_overrides.{stage}.{name}"
            if name in {"cpus_per_task", "max_concurrent"}:
                checked[name] = _positive_integer(value, qualified)
            elif name == "memory":
                checked[name] = _memory(value, qualified)
            else:
                checked[name] = _time(value, qualified)
        result[stage] = checked
    return result


def resolve_resource_plan(config, raw_slurm: Mapping[str, object]):
    """Derive a reproducible CPU-only resource plan from scientific settings."""
    if not isinstance(raw_slurm, Mapping):
        raise ValueError("slurm must be an object")
    profile = raw_slurm.get("resource_profile", "auto")
    if profile != "auto":
        raise ValueError("slurm.resource_profile currently supports only 'auto'")
    partition = raw_slurm.get("partition")
    if not isinstance(partition, str) or not partition.strip():
        raise ValueError("slurm.partition must be a non-empty string")
    account = raw_slurm.get("account")
    if account is not None and (not isinstance(account, str) or not account.strip()):
        raise ValueError("slurm.account must be null or a non-empty string")
    concurrency = _positive_integer(
        raw_slurm.get("max_concurrent"), "slurm.max_concurrent"
    )
    bands = resource_bands(raw_slurm, config.subgoal_selection.rank_range)
    highest_rank_band = next(
        band
        for band in bands
        if band["rank_range"][0]
        <= config.subgoal_selection.rank_range[1]
        <= band["rank_range"][1]
    )

    ordinary = _resource(partition, account, 1, "4G", "02:00:00", concurrency)
    stages = {stage: deepcopy(ordinary) for stage in STAGES}
    stages["selected-refit"].update(
        memory=highest_rank_band["memory"], time=highest_rank_band["time"]
    )
    # Provisional ceiling from simulated cells, pending completion of the
    # real K=6 pilot. One CPU follows the thread benchmark; the fit records
    # its actual wall time/RSS. Do not describe this estimate as calibrated.
    if config.behavioral:
        for stage in ("behavioral-cell", "behavioral-local-cell", "behavioral-refit"):
            stages[stage].update(cpus_per_task=1, memory="8G", time="12:00:00")
        # A behavioral-cell task fits every coarse theta cell of one fold and
        # K in sequence. Budget the documented simulated estimate of ~3
        # CPU-hours per nine-start cell (1/3 h per start); cells checkpoint,
        # so a timed-out task resumes.
        from andrew_mlmdp.hierarchy.behavioral_profiles import coarse_theta_grid

        settings = config.mlmdp.settings
        n_starts = len(coarse_theta_grid(settings)) * len(settings.start_names)
        stages["behavioral-cell"]["time"] = _slurm_hours(n_starts / 3.0)
    stages["feature-generation"]["time"] = "01:00:00"
    stages["policy-overlap"]["time"] = "01:00:00"
    stages["lattice-precompute"]["time"] = "01:00:00"
    if config.regression_cv.method == "blocked_trial_learning_curve":
        stages["regression"]["time"] = "04:00:00"

    route_family = config.predictors.route_family
    route_count_bound = None
    hmm_epochs = None
    calibrated = True
    calibration_limits = None
    if route_family == "pca":
        stages["predictor-bundle"]["time"] = "02:00:00"
    else:
        hmm = config.predictors.selected_route_settings
        configured_routes = hmm.get("n_routes", 6)  # fitting default
        route_count_bound = (
            config.subgoal_selection.rank_range[1] + 1
            if configured_routes == "auto"
            else int(configured_routes)
        )
        hmm_epochs = int(hmm.get("epochs", 2000))  # fitting default
        hmm_concurrency = min(
            96 if partition == "cpu_backfill" else 8,
            concurrency,
        )
        estimate = hmm_resource_estimate(route_count_bound, hmm_epochs)
        stages["predictor-bundle"].update(
            cpus_per_task=_HMM_CPUS,
            max_concurrent=hmm_concurrency,
            **estimate["predictor_bundle"],
        )
        stages["feature-generation"].update(
            cpus_per_task=_HMM_CPUS,
            max_concurrent=hmm_concurrency,
            **estimate["feature_generation"],
        )
        calibration_limits = [{"max_routes": _HMM_MEASURED_MAX_ROUTES}]
        calibrated = estimate["calibrated"]

    overrides = _validated_overrides(raw_slurm)
    for stage, values in overrides.items():
        stages[stage].update(values)

    candidate_bands = []
    for band in bands:
        resource = _resource(
            partition,
            account,
            1,
            band["memory"],
            band["time"],
            concurrency,
        )
        resource.update(overrides.get("candidate", {}))
        candidate_bands.append({"rank_range": band["rank_range"], **resource})

    # Estimates beyond the measured route counts are conservative
    # extrapolations, reported through ``calibrated`` but not blocking.
    launchable = True
    reason = None

    return {
        "schema_version": RESOURCE_PROFILE_VERSION,
        "profile": RESOURCE_PROFILE_NAME,
        "route_family": route_family,
        "route_count_bound": route_count_bound,
        "hmm_epochs": hmm_epochs,
        "calibration_limits": calibration_limits,
        "calibrated": calibrated,
        **({"behavioral_calibrated": False} if config.behavioral else {}),
        "launchable": launchable,
        "launch_block_reason": reason if not launchable else None,
        "stages": stages,
        "candidate_bands": candidate_bands,
    }


def resource_plan_rows(plan):
    """Flatten a resource plan for CLI and web tables."""
    rows = []
    for stage in STAGES:
        if stage == "candidate":
            rows.extend(
                {
                    "stage": stage,
                    "rank_range": f"{band['rank_range'][0]}-{band['rank_range'][1]}",
                    **{
                        key: band[key]
                        for key in (
                            "cpus_per_task",
                            "memory",
                            "time",
                            "max_concurrent",
                        )
                    },
                }
                for band in plan["candidate_bands"]
            )
        else:
            rows.append(
                {
                    "stage": stage,
                    "rank_range": None,
                    **{
                        key: plan["stages"][stage][key]
                        for key in (
                            "cpus_per_task",
                            "memory",
                            "time",
                            "max_concurrent",
                        )
                    },
                }
            )
    return rows
