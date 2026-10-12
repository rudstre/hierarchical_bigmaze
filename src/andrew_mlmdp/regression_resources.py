"""CPU resource plans for the held-out regression workflow."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Mapping

RESOURCE_PROFILE_VERSION = 1
RESOURCE_PROFILE_NAME = "auto-cpu-v1"

STAGES = (
    "discovery",
    "candidate",
    "selection",
    "selected-refit",
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
                "time": _time(
                    raw_band["time"], f"slurm.resource_bands[{index}].time"
                ),
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
        configured_routes = hmm.get("n_routes", 7)
        route_count_bound = (
            config.subgoal_selection.rank_range[1] + 1
            if configured_routes == "auto"
            else int(configured_routes)
        )
        hmm_epochs = int(hmm.get("epochs", 500))
        hmm_concurrency = min(
            96 if partition == "cpu_backfill" else 8,
            concurrency,
        )
        stages["predictor-bundle"].update(
            cpus_per_task=8,
            memory="16G",
            time="04:00:00",
            max_concurrent=hmm_concurrency,
        )
        stages["feature-generation"].update(
            cpus_per_task=8,
            memory="16G",
            time="01:00:00",
            max_concurrent=hmm_concurrency,
        )
        calibration_limits = {"max_routes": 10, "max_epochs": 500}
        calibrated = route_count_bound <= 10 and hmm_epochs <= 500

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

    required_overrides = []
    if not calibrated:
        required_fields = _OVERRIDE_FIELDS
        required_overrides = [
            stage
            for stage in ("predictor-bundle", "feature-generation")
            if not required_fields.issubset(overrides.get(stage, {}))
        ]
    launchable = not required_overrides
    reason = None
    if not calibrated:
        reason = (
            "HMM resources are uncalibrated for "
            f"{route_count_bound} possible routes and {hmm_epochs} epochs."
        )
        if required_overrides:
            reason += (
                " Supply cpus_per_task, memory, time, and max_concurrent for: "
                + ", ".join(required_overrides)
                + "."
            )

    return {
        "schema_version": RESOURCE_PROFILE_VERSION,
        "profile": RESOURCE_PROFILE_NAME,
        "route_family": route_family,
        "route_count_bound": route_count_bound,
        "hmm_epochs": hmm_epochs,
        "calibration_limits": calibration_limits,
        "calibrated": calibrated,
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
