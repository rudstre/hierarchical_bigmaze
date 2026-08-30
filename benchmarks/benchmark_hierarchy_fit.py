"""Validate and benchmark rank-eight production hierarchy fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from andrew_mlmdp import (  # noqa: E402
    NMFConfig,
    NMFConnectivityConfig,
    SubgoalBasis,
    discover_subgoals,
    fittable_parameters,
    soft_parameters,
)
from andrew_mlmdp.hierarchy import fitting as hierarchy_fitting  # noqa: E402
from andrew_mlmdp.hierarchy.likelihood import (  # noqa: E402
    BatchTimings,
    _reference_prepared_log_likelihoods,
    prepare_batch,
    prepared_log_likelihoods,
)
from andrew_mlmdp.validation import (  # noqa: E402
    _load_problem_context,
    _rank_result_payload,
    _strict_score,
    load_validation_config,
)
from doohan_data_interaction.fit_workflow import (  # noqa: E402
    fit_result_to_payload,
)

RANK = 8
RTOL = 1e-10
ATOL = 1e-11
THREADS = 8  # Validated for this benchmark host/workload, not a library default.
CACHE_VERSION = 1
CONFIG_PATH = PROJECT_ROOT / "configs" / "hierarchy_rank_validation_production.json"
DEFAULT_CACHE = PROJECT_ROOT / "output" / "benchmark_basis_cache" / "rank_08.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--basis-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cold-basis", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--fit-mode", choices=("none", "fixed", "convergence", "cold"), default="none"
    )
    parser.add_argument(
        "--recursion", choices=("reference", "batched"), default="batched"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-fit-results", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare_fit_results:
        _compare_fit_payloads(*args.compare_fit_results)
        return
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    torch.set_num_threads(THREADS)
    config = load_validation_config(CONFIG_PATH)
    context = _load_problem_context(config)
    started = perf_counter()
    profiles, discovery, loaded = _load_or_discover_basis(
        config,
        context,
        args.basis_cache,
        cold=args.cold_basis or args.fit_mode == "cold",
    )
    basis_seconds = perf_counter() - started
    template, threshold_cap = _initial_template(config, context, profiles)
    print(
        f"rank={RANK} training_trials={len(context.training_trials)} "
        f"validation_trials={len(context.validation_trials)} "
        f"nmf_restarts={len(discovery['restarts'])} "
        f"selected_seed={discovery['selected_seed']} "
        f"basis_cache={'hit' if loaded else 'miss'} basis_seconds={basis_seconds:.6f} "
        f"threads={THREADS}",
        flush=True,
    )
    if args.fit_mode == "none":
        payload = _likelihood_benchmark(template, context.training_trials, args)
    elif args.fit_mode in {"fixed", "convergence"}:
        payload = _fit_benchmark(
            template,
            context.training_trials,
            config,
            threshold_cap,
            recursion=args.recursion,
            fixed=args.fit_mode == "fixed",
        )
    else:
        payload = _cold_pipeline(
            template, profiles, context, config, threshold_cap, basis_seconds
        )
    payload.update(
        {
            "rank": RANK,
            "basis_cache_hit": loaded,
            "basis_seconds": basis_seconds,
            "selected_seed": discovery["selected_seed"],
            "threads": THREADS,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if args.output is not None:
        _atomic_write_json(args.output, payload)


def _load_or_discover_basis(config, context, path: Path, *, cold: bool):
    specification = {
        "cache_version": CACHE_VERSION,
        "rank": RANK,
        "configuration": config.normalized_payload(),
        "compatibility": context.compatibility,
    }
    signature = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if path.is_file() and not cold:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("signature") != signature
            or payload.get("specification") != specification
        ):
            raise RuntimeError("Production basis cache signature mismatch")
        discovery = _validated_discovery(payload.get("discovery"))
        return _profiles(discovery), discovery, True
    settings = config.discovery
    result = discover_subgoals(
        context.environment,
        ranks=(RANK,),
        parameters=NMFConfig(
            interior_reward=settings.interior_reward,
            goal_reward=settings.goal_reward,
            control_cost=settings.control_cost,
            profile_normalization=settings.profile_normalization,
        ),
        connectivity=NMFConnectivityConfig(
            support_mass=settings.support_mass,
            max_prune_refits=settings.max_prune_refits,
            positive_fallback_attempts=settings.positive_fallback_attempts,
            restart_seeds=settings.restart_seeds,
        ),
        max_iter=settings.max_iter,
        tolerance=settings.tolerance,
    ).rank_result(RANK)
    if result.discovery is None:
        raise RuntimeError("Every production rank-eight NMF restart failed")
    discovery = _validated_discovery(_rank_result_payload(result))
    _atomic_write_json(
        path,
        {
            "signature": signature,
            "specification": specification,
            "discovery": discovery,
        },
    )
    return _profiles(discovery), discovery, False


def _validated_discovery(value):
    if not isinstance(value, dict):
        raise RuntimeError("Production basis cache has no discovery payload")
    restarts = value.get("restarts")
    if not isinstance(restarts, list) or len(restarts) != 50:
        raise RuntimeError("Production basis cache must contain 50 NMF restarts")
    if [row.get("seed") for row in restarts] != list(range(50)):
        raise RuntimeError("Production basis cache restart seeds must be 0..49")
    if value.get("selected_restart_id") is None:
        raise RuntimeError("Production basis cache has no selected restart")
    selected = value.get("selected_discovery")
    if not isinstance(selected, dict) or "profiles" not in selected:
        raise RuntimeError("Production basis cache has no selected profiles")
    profiles = np.asarray(selected["profiles"], dtype=np.float64)
    if hashlib.sha256(profiles.tobytes()).hexdigest() != selected.get("profile_sha256"):
        raise RuntimeError("Production basis cache profile digest mismatch")
    return value


def _profiles(discovery):
    return np.asarray(discovery["selected_discovery"]["profiles"], dtype=np.float64)


def _initial_template(config, context, profiles):
    values = {**config.adam.initial_values, "core_threshold": 0.0}
    probe_basis = SubgoalBasis.from_profiles(
        context.environment.maze,
        profiles,
        core_threshold=0.0,
        core_exponent=values["core_exponent"],
        profile_normalization=config.discovery.profile_normalization,
    )
    probe = context.environment.hierarchy(
        probe_basis, parameters=soft_parameters(RANK, **values)
    )
    cap = float(
        probe.threshold_range({t.goal for t in context.training_trials}).maximum
    )
    values["core_threshold"] = config.adam.initial_core_threshold_fraction * cap
    basis = SubgoalBasis.from_profiles(
        context.environment.maze,
        profiles,
        core_threshold=values["core_threshold"],
        core_exponent=values["core_exponent"],
        profile_normalization=config.discovery.profile_normalization,
    )
    return context.environment.hierarchy(
        basis, parameters=soft_parameters(RANK, **values)
    ), cap


def _parameter_values(template):
    fitted = set(fittable_parameters(template))
    return {
        name: value.detach().clone().requires_grad_(name in fitted)
        for name, value in template.parameter_values().items()
    }


def _evaluate(function, template, prepared):
    values = _parameter_values(template)
    diagnostics = BatchTimings()
    started = perf_counter()
    scores = function(
        template, prepared, parameter_values=values, diagnostics=diagnostics
    )
    forward = perf_counter() - started
    started = perf_counter()
    (-scores.sum()).backward()
    backward = perf_counter() - started
    return scores, values, forward, backward, diagnostics


def _assert_parity(reference, batched, reference_values, batched_values, names):
    if not torch.equal(torch.isfinite(reference), torch.isfinite(batched)):
        raise AssertionError("Finite likelihood masks differ")
    if not torch.equal(torch.isneginf(reference), torch.isneginf(batched)):
        raise AssertionError("Impossible-event masks differ")
    finite = torch.isfinite(reference)
    torch.testing.assert_close(reference[finite], batched[finite], rtol=RTOL, atol=ATOL)
    for name in names:
        torch.testing.assert_close(
            reference_values[name].grad, batched_values[name].grad, rtol=RTOL, atol=ATOL
        )


def _likelihood_benchmark(template, trials, args):
    prepared = prepare_batch(template, trials)
    reference = _evaluate(_reference_prepared_log_likelihoods, template, prepared)
    batched = _evaluate(prepared_log_likelihoods, template, prepared)
    _assert_parity(
        reference[0],
        batched[0],
        reference[1],
        batched[1],
        fittable_parameters(template),
    )
    print("production_value_and_six_gradient_parity=passed", flush=True)
    _evaluate(_reference_prepared_log_likelihoods, template, prepared)
    _evaluate(prepared_log_likelihoods, template, prepared)
    rows = {"reference": [], "batched": []}
    for _ in range(args.repeats):
        for name, function in (
            ("reference", _reference_prepared_log_likelihoods),
            ("batched", prepared_log_likelihoods),
        ):
            result = _evaluate(function, template, prepared)
            rows[name].append(result[2:])
    summary = {}
    for name, observations in rows.items():
        forward = [row[0] for row in observations]
        backward = [row[1] for row in observations]
        summary[name] = {
            "forward_seconds": forward,
            "backward_seconds": backward,
            "median_forward_seconds": statistics.median(forward),
            "median_backward_seconds": statistics.median(backward),
            "median_total_seconds": statistics.median(
                a + b for a, b in zip(forward, backward, strict=True)
            ),
        }
    summary["speedup"] = (
        summary["reference"]["median_total_seconds"]
        / summary["batched"]["median_total_seconds"]
    )
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as profile:
            _evaluate(prepared_log_likelihoods, template, prepared)
        print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    return {"mode": "likelihood", "timings": summary}


@contextmanager
def _recursion_mode(name):
    original = hierarchy_fitting.total_prepared_log_likelihood
    if name == "reference":
        hierarchy_fitting.total_prepared_log_likelihood = (
            lambda template, prepared, *, parameter_values: (
                _reference_prepared_log_likelihoods(
                    template, prepared, parameter_values=parameter_values
                ).sum()
            )
        )
    try:
        yield
    finally:
        hierarchy_fitting.total_prepared_log_likelihood = original


def _fit_benchmark(template, trials, config, threshold_cap, *, recursion, fixed):
    adam = config.adam
    settings = {
        "lr": adam.learning_rate,
        "max_steps": 50 if fixed else adam.max_steps,
        "tolerance": adam.convergence_tolerance,
        "convergence_tolerance": adam.convergence_tolerance,
        "scheduler_tolerance": adam.scheduler_tolerance,
        "patience": 100 if fixed else adam.patience,
        "lr_decay": adam.lr_decay,
        "lr_patience": 100 if fixed else adam.lr_patience,
        "min_lr": adam.min_lr,
    }
    started = perf_counter()
    with _recursion_mode(recursion):
        result = hierarchy_fitting.fit_parameters(
            template, trials, names=adam.fitted_names, **settings
        )
    payload = fit_result_to_payload(result)
    payload.update(
        {
            "mode": "fixed" if fixed else "convergence",
            "recursion": recursion,
            "seconds": perf_counter() - started,
            "threshold_cap": threshold_cap,
        }
    )
    return payload


def _compare_fit_payloads(reference_path, batched_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    batched = json.loads(batched_path.read_text(encoding="utf-8"))
    if reference["updates"] != batched["updates"] or len(reference["history"]) != len(
        batched["history"]
    ):
        raise AssertionError("Fit update or evaluation counts differ")
    if reference["mode"] == "convergence" and (
        reference["reason"] != batched["reason"]
        or reference["converged"] != batched["converged"]
    ):
        raise AssertionError("Convergence termination differs")
    for left, right in zip(reference["history"], batched["history"], strict=True):
        np.testing.assert_allclose(left["loss"], right["loss"], rtol=RTOL, atol=ATOL)
        for name in reference["names"]:
            np.testing.assert_allclose(
                left["parameter_values"][name],
                right["parameter_values"][name],
                rtol=RTOL,
                atol=ATOL,
            )
    if reference["mode"] == "fixed":
        speedup = reference["seconds"] / batched["seconds"]
        print(f"fit_parity=passed fixed_work_speedup={speedup:.6f}")
    else:
        print("convergence_parity=passed timing_not_used_for_speed_claim")


def _cold_pipeline(
    template, profiles, context, config, threshold_cap, discovery_seconds
):
    started = perf_counter()
    initial = _strict_score(template, context.training_trials)
    initial_seconds = perf_counter() - started
    fit = _fit_benchmark(
        template,
        context.training_trials,
        config,
        threshold_cap,
        recursion="batched",
        fixed=False,
    )
    best = fit["best_values"]
    assert best is not None
    basis = SubgoalBasis.from_profiles(
        context.environment.maze,
        profiles,
        core_threshold=best["core_threshold"],
        core_exponent=best["core_exponent"],
        profile_normalization=config.discovery.profile_normalization,
    )
    fitted = context.environment.hierarchy(
        basis, parameters=soft_parameters(RANK, **best)
    )
    started = perf_counter()
    training = _strict_score(fitted, context.training_trials)
    validation = _strict_score(fitted, context.validation_trials)
    scoring_seconds = perf_counter() - started
    return {
        "mode": "cold",
        "discovery_seconds": discovery_seconds,
        "initial_scoring_seconds": initial_seconds,
        "fit_seconds": fit["seconds"],
        "final_scoring_seconds": scoring_seconds,
        "total_seconds": discovery_seconds
        + initial_seconds
        + fit["seconds"]
        + scoring_seconds,
        "fit": fit,
        "initial_training": initial,
        "fitted_training": training,
        "validation": validation,
    }


def _atomic_write_json(path: Path, payload) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
