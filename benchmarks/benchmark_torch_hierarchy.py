"""Benchmark the exact Doohan full-batch Torch hierarchy without fitting."""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp import (  # noqa: E402
    DoohanDataset,
    Environment,
    NMFConfig,
    NMFConnectivityConfig,
    SubgoalBasis,
    discover_subgoals,
    parameter_values,
    soft_parameters,
)
from andrew_mlmdp.hierarchy.likelihood import (  # noqa: E402
    BatchTimings,
    prepare_batch,
    total_prepared_log_likelihood,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    template, trials = _load_problem()
    prepared = prepare_batch(template, trials)
    print(
        f"trials={len(trials)} goals={len(prepared.goals)} "
        f"shared={prepared.n_shared} "
        f"closures={prepared.n_closures} "
        f"operators={prepared.n_operators}"
    )

    _evaluate(template, prepared, backward=True)
    forward_times = []
    backward_times = []
    diagnostics_by_run = []
    for _ in range(args.repeats):
        forward, backward, diagnostics = _evaluate(
            template, prepared, backward=True
        )
        forward_times.append(forward)
        backward_times.append(backward)
        diagnostics_by_run.append(diagnostics)
    diagnostics = diagnostics_by_run[-1]
    median_stages = {
        name: statistics.median(
            run.stage_seconds[name] for run in diagnostics_by_run
        )
        for name in diagnostics.stage_seconds
    }
    recursion_fractions = [
        run.stage_seconds["batched_trajectory_recursion"] / forward
        for run, forward in zip(diagnostics_by_run, forward_times, strict=True)
    ]
    print(f"median_forward_seconds={statistics.median(forward_times):.6f}")
    print(f"median_backward_seconds={statistics.median(backward_times):.6f}")
    print(f"median_stage_seconds={median_stages}")
    print(
        "median_batched_recursion_fraction="
        f"{statistics.median(recursion_fractions):.6f}"
    )
    print(
        f"shared_bank_shape={diagnostics.shared_bank_shape} "
        f"elements={diagnostics.shared_bank_elements} "
        f"payload_mib={diagnostics.shared_bank_payload_mib:.3f}"
    )
    print(
        "linear_algebra_dispatches_per_forward="
        "common_pinv:1,batched_closure_solve:1"
    )
    print(
        "peak_rss_mib="
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.1f}"
    )
    if args.profile:
        _profile(template, prepared)


def _load_problem():
    data_root = (
        PROJECT_ROOT / "external" / "GridMaze-mFC-ephys-DATA" / "data"
    )
    dataset = DoohanDataset.from_data_root(
        data_root,
        subject_ids=["m2"],
        start_date="2022-06-28",
        end_date="2022-07-05",
        maze_name="maze_1",
    )
    environment = Environment(dataset.definition.maze)
    discovery = discover_subgoals(
        environment,
        ranks=(8,),
        parameters=NMFConfig(),
        connectivity=NMFConnectivityConfig(restart_seeds=(0, 1, 2, 3)),
    ).result(8)
    basis = SubgoalBasis.from_profiles(
        dataset.definition.maze,
        discovery.profiles,
        core_threshold=0.8,
    )
    template = environment.hierarchy(
        basis,
        parameters=soft_parameters(8, upper_control_cost=1.8),
    )
    return template, tuple(dataset.trials)


def _evaluate(template, prepared, *, backward: bool):
    values = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in parameter_values(template).items()
    }
    diagnostics = BatchTimings()
    started = perf_counter()
    total = total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=values,
        diagnostics=diagnostics,
    )
    forward_seconds = perf_counter() - started
    backward_seconds = 0.0
    if backward:
        started = perf_counter()
        (-total).backward()
        backward_seconds = perf_counter() - started
    return forward_seconds, backward_seconds, diagnostics


def _profile(template, prepared) -> None:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=False,
    ) as profile:
        _evaluate(template, prepared, backward=True)
    print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
