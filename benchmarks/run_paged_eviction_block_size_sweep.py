# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep PagedEviction sensitivity to physical KV cache block size.

Run from an initialized vLLM environment, for example:

    .venv/bin/python benchmarks/run_paged_eviction_block_size_sweep.py

The assumed current/default block size is 16 tokens, matching
``CacheConfig.DEFAULT_BLOCK_SIZE``. This sweep evaluates physical KV cache block
size by passing ``--block-size`` to the vLLM server. The default sweep is
16,32,64,128 because the PagedEviction path requires FlashAttention and this
setup supports physical block sizes that are multiples of 16.
"""

from __future__ import annotations

import argparse
import copy
import csv
import shlex
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm.auto import tqdm  # noqa: E402

import benchmarks.run_paged_eviction_memory_bench as bench  # noqa: E402

ASSUMED_DEFAULT_BLOCK_SIZE = bench.DEFAULT_KV_CACHE_BLOCK_SIZE
DEFAULT_BLOCK_SIZES = (16, 32, 64, 128)
SUPPORTED_BLOCK_SIZE_MULTIPLE = 16


def build_arg_parser() -> argparse.ArgumentParser:
    parser = bench.build_arg_parser(
        description=(
            "Run disabled/enabled PagedEviction benchmarks across physical "
            "KV cache block sizes."
        ),
        default_results_dir=Path("benchmarks/results/paged_eviction_block_size"),
        include_block_size=False,
    )
    sweep = parser.add_argument_group("block size sweep")
    sweep.add_argument(
        "--block-sizes",
        default=",".join(str(size) for size in DEFAULT_BLOCK_SIZES),
        help="Comma-separated physical KV cache block sizes to evaluate.",
    )
    sweep.add_argument(
        "--record-unsupported",
        action="store_true",
        help=(
            "Record unsupported block sizes as unsupported summary rows "
            "without starting a server."
        ),
    )
    progress = parser.add_argument_group("progress")
    progress.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable the outer sweep progress bar.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.quantization.lower() in ("", "none"):
        args.quantization = None
    try:
        args.block_sizes = parse_block_sizes(args.block_sizes)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    unsupported = unsupported_block_sizes(args.block_sizes)
    if unsupported and not args.record_unsupported:
        parser.error(
            "unsupported --block-sizes values: "
            f"{','.join(str(size) for size in unsupported)}; supported values "
            f"must be positive multiples of {SUPPORTED_BLOCK_SIZE_MULTIPLE}"
        )
    if args.smoke:
        args.num_prompts = 8
        args.random_input_len = 512
        args.random_output_len = 64
        args.max_concurrency = 2
        args.gsm8k_limit = min(args.gsm8k_limit, 2)
        args.wikitext_limit = min(args.wikitext_limit, 2)
    return args


def parse_block_sizes(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("block size list contains an empty item")
        try:
            block_size = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"block size {item!r} is not an integer"
            ) from exc
        if block_size <= 0:
            raise argparse.ArgumentTypeError("block sizes must be positive")
        values.append(block_size)
    return values


def unsupported_block_sizes(block_sizes: Sequence[int]) -> list[int]:
    return [size for size in block_sizes if not is_supported_block_size(size)]


def is_supported_block_size(block_size: int) -> bool:
    return block_size % SUPPORTED_BLOCK_SIZE_MULTIPLE == 0


def block_root_dir(root_dir: Path, block_size: int) -> Path:
    return root_dir / f"block_size_{block_size}"


def run_block_size(
    args: argparse.Namespace,
    root_dir: Path,
    block_size: int,
    progress: tqdm | None = None,
) -> dict[str, dict[str, bench.RunSummary]]:
    block_args = copy.copy(args)
    block_args.block_size = block_size
    root = block_root_dir(root_dir, block_size)
    root.mkdir(parents=True, exist_ok=True)
    if block_args.dry_run:
        print_dry_run(root, block_args)

    runs: dict[str, dict[str, bench.RunSummary]] = {}
    for runner in bench.selected_runners(block_args.runner):
        runs[runner] = {}
        for label, enabled in (("disabled", False), ("enabled", True)):
            if progress is not None:
                progress.set_postfix(
                    block_size=block_size,
                    runner=runner,
                    mode=label,
                    refresh=True,
                )
            runs[runner][label] = bench.run_one(
                block_args,
                root,
                runner,
                label,
                enabled=enabled,
                **inner_progress_kwargs(
                    block_args,
                    progress,
                    f"block={block_size} {runner}/{label}",
                ),
            )
            if progress is not None:
                progress.update()
    return runs


def record_unsupported_block_size(
    args: argparse.Namespace,
    root_dir: Path,
    block_size: int,
    progress: tqdm | None = None,
) -> dict[str, dict[str, bench.RunSummary]]:
    root = block_root_dir(root_dir, block_size)
    root.mkdir(parents=True, exist_ok=True)
    reason = (
        f"block size {block_size} is unsupported for this sweep; block sizes "
        f"must be multiples of {SUPPORTED_BLOCK_SIZE_MULTIPLE}"
    )
    bench.write_json(
        root / "unsupported.json",
        {
            "block_size": block_size,
            "completion_status": "unsupported",
            "default_block_sizes": list(DEFAULT_BLOCK_SIZES),
            "reason": reason,
            "supported_block_size_multiple": SUPPORTED_BLOCK_SIZE_MULTIPLE,
        },
    )

    runs: dict[str, dict[str, bench.RunSummary]] = {}
    for runner in bench.selected_runners(args.runner):
        runs[runner] = {}
        for label, enabled in (("disabled", False), ("enabled", True)):
            if progress is not None:
                progress.set_postfix(
                    block_size=block_size,
                    runner=runner,
                    mode=label,
                    refresh=True,
                )
            artifacts = make_artifacts(root / runner / label)
            Path(artifacts.run_dir).mkdir(parents=True, exist_ok=True)
            bench.write_json(
                Path(artifacts.command_json),
                {
                    "block_size": block_size,
                    "completion_status": "unsupported",
                    "environment": {
                        "VLLM_USE_V2_MODEL_RUNNER": bench.RUNNER_ENV[runner],
                    },
                    "paged_eviction_enabled": enabled,
                    "reason": reason,
                    "runner": runner,
                    "server_command": None,
                    "benchmark_command": None,
                },
            )
            run = bench.empty_summary(runner, label, enabled, artifacts)
            run.completion_status = "unsupported"
            runs[runner][label] = run
            if progress is not None:
                progress.update()
    return runs


def progress_total(args: argparse.Namespace) -> int:
    return len(args.block_sizes) * len(bench.selected_runners(args.runner)) * 2


def inner_progress_kwargs(
    args: argparse.Namespace,
    progress: tqdm | None,
    label: str,
) -> dict[str, Any]:
    if progress is None or args.disable_tqdm or args.dry_run:
        return {}
    return {"show_progress": True, "progress_label": label}


def make_artifacts(run_dir: Path) -> bench.RunArtifacts:
    return bench.RunArtifacts(
        run_dir=str(run_dir),
        server_log=str(run_dir / "server.log"),
        benchmark_log=str(run_dir / "bench_stdout.log"),
        benchmark_json=str(run_dir / "bench.json"),
        nvidia_smi_csv=str(run_dir / "nvidia_smi.csv"),
        metrics_jsonl=str(run_dir / "metrics_samples.jsonl"),
        command_json=str(run_dir / "commands.json"),
        gsm8k_json=str(run_dir / "gsm8k.json"),
        wikitext_json=str(run_dir / "wikitext.json"),
    )


def print_dry_run(root_dir: Path, args: argparse.Namespace) -> None:
    for runner in bench.selected_runners(args.runner):
        for label, enabled in (("disabled", False), ("enabled", True)):
            run_dir = root_dir / runner / label
            print(
                f"\n[block_size={args.block_size}/{runner}/{label}] "
                f"VLLM_USE_V2_MODEL_RUNNER={bench.RUNNER_ENV[runner]} server:"
            )
            print(shlex.join(bench.build_server_command(args, enabled)))
            print(f"[block_size={args.block_size}/{runner}/{label}] benchmark:")
            print(shlex.join(bench.build_benchmark_command(args, run_dir, label)))
    print(f"\nArtifacts would be written under {root_dir}")


def write_summary(
    root_dir: Path,
    args: argparse.Namespace,
    runs_by_block_size: dict[int, dict[str, dict[str, bench.RunSummary]]],
) -> None:
    enabled_deltas = compute_enabled_deltas(runs_by_block_size)
    sensitivity_deltas = compute_sensitivity_deltas(runs_by_block_size)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root_dir),
        "assumed_default_block_size": ASSUMED_DEFAULT_BLOCK_SIZE,
        "block_sizes": args.block_sizes,
        "default_block_sizes": list(DEFAULT_BLOCK_SIZES),
        "supported_block_size_multiple": SUPPORTED_BLOCK_SIZE_MULTIPLE,
        "measurement_notes": {
            "block_size": (
                "Physical KV cache block size passed to vLLM serve with "
                "--block-size. Omitted runs use the assumed default block "
                f"size {ASSUMED_DEFAULT_BLOCK_SIZE}."
            ),
            "sensitivity_deltas_vs_block_size_16_pct": (
                "Percent deltas compare each run against the same runner and "
                "eviction mode at block size 16."
            ),
        },
        "config": {
            "model": args.model,
            "server": {
                "quantization": args.quantization,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "cache_budget_tokens": args.cache_budget_tokens,
            },
            "workload": {
                "num_prompts": args.num_prompts,
                "random_input_len": args.random_input_len,
                "random_output_len": args.random_output_len,
                "random_range_ratio": args.random_range_ratio,
                "request_rate": args.request_rate,
                "max_concurrency": args.max_concurrency,
                "seed": args.seed,
            },
            "bytes_per_kv_token": args.bytes_per_kv_token,
            "gpu_index": args.gpu_index,
            "runners": bench.selected_runners(args.runner),
            "quality": {
                "enabled": not args.skip_quality,
                "gsm8k_limit": args.gsm8k_limit,
                "gsm8k_max_tokens": args.gsm8k_max_tokens,
                "wikitext_limit": args.wikitext_limit,
                "wikitext_max_words": args.wikitext_max_words,
            },
        },
        "runs": {
            str(block_size): {
                runner: {mode: asdict(run) for mode, run in runner_runs.items()}
                for runner, runner_runs in block_runs.items()
            }
            for block_size, block_runs in runs_by_block_size.items()
        },
        "deltas_enabled_vs_disabled_pct": enabled_deltas,
        "sensitivity_deltas_vs_block_size_16_pct": sensitivity_deltas,
    }
    bench.write_json(root_dir / "summary.json", summary)
    write_summary_csv(
        root_dir / "summary.csv",
        runs_by_block_size,
        enabled_deltas,
        sensitivity_deltas,
    )


def compute_enabled_deltas(
    runs_by_block_size: dict[int, dict[str, dict[str, bench.RunSummary]]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    return {
        str(block_size): {
            runner: bench.compute_deltas(
                runner_runs["disabled"], runner_runs["enabled"]
            )
            for runner, runner_runs in block_runs.items()
        }
        for block_size, block_runs in runs_by_block_size.items()
    }


def compute_sensitivity_deltas(
    runs_by_block_size: dict[int, dict[str, dict[str, bench.RunSummary]]],
    baseline_block_size: int = ASSUMED_DEFAULT_BLOCK_SIZE,
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    baseline = runs_by_block_size.get(baseline_block_size, {})
    deltas: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for block_size, block_runs in runs_by_block_size.items():
        block_deltas: dict[str, dict[str, dict[str, float | None]]] = {}
        for runner, runner_runs in block_runs.items():
            runner_deltas: dict[str, dict[str, float | None]] = {}
            for mode, run in runner_runs.items():
                base = baseline.get(runner, {}).get(mode)
                runner_deltas[mode] = metric_deltas(base, run)
            block_deltas[runner] = runner_deltas
        deltas[str(block_size)] = block_deltas
    return deltas


def metric_deltas(
    baseline: bench.RunSummary | None,
    run: bench.RunSummary,
) -> dict[str, float | None]:
    if baseline is None:
        return {metric: None for metric in bench.summary_metric_names()}
    return {
        metric: bench.percent_delta(
            bench.as_float(getattr(baseline, metric)),
            bench.as_float(getattr(run, metric)),
        )
        for metric in bench.summary_metric_names()
    }


def write_summary_csv(
    path: Path,
    runs_by_block_size: dict[int, dict[str, dict[str, bench.RunSummary]]],
    enabled_deltas: dict[str, dict[str, dict[str, float | None]]],
    sensitivity_deltas: dict[str, dict[str, dict[str, dict[str, float | None]]]],
) -> None:
    metric_names = bench.summary_metric_names()
    enabled_delta_fields = [
        f"delta_enabled_vs_disabled_pct_{metric}" for metric in metric_names
    ]
    sensitivity_delta_fields = [
        f"sensitivity_delta_vs_16_pct_{metric}" for metric in metric_names
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "block_size",
                "runner",
                "eviction_mode",
                "completion_status",
                "validation_passed",
                "validation_errors",
                *metric_names,
                *enabled_delta_fields,
                *sensitivity_delta_fields,
            ],
        )
        writer.writeheader()
        for block_size, block_runs in runs_by_block_size.items():
            block_key = str(block_size)
            for runner, runner_runs in block_runs.items():
                for mode, run in runner_runs.items():
                    writer.writerow(
                        {
                            "block_size": block_size,
                            "runner": runner,
                            "eviction_mode": mode,
                            "completion_status": run.completion_status,
                            "validation_passed": run.validation_passed,
                            "validation_errors": "; ".join(run.validation_errors),
                            **{
                                metric: getattr(run, metric)
                                for metric in metric_names
                            },
                            **{
                                field: enabled_deltas[block_key][runner][metric]
                                for field, metric in zip(
                                    enabled_delta_fields, metric_names
                                )
                            },
                            **{
                                field: sensitivity_deltas[block_key][runner][mode][
                                    metric
                                ]
                                for field, metric in zip(
                                    sensitivity_delta_fields, metric_names
                                )
                            },
                        }
                    )


def has_validation_errors(
    runs_by_block_size: dict[int, dict[str, dict[str, bench.RunSummary]]],
) -> bool:
    return any(
        run.validation_errors
        for block_runs in runs_by_block_size.values()
        for runner_runs in block_runs.values()
        for run in runner_runs.values()
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root_dir = bench.make_root_dir(args)
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {root_dir}")

    runs_by_block_size = {}
    with tqdm(
        total=progress_total(args),
        desc="Block size sweep",
        unit="run",
        disable=args.disable_tqdm or args.dry_run,
    ) as progress:
        for block_size in args.block_sizes:
            if is_supported_block_size(block_size):
                runs_by_block_size[block_size] = run_block_size(
                    args, root_dir, block_size, progress
                )
            else:
                runs_by_block_size[block_size] = record_unsupported_block_size(
                    args, root_dir, block_size, progress
                )

    write_summary(root_dir, args, runs_by_block_size)
    print(f"\nsummary: {root_dir / 'summary.json'}")
    print(f"csv:     {root_dir / 'summary.csv'}")
    return int(has_validation_errors(runs_by_block_size))


if __name__ == "__main__":
    raise SystemExit(main())
