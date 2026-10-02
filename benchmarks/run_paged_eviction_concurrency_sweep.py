# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep concurrency to find PagedEviction's latency-preserving capacity.

For every (context length, KV budget) cell the script launches a vLLM server
with (or without) ``--paged-eviction-config``, then runs ``vllm bench serve`` at
increasing ``--max-concurrency`` levels on the same fixed KV pool. It records
TTFT/TPOT/ITL percentiles, throughput, failures, and peak KV-cache occupancy,
then derives the largest concurrency that keeps eviction latency within a
relative budget of the full-cache baseline at the same context length.

The derived multiplier answers: with a fixed DRAM budget, how many more
concurrent requests can be served before latency degrades beyond the target.

Example:

    .venv/bin/python benchmarks/run_paged_eviction_concurrency_sweep.py \
        --context-lengths 32768 --budgets 4096,8192,16384 \
        --concurrency-levels 1,2,3,4,6,8,10,12,16,20,24,32,40 \
        --relative-latency-tolerance 0.10

Artifacts (per cell) reuse the ``run_paged_eviction_memory_bench`` layout so the
memory/quality/RULER harnesses and the analysis script can read them together.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm.auto import tqdm  # noqa: E402

import benchmarks.run_paged_eviction_memory_bench as bench  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FULL_CACHE_LABEL = "full_cache"
DEFAULT_CONTEXT_LENGTHS = (32768,)
DEFAULT_BUDGETS = (4096, 8192, 16384)
DEFAULT_CONCURRENCY_LEVELS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40)
DEFAULT_INPUT_LEN = 0  # 0 means "use the cell context length"
SUMMARY_CSV_COLUMNS = (
    "mode",
    "context_length",
    "cache_budget_tokens",
    "max_concurrency",
    "completion_status",
    "validation_passed",
    "completed",
    "failed",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "peak_kv_cache_usage_fraction",
    "derived_peak_kv_cache_bytes",
)


@dataclass
class CellRun:
    budget: int | None
    context_length: int
    max_concurrency: int
    label: str
    summary: dict[str, Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep serving concurrency for PagedEviction and derive the "
            "latency-preserving concurrency multiplier on a fixed KV pool."
        )
    )

    server = parser.add_argument_group("server")
    server.add_argument("--model", default=MODEL)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--startup-timeout-s", type=float, default=1800.0)
    server.add_argument("--shutdown-timeout-s", type=float, default=60.0)
    server.add_argument("--post-load-sleep-s", type=float, default=5.0)
    server.add_argument("--max-model-len", type=int, default=40960)
    server.add_argument("--max-num-seqs", type=int, default=64)
    server.add_argument("--max-num-batched-tokens", type=int, default=8192)
    server.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    server.add_argument("--quantization", default="fp8")
    server.add_argument("--kv-cache-dtype", default="bfloat16")
    server.add_argument("--attention-backend", default="FLASH_ATTN")
    server.add_argument("--block-size", type=int, default=16)
    server.add_argument("--runner", choices=("legacy", "v2", "both"), default="v2")

    grid = parser.add_argument_group("grid")
    grid.add_argument(
        "--context-lengths",
        default=",".join(str(length) for length in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated context lengths to sweep.",
    )
    grid.add_argument(
        "--budgets",
        default=",".join(str(budget) for budget in DEFAULT_BUDGETS),
        help=(
            "Comma-separated retained KV-token budgets. A full-cache baseline "
            "is always included."
        ),
    )
    grid.add_argument(
        "--concurrency-levels",
        default=",".join(str(level) for level in DEFAULT_CONCURRENCY_LEVELS),
        help="Comma-separated --max-concurrency levels to test.",
    )

    workload = parser.add_argument_group("workload")
    workload.add_argument("--num-prompts", type=int, default=128)
    workload.add_argument("--random-input-len", type=int, default=DEFAULT_INPUT_LEN)
    workload.add_argument("--random-output-len", type=int, default=128)
    workload.add_argument("--random-range-ratio", default="0")
    workload.add_argument("--request-rate", default="inf")
    workload.add_argument("--seed", type=int, default=0)
    workload.add_argument(
        "--relative-latency-tolerance",
        type=float,
        default=0.10,
        help=(
            "Maximum allowed relative increase of eviction p99 TTFT and TPOT "
            "versus the full-cache baseline at the same context length."
        ),
    )

    artifacts = parser.add_argument_group("artifacts")
    artifacts.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/paged_eviction_concurrency"),
    )
    artifacts.add_argument("--name", default=None)
    artifacts.add_argument("--gpu-index", default=bench.default_gpu_index())
    artifacts.add_argument("--poll-interval-s", type=float, default=1.0)
    artifacts.add_argument(
        "--bytes-per-kv-token", type=int, default=bench.BYTES_PER_KV_TOKEN_LLAMA_3_1_8B
    )
    artifacts.add_argument("--disable-tqdm", action="store_true")
    artifacts.add_argument("--dry-run", action="store_true")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.quantization.lower() in ("", "none"):
        args.quantization = None
    if args.kv_cache_dtype.lower() in ("", "none"):
        args.kv_cache_dtype = None
    if args.attention_backend.lower() in ("", "none"):
        args.attention_backend = None

    args.context_lengths = parse_int_list(args.context_lengths, "context-lengths")
    args.budgets = parse_int_list(args.budgets, "budgets")
    args.concurrency_levels = parse_int_list(
        args.concurrency_levels, "concurrency-levels"
    )
    if not args.context_lengths:
        parser.error("--context-lengths must contain at least one length")
    if not args.concurrency_levels:
        parser.error("--concurrency-levels must contain at least one level")
    if any(level < 1 for level in args.concurrency_levels):
        parser.error("--concurrency-levels entries must be at least 1")

    input_len = args.random_input_len or max(args.context_lengths)
    if input_len + args.random_output_len > args.max_model_len:
        parser.error(
            f"--max-model-len {args.max_model_len} is too small for input "
            f"{input_len} + output {args.random_output_len}"
        )
    return args


def parse_int_list(value: str, flag: str) -> tuple[int, ...]:
    entries: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            entries.append(int(part))
        except ValueError as exc:
            raise SystemExit(
                f"--{flag} contains a non-integer entry: {part!r}"
            ) from exc
    return tuple(entries)


def cell_label(budget: int | None) -> str:
    return FULL_CACHE_LABEL if budget is None else f"budget_{budget}"


def build_cell_list(
    args: argparse.Namespace,
) -> list[tuple[int | None, int, int]]:
    cells: list[tuple[int | None, int, int]] = []
    for context_length in args.context_lengths:
        for budget in (*args.budgets, None):
            for concurrency in args.concurrency_levels:
                cells.append((budget, context_length, concurrency))
    return cells


def build_run_dir(
    root_dir: Path,
    runner: str,
    label: str,
    context_length: int,
    max_concurrency: int,
) -> Path:
    return (
        root_dir
        / label
        / f"context_{context_length}"
        / f"concurrency_{max_concurrency}"
        / runner
        / label
    )


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


def resolved_input_len(args: argparse.Namespace, context_length: int) -> int:
    return args.random_input_len or context_length


def build_server_command(
    args: argparse.Namespace, enabled: bool, budget: int | None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enforce-eager",
        "--disable-cascade-attn",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.quantization is not None:
        command.extend(["--quantization", args.quantization])
    if args.kv_cache_dtype is not None:
        command.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    if args.attention_backend is not None:
        command.extend(["--attention-config.backend", args.attention_backend])
    if args.block_size is not None:
        command.extend(["--block-size", str(args.block_size)])
    if enabled:
        command.extend(
            [
                "--paged-eviction-config",
                json.dumps({"cache_budget_tokens": budget}),
            ]
        )
    return command


def build_benchmark_command(
    args: argparse.Namespace,
    run_dir: Path,
    label: str,
    context_length: int,
    max_concurrency: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        bench.base_url(args),
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(resolved_input_len(args, context_length)),
        "--random-output-len",
        str(args.random_output_len),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(max_concurrency),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(run_dir),
        "--result-filename",
        "bench.json",
        "--seed",
        str(args.seed),
        "--label",
        f"{label}_c{max_concurrency}",
        "--disable-tqdm",
    ]


def run_phase_count() -> int:
    return 6


def run_cell(
    args: argparse.Namespace,
    root_dir: Path,
    runner: str,
    budget: int | None,
    context_length: int,
    max_concurrency: int,
    progress: tqdm | None = None,
) -> CellRun:
    label = cell_label(budget)
    enabled = budget is not None
    run_dir = build_run_dir(root_dir, runner, label, context_length, max_concurrency)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = make_artifacts(run_dir)

    server_command = build_server_command(args, enabled, budget)
    benchmark_command = build_benchmark_command(
        args, run_dir, label, context_length, max_concurrency
    )
    bench.write_json(
        Path(artifacts.command_json),
        {
            "attention_backend": args.attention_backend,
            "benchmark_command": benchmark_command,
            "block_size": args.block_size,
            "chunked_prefill_enabled": True,
            "environment": {"VLLM_USE_V2_MODEL_RUNNER": bench.RUNNER_ENV[runner]},
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_concurrency": max_concurrency,
            "paged_eviction_enabled": enabled,
            "prefix_caching_enabled": False,
            "runner": runner,
            "server_command": server_command,
        },
    )

    if args.dry_run:
        cell_desc = f"{runner}/{label}/ctx{context_length}/c{max_concurrency}"
        print(f"\n[{cell_desc}] server:")
        print("  " + " ".join(server_command))
        print(f"[{cell_desc}] benchmark:")
        print("  " + " ".join(benchmark_command))
        return CellRun(
            budget=budget,
            context_length=context_length,
            max_concurrency=max_concurrency,
            label=label,
            summary=asdict(bench.empty_summary(runner, label, enabled, artifacts)),
        )

    phase_progress = None
    if progress is not None:
        phase_progress = tqdm(
            total=run_phase_count(),
            desc=f"{runner}/{label}/ctx{context_length}/c{max_concurrency}",
            unit="phase",
            leave=False,
            position=1,
        )

    phase = bench.Phase("idle")
    memory_sampler = bench.NvidiaSmiSampler(
        Path(artifacts.nvidia_smi_csv),
        args.gpu_index,
        args.poll_interval_s,
        phase.get,
    )
    metrics_sampler: bench.MetricsSampler | None = None
    server_proc = None

    try:
        bench.set_run_phase_progress(phase_progress, "idle")
        with bench.detail_progress(phase_progress, "idle sample", total=1) as detail:
            memory_sampler.sample_now("idle")
            bench.advance_progress(detail)
        bench.advance_run_phase_progress(phase_progress)

        phase.set("startup")
        bench.set_run_phase_progress(phase_progress, "startup")
        memory_sampler.start()
        bench.assert_port_available(args)

        with Path(artifacts.server_log).open("w", encoding="utf-8") as server_log:
            server_proc = bench.subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=bench.subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=bench.server_environment(runner),
            )
            with bench.detail_progress(
                phase_progress, "startup health", unit="poll"
            ) as detail:
                bench.wait_for_health(
                    args, server_proc, Path(artifacts.server_log), progress=detail
                )
            bench.advance_run_phase_progress(phase_progress)

            metrics_sampler = bench.MetricsSampler(
                bench.base_url(args),
                Path(artifacts.metrics_jsonl),
                args.poll_interval_s,
                phase.get,
            )
            metrics_sampler.start()

            phase.set("post_load")
            bench.set_run_phase_progress(phase_progress, "post_load")
            with bench.detail_progress(
                phase_progress, "post-load", total=2 + max(args.post_load_sleep_s, 0.0)
            ) as detail:
                memory_sampler.sample_now("post_load")
                bench.advance_progress(detail)
                metrics_sampler.sample_now("post_load")
                bench.advance_progress(detail)
                bench.sleep_with_progress(args.post_load_sleep_s, detail)
            bench.advance_run_phase_progress(phase_progress)

            phase.set("benchmark")
            bench.set_run_phase_progress(phase_progress, "benchmark")
            with bench.detail_progress(
                phase_progress, "benchmark elapsed", unit="s"
            ) as detail:
                bench.run_benchmark(
                    benchmark_command,
                    Path(artifacts.benchmark_log),
                    progress=detail,
                )
            bench.advance_run_phase_progress(phase_progress)

            phase.set("post_bench")
            bench.set_run_phase_progress(phase_progress, "post_bench")
            with bench.detail_progress(
                phase_progress, "post-bench sample", total=2
            ) as detail:
                memory_sampler.sample_now("post_bench")
                bench.advance_progress(detail)
                metrics_sampler.sample_now("post_bench")
                bench.advance_progress(detail)
            bench.advance_run_phase_progress(phase_progress)
    finally:
        bench.set_run_phase_progress(phase_progress, "shutdown")
        if metrics_sampler is not None:
            metrics_sampler.stop()
        if server_proc is not None:
            bench.terminate_process(server_proc, args.shutdown_timeout_s)
            with bench.detail_progress(
                phase_progress, "shutdown health", unit="poll"
            ) as detail:
                bench.wait_for_health_down(args, timeout_s=30.0, progress=detail)
        memory_sampler.stop()
        bench.advance_run_phase_progress(phase_progress)
        if phase_progress is not None:
            phase_progress.close()

    summary = summarize_cell(args, runner, label, enabled, artifacts)
    summary["max_concurrency"] = max_concurrency
    return CellRun(
        budget=budget,
        context_length=context_length,
        max_concurrency=max_concurrency,
        label=label,
        summary=summary,
    )


def summarize_cell(
    args: argparse.Namespace,
    runner: str,
    label: str,
    enabled: bool,
    artifacts: bench.RunArtifacts,
) -> dict[str, Any]:
    benchmark = bench.load_json(Path(artifacts.benchmark_json))
    validation_errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=args.num_prompts,
        quality_required=False,
        serving_required=True,
    )
    memory_stats = bench.load_memory_stats(Path(artifacts.nvidia_smi_csv))
    peak_usage = bench.load_peak_kv_usage(Path(artifacts.metrics_jsonl))
    capacity_tokens = bench.parse_kv_cache_capacity_tokens(Path(artifacts.server_log))
    capacity_bytes = (
        capacity_tokens * args.bytes_per_kv_token
        if capacity_tokens is not None
        else None
    )
    derived_peak_kv = (
        capacity_bytes * peak_usage
        if capacity_bytes is not None and peak_usage is not None
        else None
    )
    idle_memory = memory_stats.get("idle")
    peak_memory = memory_stats.get("peak_benchmark")
    completed = bench.as_int(benchmark.get("completed")) or 0
    failed = bench.as_int(benchmark.get("failed")) or 0

    summary = bench.RunSummary(
        runner=runner,
        label=label,
        paged_eviction_enabled=enabled,
        completion_status=("complete" if not validation_errors else "invalid"),
        artifacts=artifacts,
        completed=completed,
        failed=failed,
        request_throughput=bench.as_float(benchmark.get("request_throughput")),
        output_throughput=bench.as_float(benchmark.get("output_throughput")),
        total_token_throughput=bench.as_float(benchmark.get("total_token_throughput")),
        mean_ttft_ms=bench.as_float(benchmark.get("mean_ttft_ms")),
        median_ttft_ms=bench.as_float(benchmark.get("median_ttft_ms")),
        p50_ttft_ms=bench.as_float(benchmark.get("p50_ttft_ms")),
        p90_ttft_ms=bench.as_float(benchmark.get("p90_ttft_ms")),
        p99_ttft_ms=bench.as_float(benchmark.get("p99_ttft_ms")),
        idle_gpu_memory_mib=idle_memory,
        post_load_gpu_memory_mib=memory_stats.get("post_load"),
        peak_benchmark_gpu_memory_mib=peak_memory,
        peak_gpu_memory_delta_mib=(
            peak_memory - idle_memory
            if peak_memory is not None and idle_memory is not None
            else None
        ),
        kv_cache_capacity_tokens=capacity_tokens,
        kv_cache_capacity_bytes=capacity_bytes,
        peak_kv_cache_usage_fraction=peak_usage,
        derived_peak_kv_cache_bytes=derived_peak_kv,
        gpqa_accuracy=None,
        gsm8k_accuracy=None,
        wikitext_continuation_f1=None,
        wikitext_word_perplexity=None,
        validation_passed=not validation_errors,
        validation_errors=validation_errors,
    )
    summary_dict = asdict(summary)
    summary_dict["median_tpot_ms"] = bench.as_float(benchmark.get("median_tpot_ms"))
    summary_dict["p99_tpot_ms"] = bench.as_float(benchmark.get("p99_tpot_ms"))
    summary_dict["median_itl_ms"] = bench.as_float(benchmark.get("median_itl_ms"))
    summary_dict["p99_itl_ms"] = bench.as_float(benchmark.get("p99_itl_ms"))
    return summary_dict


def default_root_dir_name() -> str:
    return f"concurrency-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def write_summary_files(
    root_dir: Path,
    args: argparse.Namespace,
    cells: list[CellRun],
) -> None:
    summary = {
        "config": {
            "attention_backend": args.attention_backend,
            "block_size": args.block_size,
            "cache_budgets": list(args.budgets),
            "concurrency_levels": list(args.concurrency_levels),
            "context_lengths": list(args.context_lengths),
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "model": args.model,
            "num_prompts": args.num_prompts,
            "output_length": args.random_output_len,
            "relative_latency_tolerance": args.relative_latency_tolerance,
            "weight_quantization": args.quantization,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root_dir),
        "runs": [
            {
                "cell": {
                    "cache_budget_tokens": cell.budget,
                    "context_length": cell.context_length,
                    "max_concurrency": cell.max_concurrency,
                },
                "summary": cell.summary,
            }
            for cell in cells
        ],
    }
    bench.write_json(root_dir / "summary.json", summary)
    write_summary_csv(root_dir / "summary.csv", cells)


def write_summary_csv(path: Path, cells: list[CellRun]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for cell in cells:
            summary = cell.summary
            writer.writerow(
                {
                    "mode": cell.label,
                    "context_length": cell.context_length,
                    "cache_budget_tokens": ("" if cell.budget is None else cell.budget),
                    "max_concurrency": cell.max_concurrency,
                    "completion_status": summary.get("completion_status"),
                    "validation_passed": summary.get("validation_passed"),
                    "completed": summary.get("completed"),
                    "failed": summary.get("failed"),
                    "p50_ttft_ms": summary.get("p50_ttft_ms"),
                    "p90_ttft_ms": summary.get("p90_ttft_ms"),
                    "p99_ttft_ms": summary.get("p99_ttft_ms"),
                    "median_tpot_ms": summary.get("median_tpot_ms"),
                    "p99_tpot_ms": summary.get("p99_tpot_ms"),
                    "p99_itl_ms": summary.get("p99_itl_ms"),
                    "request_throughput": summary.get("request_throughput"),
                    "output_throughput": summary.get("output_throughput"),
                    "total_token_throughput": summary.get("total_token_throughput"),
                    "peak_kv_cache_usage_fraction": summary.get(
                        "peak_kv_cache_usage_fraction"
                    ),
                    "derived_peak_kv_cache_bytes": summary.get(
                        "derived_peak_kv_cache_bytes"
                    ),
                }
            )


def find_baseline(cells: list[CellRun], cell: CellRun) -> CellRun | None:
    """Lowest full-cache concurrency cell at the same context length."""
    candidates = [
        other
        for other in cells
        if other.label == FULL_CACHE_LABEL
        and other.context_length == cell.context_length
        and other.summary.get("validation_passed")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda other: other.max_concurrency)


def latency_within_budget(
    cell: CellRun, baseline: CellRun | None, tolerance: float
) -> bool:
    if baseline is None:
        return False
    summary = cell.summary
    if summary.get("failed"):
        return False
    for key in ("p99_ttft_ms", "p99_tpot_ms"):
        value = bench.as_float(summary.get(key))
        base = bench.as_float(baseline.summary.get(key))
        if value is None or base is None or base <= 0:
            continue
        if value > base * (1.0 + tolerance):
            return False
    return True


def derive_capacity(
    cells: list[CellRun], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Largest latency-preserving concurrency per (context, mode)."""
    rows: list[dict[str, Any]] = []
    modes = list(dict.fromkeys(cell.label for cell in cells))
    contexts = list(dict.fromkeys(cell.context_length for cell in cells))
    for context in contexts:
        for mode in modes:
            mode_cells = sorted(
                (
                    cell
                    for cell in cells
                    if cell.label == mode and cell.context_length == context
                ),
                key=lambda cell: cell.max_concurrency,
            )
            if not mode_cells:
                continue
            baseline = find_baseline(cells, mode_cells[0])
            allowed = [
                cell
                for cell in mode_cells
                if cell.summary.get("validation_passed")
                and latency_within_budget(
                    cell, baseline, args.relative_latency_tolerance
                )
            ]
            best = (
                max(allowed, key=lambda cell: cell.max_concurrency) if allowed else None
            )
            baseline_concurrency = baseline.max_concurrency if baseline else None
            multiplier = (
                best.max_concurrency / baseline_concurrency
                if best is not None and baseline_concurrency
                else None
            )
            rows.append(
                {
                    "context_length": context,
                    "mode": mode,
                    "cache_budget_tokens": mode_cells[0].budget,
                    "max_latency_preserving_concurrency": (
                        best.max_concurrency if best else None
                    ),
                    "full_cache_baseline_concurrency": baseline_concurrency,
                    "concurrency_multiplier": multiplier,
                    "tested_levels": [cell.max_concurrency for cell in mode_cells],
                }
            )
    return rows


def write_report(
    root_dir: Path,
    args: argparse.Namespace,
    cells: list[CellRun],
) -> list[dict[str, Any]]:
    capacity = derive_capacity(cells, args)
    lines = [
        "# PagedEviction concurrency / latency sweep",
        "",
        (
            f"Model: `{args.model}`. KV cache: `{args.kv_cache_dtype}`. "
            f"Attention backend: `{args.attention_backend}`. "
            f"Relative latency tolerance: {args.relative_latency_tolerance:.0%} "
            "on p99 TTFT and TPOT versus the lowest full-cache concurrency "
            "at the same context length."
        ),
        "",
        "| Context | Mode | Budget | Max latency-preserving concurrency | "
        "Full-cache baseline | Multiplier |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in capacity:
        multiplier = row["concurrency_multiplier"]
        lines.append(
            "| {context} | {mode} | {budget} | {best} | {base} | {mult} |".format(
                context=row["context_length"],
                mode=row["mode"],
                budget=(
                    "n/a"
                    if row["cache_budget_tokens"] is None
                    else row["cache_budget_tokens"]
                ),
                best=(
                    "n/a"
                    if row["max_latency_preserving_concurrency"] is None
                    else f"{row['max_latency_preserving_concurrency']}x"
                ),
                base=(
                    "n/a"
                    if row["full_cache_baseline_concurrency"] is None
                    else f"{row['full_cache_baseline_concurrency']}x"
                ),
                mult="n/a" if multiplier is None else f"{multiplier:.2f}x",
            )
        )

    lines.extend(
        [
            "",
            "## Raw sweep",
            "",
            "| Context | Mode | Concurrency | completed | failed | p50 TTFT (ms) | "
            "p99 TTFT (ms) | p99 TPOT (ms) | Output tok/s | Peak KV (GiB) | "
            "Within budget |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in sorted(
        cells, key=lambda cell: (cell.context_length, cell.label, cell.max_concurrency)
    ):
        summary = cell.summary
        baseline = find_baseline(cells, cell)
        peak = bench.bytes_to_gib(
            bench.as_float(summary.get("derived_peak_kv_cache_bytes"))
        )
        within = latency_within_budget(cell, baseline, args.relative_latency_tolerance)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cell.context_length,
                cell.label,
                cell.max_concurrency,
                summary.get("completed"),
                summary.get("failed"),
                format_value(bench.as_float(summary.get("p50_ttft_ms"))),
                format_value(bench.as_float(summary.get("p99_ttft_ms"))),
                format_value(bench.as_float(summary.get("p99_tpot_ms"))),
                format_value(bench.as_float(summary.get("output_throughput"))),
                format_value(peak),
                "yes" if within else "no",
            )
        )
    (root_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bench.write_json(root_dir / "capacity.json", {"capacity": capacity})
    return capacity


def format_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def write_plots(
    root_dir: Path,
    args: argparse.Namespace,
    cells: list[CellRun],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots")
        return []

    plots_dir = root_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    modes = list(dict.fromkeys(cell.label for cell in cells))
    contexts = list(dict.fromkeys(cell.context_length for cell in cells))

    def line_plot(
        filename: str,
        title: str,
        ylabel: str,
        pick: Callable[[CellRun], Any],
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        plotted = False
        for context in contexts:
            for mode in modes:
                series = sorted(
                    (
                        cell
                        for cell in cells
                        if cell.label == mode and cell.context_length == context
                    ),
                    key=lambda cell: cell.max_concurrency,
                )
                xs = [cell.max_concurrency for cell in series]
                ys = [bench.as_float(pick(cell)) for cell in series]
                pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if not pairs:
                    continue
                plotted = True
                ax.plot(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    marker="o",
                    label=f"ctx {context} {mode}",
                )
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("max concurrency")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

    line_plot(
        "latency_ttft_p99.png",
        "p99 TTFT vs concurrency",
        "milliseconds",
        lambda cell: cell.summary.get("p99_ttft_ms"),
    )
    line_plot(
        "latency_tpot_p99.png",
        "p99 TPOT vs concurrency",
        "milliseconds",
        lambda cell: cell.summary.get("p99_tpot_ms"),
    )
    line_plot(
        "throughput_output.png",
        "Output throughput vs concurrency",
        "tokens / second",
        lambda cell: cell.summary.get("output_throughput"),
    )
    line_plot(
        "kv_usage.png",
        "Peak occupied KV memory vs concurrency",
        "GiB",
        lambda cell: bench.bytes_to_gib(
            bench.as_float(cell.summary.get("derived_peak_kv_cache_bytes"))
        ),
    )
    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root_dir = args.results_dir / (args.name or default_root_dir_name())
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {root_dir}")

    cells_spec = build_cell_list(args)
    runners = bench.selected_runners(args.runner)
    cells: list[CellRun] = []
    with tqdm(
        total=len(cells_spec) * len(runners),
        desc="PagedEviction concurrency",
        unit="cell",
        disable=args.dry_run or args.disable_tqdm,
    ) as progress:
        for runner in runners:
            for budget, context_length, concurrency in cells_spec:
                progress.set_postfix(
                    cell=cell_label(budget),
                    context=context_length,
                    concurrency=concurrency,
                    refresh=True,
                )
                cells.append(
                    run_cell(
                        args,
                        root_dir,
                        runner,
                        budget,
                        context_length,
                        concurrency,
                        progress=progress,
                    )
                )
                progress.update(1)

    write_summary_files(root_dir, args, cells)
    write_plots(root_dir, args, cells)
    capacity = write_report(root_dir, args, cells)

    print(f"\nsummary:  {root_dir / 'summary.json'}")
    print(f"csv:      {root_dir / 'summary.csv'}")
    print(f"report:   {root_dir / 'REPORT.md'}")
    print(f"capacity: {root_dir / 'capacity.json'}")
    if args.dry_run:
        return 0
    for row in capacity:
        print(
            "  ctx {ctx} {mode}: best {best} vs baseline {base} -> {mult}".format(
                ctx=row["context_length"],
                mode=row["mode"],
                best=row["max_latency_preserving_concurrency"],
                base=row["full_cache_baseline_concurrency"],
                mult=(
                    "n/a"
                    if row["concurrency_multiplier"] is None
                    else f"{row['concurrency_multiplier']:.2f}x"
                ),
            )
        )
    return int(any(not cell.summary.get("validation_passed") for cell in cells))


if __name__ == "__main__":
    raise SystemExit(main())
