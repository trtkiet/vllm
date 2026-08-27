# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark PagedEviction long-context quality with RULER.

For every (context length, KV budget, repetition) cell the script:

1. Launches a vLLM server with (or without) a ``--paged-eviction-config``
   budget.
2. Runs a synthetic ``vllm bench serve`` workload at that context length.
3. Evaluates RULER quality against
   ``lighteval/RULER-{context_length}-{model}`` (13 task splits; prompts are
   ``input + answer_prefix`` rows and predictions are scored with the official
   substring match: fraction of references found for retrieval/counting tasks,
   any reference for QA tasks).
4. Aggregates ``summary.json``/``summary.csv``, plots, and ``REPORT.md``.

Example pilot run that reproduces the recorded 1-sample-per-task results:

    .venv/bin/python benchmarks/run_paged_eviction_ruler.py \
        --context-lengths 8192,16384,32768 \
        --budgets 2048,4096,8192 \
        --results-dir benchmarks/results/paged_eviction_long_context \
        --name ruler-8k-32k-all-tasks-repro

Use ``--ruler-samples-per-task 500`` for the full published protocol and
``--dry-run`` to print the exact commands without starting servers.

Requirements: Hugging Face authentication for the gated default model, plus the
``datasets`` and ``matplotlib`` packages used for RULER data and plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
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
RULER_DATASET_TEMPLATE = "lighteval/RULER-{context_length}-llama3.1-8b-chat"
RULER_SCORING_METHOD = "official substring match (all; any for QA)"
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
QA_TASKS = ("qa_1", "qa_2")
FULL_CACHE_LABEL = "full_cache"
DEFAULT_CONTEXT_LENGTHS = (8192, 16384, 32768)
DEFAULT_BUDGETS = (2048, 4096, 8192)
REPORT_TITLE = "# FlashAttention chunked-prefill PagedEviction benchmark"
SUMMARY_CSV_COLUMNS = (
    "mode",
    "context_length",
    "cache_budget_tokens",
    "repetition",
    "completion_status",
    "validation_passed",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "peak_gpu_memory_delta_mib",
    "derived_peak_kv_cache_bytes",
    "ruler_score_percent",
)


@dataclass
class CellRun:
    budget: int | None
    context_length: int
    repetition: int
    label: str
    summary: dict[str, Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PagedEviction long-context serving benchmarks with RULER "
            "quality evaluation over context-length/budget grids."
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
    server.add_argument("--max-num-seqs", type=int, default=2)
    server.add_argument("--max-num-batched-tokens", type=int, default=2048)
    server.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    server.add_argument(
        "--quantization",
        default="fp8",
        help=(
            "Weight quantization passed to vLLM. Defaults to online FP8 to "
            "leave more GPU memory for the KV cache. Use 'none' to disable."
        ),
    )
    server.add_argument("--kv-cache-dtype", default="bfloat16")
    server.add_argument(
        "--attention-backend",
        default="FLASH_ATTN",
        help="Passed to vLLM as --attention-config.backend.",
    )
    server.add_argument("--block-size", type=int, default=16)
    server.add_argument(
        "--runner",
        choices=("legacy", "v2", "both"),
        default="v2",
        help="Model runner(s) to benchmark. Each runner uses a fresh server.",
    )

    grid = parser.add_argument_group("grid")
    grid.add_argument(
        "--context-lengths",
        default=",".join(str(length) for length in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated RULER context lengths to evaluate.",
    )
    grid.add_argument(
        "--budgets",
        default=",".join(str(budget) for budget in DEFAULT_BUDGETS),
        help=(
            "Comma-separated retained KV-token budgets to evaluate. A "
            "full-cache baseline is always included."
        ),
    )
    grid.add_argument("--repetitions", type=int, default=1)

    workload = parser.add_argument_group("workload")
    workload.add_argument("--num-prompts", type=int, default=4)
    workload.add_argument("--random-output-len", type=int, default=128)
    workload.add_argument("--random-range-ratio", default="0")
    workload.add_argument("--request-rate", default="inf")
    workload.add_argument("--max-concurrency", type=int, default=2)
    workload.add_argument("--seed", type=int, default=0)
    workload.add_argument(
        "--skip-serving-benchmark",
        action="store_true",
        help="Skip the synthetic serving benchmark and only run RULER.",
    )

    ruler = parser.add_argument_group("ruler")
    ruler.add_argument(
        "--ruler-samples-per-task",
        type=int,
        default=1,
        help=("RULER samples per task split. Use 500 for the full published protocol."),
    )
    ruler.add_argument(
        "--ruler-tasks",
        default=",".join(RULER_TASKS),
        help="Comma-separated RULER task splits to evaluate.",
    )
    ruler.add_argument("--ruler-max-tokens", type=int, default=128)
    ruler.add_argument("--ruler-temperature", type=float, default=0.0)
    ruler.add_argument(
        "--ruler-dataset-template",
        default=RULER_DATASET_TEMPLATE,
        help=(
            "Dataset template with a {context_length} placeholder, for "
            f"example {RULER_DATASET_TEMPLATE}."
        ),
    )
    ruler.add_argument(
        "--skip-ruler",
        action="store_true",
        help="Skip the RULER evaluation and only run the serving benchmark.",
    )
    ruler.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose serving and RULER artifacts already exist.",
    )

    artifacts = parser.add_argument_group("artifacts")
    artifacts.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/paged_eviction_long_context"),
    )
    artifacts.add_argument(
        "--name",
        default=None,
        help=(
            "Result directory name under --results-dir. Defaults to a "
            "timestamped ruler-* name."
        ),
    )
    artifacts.add_argument(
        "--gpu-index",
        default=bench.default_gpu_index(),
        help=(
            "Physical GPU index or UUID for nvidia-smi. Defaults to the first "
            "numeric CUDA_VISIBLE_DEVICES entry, otherwise 0."
        ),
    )
    artifacts.add_argument("--poll-interval-s", type=float, default=1.0)
    artifacts.add_argument(
        "--bytes-per-kv-token",
        type=int,
        default=bench.BYTES_PER_KV_TOKEN_LLAMA_3_1_8B,
        help=(
            "Bytes per KV token for derived KV memory. The default is "
            "Llama-3.1-8B TP=1 fp16/bf16: 32*2*8*128*2."
        ),
    )
    artifacts.add_argument("--disable-tqdm", action="store_true")
    artifacts.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned output paths without starting servers.",
    )

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
    if not args.context_lengths:
        parser.error("--context-lengths must contain at least one length")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    tasks = tuple(task.strip() for task in args.ruler_tasks.split(",") if task.strip())
    unknown_tasks = [task for task in tasks if task not in RULER_TASKS]
    if unknown_tasks:
        parser.error(
            f"unknown --ruler-tasks entries: {unknown_tasks}; "
            f"expected a subset of {list(RULER_TASKS)}"
        )
    args.ruler_tasks = tasks
    if not args.skip_ruler and args.ruler_samples_per_task < 1:
        parser.error("--ruler-samples-per-task must be at least 1")

    needed_lengths = [args.random_output_len]
    if not args.skip_ruler:
        needed_lengths.append(args.ruler_max_tokens)
    required_model_len = max(args.context_lengths) + max(needed_lengths)
    if required_model_len > args.max_model_len:
        parser.error(
            f"--max-model-len {args.max_model_len} is too small for the "
            f"largest context plus generation tokens "
            f"({required_model_len} tokens needed)"
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
        for repetition in range(1, args.repetitions + 1):
            for budget in (*args.budgets, None):
                cells.append((budget, context_length, repetition))
    return cells


def build_run_dir(
    root_dir: Path,
    runner: str,
    label: str,
    context_length: int,
    repetition: int,
) -> Path:
    return (
        root_dir
        / label
        / f"context_{context_length}"
        / f"rep_{repetition}"
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
        str(context_length),
        "--random-output-len",
        str(args.random_output_len),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.max_concurrency),
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
        label,
        "--disable-tqdm",
    ]


def run_phase_count(args: argparse.Namespace) -> int:
    total = 5
    if not args.skip_serving_benchmark:
        total += 1
    if not args.skip_ruler:
        total += 1
    return total


def run_ruler_cell(
    args: argparse.Namespace,
    root_dir: Path,
    runner: str,
    budget: int | None,
    context_length: int,
    repetition: int,
    progress: tqdm | None = None,
) -> CellRun:
    label = cell_label(budget)
    enabled = budget is not None
    run_dir = build_run_dir(root_dir, runner, label, context_length, repetition)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = make_artifacts(run_dir)

    server_command = build_server_command(args, enabled, budget)
    benchmark_command = build_benchmark_command(args, run_dir, label, context_length)
    bench.write_json(
        Path(artifacts.command_json),
        {
            "attention_backend": args.attention_backend,
            "benchmark_command": benchmark_command,
            "block_size": args.block_size,
            "chunked_prefill_enabled": True,
            "environment": {"VLLM_USE_V2_MODEL_RUNNER": bench.RUNNER_ENV[runner]},
            "kv_cache_dtype": args.kv_cache_dtype,
            "paged_eviction_enabled": enabled,
            "prefix_caching_enabled": False,
            "runner": runner,
            "server_command": server_command,
            "skip_serving_benchmark": args.skip_serving_benchmark,
        },
    )

    if args.dry_run:
        cell_desc = f"{runner}/{label}/context_{context_length}"
        runner_env = bench.RUNNER_ENV[runner]
        print(f"\n[{cell_desc}] VLLM_USE_V2_MODEL_RUNNER={runner_env} server:")
        print(shlex.join(server_command))
        if args.skip_serving_benchmark:
            print(f"[{cell_desc}] benchmark: skipped")
        else:
            print(f"[{cell_desc}] benchmark:")
            print(shlex.join(benchmark_command))
        return CellRun(
            budget=budget,
            context_length=context_length,
            repetition=repetition,
            label=label,
            summary=asdict(bench.empty_summary(runner, label, enabled, artifacts)),
        )

    if args.resume and cell_artifacts_complete(args, run_dir):
        print(f"Skipping completed cell {run_dir}")
        return CellRun(
            budget=budget,
            context_length=context_length,
            repetition=repetition,
            label=label,
            summary=summarize_ruler_run(args, runner, label, enabled, artifacts),
        )

    phase_progress = None
    if progress is not None:
        phase_progress = tqdm(
            total=run_phase_count(args),
            desc=f"{runner}/{label}/ctx{context_length}",
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
    server_proc: subprocess.Popen[str] | None = None

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
            server_proc = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=bench.server_environment(runner),
            )

            with bench.detail_progress(
                phase_progress, "startup health", unit="poll"
            ) as detail:
                bench.wait_for_health(
                    args,
                    server_proc,
                    Path(artifacts.server_log),
                    progress=detail,
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
                phase_progress,
                "post-load",
                total=2 + max(args.post_load_sleep_s, 0.0),
            ) as detail:
                memory_sampler.sample_now("post_load")
                bench.advance_progress(detail)
                metrics_sampler.sample_now("post_load")
                bench.advance_progress(detail)
                bench.sleep_with_progress(args.post_load_sleep_s, detail)
            bench.advance_run_phase_progress(phase_progress)

            if not args.skip_serving_benchmark:
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

            if not args.skip_ruler:
                phase.set("ruler")
                bench.set_run_phase_progress(phase_progress, "ruler")
                ruler_path = run_dir / "ruler.json"
                sample_total = len(args.ruler_tasks) * args.ruler_samples_per_task
                with bench.detail_progress(
                    phase_progress,
                    "ruler samples",
                    total=sample_total,
                    unit="sample",
                ) as detail:
                    try:
                        ruler_result = evaluate_ruler(
                            args, context_length, progress=detail
                        )
                    except Exception as exc:
                        ruler_result = {"error": f"{type(exc).__name__}: {exc}"}
                    bench.write_json(ruler_path, ruler_result)
                bench.advance_run_phase_progress(phase_progress)

            phase.set("post_bench")
            bench.set_run_phase_progress(phase_progress, "post_bench")
            with bench.detail_progress(
                phase_progress,
                "post-bench sample",
                total=2,
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

    return CellRun(
        budget=budget,
        context_length=context_length,
        repetition=repetition,
        label=label,
        summary=summarize_ruler_run(args, runner, label, enabled, artifacts),
    )


def cell_artifacts_complete(args: argparse.Namespace, run_dir: Path) -> bool:
    if not (run_dir / "commands.json").exists():
        return False
    if not args.skip_serving_benchmark and not (run_dir / "bench.json").exists():
        return False
    if args.skip_ruler:
        return True
    ruler = bench.load_json(run_dir / "ruler.json")
    return bench.as_float(ruler.get("score_percent")) is not None


def score_sample(prediction: str | None, references: list[str], task: str) -> float:
    """Score one sample with the official substring match.

    QA tasks score 1.0 when any reference appears in the prediction; every
    other task scores the fraction of references found.
    """
    if prediction is None or not references:
        return 0.0
    if task in QA_TASKS:
        return 1.0 if any(ref in prediction for ref in references) else 0.0
    found = sum(1 for ref in references if ref in prediction)
    return found / len(references)


def evaluate_ruler(
    args: argparse.Namespace,
    context_length: int,
    progress: tqdm | None = None,
) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_name = args.ruler_dataset_template.format(context_length=context_length)
    tasks_result: dict[str, Any] = {}
    num_samples = 0
    for task in args.ruler_tasks:
        split = load_dataset(dataset_name, split=task)
        sample_count = min(args.ruler_samples_per_task, len(split))
        samples: list[dict[str, Any]] = []
        for sample_number in range(sample_count):
            row = split[sample_number]
            prompt = row["input"] + (row.get("answer_prefix") or "")
            references = [str(reference) for reference in row["outputs"]]
            sample: dict[str, Any] = {
                "dataset_index": bench.as_int(row.get("index")),
                "declared_length": bench.as_int(row.get("length")),
                "elapsed_s": None,
                "prediction": None,
                "references": references,
                "sample_number": sample_number,
                "score": 0.0,
                "usage": None,
            }
            started = time.perf_counter()
            try:
                response = bench.post_completion(
                    args,
                    {
                        "prompt": prompt,
                        "max_tokens": args.ruler_max_tokens,
                        "temperature": args.ruler_temperature,
                        "seed": args.seed,
                    },
                )
                sample["prediction"] = response["choices"][0]["text"]
                sample["usage"] = response.get("usage")
                sample["score"] = score_sample(sample["prediction"], references, task)
            except Exception as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                sample["elapsed_s"] = time.perf_counter() - started
            samples.append(sample)
            bench.set_detail_postfix(progress, ruler=task, sample=len(samples))
            bench.advance_progress(progress)
        task_accuracy = (
            sum(sample["score"] for sample in samples) / len(samples)
            if samples
            else 0.0
        )
        tasks_result[task] = {
            "num_samples": len(samples),
            "samples": samples,
            "score_percent": round(100.0 * task_accuracy, 2),
        }
        num_samples += len(samples)

    task_percents = [tasks_result[task]["score_percent"] for task in args.ruler_tasks]
    mean_percent = sum(task_percents) / len(task_percents) if task_percents else None
    return {
        "accuracy": (mean_percent / 100.0 if mean_percent is not None else None),
        "benchmark": "RULER",
        "context_length": context_length,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "num_tasks": len(args.ruler_tasks),
        "samples_per_task": args.ruler_samples_per_task,
        "score_percent": (round(mean_percent, 2) if mean_percent is not None else None),
        "scoring_method": RULER_SCORING_METHOD,
        "tasks": tasks_result,
    }


def summarize_ruler_run(
    args: argparse.Namespace,
    runner: str,
    label: str,
    enabled: bool,
    artifacts: bench.RunArtifacts,
) -> dict[str, Any]:
    benchmark = bench.load_json(Path(artifacts.benchmark_json))
    ruler = bench.load_json(Path(artifacts.run_dir) / "ruler.json")
    validation_errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=args.num_prompts,
        quality_required=False,
        serving_required=not args.skip_serving_benchmark,
    )
    if not args.skip_ruler and bench.as_float(ruler.get("score_percent")) is None:
        validation_errors.append("RULER result has no score metric")

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

    summary = bench.RunSummary(
        runner=runner,
        label=label,
        paged_eviction_enabled=enabled,
        completion_status=("complete" if not validation_errors else "invalid"),
        artifacts=artifacts,
        completed=bench.as_int(benchmark.get("completed")),
        failed=bench.as_int(benchmark.get("failed")),
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
    summary_dict["ruler_score_percent"] = bench.as_float(ruler.get("score_percent"))
    return summary_dict


def default_root_dir_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"ruler-{timestamp}"


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
            "chunked_prefill": True,
            "context_lengths": list(args.context_lengths),
            "kv_cache_dtype": args.kv_cache_dtype,
            "max_concurrency": args.max_concurrency,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "model": args.model,
            "num_prompts": args.num_prompts,
            "output_length": args.random_output_len,
            "prefix_caching": False,
            "quality_repetitions": args.repetitions,
            "repetitions": args.repetitions,
            "ruler_samples_per_task": args.ruler_samples_per_task,
            "ruler_tasks": list(args.ruler_tasks),
            "weight_quantization": args.quantization,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root_dir),
        "runs": [
            {
                "cell": {
                    "cache_budget_tokens": cell.budget,
                    "context_length": cell.context_length,
                    "repetition": cell.repetition,
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
                    "repetition": cell.repetition,
                    "completion_status": summary.get("completion_status"),
                    "validation_passed": summary.get("validation_passed"),
                    "p50_ttft_ms": summary.get("p50_ttft_ms"),
                    "p90_ttft_ms": summary.get("p90_ttft_ms"),
                    "p99_ttft_ms": summary.get("p99_ttft_ms"),
                    "request_throughput": summary.get("request_throughput"),
                    "output_throughput": summary.get("output_throughput"),
                    "total_token_throughput": summary.get("total_token_throughput"),
                    "peak_gpu_memory_delta_mib": summary.get(
                        "peak_gpu_memory_delta_mib"
                    ),
                    "derived_peak_kv_cache_bytes": summary.get(
                        "derived_peak_kv_cache_bytes"
                    ),
                    "ruler_score_percent": summary.get("ruler_score_percent"),
                }
            )


def retention_percent(cells: list[CellRun], cell: CellRun) -> float | None:
    """RULER score relative to the full-cache baseline of the same context."""
    score = bench.as_float(cell.summary.get("ruler_score_percent"))
    if score is None:
        return None
    baseline = next(
        (
            bench.as_float(other.summary.get("ruler_score_percent"))
            for other in cells
            if other.label == FULL_CACHE_LABEL
            and other.context_length == cell.context_length
            and other.repetition == cell.repetition
        ),
        None,
    )
    if baseline is None or baseline == 0:
        return None
    return score / baseline * 100.0


def format_report_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def write_report(
    root_dir: Path,
    args: argparse.Namespace,
    cells: list[CellRun],
) -> None:
    lines = [
        REPORT_TITLE,
        "",
        (
            f"Model: `{args.model}`. KV cache: `{args.kv_cache_dtype}`. "
            f"Attention backend: `{args.attention_backend}`."
        ),
        "",
    ]
    if args.skip_ruler:
        lines.extend(
            [
                ("| Mode | Context | TTFT p50 (ms) | Output tok/s | Peak KV (GiB) |"),
                "|---|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| Mode | Context | TTFT p50 (ms) | Output tok/s | "
                    "Peak KV (GiB) | RULER score | Retention |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for cell in cells:
        summary = cell.summary
        peak_kv_gib = bench.bytes_to_gib(
            bench.as_float(summary.get("derived_peak_kv_cache_bytes"))
        )
        row = [
            cell.label,
            str(cell.context_length),
            format_report_value(bench.as_float(summary.get("p50_ttft_ms"))),
            format_report_value(bench.as_float(summary.get("output_throughput"))),
            format_report_value(peak_kv_gib),
        ]
        if not args.skip_ruler:
            row.append(
                format_report_value(bench.as_float(summary.get("ruler_score_percent")))
            )
            row.append(format_report_value(retention_percent(cells, cell), suffix="%"))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            (
                "Raw commands, server logs, request samples, and memory "
                "traces are stored below each run directory."
            ),
        ]
    )
    if not args.skip_ruler:
        lines.extend(
            [
                "",
                (
                    f"RULER uses {args.ruler_samples_per_task} sample(s) per "
                    "task. Use `--ruler-samples-per-task 500` for the full "
                    "published protocol."
                ),
            ]
        )
    (root_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    modes = list(dict.fromkeys(cell.label for cell in cells))
    context_lengths = list(dict.fromkeys(cell.context_length for cell in cells))
    if not modes or not context_lengths:
        return []

    plots_dir = root_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def grouped_bar(
        filename: str,
        title: str,
        ylabel: str,
        pick: Callable[[CellRun], Any],
        baseline: bool = False,
    ) -> None:
        series = []
        has_value = False
        for context in context_lengths:
            values = [
                bench.as_float(pick(cell))
                for mode in modes
                for cell in cells
                if cell.label == mode and cell.context_length == context
            ]
            has_value = has_value or any(value is not None for value in values)
            series.append((context, values))
        if not has_value:
            return

        positions = list(range(len(modes)))
        width = 0.8 / max(len(series), 1)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for series_index, (context, values) in enumerate(series):
            offset = (series_index - (len(series) - 1) / 2) * width
            bars = ax.bar(
                [position + offset for position in positions],
                [float("nan") if value is None else value for value in values],
                width,
                label=f"ctx {context}",
            )
            for bar, value in zip(bars, values):
                if value is not None:
                    annotation = (
                        f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"
                    )
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        annotation,
                        ha="center",
                        va="bottom",
                    )
        if baseline:
            ax.axhline(100, color="#6c757d", linewidth=1.2, linestyle="--")
        ax.set_xticks(positions, modes)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

    grouped_bar(
        "output_throughput.png",
        "Output-token throughput",
        "tokens / second",
        lambda cell: cell.summary.get("output_throughput"),
    )
    grouped_bar(
        "ttft_p50_ms.png",
        "TTFT p50",
        "milliseconds",
        lambda cell: cell.summary.get("p50_ttft_ms"),
    )
    grouped_bar(
        "peak_kv_cache_gib.png",
        "Peak occupied KV-cache memory",
        "GiB",
        lambda cell: bench.bytes_to_gib(
            bench.as_float(cell.summary.get("derived_peak_kv_cache_bytes"))
        ),
    )
    if not args.skip_ruler:
        grouped_bar(
            "ruler_score.png",
            "RULER score",
            "score (percent)",
            lambda cell: cell.summary.get("ruler_score_percent"),
        )
        grouped_bar(
            "ruler_score_retention.png",
            "RULER score retention vs full cache",
            "retention (%)",
            lambda cell: retention_percent(cells, cell),
            baseline=True,
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

    dry_run = args.dry_run
    with tqdm(
        total=len(cells_spec) * len(runners),
        desc="RULER PagedEviction",
        unit="cell",
        disable=dry_run or args.disable_tqdm,
    ) as progress:
        for runner in runners:
            for budget, context_length, repetition in cells_spec:
                if progress is not None:
                    progress.set_postfix(
                        cell=cell_label(budget),
                        context=context_length,
                        runner=runner,
                        refresh=True,
                    )
                cells.append(
                    run_ruler_cell(
                        args,
                        root_dir,
                        runner,
                        budget,
                        context_length,
                        repetition,
                        progress=progress,
                    )
                )
                progress.update(1)

    write_summary_files(root_dir, args, cells)
    plot_paths = write_plots(root_dir, args, cells)
    write_report(root_dir, args, cells)

    print(f"\nsummary: {root_dir / 'summary.json'}")
    print(f"csv:     {root_dir / 'summary.csv'}")
    print(f"report:  {root_dir / 'REPORT.md'}")
    if plot_paths:
        print(f"plots:   {root_dir / 'plots'}")
    if dry_run:
        return 0
    return int(any(not cell.summary.get("validation_passed") for cell in cells))


if __name__ == "__main__":
    raise SystemExit(main())
