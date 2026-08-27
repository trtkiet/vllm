# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare PagedEviction and KVPress across retained-KV budgets.

Example dry run:

    .venv/bin/python benchmarks/run_paged_eviction_kvpress_comparison.py --dry-run

Install optional HF-side dependencies before running KVPress measurements:

    uv pip install bitsandbytes kvpress matplotlib
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm.auto import tqdm  # noqa: E402

import benchmarks.run_paged_eviction_memory_bench as bench  # noqa: E402

DEFAULT_BUDGETS = (64, 128, 256, 512, 1024)
DEFAULT_KVPRESS_METHODS = (
    "KnormPress",
    "ExpectedAttentionPress",
    "StreamingLLMPress",
    "SnapKVPress",
    "PyramidKVPress",
)
DEFAULT_VLLM_RUNNER = "legacy"
DEFAULT_TRANSFORMERS_QUANTIZATION = "bitsandbytes-4bit"
TRANSFORMERS_QUANTIZATION_CHOICES = (
    "bitsandbytes-4bit",
    "bitsandbytes-8bit",
    "none",
)
FULL_CACHE_METHOD = "full_cache"
NORMALIZED_FIELDS = [
    "backend",
    "method",
    "runner",
    "model",
    "dataset",
    "budget_tokens",
    "metric_name",
    "metric_value",
    "baseline_delta",
    "latency_s",
    "peak_memory_mib",
    "artifact_json",
    "artifact_command_json",
    "artifact_run_dir",
    "completion_status",
]


@dataclass
class KvpressArtifacts:
    run_dir: str
    result_json: str
    command_json: str
    stdout_log: str


@dataclass
class KvpressRun:
    method: str
    budget_tokens: int | None
    completion_status: str
    artifacts: KvpressArtifacts
    latency_s: float | None
    peak_memory_mib: float | None
    gsm8k_accuracy: float | None
    wikitext_continuation_f1: float | None
    wikitext_word_perplexity: float | None
    validation_errors: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = bench.build_arg_parser(
        description=(
            "Compare vLLM PagedEviction against transformers/KVPress over "
            "retained KV-cache budgets."
        ),
        default_results_dir=Path("benchmarks/results/paged_eviction_kvpress"),
        include_block_size=True,
    )
    comparison = parser.add_argument_group("comparison")
    comparison.add_argument(
        "--budgets",
        default=",".join(str(budget) for budget in DEFAULT_BUDGETS),
        help="Comma-separated retained KV-token budgets to evaluate.",
    )
    comparison.add_argument(
        "--kvpress-methods",
        default="auto",
        help=(
            "Comma-separated KVPress methods, or 'auto' to use installed "
            "supported methods."
        ),
    )
    comparison.add_argument(
        "--skip-paged-eviction",
        action="store_true",
        help="Only run the transformers/KVPress side.",
    )
    comparison.add_argument(
        "--skip-kvpress",
        action="store_true",
        help="Only run the vLLM PagedEviction side.",
    )
    comparison.add_argument("--kvpress-device", default="auto")
    comparison.add_argument("--kvpress-dtype", default="auto")
    comparison.add_argument("--kvpress-compression-interval", type=int, default=64)
    comparison.add_argument(
        "--kvpress-hidden-states-buffer-size",
        type=int,
        default=256,
    )
    comparison.add_argument(
        "--transformers-quantization",
        choices=TRANSFORMERS_QUANTIZATION_CHOICES,
        default=DEFAULT_TRANSFORMERS_QUANTIZATION,
        help=(
            "Transformers quantization for KVPress evaluations. Defaults to "
            "bitsandbytes 4-bit. Use 'none' to disable."
        ),
    )
    comparison.add_argument(
        "--quality-min-prompt-budget-tokens",
        type=int,
        default=None,
        help=(
            "Only evaluate quality samples whose prompt fits this many "
            "block-aligned tokens. Defaults to the smallest retained-KV budget "
            "so low-budget vLLM runs do not reject oversized prompts."
        ),
    )
    restrict_runner_to_single_default(parser)
    progress = parser.add_argument_group("progress")
    progress.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable the outer comparison progress bar.",
    )
    return parser


def build_kvpress_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one KVPress HF evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gsm8k-limit", type=int, default=8)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=2048)
    parser.add_argument("--wikitext-limit", type=int, default=4)
    parser.add_argument("--wikitext-max-words", type=int, default=256)
    parser.add_argument("--wikitext-continuation-words", type=int, default=64)
    parser.add_argument("--skip-wikitext-continuation", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--min-prompt-budget-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--transformers-quantization",
        choices=TRANSFORMERS_QUANTIZATION_CHOICES,
        default=DEFAULT_TRANSFORMERS_QUANTIZATION,
    )
    parser.add_argument("--compression-interval", type=int, default=512)
    parser.add_argument("--hidden-states-buffer-size", type=int, default=256)
    return parser


def build_kvpress_batch_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batched KVPress HF evaluations.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gsm8k-limit", type=int, default=8)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=2048)
    parser.add_argument("--wikitext-limit", type=int, default=4)
    parser.add_argument("--wikitext-max-words", type=int, default=256)
    parser.add_argument("--wikitext-continuation-words", type=int, default=64)
    parser.add_argument("--skip-wikitext-continuation", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--min-prompt-budget-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--transformers-quantization",
        choices=TRANSFORMERS_QUANTIZATION_CHOICES,
        default=DEFAULT_TRANSFORMERS_QUANTIZATION,
    )
    parser.add_argument("--compression-interval", type=int, default=512)
    parser.add_argument("--hidden-states-buffer-size", type=int, default=256)
    return parser


def restrict_runner_to_single_default(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if "--runner" not in action.option_strings:
            continue
        action.choices = ("legacy", "v2")
        action.default = DEFAULT_VLLM_RUNNER
        action.help = (
            "vLLM model runner to benchmark. Defaults to legacy; use --runner "
            "v2 to force the V2 runner."
        )
        return
    raise RuntimeError("benchmark parser does not define --runner")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.quantization.lower() in ("", "none"):
        args.quantization = None
    args.block_size = args.block_size or bench.DEFAULT_KV_CACHE_BLOCK_SIZE
    try:
        args.budgets = parse_budgets(args.budgets, args.block_size)
        args.kvpress_methods = parse_methods(args.kvpress_methods)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.quality_min_prompt_budget_tokens is None:
        args.quality_min_prompt_budget_tokens = min(args.budgets)
    if args.quality_min_prompt_budget_tokens <= 0:
        parser.error("--quality-min-prompt-budget-tokens must be positive")
    if args.kvpress_methods == ["auto"]:
        args.kvpress_methods = discover_kvpress_methods()
        if not args.kvpress_methods and not args.skip_kvpress:
            parser.error(
                "no supported KVPress methods were importable; install kvpress "
                "or pass --skip-kvpress"
            )
    if args.smoke:
        args.num_prompts = 8
        args.random_input_len = 512
        args.random_output_len = 64
        args.max_concurrency = 2
        args.gsm8k_limit = min(args.gsm8k_limit, 2)
        args.wikitext_limit = min(args.wikitext_limit, 2)
    return args


def parse_budgets(value: str, block_size: int) -> list[int]:
    budgets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("budget list contains an empty item")
        try:
            budget = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"budget {item!r} is not an integer"
            ) from exc
        if budget <= 0:
            raise argparse.ArgumentTypeError("budgets must be positive")
        if budget % block_size != 0:
            raise argparse.ArgumentTypeError(
                f"budget {budget} must be aligned to block size {block_size}"
            )
        budgets.append(budget)
    return budgets


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",")]
    if not methods or any(not method for method in methods):
        raise argparse.ArgumentTypeError("KVPress method list contains an empty item")
    if methods == ["auto"]:
        return methods
    unknown = [method for method in methods if method not in DEFAULT_KVPRESS_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unsupported KVPress methods: " + ",".join(unknown)
        )
    return methods


def discover_kvpress_methods() -> list[str]:
    try:
        import kvpress  # type: ignore[import-not-found]
    except ImportError:
        return []
    return [
        method
        for method in DEFAULT_KVPRESS_METHODS
        if getattr(kvpress, method, None) is not None
    ]


def run_paged_sweep(
    args: argparse.Namespace,
    root_dir: Path,
    progress: tqdm | None = None,
) -> dict[str, dict[int | None, bench.RunSummary]]:
    runs: dict[str, dict[int | None, bench.RunSummary]] = {}
    paged_root = root_dir / "paged_eviction"
    for runner in bench.selected_runners(args.runner):
        runs[runner] = {}
        baseline_args = paged_args_for_budget(args, None)
        if args.dry_run:
            print_paged_command(paged_root / "full_cache", baseline_args, runner, None)
        if progress is not None:
            progress.set_postfix(
                backend="vllm",
                runner=runner,
                budget="full_cache",
                refresh=True,
            )
        runs[runner][None] = bench.run_one(
            baseline_args,
            paged_root / "full_cache",
            runner,
            "full_cache",
            enabled=False,
            **inner_progress_kwargs(
                args,
                progress,
                f"paged {runner}/full_cache",
            ),
        )
        if progress is not None:
            progress.update()
        for budget in args.budgets:
            budget_args = paged_args_for_budget(args, budget)
            budget_root = paged_root / f"budget_{budget}"
            if args.dry_run:
                print_paged_command(budget_root, budget_args, runner, budget)
            if progress is not None:
                progress.set_postfix(
                    backend="vllm",
                    runner=runner,
                    budget=budget,
                    refresh=True,
                )
            runs[runner][budget] = bench.run_one(
                budget_args,
                budget_root,
                runner,
                f"budget_{budget}",
                enabled=True,
                **inner_progress_kwargs(
                    args,
                    progress,
                    f"paged {runner}/budget_{budget}",
                ),
            )
            if progress is not None:
                progress.update()
    return runs


def paged_args_for_budget(
    args: argparse.Namespace,
    budget: int | None,
) -> argparse.Namespace:
    budget_args = argparse.Namespace(**vars(args))
    budget_args.cache_budget_tokens = budget or args.cache_budget_tokens
    budget_args.quality_min_prompt_budget_tokens = args.quality_min_prompt_budget_tokens
    budget_args.quality_block_size = args.block_size
    budget_args.skip_serving_benchmark = True
    return budget_args


def print_paged_command(
    root_dir: Path,
    args: argparse.Namespace,
    runner: str,
    budget: int | None,
) -> None:
    enabled = budget is not None
    label = "full_cache" if budget is None else f"budget_{budget}"
    run_dir = root_dir / runner / label
    print(f"\n[paged/{runner}/{label}] server:")
    print(shlex.join(bench.build_server_command(args, enabled)))
    if args.skip_serving_benchmark:
        print(f"[paged/{runner}/{label}] benchmark: skipped")
    else:
        print(f"[paged/{runner}/{label}] benchmark:")
        print(shlex.join(bench.build_benchmark_command(args, run_dir, label)))


def run_kvpress_sweep(
    args: argparse.Namespace,
    root_dir: Path,
    progress: tqdm | None = None,
) -> dict[str, dict[int | None, KvpressRun]]:
    if not args.dry_run:
        return run_kvpress_batch(args, root_dir, progress)

    runs: dict[str, dict[int | None, KvpressRun]] = {}
    kvpress_root = root_dir / "kvpress"
    for method in [FULL_CACHE_METHOD, *args.kvpress_methods]:
        budgets = [None] if method == FULL_CACHE_METHOD else args.budgets
        runs[method] = {}
        for budget in budgets:
            if progress is not None:
                progress.set_postfix(
                    backend="kvpress",
                    method=method,
                    budget=budget or "full_cache",
                    refresh=True,
                )
            run = run_kvpress_one(
                args,
                kvpress_root,
                method,
                budget,
                **inner_progress_kwargs(
                    args,
                    progress,
                    f"kvpress {method}/{budget or 'full_cache'}",
                ),
            )
            runs[method][budget] = run
            if progress is not None:
                progress.update()
    return runs


def run_kvpress_batch(
    args: argparse.Namespace,
    root_dir: Path,
    progress: tqdm | None = None,
) -> dict[str, dict[int | None, KvpressRun]]:
    runs: dict[str, dict[int | None, KvpressRun]] = {}
    kvpress_root = root_dir / "kvpress"
    batch_dir = kvpress_root / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    artifacts_by_job: dict[tuple[str, int | None], KvpressArtifacts] = {}
    for method in [FULL_CACHE_METHOD, *args.kvpress_methods]:
        budgets = [None] if method == FULL_CACHE_METHOD else args.budgets
        runs[method] = {}
        for budget in budgets:
            artifacts = make_kvpress_artifacts(
                kvpress_root,
                method,
                budget,
                stdout_log=batch_dir / "stdout.log",
            )
            artifacts_by_job[(method, budget)] = artifacts
            jobs.append(
                {
                    "method": method,
                    "budget_tokens": budget,
                    "output_json": artifacts.result_json,
                }
            )

    jobs_json = batch_dir / "jobs.json"
    command_json = batch_dir / "commands.json"
    bench.write_json(jobs_json, {"jobs": jobs})
    command = build_kvpress_batch_eval_command(args, jobs_json)
    bench.write_json(command_json, {"command": command, "jobs": jobs})
    for job in jobs:
        method = str(job["method"])
        budget = job["budget_tokens"]
        assert budget is None or isinstance(budget, int)
        artifacts = artifacts_by_job[(method, budget)]
        bench.write_json(
            Path(artifacts.command_json),
            {
                "backend": "transformers",
                "method": method,
                "budget_tokens": budget,
                "command": command,
                "batch_jobs_json": str(jobs_json),
            },
        )

    started = time.monotonic()
    if progress is not None:
        progress.set_postfix(
            backend="kvpress",
            method="batch",
            budget="all",
            refresh=True,
        )
    with Path(batch_dir / "stdout.log").open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        completed_job_indexes: set[int] = set()
        while proc.poll() is None:
            time.sleep(1)
            update_completed_kvpress_jobs(jobs, completed_job_indexes, progress)
        update_completed_kvpress_jobs(jobs, completed_job_indexes, progress)

    for index, job in enumerate(jobs):
        method = str(job["method"])
        budget = job["budget_tokens"]
        assert budget is None or isinstance(budget, int)
        artifacts = artifacts_by_job[(method, budget)]
        run = collect_kvpress_run(
            method,
            budget,
            artifacts,
            command_failed=proc.returncode != 0,
            fallback_latency_s=time.monotonic() - started,
        )
        runs[method][budget] = run
        if progress is not None and index not in completed_job_indexes:
            progress.update()
    return runs


def update_completed_kvpress_jobs(
    jobs: list[dict[str, Any]],
    completed_job_indexes: set[int],
    progress: tqdm | None,
) -> None:
    if progress is None:
        return
    for index, job in enumerate(jobs):
        if index in completed_job_indexes:
            continue
        if not Path(str(job["output_json"])).exists():
            continue
        completed_job_indexes.add(index)
        progress.set_postfix(
            backend="kvpress",
            method=str(job["method"]),
            budget=job["budget_tokens"] or "full_cache",
            completed=f"{len(completed_job_indexes)}/{len(jobs)}",
            refresh=True,
        )
        progress.update()


def progress_total(args: argparse.Namespace) -> int:
    total = 0
    if not args.skip_paged_eviction:
        total += len(bench.selected_runners(args.runner)) * (1 + len(args.budgets))
    if not args.skip_kvpress:
        total += 1 + len(args.kvpress_methods) * len(args.budgets)
    return total


def inner_progress_kwargs(
    args: argparse.Namespace,
    progress: tqdm | None,
    label: str,
) -> dict[str, Any]:
    if progress is None or args.disable_tqdm or args.dry_run:
        return {}
    return {"show_progress": True, "progress_label": label}


def make_kvpress_artifacts(
    root_dir: Path,
    method: str,
    budget: int | None,
    *,
    stdout_log: Path | None = None,
) -> KvpressArtifacts:
    label = "full_cache" if budget is None else f"budget_{budget}"
    run_dir = root_dir / method / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return KvpressArtifacts(
        run_dir=str(run_dir),
        result_json=str(run_dir / "result.json"),
        command_json=str(run_dir / "commands.json"),
        stdout_log=str(stdout_log or run_dir / "stdout.log"),
    )


def collect_kvpress_run(
    method: str,
    budget: int | None,
    artifacts: KvpressArtifacts,
    *,
    command_failed: bool,
    fallback_latency_s: float | None,
) -> KvpressRun:
    result_path = Path(artifacts.result_json)
    result = bench.load_json(result_path)
    validation_errors = []
    if command_failed and not result_path.exists():
        validation_errors.append("KVPress command failed before writing result")
    if result.get("error"):
        validation_errors.append(f"KVPress evaluation failed: {result['error']}")
    if not result.get("quality_skipped"):
        if bench.as_float(result.get("gsm8k_accuracy")) is None:
            validation_errors.append("KVPress result has no GSM8K accuracy")
        if bench.as_float(result.get("wikitext_word_perplexity")) is None:
            validation_errors.append("KVPress result has no WikiText word_perplexity")
    latency_s = bench.as_float(result.get("latency_s"))
    return KvpressRun(
        method=method,
        budget_tokens=budget,
        completion_status="invalid" if validation_errors else "complete",
        artifacts=artifacts,
        latency_s=latency_s if latency_s is not None else fallback_latency_s,
        peak_memory_mib=bench.as_float(result.get("peak_memory_mib")),
        gsm8k_accuracy=bench.as_float(result.get("gsm8k_accuracy")),
        wikitext_continuation_f1=bench.as_float(result.get("wikitext_continuation_f1")),
        wikitext_word_perplexity=bench.as_float(result.get("wikitext_word_perplexity")),
        validation_errors=validation_errors,
    )


def run_kvpress_one(
    args: argparse.Namespace,
    root_dir: Path,
    method: str,
    budget: int | None,
    *,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> KvpressRun:
    phase_progress = None
    if show_progress:
        phase_progress = tqdm(
            total=3,
            desc=progress_label or f"kvpress {method}",
            unit="phase",
            leave=False,
            position=1,
        )
    label = "full_cache" if budget is None else f"budget_{budget}"
    artifacts = make_kvpress_artifacts(root_dir, method, budget)
    set_phase_progress(phase_progress, "prepare")
    with detail_progress(phase_progress, "kvpress prepare", total=2) as detail:
        command = build_kvpress_eval_command(
            args,
            Path(artifacts.result_json),
            method,
            budget,
        )
        advance_detail_progress(detail)
        bench.write_json(
            Path(artifacts.command_json),
            {
                "backend": "transformers",
                "method": method,
                "budget_tokens": budget,
                "command": command,
            },
        )
        advance_detail_progress(detail)
    advance_phase_progress(phase_progress)
    if args.dry_run:
        close_phase_progress(phase_progress)
        print(f"\n[kvpress/{method}/{label}]:")
        print(shlex.join(command))
        return KvpressRun(
            method=method,
            budget_tokens=budget,
            completion_status="not_run",
            artifacts=artifacts,
            latency_s=None,
            peak_memory_mib=None,
            gsm8k_accuracy=None,
            wikitext_continuation_f1=None,
            wikitext_word_perplexity=None,
            validation_errors=[],
        )

    started = time.monotonic()
    set_phase_progress(phase_progress, "evaluate")
    with (
        detail_progress(phase_progress, "kvpress evaluate elapsed", unit="s") as detail,
        Path(artifacts.stdout_log).open("w", encoding="utf-8") as log_file,
    ):
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while proc.poll() is None:
            time.sleep(1)
            advance_detail_progress(detail)
    advance_phase_progress(phase_progress)
    set_phase_progress(phase_progress, "collect")
    with detail_progress(phase_progress, "kvpress collect", total=1) as detail:
        run = collect_kvpress_run(
            method,
            budget,
            artifacts,
            command_failed=proc.returncode != 0,
            fallback_latency_s=time.monotonic() - started,
        )
        advance_detail_progress(detail)
    advance_phase_progress(phase_progress)
    close_phase_progress(phase_progress)
    return run


def set_phase_progress(progress: tqdm | None, phase: str) -> None:
    if progress is not None:
        progress.set_postfix(phase=phase, refresh=True)


def advance_phase_progress(progress: tqdm | None) -> None:
    if progress is not None:
        progress.update()


def close_phase_progress(progress: tqdm | None) -> None:
    if progress is not None:
        progress.close()


def detail_progress(
    parent: tqdm | None,
    desc: str,
    *,
    total: float | None = None,
    unit: str = "step",
) -> AbstractContextManager[tqdm | None]:
    if parent is None:
        return contextlib.nullcontext(None)
    return tqdm(
        total=total,
        desc=f"  {desc}",
        unit=unit,
        leave=False,
        position=2,
    )


def advance_detail_progress(progress: tqdm | None, amount: float = 1) -> None:
    if progress is not None:
        progress.update(amount)


def build_kvpress_eval_command(
    args: argparse.Namespace,
    output_json: Path,
    method: str,
    budget: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "kvpress-eval",
        "--model",
        args.model,
        "--method",
        method,
        "--output-json",
        str(output_json),
        "--seed",
        str(args.seed),
        "--gsm8k-limit",
        str(args.gsm8k_limit),
        "--gsm8k-max-tokens",
        str(args.gsm8k_max_tokens),
        "--wikitext-limit",
        str(args.wikitext_limit),
        "--wikitext-max-words",
        str(args.wikitext_max_words),
        "--wikitext-continuation-words",
        str(args.wikitext_continuation_words),
        "--block-size",
        str(args.block_size),
        "--device",
        args.kvpress_device,
        "--dtype",
        args.kvpress_dtype,
        "--transformers-quantization",
        args.transformers_quantization,
        "--compression-interval",
        str(args.kvpress_compression_interval),
        "--hidden-states-buffer-size",
        str(args.kvpress_hidden_states_buffer_size),
    ]
    if budget is not None:
        command.extend(["--budget-tokens", str(budget)])
    if getattr(args, "quality_min_prompt_budget_tokens", None) is not None:
        command.extend(
            [
                "--min-prompt-budget-tokens",
                str(args.quality_min_prompt_budget_tokens),
            ]
        )
    if args.skip_wikitext_continuation:
        command.append("--skip-wikitext-continuation")
    if args.skip_quality:
        command.append("--skip-quality")
    return command


def build_kvpress_batch_eval_command(
    args: argparse.Namespace,
    jobs_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "kvpress-eval-batch",
        "--model",
        args.model,
        "--jobs-json",
        str(jobs_json),
        "--seed",
        str(args.seed),
        "--gsm8k-limit",
        str(args.gsm8k_limit),
        "--gsm8k-max-tokens",
        str(args.gsm8k_max_tokens),
        "--wikitext-limit",
        str(args.wikitext_limit),
        "--wikitext-max-words",
        str(args.wikitext_max_words),
        "--wikitext-continuation-words",
        str(args.wikitext_continuation_words),
        "--block-size",
        str(args.block_size),
        "--device",
        args.kvpress_device,
        "--dtype",
        args.kvpress_dtype,
        "--transformers-quantization",
        args.transformers_quantization,
        "--compression-interval",
        str(args.kvpress_compression_interval),
        "--hidden-states-buffer-size",
        str(args.kvpress_hidden_states_buffer_size),
    ]
    if getattr(args, "quality_min_prompt_budget_tokens", None) is not None:
        command.extend(
            [
                "--min-prompt-budget-tokens",
                str(args.quality_min_prompt_budget_tokens),
            ]
        )
    if args.skip_wikitext_continuation:
        command.append("--skip-wikitext-continuation")
    if args.skip_quality:
        command.append("--skip-quality")
    return command


def normalize_rows(
    model: str,
    paged_runs: dict[str, dict[int | None, bench.RunSummary]],
    kvpress_runs: dict[str, dict[int | None, KvpressRun]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for runner, runs_by_budget in paged_runs.items():
        for budget, run in runs_by_budget.items():
            method = "full_cache" if budget is None else f"PagedEviction-{runner}"
            rows.extend(
                normalized_metric_rows(
                    backend="vllm",
                    method=method,
                    runner=runner,
                    model=model,
                    budget=budget,
                    metrics={
                        "gsm8k": {"accuracy": run.gsm8k_accuracy},
                        "wikitext": {
                            "continuation_f1": run.wikitext_continuation_f1,
                            "word_perplexity": run.wikitext_word_perplexity,
                        },
                    },
                    latency_s=ms_to_s(run.median_ttft_ms),
                    peak_memory_mib=run.peak_benchmark_gpu_memory_mib,
                    artifact_json=run.artifacts.benchmark_json,
                    artifact_command_json=run.artifacts.command_json,
                    artifact_run_dir=run.artifacts.run_dir,
                    completion_status=run.completion_status,
                )
            )
    for method, runs_by_budget in kvpress_runs.items():
        for budget, run in runs_by_budget.items():
            rows.extend(
                normalized_metric_rows(
                    backend="transformers",
                    method=method,
                    runner="",
                    model=model,
                    budget=budget,
                    metrics={
                        "gsm8k": {"accuracy": run.gsm8k_accuracy},
                        "wikitext": {
                            "continuation_f1": run.wikitext_continuation_f1,
                            "word_perplexity": run.wikitext_word_perplexity,
                        },
                    },
                    latency_s=run.latency_s,
                    peak_memory_mib=run.peak_memory_mib,
                    artifact_json=run.artifacts.result_json,
                    artifact_command_json=run.artifacts.command_json,
                    artifact_run_dir=run.artifacts.run_dir,
                    completion_status=run.completion_status,
                )
            )
    add_baseline_deltas(rows)
    return rows


def normalized_metric_rows(
    *,
    backend: str,
    method: str,
    runner: str,
    model: str,
    budget: int | None,
    metrics: dict[str, dict[str, float | None]],
    latency_s: float | None,
    peak_memory_mib: float | None,
    artifact_json: str,
    artifact_command_json: str,
    artifact_run_dir: str,
    completion_status: str,
) -> list[dict[str, Any]]:
    rows = []
    for dataset, dataset_metrics in metrics.items():
        for metric_name, metric_value in dataset_metrics.items():
            if metric_value is None:
                continue
            rows.append(
                {
                    "backend": backend,
                    "method": method,
                    "runner": runner,
                    "model": model,
                    "dataset": dataset,
                    "budget_tokens": budget,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "baseline_delta": None,
                    "latency_s": latency_s,
                    "peak_memory_mib": peak_memory_mib,
                    "artifact_json": artifact_json,
                    "artifact_command_json": artifact_command_json,
                    "artifact_run_dir": artifact_run_dir,
                    "completion_status": completion_status,
                }
            )
    return rows


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["budget_tokens"] is not None or row["method"] != FULL_CACHE_METHOD:
            continue
        key = baseline_key(row)
        baselines[key] = float(row["metric_value"])

    for row in rows:
        baseline = baselines.get(baseline_key(row))
        value = bench.as_float(row["metric_value"])
        if baseline is None or value is None or baseline == 0:
            row["baseline_delta"] = None
        else:
            row["baseline_delta"] = (value - baseline) / baseline * 100


def baseline_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["backend"]),
        str(row["runner"]),
        str(row["dataset"]),
        str(row["metric_name"]),
    )


def ms_to_s(value: float | None) -> float | None:
    return value / 1000 if value is not None else None


def write_normalized_outputs(root_dir: Path, rows: list[dict[str, Any]]) -> None:
    bench.write_json(root_dir / "normalized_results.json", {"rows": rows})
    with (root_dir / "normalized_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_file_name(dataset: str, metric_name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in f"{dataset}_{metric_name}".lower()
    )
    return f"{safe}.png"


def write_plots(root_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return []

    plots_dir = root_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["metric_name"]), []).append(row)

    for (dataset, metric_name), group_rows in grouped.items():
        fig, ax = plt.subplots()
        series: dict[str, list[tuple[int, float]]] = {}
        baselines = []
        for row in group_rows:
            value = bench.as_float(row["metric_value"])
            if value is None:
                continue
            label = f"{row['backend']}:{row['method']}"
            if row["runner"]:
                label = f"{label}:{row['runner']}"
            budget = row["budget_tokens"]
            if budget is None:
                baselines.append((label, value))
            else:
                series.setdefault(label, []).append((int(budget), value))
        for label, points in series.items():
            points = sorted(points)
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=label,
            )
        for label, value in baselines:
            ax.axhline(value, linestyle="--", linewidth=1, label=label)
        ax.set_xlabel("retained KV budget tokens")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{dataset} {metric_name}")
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = plots_dir / plot_file_name(dataset, metric_name)
        fig.savefig(path)
        plt.close(fig)
        plot_paths.append(str(path))
    return plot_paths


def write_summary(
    root_dir: Path,
    args: argparse.Namespace,
    paged_runs: dict[str, dict[int | None, bench.RunSummary]],
    kvpress_runs: dict[str, dict[int | None, KvpressRun]],
    rows: list[dict[str, Any]],
    plot_paths: list[str],
) -> None:
    bench.write_json(
        root_dir / "summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(root_dir),
            "config": {
                "model": args.model,
                "budgets": args.budgets,
                "block_size": args.block_size,
                "kvpress_methods": args.kvpress_methods,
                "runner": args.runner,
                "vllm_quantization": args.quantization,
                "transformers_quantization": args.transformers_quantization,
                "quality_skipped": args.skip_quality,
                "quality_min_prompt_budget_tokens": (
                    getattr(args, "quality_min_prompt_budget_tokens", None)
                ),
                "skip_wikitext_continuation": args.skip_wikitext_continuation,
            },
            "paged_eviction_runs": {
                runner: {
                    str(budget) if budget is not None else "full_cache": asdict(run)
                    for budget, run in runs_by_budget.items()
                }
                for runner, runs_by_budget in paged_runs.items()
            },
            "kvpress_runs": {
                method: {
                    str(budget) if budget is not None else "full_cache": asdict(run)
                    for budget, run in runs_by_budget.items()
                }
                for method, runs_by_budget in kvpress_runs.items()
            },
            "normalized_results_json": str(root_dir / "normalized_results.json"),
            "normalized_results_csv": str(root_dir / "normalized_results.csv"),
            "plots": plot_paths,
            "row_count": len(rows),
        },
    )


def has_validation_errors(
    paged_runs: dict[str, dict[int | None, bench.RunSummary]],
    kvpress_runs: dict[str, dict[int | None, KvpressRun]],
) -> bool:
    return any(
        run.validation_errors
        for runs_by_budget in paged_runs.values()
        for run in runs_by_budget.values()
    ) or any(
        run.validation_errors
        for runs_by_budget in kvpress_runs.values()
        for run in runs_by_budget.values()
    )


def validate_comparable_sample_sets(
    paged_runs: dict[str, dict[int | None, bench.RunSummary]],
    kvpress_runs: dict[str, dict[int | None, KvpressRun]],
) -> None:
    for runner, runs_by_budget in paged_runs.items():
        baseline = runs_by_budget.get(None)
        if baseline is None:
            continue
        baseline_samples = paged_sample_sets(baseline)
        for budget, run in runs_by_budget.items():
            if budget is None:
                continue
            append_sample_set_errors(
                run.validation_errors,
                baseline_samples,
                paged_sample_sets(run),
                f"vLLM {runner} budget {budget}",
            )
            if run.validation_errors:
                run.completion_status = "invalid"

    kvpress_baseline = kvpress_runs.get(FULL_CACHE_METHOD, {}).get(None)
    if kvpress_baseline is None:
        return
    baseline_samples = kvpress_sample_sets(kvpress_baseline)
    for method, runs_by_budget in kvpress_runs.items():
        if method == FULL_CACHE_METHOD:
            continue
        for budget, run in runs_by_budget.items():
            append_sample_set_errors(
                run.validation_errors,
                baseline_samples,
                kvpress_sample_sets(run),
                f"KVPress {method} budget {budget}",
            )
            if run.validation_errors:
                run.completion_status = "invalid"


def paged_sample_sets(run: bench.RunSummary) -> dict[str, list[int] | None]:
    return {
        "gsm8k": sample_indices_from_json(Path(run.artifacts.gsm8k_json)),
        "wikitext": sample_indices_from_json(Path(run.artifacts.wikitext_json)),
    }


def kvpress_sample_sets(run: KvpressRun) -> dict[str, list[int] | None]:
    result = bench.load_json(Path(run.artifacts.result_json))
    return {
        "gsm8k": nested_sample_indices(result, "gsm8k"),
        "wikitext": nested_sample_indices(result, "wikitext"),
    }


def sample_indices_from_json(path: Path) -> list[int] | None:
    return normalize_sample_indices(bench.load_json(path).get("sample_indices"))


def nested_sample_indices(result: dict[str, Any], dataset: str) -> list[int] | None:
    value = result.get(dataset)
    if not isinstance(value, dict):
        return None
    return normalize_sample_indices(value.get("sample_indices"))


def normalize_sample_indices(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    indices = []
    for item in value:
        index = bench.as_int(item)
        if index is None:
            return None
        indices.append(index)
    return indices


def append_sample_set_errors(
    errors: list[str],
    baseline_samples: dict[str, list[int] | None],
    run_samples: dict[str, list[int] | None],
    label: str,
) -> None:
    for dataset, baseline_indices in baseline_samples.items():
        run_indices = run_samples.get(dataset)
        if baseline_indices is None or run_indices is None:
            continue
        if run_indices != baseline_indices:
            errors.append(f"{label} {dataset} sample_indices differ from full_cache")


def run_kvpress_eval(args: argparse.Namespace) -> int:
    if args.skip_quality:
        write_skipped_kvpress_result(args)
        return 0
    torch, load_dataset, model, tokenizer, device = load_transformers_eval_model(args)
    run_kvpress_eval_loaded(args, torch, load_dataset, model, tokenizer, device)
    return 0


def run_kvpress_eval_batch(args: argparse.Namespace) -> int:
    jobs = bench.load_json(args.jobs_json).get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError(f"Expected jobs list in {args.jobs_json}")
    if args.skip_quality:
        write_skipped_kvpress_batch_results(args, jobs)
        return 0

    torch, load_dataset, model, tokenizer, device = load_transformers_eval_model(args)
    failed = False
    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            failed = True
            continue
        job_args = argparse.Namespace(**vars(args))
        job_args.method = job.get("method")
        job_args.budget_tokens = job.get("budget_tokens")
        job_args.output_json = Path(str(job.get("output_json")))
        label = kvpress_job_label(job_args.method, job_args.budget_tokens)
        kvpress_log(f"[kvpress-batch] start {index}/{total_jobs} {label}")
        job_started = time.monotonic()
        try:
            run_kvpress_eval_loaded(
                job_args,
                torch,
                load_dataset,
                model,
                tokenizer,
                device,
            )
        except Exception as exc:
            failed = True
            kvpress_log(
                f"[kvpress-batch] failed {index}/{total_jobs} {label}: "
                f"{type(exc).__name__}: {exc}"
            )
            bench.write_json(
                job_args.output_json,
                {
                    "backend": "transformers",
                    "method": job_args.method,
                    "budget_tokens": job_args.budget_tokens,
                    "model": args.model,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        else:
            kvpress_log(
                f"[kvpress-batch] complete {index}/{total_jobs} {label} "
                f"in {time.monotonic() - job_started:.1f}s"
            )
    return int(failed)


def write_skipped_kvpress_batch_results(
    args: argparse.Namespace,
    jobs: list[Any],
) -> None:
    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            continue
        job_args = argparse.Namespace(**vars(args))
        job_args.method = job.get("method")
        job_args.budget_tokens = job.get("budget_tokens")
        job_args.output_json = Path(str(job.get("output_json")))
        kvpress_log(
            "[kvpress-batch] skipped "
            f"{index}/{total_jobs} "
            f"{kvpress_job_label(job_args.method, job_args.budget_tokens)} "
            "because --skip-quality was set"
        )
        write_skipped_kvpress_result(job_args)


def write_skipped_kvpress_result(args: argparse.Namespace) -> None:
    bench.write_json(
        args.output_json,
        {
            "backend": "transformers",
            "method": args.method,
            "budget_tokens": args.budget_tokens,
            "model": args.model,
            "quality_skipped": True,
            "latency_s": 0.0,
            "peak_memory_mib": None,
            "gsm8k_accuracy": None,
            "wikitext_continuation_f1": None,
            "wikitext_word_perplexity": None,
            "gsm8k": {"skipped": True},
            "wikitext": {"skipped": True},
        },
    )


def kvpress_job_label(method: Any, budget: Any) -> str:
    budget_label = "full_cache" if budget is None else f"budget_{budget}"
    return f"{method}/{budget_label}"


def kvpress_log(message: str) -> None:
    print(message, flush=True)


def load_transformers_eval_model(args: argparse.Namespace) -> tuple[Any, ...]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = build_transformers_model_kwargs(args, torch, BitsAndBytesConfig)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto" and args.transformers_quantization == "none":
        model = model.to(args.device)
    model.eval()
    device = next(model.parameters()).device
    return torch, load_dataset, model, tokenizer, device


def run_kvpress_eval_loaded(
    args: argparse.Namespace,
    torch: Any,
    load_dataset: Any,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> None:
    started = time.monotonic()
    if torch.accelerator.is_available():
        torch.accelerator.reset_peak_memory_stats()

    press_context = contextlib.nullcontext()
    if args.method != FULL_CACHE_METHOD:
        press_context = build_decoding_press(args)(model)

    with torch.inference_mode(), press_context:
        gsm8k = eval_gsm8k_hf(args, model, tokenizer, device, load_dataset)
        wikitext = eval_wikitext_hf(args, model, tokenizer, device, load_dataset)

    peak_memory_mib = None
    if torch.accelerator.is_available():
        peak_memory_mib = torch.accelerator.max_memory_allocated() / (1024**2)
    result = {
        "backend": "transformers",
        "method": args.method,
        "budget_tokens": args.budget_tokens,
        "model": args.model,
        "latency_s": time.monotonic() - started,
        "peak_memory_mib": peak_memory_mib,
        "gsm8k_accuracy": gsm8k["accuracy"],
        "wikitext_continuation_f1": wikitext.get("continuation_f1"),
        "wikitext_word_perplexity": wikitext["word_perplexity"],
        "gsm8k": gsm8k,
        "wikitext": wikitext,
    }
    bench.write_json(args.output_json, result)


def build_transformers_model_kwargs(
    args: argparse.Namespace,
    torch: Any,
    bitsandbytes_config_cls: Any,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    if args.dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.dtype)

    quantization = args.transformers_quantization
    if quantization == "bitsandbytes-4bit":
        model_kwargs["quantization_config"] = bitsandbytes_config_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype", torch.bfloat16),
        )
    elif quantization == "bitsandbytes-8bit":
        model_kwargs["quantization_config"] = bitsandbytes_config_cls(load_in_8bit=True)
    elif quantization != "none":
        raise ValueError(f"unsupported transformers quantization: {quantization}")

    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    elif quantization != "none":
        model_kwargs["device_map"] = {"": args.device}
    return model_kwargs


def build_decoding_press(args: argparse.Namespace) -> Any:
    import kvpress  # type: ignore[import-not-found]

    method_cls = getattr(kvpress, args.method)
    prefill_press = build_target_size_prefill_press(
        kvpress,
        method_cls(),
        args.budget_tokens,
    )
    decoding_press_cls = compatible_decoding_press_cls(kvpress.DecodingPress)
    decoding_press = decoding_press_cls(
        base_press=method_cls(),
        compression_interval=args.compression_interval,
        target_size=args.budget_tokens,
        hidden_states_buffer_size=args.hidden_states_buffer_size,
    )
    prefill_decoding_press_cls = compatible_decoding_press_cls(
        kvpress.PrefillDecodingPress
    )
    return prefill_decoding_press_cls(
        prefilling_press=prefill_press,
        decoding_press=decoding_press,
    )


def build_target_size_prefill_press(
    kvpress: Any,
    base_press: Any,
    target_size: int,
) -> Any:
    class TargetSizePrefillPress(kvpress.BasePress):  # type: ignore[misc, valid-type]
        def __init__(self, base_press: Any, target_size: int) -> None:
            self.base_press = base_press
            self.target_size = target_size

        def post_init_from_model(self, model: Any) -> None:
            self.base_press.post_init_from_model(model)

        def compress(
            self,
            module: Any,
            hidden_states: Any,
            keys: Any,
            values: Any,
            attentions: Any,
            kwargs: dict,
        ) -> tuple[Any, Any]:
            original_ratio = getattr(self.base_press, "compression_ratio", None)
            if original_ratio is None:
                raise TypeError(
                    f"{type(self.base_press).__name__} does not expose "
                    "compression_ratio"
                )
            self.base_press.compression_ratio = target_size_compression_ratio(
                int(keys.shape[2]),
                self.target_size,
            )
            try:
                return self.base_press.compress(
                    module,
                    hidden_states,
                    keys,
                    values,
                    attentions,
                    kwargs,
                )
            finally:
                self.base_press.compression_ratio = original_ratio

    return TargetSizePrefillPress(base_press, target_size)


def target_size_compression_ratio(seq_len: int, target_tokens: int) -> float:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if seq_len <= target_tokens:
        return 0.0

    ratio = 1.0 - (target_tokens / seq_len)
    low = 0.0
    high = 1.0
    for _ in range(20):
        n_kept = int(seq_len * (1.0 - ratio))
        if n_kept == target_tokens:
            return ratio
        if n_kept > target_tokens:
            low = ratio
        else:
            high = ratio
        ratio = (low + high) / 2.0
    return ratio


def compatible_decoding_press_cls(decoding_press_cls: Any) -> Any:
    class CompatibleDecodingPress(decoding_press_cls):  # type: ignore[misc, valid-type]
        def forward_hook(
            self,
            module: Any,
            input: list[Any],
            kwargs: dict,
            output: list,
        ):
            if "cache_position" not in kwargs:
                cache_position = cache_position_from_kwargs(kwargs)
                if cache_position is not None:
                    kwargs = {**kwargs, "cache_position": cache_position}
            return super().forward_hook(module, input, kwargs, output)

    return CompatibleDecodingPress


def cache_position_from_kwargs(kwargs: dict[str, Any]) -> Any | None:
    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        return None
    if getattr(position_ids, "ndim", None) == 2:
        return position_ids[0]
    return position_ids


def eval_gsm8k_hf(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: Any,
    load_dataset: Any,
) -> dict[str, Any]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    correct = 0
    invalid = 0
    evaluated = 0
    samples = []
    for sample_index, item in enumerate(dataset):
        prompt = f"Question: {item['question']}\nAnswer: Let's think step by step."
        if not hf_prompt_fits_budget(args, tokenizer, prompt):
            continue
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.gsm8k_max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(
            output_ids[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        prediction = bench.extract_gsm8k_answer(completion)
        expected = bench.extract_gsm8k_answer(item["answer"])
        invalid += prediction is None
        is_correct = prediction is not None and prediction == expected
        correct += is_correct
        samples.append(
            {
                "sample_index": sample_index,
                "prediction": prediction,
                "expected": expected,
                "correct": is_correct,
            }
        )
        evaluated += 1
        if evaluated == args.gsm8k_limit:
            break
    if evaluated != args.gsm8k_limit:
        raise ValueError(
            f"evaluated {evaluated} GSM8K samples; expected {args.gsm8k_limit}"
        )
    return {
        "accuracy": correct / evaluated if evaluated else None,
        "correct": correct,
        "invalid": invalid,
        "num_questions": evaluated,
        "sample_indices": [sample["sample_index"] for sample in samples],
        "samples": samples,
    }


def eval_wikitext_hf(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: Any,
    load_dataset: Any,
) -> dict[str, Any]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [
        (sample_index, text.strip())
        for sample_index, text in enumerate(dataset["text"])
        if text.strip()
    ]
    total_nll = 0.0
    total_words = 0
    total_f1 = 0.0
    evaluated = 0
    sample_indices = []
    skip_continuation = getattr(args, "skip_wikitext_continuation", False)
    for sample_index, text in texts:
        words = text.split()
        prefix_words = words[: args.wikitext_max_words]
        continuation_words = words[
            args.wikitext_max_words : args.wikitext_max_words
            + args.wikitext_continuation_words
        ]
        if not prefix_words or not continuation_words:
            continue
        prompt = " ".join(prefix_words)
        if not hf_prompt_fits_budget(args, tokenizer, prompt):
            continue
        continuation = " ".join(continuation_words)
        nll, _ = score_continuation_hf(
            args,
            model,
            tokenizer,
            device,
            prompt,
            continuation,
        )
        total_nll += nll
        total_words += len(continuation_words)
        if not skip_continuation:
            generated_ids = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(device),
                do_sample=False,
                max_new_tokens=args.wikitext_continuation_words,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            prediction = tokenizer.decode(
                generated_ids[0, prompt_tokens:],
                skip_special_tokens=True,
            )
            total_f1 += bench.word_overlap_f1(prediction, continuation)
        sample_indices.append(sample_index)
        evaluated += 1
        if evaluated == args.wikitext_limit:
            break
    if evaluated != args.wikitext_limit:
        raise ValueError(
            f"evaluated {evaluated} WikiText samples; expected {args.wikitext_limit}"
        )
    result = {
        "word_perplexity": math.exp(total_nll / total_words),
        "num_samples": evaluated,
        "num_words": total_words,
        "max_words_per_sample": args.wikitext_max_words,
        "continuation_words_per_sample": args.wikitext_continuation_words,
        "sample_indices": sample_indices,
    }
    if not skip_continuation:
        result["continuation_f1"] = total_f1 / evaluated if evaluated else None
    return result


def score_continuation_hf(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    continuation: str,
) -> tuple[float, int]:
    import torch

    prefix_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    continuation_inputs = tokenizer(
        " " + continuation,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    continuation_ids = continuation_inputs.input_ids
    if int(continuation_ids.numel()) == 0:
        raise ValueError("WikiText continuation tokenized to zero tokens")

    prefix_outputs = model(**prefix_inputs, use_cache=True)
    attention_mask = continuation_attention_mask(
        prefix_inputs,
        continuation_ids,
    )
    continuation_outputs = model(
        input_ids=continuation_ids,
        attention_mask=attention_mask,
        past_key_values=prefix_outputs.past_key_values,
        use_cache=True,
    )

    logits = prefix_outputs.logits[:, -1:, :]
    if continuation_ids.shape[1] > 1:
        logits = torch.cat(
            (logits, continuation_outputs.logits[:, :-1, :]),
            dim=1,
        )
    log_probs = logits.log_softmax(dim=-1)
    token_log_probs = log_probs.gather(
        -1,
        continuation_ids.unsqueeze(-1),
    ).squeeze(-1)
    return -float(token_log_probs.sum()), int(continuation_ids.numel())


def continuation_attention_mask(
    prefix_inputs: Any,
    continuation_ids: Any,
) -> Any:
    attention_mask = getattr(prefix_inputs, "attention_mask", None)
    if attention_mask is None:
        return None
    return attention_mask.new_ones(
        (
            attention_mask.shape[0],
            attention_mask.shape[1] + continuation_ids.shape[1],
        )
    )


def hf_prompt_fits_budget(
    args: argparse.Namespace,
    tokenizer: Any,
    prompt: str,
) -> bool:
    if args.min_prompt_budget_tokens is None:
        return True
    token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    rounded_tokens = math.ceil(token_count / args.block_size) * args.block_size
    return rounded_tokens <= args.min_prompt_budget_tokens


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] == "kvpress-eval":
        eval_args = build_kvpress_eval_parser().parse_args(argv[1:])
        return run_kvpress_eval(eval_args)
    if argv and argv[0] == "kvpress-eval-batch":
        eval_args = build_kvpress_batch_eval_parser().parse_args(argv[1:])
        return run_kvpress_eval_batch(eval_args)

    args = parse_args(argv)
    root_dir = bench.make_root_dir(args)
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to {root_dir}")

    paged_runs: dict[str, dict[int | None, bench.RunSummary]] = {}
    kvpress_runs: dict[str, dict[int | None, KvpressRun]] = {}
    with tqdm(
        total=progress_total(args),
        desc="PagedEviction/KVPress",
        unit="run",
        disable=args.disable_tqdm or args.dry_run,
    ) as progress:
        if not args.skip_paged_eviction:
            paged_runs = run_paged_sweep(args, root_dir, progress)
        if not args.skip_kvpress:
            kvpress_runs = run_kvpress_sweep(args, root_dir, progress)

    validate_comparable_sample_sets(paged_runs, kvpress_runs)
    rows = normalize_rows(args.model, paged_runs, kvpress_runs)
    write_normalized_outputs(root_dir, rows)
    plot_paths = [] if args.dry_run else write_plots(root_dir, rows)
    write_summary(root_dir, args, paged_runs, kvpress_runs, rows, plot_paths)
    print(f"\nsummary: {root_dir / 'summary.json'}")
    print(f"json:    {root_dir / 'normalized_results.json'}")
    print(f"csv:     {root_dir / 'normalized_results.csv'}")
    if plot_paths:
        print(f"plots:   {root_dir / 'plots'}")
    return int(has_validation_errors(paged_runs, kvpress_runs))


if __name__ == "__main__":
    raise SystemExit(main())
