# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark PagedEviction memory use and serving latency.

Run from an initialized vLLM environment, for example:

    .venv/bin/python benchmarks/run_paged_eviction_memory_bench.py --smoke

The default workload is intentionally large and targets an online FP8-quantized
``meta-llama/Llama-3.1-8B`` on a single CUDA GPU. Its decode-heavy shape makes
the difference in occupied KV blocks visible. Use ``--smoke`` first to validate
model access, server startup, metrics scraping, and artifact writing.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = "meta-llama/Llama-3.1-8B"
BYTES_PER_KV_TOKEN_LLAMA_3_1_8B = 32 * 2 * 8 * 128 * 2
KV_CACHE_SIZE_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s+tokens")
KV_CACHE_USAGE_METRIC = "vllm:kv_cache_usage_perc"
SERVER_ERROR_RE = re.compile(
    r"(?:\b(?:ERROR|CRITICAL)\b|Traceback \(most recent call last\)|"
    r"\bRuntimeError:|[Uu]ncaught (?:runtime )?[Ee]xception)"
)
RUNNER_ENV = {"legacy": "0", "v2": "1"}


@dataclass
class RunArtifacts:
    run_dir: str
    server_log: str
    benchmark_log: str
    benchmark_json: str
    nvidia_smi_csv: str
    metrics_jsonl: str
    command_json: str
    gsm8k_json: str
    wikitext_json: str


@dataclass
class RunSummary:
    runner: str
    label: str
    paged_eviction_enabled: bool
    completion_status: str
    artifacts: RunArtifacts
    completed: int | None
    failed: int | None
    request_throughput: float | None
    output_throughput: float | None
    total_token_throughput: float | None
    mean_ttft_ms: float | None
    median_ttft_ms: float | None
    p50_ttft_ms: float | None
    p90_ttft_ms: float | None
    p99_ttft_ms: float | None
    idle_gpu_memory_mib: float | None
    post_load_gpu_memory_mib: float | None
    peak_benchmark_gpu_memory_mib: float | None
    peak_gpu_memory_delta_mib: float | None
    kv_cache_capacity_tokens: int | None
    kv_cache_capacity_bytes: int | None
    peak_kv_cache_usage_fraction: float | None
    derived_peak_kv_cache_bytes: float | None
    gsm8k_accuracy: float | None
    wikitext_word_perplexity: float | None
    validation_passed: bool
    validation_errors: list[str]


class Phase:
    def __init__(self, initial: str) -> None:
        self._phase = initial
        self._lock = threading.Lock()

    def set(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def get(self) -> str:
        with self._lock:
            return self._phase


class NvidiaSmiSampler:
    def __init__(
        self,
        output_path: Path,
        gpu_index: str,
        interval_s: float,
        phase_getter: Callable[[], str],
    ) -> None:
        self.output_path = output_path
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.phase_getter = phase_getter
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._file = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "monotonic_s",
                "iso_time",
                "phase",
                "nvidia_timestamp",
                "gpu_index",
                "gpu_name",
                "memory_used_mib",
                "memory_total_mib",
                "error",
            ],
        )
        self._writer.writeheader()
        self._reported_unavailable = False

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval_s * 2))
        with self._lock:
            self._file.close()

    def sample_now(self, phase: str | None = None) -> None:
        phase = phase or self.phase_getter()
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=timestamp,index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        if shutil.which("nvidia-smi") is None:
            if not self._reported_unavailable:
                self._write_error_row(phase, "nvidia-smi not found")
                self._reported_unavailable = True
            return

        try:
            proc = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            self._write_error_row(phase, str(exc))
            return

        rows = list(csv.reader(proc.stdout.splitlines()))
        if not rows:
            self._write_error_row(phase, "nvidia-smi returned no rows")
            return

        now = datetime.now(timezone.utc).isoformat()
        monotonic_s = f"{time.monotonic():.6f}"
        with self._lock:
            for row in rows:
                if len(row) < 5:
                    self._writer.writerow(
                        {
                            "monotonic_s": monotonic_s,
                            "iso_time": now,
                            "phase": phase,
                            "error": f"unexpected nvidia-smi row: {row}",
                        }
                    )
                    continue
                timestamp, idx, name, used, total = [part.strip() for part in row[:5]]
                self._writer.writerow(
                    {
                        "monotonic_s": monotonic_s,
                        "iso_time": now,
                        "phase": phase,
                        "nvidia_timestamp": timestamp,
                        "gpu_index": idx,
                        "gpu_name": name,
                        "memory_used_mib": used,
                        "memory_total_mib": total,
                        "error": "",
                    }
                )
            self._file.flush()

    def _write_error_row(self, phase: str, error: str) -> None:
        with self._lock:
            self._writer.writerow(
                {
                    "monotonic_s": f"{time.monotonic():.6f}",
                    "iso_time": datetime.now(timezone.utc).isoformat(),
                    "phase": phase,
                    "gpu_index": self.gpu_index,
                    "error": error,
                }
            )
            self._file.flush()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.sample_now()
            self.stop_event.wait(self.interval_s)


class MetricsSampler:
    def __init__(
        self,
        base_url: str,
        output_path: Path,
        interval_s: float,
        phase_getter: Callable[[], str],
    ) -> None:
        self.metrics_url = f"{base_url}/metrics"
        self.output_path = output_path
        self.interval_s = interval_s
        self.phase_getter = phase_getter
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._file = output_path.open("w", encoding="utf-8")

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval_s * 2))
        with self._lock:
            self._file.close()

    def sample_now(self, phase: str | None = None) -> None:
        phase = phase or self.phase_getter()
        sample: dict[str, Any] = {
            "monotonic_s": time.monotonic(),
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "url": self.metrics_url,
        }
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=10) as response:
                body = response.read().decode("utf-8", "replace")
            values = parse_prometheus_values(body, KV_CACHE_USAGE_METRIC)
            sample.update(
                {
                    "ok": True,
                    "kv_cache_usage_perc_values": values,
                    "kv_cache_usage_perc_max": max(values) if values else None,
                    "metric_line_count": len(body.splitlines()),
                }
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            sample.update({"ok": False, "error": str(exc)})

        with self._lock:
            self._file.write(json.dumps(sample, sort_keys=True) + "\n")
            self._file.flush()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.sample_now()
            self.stop_event.wait(self.interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run disabled/enabled PagedEviction serving benchmarks and collect "
            "GPU memory, vLLM metrics, and benchmark latency artifacts."
        )
    )

    server = parser.add_argument_group("server")
    server.add_argument("--model", default=MODEL)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--startup-timeout-s", type=float, default=1800.0)
    server.add_argument("--shutdown-timeout-s", type=float, default=60.0)
    server.add_argument("--post-load-sleep-s", type=float, default=5.0)
    server.add_argument("--max-model-len", type=int, default=5120)
    server.add_argument("--max-num-seqs", type=int, default=64)
    server.add_argument("--max-num-batched-tokens", type=int, default=8192)
    server.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    server.add_argument(
        "--quantization",
        default="fp8",
        help=(
            "Weight quantization passed to vLLM. Defaults to online FP8 to "
            "leave more GPU memory for the KV cache. Use 'none' to disable."
        ),
    )
    server.add_argument("--cache-budget-tokens", type=int, default=1024)
    server.add_argument(
        "--runner",
        choices=("legacy", "v2", "both"),
        default="both",
        help="Model runner(s) to benchmark. Each runner uses a fresh server.",
    )

    workload = parser.add_argument_group("workload")
    workload.add_argument("--num-prompts", type=int, default=128)
    workload.add_argument("--random-input-len", type=int, default=1024)
    workload.add_argument("--random-output-len", type=int, default=4096)
    workload.add_argument("--random-range-ratio", default="0")
    workload.add_argument("--request-rate", default="inf")
    workload.add_argument("--max-concurrency", type=int, default=64)
    workload.add_argument("--seed", type=int, default=0)
    workload.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use a short smoke workload: 8 prompts, 512 input tokens, "
            "64 output tokens, and concurrency 2."
        ),
    )

    quality = parser.add_argument_group("quality")
    quality.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip GSM8K accuracy and WikiText perplexity evaluation.",
    )
    quality.add_argument("--gsm8k-limit", type=int, default=8)
    quality.add_argument("--gsm8k-max-tokens", type=int, default=2048)
    quality.add_argument("--wikitext-limit", type=int, default=4)
    quality.add_argument("--wikitext-max-words", type=int, default=256)

    artifacts = parser.add_argument_group("artifacts")
    artifacts.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/paged_eviction"),
    )
    artifacts.add_argument(
        "--timestamp",
        default=None,
        help="Override the timestamped result directory name.",
    )
    artifacts.add_argument(
        "--gpu-index",
        default=default_gpu_index(),
        help=(
            "Physical GPU index or UUID for nvidia-smi. Defaults to the first "
            "numeric CUDA_VISIBLE_DEVICES entry, otherwise 0."
        ),
    )
    artifacts.add_argument("--poll-interval-s", type=float, default=1.0)
    artifacts.add_argument(
        "--bytes-per-kv-token",
        type=int,
        default=BYTES_PER_KV_TOKEN_LLAMA_3_1_8B,
        help=(
            "Bytes per KV token for derived KV memory. The default is "
            "Llama-3.1-8B TP=1 fp16/bf16: 32*2*8*128*2."
        ),
    )
    artifacts.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned output paths without starting servers.",
    )

    args = parser.parse_args()
    if args.quantization.lower() in ("", "none"):
        args.quantization = None
    if args.smoke:
        args.num_prompts = 8
        args.random_input_len = 512
        args.random_output_len = 64
        args.max_concurrency = 2
        args.gsm8k_limit = min(args.gsm8k_limit, 2)
        args.wikitext_limit = min(args.wikitext_limit, 2)
    return args


def selected_runners(runner: str) -> list[str]:
    return list(RUNNER_ENV) if runner == "both" else [runner]


def server_environment(runner: str) -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_USE_V2_MODEL_RUNNER"] = RUNNER_ENV[runner]
    return env


def default_gpu_index() -> str:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        first = cuda_visible_devices.split(",", 1)[0].strip()
        if first:
            return first
    return "0"


def build_server_command(args: argparse.Namespace, enabled: bool) -> list[str]:
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
        "--no-enable-chunked-prefill",
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
    if enabled:
        command.extend(
            [
                "--paged-eviction-config",
                json.dumps({"cache_budget_tokens": args.cache_budget_tokens}),
            ]
        )
    return command


def build_benchmark_command(
    args: argparse.Namespace,
    run_dir: Path,
    label: str,
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
        base_url(args),
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.random_input_len),
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


def base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def health_url(args: argparse.Namespace) -> str:
    return f"{base_url(args)}/health"


def assert_port_available(args: argparse.Namespace) -> None:
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    with (
        contextlib.suppress(OSError),
        socket.create_connection(
            (host, args.port),
            timeout=1,
        ),
    ):
        raise RuntimeError(
            f"{host}:{args.port} is already accepting connections. "
            "Use --port to choose a free port."
        )


def run_one(
    args: argparse.Namespace,
    root_dir: Path,
    runner: str,
    label: str,
    enabled: bool,
) -> RunSummary:
    run_dir = root_dir / runner / label
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(
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

    server_command = build_server_command(args, enabled)
    benchmark_command = build_benchmark_command(args, run_dir, label)
    write_json(
        Path(artifacts.command_json),
        {
            "server_command": server_command,
            "benchmark_command": benchmark_command,
            "environment": {"VLLM_USE_V2_MODEL_RUNNER": RUNNER_ENV[runner]},
            "runner": runner,
            "paged_eviction_enabled": enabled,
        },
    )

    if args.dry_run:
        return empty_summary(runner, label, enabled, artifacts)

    phase = Phase("idle")
    memory_sampler = NvidiaSmiSampler(
        Path(artifacts.nvidia_smi_csv),
        args.gpu_index,
        args.poll_interval_s,
        phase.get,
    )
    metrics_sampler: MetricsSampler | None = None
    server_proc: subprocess.Popen[str] | None = None

    try:
        memory_sampler.sample_now("idle")
        phase.set("startup")
        memory_sampler.start()
        assert_port_available(args)

        with Path(artifacts.server_log).open("w", encoding="utf-8") as server_log:
            server_proc = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=server_environment(runner),
            )

            wait_for_health(args, server_proc, Path(artifacts.server_log))

            metrics_sampler = MetricsSampler(
                base_url(args),
                Path(artifacts.metrics_jsonl),
                args.poll_interval_s,
                phase.get,
            )
            metrics_sampler.start()

            phase.set("post_load")
            memory_sampler.sample_now("post_load")
            metrics_sampler.sample_now("post_load")
            time.sleep(args.post_load_sleep_s)

            phase.set("benchmark")
            run_benchmark(benchmark_command, Path(artifacts.benchmark_log))

            if not args.skip_quality:
                phase.set("quality")
                run_quality_evaluation(args, artifacts)

            phase.set("post_bench")
            memory_sampler.sample_now("post_bench")
            metrics_sampler.sample_now("post_bench")
    finally:
        if metrics_sampler is not None:
            metrics_sampler.stop()
        if server_proc is not None:
            terminate_process(server_proc, args.shutdown_timeout_s)
            wait_for_health_down(args, timeout_s=30.0)
        memory_sampler.stop()

    return summarize_run(args, runner, label, enabled, artifacts)


def empty_summary(
    runner: str,
    label: str,
    enabled: bool,
    artifacts: RunArtifacts,
) -> RunSummary:
    return RunSummary(
        runner=runner,
        label=label,
        paged_eviction_enabled=enabled,
        completion_status="not_run",
        artifacts=artifacts,
        completed=None,
        failed=None,
        request_throughput=None,
        output_throughput=None,
        total_token_throughput=None,
        mean_ttft_ms=None,
        median_ttft_ms=None,
        p50_ttft_ms=None,
        p90_ttft_ms=None,
        p99_ttft_ms=None,
        idle_gpu_memory_mib=None,
        post_load_gpu_memory_mib=None,
        peak_benchmark_gpu_memory_mib=None,
        peak_gpu_memory_delta_mib=None,
        kv_cache_capacity_tokens=None,
        kv_cache_capacity_bytes=None,
        peak_kv_cache_usage_fraction=None,
        derived_peak_kv_cache_bytes=None,
        gsm8k_accuracy=None,
        wikitext_word_perplexity=None,
        validation_passed=True,
        validation_errors=[],
    )


def wait_for_health(
    args: argparse.Namespace,
    proc: subprocess.Popen[str],
    server_log_path: Path,
) -> None:
    deadline = time.monotonic() + args.startup_timeout_s
    url = health_url(args)
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {proc.returncode} before /health "
                f"became ready. See {server_log_path}.\n"
                f"Last log lines:\n{tail_file(server_log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(
        f"Timed out after {args.startup_timeout_s:.0f}s waiting for {url}. "
        f"Last error: {last_error}. See {server_log_path}."
    )


def wait_for_health_down(args: argparse.Namespace, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = health_url(args)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    time.sleep(1)
                    continue
        except (urllib.error.URLError, TimeoutError, OSError):
            return
        time.sleep(1)


def run_benchmark(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed with code {proc.returncode}. "
            f"See {log_path}.\nLast log lines:\n{tail_file(log_path)}"
        )


def post_completion(
    args: argparse.Namespace, payload: dict[str, Any]
) -> dict[str, Any]:
    payload = {"model": args.model, **payload}
    request = urllib.request.Request(
        f"{base_url(args)}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("completion response is not a JSON object")
    return result


def extract_gsm8k_answer(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def evaluate_gsm8k(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    limit = min(args.gsm8k_limit, len(dataset))
    correct = 0
    invalid = 0
    for item in dataset.select(range(limit)):
        prompt = f"Question: {item['question']}\nAnswer: Let's think step by step."
        response = post_completion(
            args,
            {
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": args.gsm8k_max_tokens,
                "stop": ["Question:"],
                "seed": args.seed,
            },
        )
        prediction = extract_gsm8k_answer(response["choices"][0]["text"])
        expected = extract_gsm8k_answer(item["answer"])
        invalid += prediction is None
        correct += prediction is not None and prediction == expected
    return {
        "accuracy": correct / limit if limit else None,
        "correct": correct,
        "invalid": invalid,
        "num_questions": limit,
        "max_tokens": args.gsm8k_max_tokens,
    }


def evaluate_wikitext(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [text.strip() for text in dataset["text"] if text.strip()]
    total_nll = 0.0
    total_words = 0
    evaluated = 0
    for text in texts:
        words = text.split()
        if len(words) < 2:
            continue
        prompt = " ".join(words[: args.wikitext_max_words])
        response = post_completion(
            args,
            {
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 1,
                "echo": True,
                "logprobs": 1,
            },
        )
        choice = response["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
        completion_tokens = as_int(response.get("usage", {}).get("completion_tokens"))
        if completion_tokens:
            token_logprobs = token_logprobs[:-completion_tokens]
        valid_logprobs = [
            float(logprob) for logprob in token_logprobs if logprob is not None
        ]
        if not valid_logprobs:
            raise ValueError("completion response has no prompt token logprobs")
        total_nll -= sum(valid_logprobs)
        total_words += len(prompt.split())
        evaluated += 1
        if evaluated == args.wikitext_limit:
            break
    if not total_words or evaluated != args.wikitext_limit:
        raise ValueError(
            f"evaluated {evaluated} WikiText samples; expected {args.wikitext_limit}"
        )
    return {
        "word_perplexity": math.exp(total_nll / total_words),
        "num_samples": evaluated,
        "num_words": total_words,
        "max_words_per_sample": args.wikitext_max_words,
    }


def run_quality_evaluation(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
) -> None:
    for evaluator, output_path in (
        (evaluate_gsm8k, Path(artifacts.gsm8k_json)),
        (evaluate_wikitext, Path(artifacts.wikitext_json)),
    ):
        try:
            result = evaluator(args)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        write_json(output_path, result)


def terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)


def summarize_run(
    args: argparse.Namespace,
    runner: str,
    label: str,
    enabled: bool,
    artifacts: RunArtifacts,
) -> RunSummary:
    benchmark = load_json(Path(artifacts.benchmark_json))
    gsm8k = load_json(Path(artifacts.gsm8k_json))
    wikitext = load_json(Path(artifacts.wikitext_json))
    validation_errors = validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=args.num_prompts,
        quality_required=not args.skip_quality,
    )
    memory_stats = load_memory_stats(Path(artifacts.nvidia_smi_csv))
    peak_usage = load_peak_kv_usage(Path(artifacts.metrics_jsonl))
    capacity_tokens = parse_kv_cache_capacity_tokens(Path(artifacts.server_log))
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
    peak_benchmark_memory = memory_stats.get("peak_benchmark")

    return RunSummary(
        runner=runner,
        label=label,
        paged_eviction_enabled=enabled,
        completion_status=(
            "complete"
            if as_int(benchmark.get("completed")) == args.num_prompts
            and as_int(benchmark.get("failed")) == 0
            else "invalid"
        ),
        artifacts=artifacts,
        completed=as_int(benchmark.get("completed")),
        failed=as_int(benchmark.get("failed")),
        request_throughput=as_float(benchmark.get("request_throughput")),
        output_throughput=as_float(benchmark.get("output_throughput")),
        total_token_throughput=as_float(benchmark.get("total_token_throughput")),
        mean_ttft_ms=as_float(benchmark.get("mean_ttft_ms")),
        median_ttft_ms=as_float(benchmark.get("median_ttft_ms")),
        p50_ttft_ms=as_float(benchmark.get("p50_ttft_ms")),
        p90_ttft_ms=as_float(benchmark.get("p90_ttft_ms")),
        p99_ttft_ms=as_float(benchmark.get("p99_ttft_ms")),
        idle_gpu_memory_mib=idle_memory,
        post_load_gpu_memory_mib=memory_stats.get("post_load"),
        peak_benchmark_gpu_memory_mib=peak_benchmark_memory,
        peak_gpu_memory_delta_mib=(
            peak_benchmark_memory - idle_memory
            if peak_benchmark_memory is not None and idle_memory is not None
            else None
        ),
        kv_cache_capacity_tokens=capacity_tokens,
        kv_cache_capacity_bytes=capacity_bytes,
        peak_kv_cache_usage_fraction=peak_usage,
        derived_peak_kv_cache_bytes=derived_peak_kv,
        gsm8k_accuracy=as_float(gsm8k.get("accuracy")),
        wikitext_word_perplexity=as_float(wikitext.get("word_perplexity")),
        validation_passed=not validation_errors,
        validation_errors=validation_errors,
    )


def validate_artifacts(
    artifacts: RunArtifacts,
    benchmark: dict[str, Any],
    expected_completed: int,
    quality_required: bool,
) -> list[str]:
    errors: list[str] = []
    required_artifacts = {
        "server log": artifacts.server_log,
        "benchmark log": artifacts.benchmark_log,
        "benchmark json": artifacts.benchmark_json,
        "nvidia-smi csv": artifacts.nvidia_smi_csv,
        "metrics jsonl": artifacts.metrics_jsonl,
        "commands json": artifacts.command_json,
    }
    if quality_required:
        required_artifacts.update(
            {
                "GSM8K json": artifacts.gsm8k_json,
                "WikiText json": artifacts.wikitext_json,
            }
        )
    for label, path_str in required_artifacts.items():
        path = Path(path_str)
        if not path.exists():
            errors.append(f"missing {label}: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"empty {label}: {path}")

    ttfts = benchmark.get("ttfts")
    if not isinstance(ttfts, list) or not ttfts:
        errors.append("benchmark JSON has no non-empty ttfts array")

    completed = as_int(benchmark.get("completed"))
    failed = as_int(benchmark.get("failed"))
    if completed != expected_completed:
        errors.append(
            f"benchmark completed {completed!r} requests; expected {expected_completed}"
        )
    if failed != 0:
        errors.append(f"benchmark failed count is {failed!r}; expected 0")

    if not has_memory_samples(Path(artifacts.nvidia_smi_csv)):
        errors.append("nvidia-smi CSV has no successful memory samples")

    if load_peak_kv_usage(Path(artifacts.metrics_jsonl)) is None:
        errors.append(f"metrics samples contain no {KV_CACHE_USAGE_METRIC} values")

    server_log_path = Path(artifacts.server_log)
    if server_log_path.exists():
        server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
        match = SERVER_ERROR_RE.search(server_log)
        if match is not None:
            errors.append(f"server log contains error marker: {match.group(0)!r}")

    if quality_required:
        gsm8k = load_json(Path(artifacts.gsm8k_json))
        wikitext = load_json(Path(artifacts.wikitext_json))
        if as_float(gsm8k.get("accuracy")) is None:
            errors.append("GSM8K result has no accuracy metric")
        if as_float(wikitext.get("word_perplexity")) is None:
            errors.append("WikiText result has no word_perplexity metric")

    return errors


def has_memory_samples(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("memory_used_mib"):
                return True
    return False


def load_memory_stats(path: Path) -> dict[str, float | None]:
    values_by_phase: dict[str, list[float]] = {}
    if not path.exists():
        return {"idle": None, "post_load": None, "peak_benchmark": None}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            value = as_float(row.get("memory_used_mib"))
            phase = row.get("phase")
            if value is None or phase is None:
                continue
            values_by_phase.setdefault(phase, []).append(value)

    idle_values = values_by_phase.get("idle", [])
    post_load_values = values_by_phase.get("post_load", [])
    benchmark_values = values_by_phase.get("benchmark", [])
    return {
        "idle": idle_values[0] if idle_values else None,
        "post_load": post_load_values[-1] if post_load_values else None,
        "peak_benchmark": max(benchmark_values) if benchmark_values else None,
    }


def parse_kv_cache_capacity_tokens(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = KV_CACHE_SIZE_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def load_peak_kv_usage(path: Path) -> float | None:
    if not path.exists():
        return None
    peak: float | None = None
    with path.open(encoding="utf-8") as file:
        for line in file:
            with contextlib.suppress(json.JSONDecodeError):
                sample = json.loads(line)
                value = as_float(sample.get("kv_cache_usage_perc_max"))
                if value is not None:
                    peak = value if peak is None else max(peak, value)
    return peak


def parse_prometheus_values(body: str, metric_name: str) -> list[float]:
    values = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sample_name = parts[0].split("{", 1)[0]
        if sample_name != metric_name:
            continue
        with contextlib.suppress(ValueError):
            values.append(float(parts[1]))
    return values


def write_summary(
    root_dir: Path,
    args: argparse.Namespace,
    runs: dict[str, dict[str, RunSummary]],
) -> None:
    deltas = {
        runner: compute_deltas(runner_runs["disabled"], runner_runs["enabled"])
        for runner, runner_runs in runs.items()
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root_dir),
        "measurement_notes": {
            "gpu_memory": (
                "vLLM preallocates the KV cache, so nvidia-smi reports the "
                "reserved pool rather than live KV-block occupancy."
            ),
            "derived_peak_kv_cache_bytes": (
                "KV-cache capacity multiplied by peak kv_cache_usage_perc; "
                "this is the primary memory comparison."
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
            "runners": selected_runners(args.runner),
            "quality": {
                "enabled": not args.skip_quality,
                "gsm8k_limit": args.gsm8k_limit,
                "gsm8k_max_tokens": args.gsm8k_max_tokens,
                "wikitext_limit": args.wikitext_limit,
                "wikitext_max_words": args.wikitext_max_words,
            },
        },
        "runs": {
            runner: {mode: asdict(run) for mode, run in runner_runs.items()}
            for runner, runner_runs in runs.items()
        },
        "deltas_enabled_vs_disabled_pct": deltas,
    }
    write_json(root_dir / "summary.json", summary)
    write_summary_csv(root_dir / "summary.csv", runs)


def compute_deltas(
    disabled: RunSummary,
    enabled: RunSummary,
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for metric in summary_metric_names():
        base = as_float(getattr(disabled, metric))
        new = as_float(getattr(enabled, metric))
        deltas[metric] = percent_delta(base, new)
    return deltas


def write_summary_csv(
    path: Path,
    runs: dict[str, dict[str, RunSummary]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        metric_names = summary_metric_names()
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "runner",
                "eviction_mode",
                "completion_status",
                "validation_passed",
                "validation_errors",
                *metric_names,
            ],
        )
        writer.writeheader()
        for runner, runner_runs in runs.items():
            for mode, run in runner_runs.items():
                writer.writerow(
                    {
                        "runner": runner,
                        "eviction_mode": mode,
                        "completion_status": run.completion_status,
                        "validation_passed": run.validation_passed,
                        "validation_errors": "; ".join(run.validation_errors),
                        **{metric: getattr(run, metric) for metric in metric_names},
                    }
                )


def summary_metric_names() -> list[str]:
    return [
        "peak_benchmark_gpu_memory_mib",
        "peak_gpu_memory_delta_mib",
        "derived_peak_kv_cache_bytes",
        "peak_kv_cache_usage_fraction",
        "median_ttft_ms",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "output_throughput",
        "total_token_throughput",
        "request_throughput",
        "completed",
        "failed",
        "gsm8k_accuracy",
        "wikitext_word_perplexity",
    ]


def print_terminal_table(runner: str, runs: dict[str, RunSummary]) -> None:
    disabled = runs["disabled"]
    enabled = runs["enabled"]
    rows = [
        (
            "preallocated_gpu_mib",
            disabled.peak_benchmark_gpu_memory_mib,
            enabled.peak_benchmark_gpu_memory_mib,
        ),
        (
            "kv_pool_capacity_gib",
            bytes_to_gib(disabled.kv_cache_capacity_bytes),
            bytes_to_gib(enabled.kv_cache_capacity_bytes),
        ),
        (
            "peak_kv_occupied_gib",
            bytes_to_gib(disabled.derived_peak_kv_cache_bytes),
            bytes_to_gib(enabled.derived_peak_kv_cache_bytes),
        ),
        ("median_ttft_ms", disabled.median_ttft_ms, enabled.median_ttft_ms),
        ("p99_ttft_ms", disabled.p99_ttft_ms, enabled.p99_ttft_ms),
        ("output_tok_s", disabled.output_throughput, enabled.output_throughput),
        (
            "total_tok_s",
            disabled.total_token_throughput,
            enabled.total_token_throughput,
        ),
        ("failed", disabled.failed, enabled.failed),
    ]
    table = [["metric", "disabled", "enabled", "delta_pct"]]
    for metric, base, new in rows:
        table.append(
            [
                metric,
                format_value(base),
                format_value(new),
                format_percent(percent_delta(as_float(base), as_float(new))),
            ]
        )
    print(f"\nRunner: {runner}")
    print(format_table(table))

    validation_errors = [
        f"{name}: {error}"
        for name, run in runs.items()
        for error in run.validation_errors
    ]
    if validation_errors:
        print("\nValidation errors:")
        for error in validation_errors:
            print(f"  - {error}")


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row_idx, row in enumerate(rows):
        line = "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if row_idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def format_value(value: Any) -> str:
    value = as_float(value)
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def percent_delta(base: float | None, new: float | None) -> float | None:
    if base is None or new is None or base == 0 or math.isnan(base) or math.isnan(new):
        return None
    return (new - base) / base * 100


def bytes_to_gib(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return int(number) if number is not None else None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def tail_file(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def make_root_dir(args: argparse.Namespace) -> Path:
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return args.results_dir / timestamp


def print_dry_run(root_dir: Path, args: argparse.Namespace) -> None:
    for runner in selected_runners(args.runner):
        for label, enabled in (("disabled", False), ("enabled", True)):
            run_dir = root_dir / runner / label
            print(
                f"\n[{runner}/{label}] "
                f"VLLM_USE_V2_MODEL_RUNNER={RUNNER_ENV[runner]} server:"
            )
            print(shlex.join(build_server_command(args, enabled)))
            print(f"[{runner}/{label}] benchmark:")
            print(shlex.join(build_benchmark_command(args, run_dir, label)))
    print(f"\nArtifacts would be written under {root_dir}")


def main() -> int:
    args = parse_args()
    root_dir = make_root_dir(args)
    root_dir.mkdir(parents=True, exist_ok=True)

    runners = selected_runners(args.runner)
    if args.dry_run:
        print_dry_run(root_dir, args)
        runs = {
            runner: {
                "disabled": run_one(args, root_dir, runner, "disabled", enabled=False),
                "enabled": run_one(args, root_dir, runner, "enabled", enabled=True),
            }
            for runner in runners
        }
        write_summary(root_dir, args, runs)
        return 0

    print(f"Writing artifacts to {root_dir}")
    runs = {}
    for runner in runners:
        runs[runner] = {
            "disabled": run_one(args, root_dir, runner, "disabled", enabled=False),
            "enabled": run_one(args, root_dir, runner, "enabled", enabled=True),
        }
    write_summary(root_dir, args, runs)
    for runner, runner_runs in runs.items():
        print_terminal_table(runner, runner_runs)
    print(f"\nsummary: {root_dir / 'summary.json'}")
    print(f"csv:     {root_dir / 'summary.csv'}")
    return int(
        any(
            run.validation_errors
            for runner_runs in runs.values()
            for run in runner_runs.values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
