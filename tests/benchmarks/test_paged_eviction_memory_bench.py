# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

import pytest

import benchmarks.run_paged_eviction_memory_bench as bench

pytestmark = pytest.mark.cpu_test


def make_artifacts(tmp_path: Path) -> bench.RunArtifacts:
    run_dir = tmp_path / "run"
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


def write_valid_artifacts(artifacts: bench.RunArtifacts) -> dict[str, object]:
    run_dir = Path(artifacts.run_dir)
    run_dir.mkdir(parents=True)
    Path(artifacts.server_log).write_text("INFO server ready\n", encoding="utf-8")
    Path(artifacts.benchmark_log).write_text("benchmark complete\n", encoding="utf-8")
    benchmark = {"completed": 4, "failed": 0, "ttfts": [1.0]}
    Path(artifacts.benchmark_json).write_text(json.dumps(benchmark), encoding="utf-8")
    Path(artifacts.nvidia_smi_csv).write_text(
        "phase,memory_used_mib\nbenchmark,100\n", encoding="utf-8"
    )
    Path(artifacts.metrics_jsonl).write_text(
        json.dumps({"kv_cache_usage_perc_max": 0.5}) + "\n",
        encoding="utf-8",
    )
    Path(artifacts.command_json).write_text("{}\n", encoding="utf-8")
    Path(artifacts.gsm8k_json).write_text(
        json.dumps({"accuracy": 0.25}) + "\n", encoding="utf-8"
    )
    Path(artifacts.wikitext_json).write_text(
        json.dumps({"word_perplexity": 12.5}) + "\n", encoding="utf-8"
    )
    return benchmark


def test_validation_fails_completion_mismatch(tmp_path: Path):
    artifacts = make_artifacts(tmp_path)
    benchmark = write_valid_artifacts(artifacts)
    benchmark["completed"] = 3

    errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=4,
        quality_required=True,
    )

    assert any("completed" in error for error in errors)


def test_validation_fails_failed_requests(tmp_path: Path):
    artifacts = make_artifacts(tmp_path)
    benchmark = write_valid_artifacts(artifacts)
    benchmark["failed"] = 1

    errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=4,
        quality_required=True,
    )

    assert any("failed count" in error for error in errors)


def test_validation_fails_server_log_error(tmp_path: Path):
    artifacts = make_artifacts(tmp_path)
    benchmark = write_valid_artifacts(artifacts)
    Path(artifacts.server_log).write_text(
        "Traceback (most recent call last):\nRuntimeError: boom\n",
        encoding="utf-8",
    )

    errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=4,
        quality_required=True,
    )

    assert any("server log contains error marker" in error for error in errors)


def test_validation_parses_quality_results(tmp_path: Path):
    artifacts = make_artifacts(tmp_path)
    benchmark = write_valid_artifacts(artifacts)

    errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=4,
        quality_required=True,
    )

    assert errors == []


def test_validation_requires_quality_metrics(tmp_path: Path):
    artifacts = make_artifacts(tmp_path)
    benchmark = write_valid_artifacts(artifacts)
    Path(artifacts.gsm8k_json).write_text('{"error": "failed"}\n', encoding="utf-8")

    errors = bench.validate_artifacts(
        artifacts,
        benchmark,
        expected_completed=4,
        quality_required=True,
    )

    assert any("GSM8K" in error and "accuracy" in error for error in errors)


@pytest.mark.parametrize(("runner", "expected"), [("legacy", "0"), ("v2", "1")])
def test_runner_environment_sets_model_runner(monkeypatch, runner: str, expected: str):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "unexpected")

    env = bench.server_environment(runner)

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == expected


def test_write_summary_uses_nested_runner_layout(tmp_path: Path):
    artifacts = make_artifacts(tmp_path)
    disabled = bench.empty_summary("legacy", "disabled", False, artifacts)
    enabled = bench.empty_summary("legacy", "enabled", True, artifacts)
    args = argparse.Namespace(
        model="model",
        quantization=None,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.9,
        cache_budget_tokens=16,
        num_prompts=1,
        random_input_len=8,
        random_output_len=8,
        random_range_ratio="0",
        request_rate="inf",
        max_concurrency=1,
        seed=0,
        bytes_per_kv_token=1,
        gpu_index="0",
        runner="legacy",
        skip_quality=False,
        gsm8k_limit=1,
        gsm8k_max_tokens=8,
        wikitext_limit=1,
        wikitext_max_words=8,
    )

    bench.write_summary(
        tmp_path,
        args,
        {"legacy": {"disabled": disabled, "enabled": enabled}},
    )

    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert data["runs"]["legacy"]["disabled"]["runner"] == "legacy"
