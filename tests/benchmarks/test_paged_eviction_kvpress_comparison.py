# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import types
from pathlib import Path

import pytest

import benchmarks.run_paged_eviction_kvpress_comparison as comparison
import benchmarks.run_paged_eviction_memory_bench as bench

pytestmark = pytest.mark.cpu_test


def make_bench_artifacts(tmp_path: Path, name: str) -> bench.RunArtifacts:
    run_dir = tmp_path / name
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


def make_kvpress_run(
    tmp_path: Path,
    method: str,
    budget: int | None,
    gsm8k_accuracy: float,
    wikitext_word_perplexity: float,
    wikitext_continuation_f1: float | None = 0.5,
) -> comparison.KvpressRun:
    run_dir = tmp_path / "kvpress" / method / str(budget)
    artifacts = comparison.KvpressArtifacts(
        run_dir=str(run_dir),
        result_json=str(run_dir / "result.json"),
        command_json=str(run_dir / "commands.json"),
        stdout_log=str(run_dir / "stdout.log"),
    )
    return comparison.KvpressRun(
        method=method,
        budget_tokens=budget,
        completion_status="complete",
        artifacts=artifacts,
        latency_s=1.5,
        peak_memory_mib=100.0,
        gsm8k_accuracy=gsm8k_accuracy,
        wikitext_continuation_f1=wikitext_continuation_f1,
        wikitext_word_perplexity=wikitext_word_perplexity,
        validation_errors=[],
    )


def test_parse_budgets_requires_block_alignment():
    assert comparison.parse_budgets("256,512", block_size=16) == [256, 512]

    with pytest.raises(Exception, match="aligned"):
        comparison.parse_budgets("257", block_size=16)


def test_parse_methods_filters_supported_names():
    assert comparison.parse_methods("KnormPress,SnapKVPress") == [
        "KnormPress",
        "SnapKVPress",
    ]

    with pytest.raises(Exception, match="unsupported"):
        comparison.parse_methods("UnknownPress")


def test_parse_args_defaults_to_single_vllm_runner_and_quantization():
    args = comparison.parse_args(["--skip-kvpress"])

    assert args.runner == "legacy"
    assert bench.selected_runners(args.runner) == ["legacy"]
    assert args.quantization == "fp8"
    assert args.transformers_quantization == "bitsandbytes-4bit"
    assert args.quality_min_prompt_budget_tokens == min(args.budgets)


def test_parse_args_rejects_both_runner():
    with pytest.raises(SystemExit):
        comparison.parse_args(["--skip-kvpress", "--runner", "both"])


def test_paged_args_skip_serving_and_filters_prompts():
    args = comparison.parse_args(["--skip-kvpress", "--budgets", "256"])

    paged_args = comparison.paged_args_for_budget(args, 256)

    assert paged_args.skip_serving_benchmark is True
    assert paged_args.skip_wikitext_continuation is False
    assert paged_args.quality_min_prompt_budget_tokens == 256


def test_discover_kvpress_methods_uses_importable_supported_methods(monkeypatch):
    fake_kvpress = types.SimpleNamespace(KnormPress=object, SnapKVPress=object)
    monkeypatch.setitem(sys.modules, "kvpress", fake_kvpress)

    assert comparison.discover_kvpress_methods() == ["KnormPress", "SnapKVPress"]


def test_normalize_rows_adds_baseline_deltas(tmp_path: Path):
    baseline = bench.empty_summary(
        "v2",
        "full_cache",
        False,
        make_bench_artifacts(tmp_path, "paged_baseline"),
    )
    baseline.gsm8k_accuracy = 0.5
    baseline.wikitext_continuation_f1 = 0.5
    baseline.wikitext_word_perplexity = 20.0
    baseline.median_ttft_ms = 100.0
    baseline.peak_benchmark_gpu_memory_mib = 1000.0

    budget = bench.empty_summary(
        "v2",
        "budget_256",
        True,
        make_bench_artifacts(tmp_path, "paged_budget"),
    )
    budget.gsm8k_accuracy = 0.4
    budget.wikitext_continuation_f1 = 0.25
    budget.wikitext_word_perplexity = 10.0

    rows = comparison.normalize_rows(
        "model",
        {"v2": {None: baseline, 256: budget}},
        {
            "full_cache": {
                None: make_kvpress_run(tmp_path, "full_cache", None, 0.5, 20.0),
            },
            "KnormPress": {
                256: make_kvpress_run(tmp_path, "KnormPress", 256, 0.25, 10.0),
            },
        },
    )

    paged_accuracy = next(
        row
        for row in rows
        if row["backend"] == "vllm"
        and row["budget_tokens"] == 256
        and row["dataset"] == "gsm8k"
        and row["metric_name"] == "accuracy"
    )
    kvpress_accuracy = next(
        row
        for row in rows
        if row["backend"] == "transformers"
        and row["method"] == "KnormPress"
        and row["dataset"] == "gsm8k"
        and row["metric_name"] == "accuracy"
    )
    metric_names = {(row["dataset"], row["metric_name"]) for row in rows}

    assert paged_accuracy["baseline_delta"] == pytest.approx(-20.0)
    assert kvpress_accuracy["baseline_delta"] == pytest.approx(-50.0)
    assert metric_names == {
        ("gsm8k", "accuracy"),
        ("wikitext", "continuation_f1"),
        ("wikitext", "word_perplexity"),
    }


def test_validate_comparable_sample_sets_flags_mismatch(tmp_path: Path):
    baseline = bench.empty_summary(
        "legacy",
        "full_cache",
        False,
        make_bench_artifacts(tmp_path, "baseline"),
    )
    budget = bench.empty_summary(
        "legacy",
        "budget_256",
        True,
        make_bench_artifacts(tmp_path, "budget"),
    )
    for run, gsm8k_indices in ((baseline, [1, 2]), (budget, [1, 3])):
        Path(run.artifacts.run_dir).mkdir(parents=True)
        Path(run.artifacts.gsm8k_json).write_text(
            json.dumps({"sample_indices": gsm8k_indices}),
            encoding="utf-8",
        )
        Path(run.artifacts.wikitext_json).write_text(
            json.dumps({"sample_indices": [10, 11]}),
            encoding="utf-8",
        )

    comparison.validate_comparable_sample_sets(
        {"legacy": {None: baseline, 256: budget}},
        {},
    )

    assert budget.completion_status == "invalid"
    assert any("sample_indices differ" in error for error in budget.validation_errors)


def test_plot_file_name_is_stable():
    assert (
        comparison.plot_file_name("WikiText", "word perplexity")
        == "wikitext_word_perplexity.png"
    )


def test_build_kvpress_eval_command_includes_decoding_budget(tmp_path: Path):
    args = comparison.parse_args(
        [
            "--budgets",
            "256",
            "--kvpress-methods",
            "KnormPress",
            "--skip-paged-eviction",
        ]
    )

    command = comparison.build_kvpress_eval_command(
        args,
        tmp_path / "result.json",
        "KnormPress",
        256,
    )

    assert "kvpress-eval" in command
    assert "--budget-tokens" in command
    assert command[command.index("--budget-tokens") + 1] == "256"
    assert "--min-prompt-budget-tokens" in command
    assert command[command.index("--min-prompt-budget-tokens") + 1] == "256"
    assert "--transformers-quantization" in command
    assert command[command.index("--transformers-quantization") + 1] == (
        "bitsandbytes-4bit"
    )
    assert "--wikitext-continuation-words" in command


def test_build_kvpress_eval_command_honors_skip_quality(tmp_path: Path):
    args = comparison.parse_args(
        [
            "--budgets",
            "256",
            "--kvpress-methods",
            "KnormPress",
            "--skip-paged-eviction",
            "--skip-quality",
        ]
    )

    command = comparison.build_kvpress_eval_command(
        args,
        tmp_path / "result.json",
        "KnormPress",
        256,
    )

    assert "--skip-quality" in command


def test_build_kvpress_batch_eval_command_reuses_model_load(tmp_path: Path):
    args = comparison.parse_args(
        [
            "--budgets",
            "256",
            "--kvpress-methods",
            "KnormPress",
            "--skip-paged-eviction",
        ]
    )

    command = comparison.build_kvpress_batch_eval_command(
        args,
        tmp_path / "jobs.json",
    )

    assert "kvpress-eval-batch" in command
    assert "--jobs-json" in command
    assert command[command.index("--jobs-json") + 1].endswith("jobs.json")
    assert "--budget-tokens" not in command
    assert "--wikitext-continuation-words" in command
    assert "--min-prompt-budget-tokens" in command


def test_target_size_compression_ratio_keeps_requested_tokens():
    ratio = comparison.target_size_compression_ratio(100, 64)

    assert int(100 * (1.0 - ratio)) == 64
    assert comparison.target_size_compression_ratio(64, 64) == 0.0


def test_target_size_prefill_press_sets_base_ratio():
    class FakeBasePress:
        pass

    class FakeMethodPress:

        def __init__(self):
            self.compression_ratio = 0.0
            self.seen_ratio = None

        def post_init_from_model(self, model):
            pass

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            self.seen_ratio = self.compression_ratio
            return keys, values

    fake_kvpress = types.SimpleNamespace(BasePress=FakeBasePress)
    base_press = FakeMethodPress()
    press = comparison.build_target_size_prefill_press(fake_kvpress, base_press, 64)
    keys = types.SimpleNamespace(shape=(1, 1, 100, 1))
    values = object()

    returned_keys, returned_values = press.compress(
        module=object(),
        hidden_states=object(),
        keys=keys,
        values=values,
        attentions=None,
        kwargs={},
    )

    assert returned_keys is keys
    assert returned_values is values
    assert int(100 * (1.0 - base_press.seen_ratio)) == 64
    assert base_press.compression_ratio == 0.0


def test_build_decoding_press_combines_prefill_and_decoding(monkeypatch):
    class FakeBasePress:
        pass

    class FakeMethodPress(FakeBasePress):

        def __init__(self):
            self.compression_ratio = 0.0

        def post_init_from_model(self, model):
            pass

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            return keys, values

    class FakeDecodingPress(FakeBasePress):

        def __init__(
            self,
            base_press,
            compression_interval,
            target_size,
            hidden_states_buffer_size,
        ):
            self.base_press = base_press
            self.compression_interval = compression_interval
            self.target_size = target_size
            self.hidden_states_buffer_size = hidden_states_buffer_size

        def forward_hook(self, module, input, kwargs, output):
            return output

    class FakePrefillDecodingPress(FakeBasePress):

        def __init__(self, prefilling_press, decoding_press):
            self.prefilling_press = prefilling_press
            self.decoding_press = decoding_press

        def forward_hook(self, module, input, kwargs, output):
            return output

    fake_kvpress = types.SimpleNamespace(
        BasePress=FakeBasePress,
        DecodingPress=FakeDecodingPress,
        PrefillDecodingPress=FakePrefillDecodingPress,
        KnormPress=FakeMethodPress,
    )
    monkeypatch.setitem(sys.modules, "kvpress", fake_kvpress)
    args = types.SimpleNamespace(
        method="KnormPress",
        budget_tokens=64,
        compression_interval=8,
        hidden_states_buffer_size=4,
    )

    press = comparison.build_decoding_press(args)

    assert press.prefilling_press.target_size == 64
    assert press.decoding_press.target_size == 64
    assert press.decoding_press.compression_interval == 8


def test_build_transformers_model_kwargs_uses_4bit_quantization():
    class FakeBitsAndBytesConfig:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args = types.SimpleNamespace(
        dtype="auto",
        device="auto",
        transformers_quantization="bitsandbytes-4bit",
    )
    fake_torch = types.SimpleNamespace(bfloat16=object())

    model_kwargs = comparison.build_transformers_model_kwargs(
        args,
        fake_torch,
        FakeBitsAndBytesConfig,
    )

    assert model_kwargs["device_map"] == "auto"
    quantization_config = model_kwargs["quantization_config"]
    assert quantization_config.kwargs["load_in_4bit"] is True
    assert quantization_config.kwargs["bnb_4bit_quant_type"] == "nf4"


def test_collect_kvpress_run_allows_skipped_quality(tmp_path: Path):
    artifacts = comparison.KvpressArtifacts(
        run_dir=str(tmp_path),
        result_json=str(tmp_path / "result.json"),
        command_json=str(tmp_path / "commands.json"),
        stdout_log=str(tmp_path / "stdout.log"),
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "quality_skipped": True,
                "latency_s": 0.0,
                "peak_memory_mib": None,
                "gsm8k_accuracy": None,
                "wikitext_word_perplexity": None,
            }
        ),
        encoding="utf-8",
    )

    run = comparison.collect_kvpress_run(
        "KnormPress",
        256,
        artifacts,
        command_failed=False,
        fallback_latency_s=None,
    )

    assert run.completion_status == "complete"
    assert run.validation_errors == []
    assert run.latency_s == 0.0


def test_kvpress_batch_skip_quality_writes_results_without_loading_model(
    tmp_path: Path,
    monkeypatch,
):
    def fail_load_model(args):
        raise AssertionError("model should not load when --skip-quality is set")

    monkeypatch.setattr(comparison, "load_transformers_eval_model", fail_load_model)
    jobs_json = tmp_path / "jobs.json"
    result_json = tmp_path / "result.json"
    jobs_json.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "method": "KnormPress",
                        "budget_tokens": 256,
                        "output_json": str(result_json),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        jobs_json=jobs_json,
        skip_quality=True,
        method=None,
        budget_tokens=None,
        output_json=None,
        model="test-model",
    )

    assert comparison.run_kvpress_eval_batch(args) == 0

    result = json.loads(result_json.read_text(encoding="utf-8"))
    assert result["quality_skipped"] is True
    assert result["method"] == "KnormPress"
    assert result["budget_tokens"] == 256


def test_cache_position_from_kwargs_uses_position_ids():
    position_ids = types.SimpleNamespace(ndim=1)
    kwargs = {"position_ids": position_ids}

    assert comparison.cache_position_from_kwargs(kwargs) is position_ids


def test_compatible_decoding_press_adds_missing_cache_position():
    class PositionIds:

        ndim = 2

        def __init__(self):
            self.first_row = object()

        def __getitem__(self, index):
            assert index == 0
            return self.first_row

    class FakeDecodingPress:

        def __init__(self):
            self.received_kwargs = None

        def forward_hook(self, module, input, kwargs, output):
            self.received_kwargs = kwargs
            return output

    press_cls = comparison.compatible_decoding_press_cls(FakeDecodingPress)
    press = press_cls()
    position_ids = PositionIds()
    output = object()

    assert press.forward_hook(
        module=object(),
        input=[],
        kwargs={"position_ids": position_ids},
        output=output,
    ) is output
    assert press.received_kwargs["cache_position"] is position_ids.first_row


def test_dry_run_writes_paged_and_kvpress_commands(tmp_path: Path):
    status = comparison.main(
        [
            "--dry-run",
            "--skip-quality",
            "--runner",
            "legacy",
            "--model",
            "test-model",
            "--budgets",
            "256",
            "--kvpress-methods",
            "KnormPress",
            "--results-dir",
            str(tmp_path),
            "--timestamp",
            "dry",
        ]
    )

    assert status == 0
    paged_commands = json.loads(
        (
            tmp_path
            / "dry"
            / "paged_eviction"
            / "budget_256"
            / "legacy"
            / "budget_256"
            / "commands.json"
        ).read_text(encoding="utf-8")
    )
    kvpress_commands = json.loads(
        (
            tmp_path
            / "dry"
            / "kvpress"
            / "KnormPress"
            / "budget_256"
            / "commands.json"
        ).read_text(encoding="utf-8")
    )

    server_command = paged_commands["server_command"]
    assert "--paged-eviction-config" in server_command
    config = json.loads(
        server_command[server_command.index("--paged-eviction-config") + 1]
    )
    assert config["cache_budget_tokens"] == 256
    assert paged_commands["skip_serving_benchmark"] is True
    assert kvpress_commands["command"][1].endswith(
        "run_paged_eviction_kvpress_comparison.py"
    )
    assert "--budget-tokens" in kvpress_commands["command"]
    assert "--transformers-quantization" in kvpress_commands["command"]
