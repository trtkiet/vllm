# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

import pytest

import benchmarks.run_paged_eviction_long_context_bench as long_bench
import benchmarks.run_paged_eviction_memory_bench as bench

pytestmark = pytest.mark.cpu_test


def test_default_matrix_targets_l4_flash_attention():
    args = long_bench.parse_args([])

    assert args.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert args.kv_cache_dtype == "bfloat16"
    assert args.attention_backend == "FLASH_ATTN"
    assert args.enable_chunked_prefill
    assert not args.enable_prefix_caching
    assert not args.skip_quality
    assert args.quality_dataset == "ruler"
    assert args.ruler_tasks == list(bench.RULER_TASKS)
    assert args.context_lengths == [8192, 16384, 32768]
    assert args.cache_budgets == [2048, 4096, 8192]
    assert len(long_bench.experiment_cells(args)) == 36


def test_smoke_matrix_crosses_eviction_budget():
    args = long_bench.parse_args(["--smoke"])
    cells = long_bench.experiment_cells(args)

    assert args.context_lengths == [4096]
    assert args.cache_budgets == [2048]
    assert args.performance_output_len == 64
    assert args.ruler_tasks == ["niah_single_1", "vt"]
    assert args.ruler_samples_per_task == 1
    assert len(cells) == 2
    assert cells[0].cache_budget_tokens is None
    assert cells[1].cache_budget_tokens == 2048


def test_server_command_enables_requested_attention_path():
    args = long_bench.parse_args(["--smoke"])
    cell = long_bench.ExperimentCell(
        context_length=4096,
        repetition=1,
        cache_budget_tokens=2048,
    )

    command = bench.build_server_command(long_bench.cell_args(args, cell), enabled=True)

    assert "--enable-chunked-prefill" in command
    assert "--no-enable-prefix-caching" in command
    assert command[command.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert command[command.index("--attention-config.backend") + 1] == "FLASH_ATTN"
    config = command[command.index("--paged-eviction-config") + 1]
    assert '"cache_budget_tokens": 2048' in config


def test_cell_args_runs_ruler_only_on_first_repetition():
    args = long_bench.parse_args([])
    first = long_bench.ExperimentCell(8192, 1, None)
    second = long_bench.ExperimentCell(8192, 2, None)

    first_args = long_bench.cell_args(args, first)
    second_args = long_bench.cell_args(args, second)

    assert first_args.ruler_context_length == 8192
    assert not first_args.skip_quality
    assert second_args.skip_quality


@pytest.mark.parametrize("value", ["1,", "0", "abc"])
def test_positive_int_list_rejects_invalid_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError):
        long_bench.parse_positive_int_list(value)


def test_cell_root_is_stable():
    cell = long_bench.ExperimentCell(8192, 2, 2048)

    assert long_bench.cell_root(Path("results"), cell) == Path(
        "results/budget_2048/context_8192/rep_2"
    )


def test_mode_order_and_labels_are_readable():
    modes = ["full_cache", "budget_8192", "budget_2048", "budget_4096"]

    ordered = sorted(modes, key=long_bench.mode_sort_key)

    assert ordered == [
        "budget_2048",
        "budget_4096",
        "budget_8192",
        "full_cache",
    ]
    assert long_bench.mode_label("full_cache") == "Full cache"
    assert long_bench.mode_label("budget_4096") == "4K-token budget"


def test_ruler_retention_is_normalized_per_context():
    artifacts = bench.RunArtifacts("", "", "", "", "", "", "", "", "")
    full = bench.empty_summary("v2", "full_cache", False, artifacts)
    budget = bench.empty_summary("v2", "budget_2048", True, artifacts)
    full.ruler_score_percent = 80.0
    budget.ruler_score_percent = 60.0
    results = [
        (long_bench.ExperimentCell(8192, 1, None), full),
        (long_bench.ExperimentCell(8192, 1, 2048), budget),
    ]

    retention = long_bench.ruler_retention_means(results)

    assert retention[("full_cache", 8192)] == 100.0
    assert retention[("budget_2048", 8192)] == 75.0
