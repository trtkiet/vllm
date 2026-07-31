# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

import pytest

import benchmarks.run_paged_eviction_block_size_sweep as sweep

pytestmark = pytest.mark.cpu_test


def test_parse_block_sizes():
    assert sweep.parse_block_sizes("16,32, 64") == [16, 32, 64]


@pytest.mark.parametrize("block_sizes", ["8", "24", "16,24"])
def test_parse_args_rejects_unsupported_block_sizes(block_sizes: str):
    with pytest.raises(SystemExit):
        sweep.parse_args(["--block-sizes", block_sizes])


def test_parse_args_allows_recording_unsupported_block_sizes():
    args = sweep.parse_args(["--block-sizes", "8,16", "--record-unsupported"])

    assert args.block_sizes == [8, 16]


def test_valid_block_size_plans_paths_and_server_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []

    def fake_run_one(args, root_dir, runner, label, enabled):
        artifacts = sweep.make_artifacts(root_dir / runner / label)
        calls.append(
            {
                "block_size": args.block_size,
                "root_dir": root_dir,
                "runner": runner,
                "label": label,
                "command": sweep.bench.build_server_command(args, enabled),
            }
        )
        return sweep.bench.empty_summary(runner, label, enabled, artifacts)

    monkeypatch.setattr(sweep.bench, "run_one", fake_run_one)
    args = sweep.parse_args(
        [
            "--block-sizes",
            "32",
            "--runner",
            "legacy",
            "--dry-run",
            "--skip-quality",
        ]
    )

    runs = sweep.run_block_size(args, tmp_path, 32)

    assert set(runs) == {"legacy"}
    assert {call["label"] for call in calls} == {"disabled", "enabled"}
    assert all(call["block_size"] == 32 for call in calls)
    assert all(call["root_dir"] == tmp_path / "block_size_32" for call in calls)
    for call in calls:
        command = call["command"]
        assert "--block-size" in command
        assert command[command.index("--block-size") + 1] == "32"


def test_record_unsupported_writes_summary_without_running_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_run_one(*args, **kwargs):
        raise AssertionError("server run should not be started")

    monkeypatch.setattr(sweep.bench, "run_one", fail_run_one)
    args = sweep.parse_args(
        [
            "--block-sizes",
            "8",
            "--record-unsupported",
            "--runner",
            "legacy",
            "--dry-run",
            "--skip-quality",
        ]
    )
    runs_by_block_size = {
        8: sweep.record_unsupported_block_size(args, tmp_path, 8),
    }

    sweep.write_summary(tmp_path, args, runs_by_block_size)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"]["8"]["legacy"]["disabled"][
        "completion_status"
    ] == "unsupported"

    commands = json.loads(
        (
            tmp_path / "block_size_8" / "legacy" / "disabled" / "commands.json"
        ).read_text(encoding="utf-8")
    )
    assert commands["server_command"] is None
    assert commands["benchmark_command"] is None

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["block_size"] == "8"
    assert rows[0]["completion_status"] == "unsupported"


def test_write_summary_includes_block_size_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_run_one(args, root_dir, runner, label, enabled):
        artifacts = sweep.make_artifacts(root_dir / runner / label)
        run = sweep.bench.empty_summary(runner, label, enabled, artifacts)
        run.completion_status = "complete"
        run.output_throughput = float(args.block_size)
        if enabled:
            run.output_throughput *= 2
        return run

    monkeypatch.setattr(sweep.bench, "run_one", fake_run_one)
    args = sweep.parse_args(
        [
            "--block-sizes",
            "16,32",
            "--runner",
            "legacy",
            "--dry-run",
            "--skip-quality",
        ]
    )
    runs_by_block_size = {
        block_size: sweep.run_block_size(args, tmp_path, block_size)
        for block_size in args.block_sizes
    }

    sweep.write_summary(tmp_path, args, runs_by_block_size)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "16" in summary["runs"]
    assert "32" in summary["runs"]
    assert summary["deltas_enabled_vs_disabled_pct"]["16"]["legacy"][
        "output_throughput"
    ] == 100.0
    assert summary["sensitivity_deltas_vs_block_size_16_pct"]["32"]["legacy"][
        "disabled"
    ]["output_throughput"] == 100.0

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        first_column = next(csv.reader(file))[0]
    assert first_column == "block_size"
