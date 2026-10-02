# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate PagedEviction experiment results into the headline numbers.

Reads the artifacts written by the benchmark harnesses and answers the project
question with measurable values:

* ``X``   - how much longer a context fits in the same KV pool,
* ``Y``   - how many more concurrent requests fit at the same latency target,
* ``Z``   - how much RULER quality is retained at those operating points.

Inputs (all optional, discovered by glob under ``--results-root``):

* ``ruler-*/summary.json``          - RULER accuracy + serving latency,
* ``concurrency-*/summary.json``    - concurrency/latency sweep,
* ``*/summary.json``                - memory/serving comparison,
* ``validation.json``               - RoPE + KV update + SDPA validation run.

Example:

    .venv/bin/python benchmarks/analyze_paged_eviction_results.py \
        --results-root benchmarks/results \
        --output-dir benchmarks/results/analysis
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

FULL_CACHE_LABEL = "full_cache"
DEFAULT_TARGETS = (0.10, 0.20)


@dataclass
class RulerCell:
    label: str
    budget: int | None
    context_length: int
    samples_per_task: int
    score_percent: float | None
    per_task: dict[str, float]
    retention_percent: float | None
    p99_ttft_ms: float | None
    p99_tpot_ms: float | None
    output_throughput: float | None
    peak_kv_gib: float | None
    peak_kv_bytes: float | None
    pool_tokens: int | None
    source: str


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def gib(value: Any) -> float | None:
    number = as_float(value)
    return None if number is None else number / (1024**3)


def wilson_interval(
    successes: float, total: int, z: float = 1.96
) -> tuple[float, float]:
    """95% Wilson interval for a score in [0, 1] (fractional successes allowed)."""
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def discover_ruler_cells(
    results_root: Path, max_samples_per_task: int | None = None
) -> list[RulerCell]:
    """Read every RULER run under ``results_root``.

    When ``max_samples_per_task`` is set, per-task scores are recomputed from
    the stored samples so every cell is compared at the same sample count.
    """
    cells: list[RulerCell] = []
    pattern = str(results_root / "**" / "ruler.json")
    for ruler_path_str in sorted(glob.glob(pattern, recursive=True)):
        ruler_path = Path(ruler_path_str)
        ruler = load_json(ruler_path)
        if not ruler:
            continue
        # .../<label>/context_<ctx>/rep_<r>/<runner>/<label>
        try:
            label = ruler_path.parent.name
            context_length = int(ruler_path.parents[3].name.split("_")[1])
        except (IndexError, ValueError):
            continue
        # The RULER harness writes the serving metrics into the run-level
        # summary.json: <results_root>/<name>/summary.json.
        summary = {}
        for parent in ruler_path.parents:
            candidate = parent / "summary.json"
            payload = load_json(candidate)
            if payload.get("runs"):
                summary = payload
                break
        serving = find_serving_summary(summary, label, context_length)

        score = as_float(ruler.get("score_percent"))
        tasks = ruler.get("tasks") or {}
        if max_samples_per_task is not None:
            per_task: dict[str, float] = {}
            num_samples = 0
            for task, payload in tasks.items():
                samples = ((payload or {}).get("samples") or [])[:max_samples_per_task]
                if samples:
                    total = sum(
                        as_float(sample.get("score")) or 0.0 for sample in samples
                    )
                    per_task[task] = 100.0 * total / len(samples)
                else:
                    per_task[task] = (
                        as_float((payload or {}).get("score_percent")) or 0.0
                    )
                num_samples += len(samples)
            num_tasks = len(per_task) or 1
            samples_per_task = num_samples // num_tasks if num_tasks else 0
            score = sum(per_task.values()) / len(per_task) if per_task else None
        else:
            per_task = {
                task: as_float((payload or {}).get("score_percent")) or 0.0
                for task, payload in tasks.items()
            }
            num_samples = sum(
                int((payload or {}).get("num_samples") or 0)
                for payload in tasks.values()
            )
            num_tasks = len(tasks) or 1
            samples_per_task = num_samples // num_tasks if num_samples else 0

        cells.append(
            RulerCell(
                label=label,
                budget=None if label == FULL_CACHE_LABEL else parse_budget(label),
                context_length=context_length,
                samples_per_task=samples_per_task,
                score_percent=score,
                per_task=per_task,
                retention_percent=None,
                p99_ttft_ms=as_float(serving.get("p99_ttft_ms")),
                p99_tpot_ms=as_float(serving.get("p99_tpot_ms")),
                output_throughput=as_float(serving.get("output_throughput")),
                peak_kv_gib=gib(serving.get("derived_peak_kv_cache_bytes")),
                peak_kv_bytes=as_float(serving.get("derived_peak_kv_cache_bytes")),
                pool_tokens=(
                    int(serving["kv_cache_capacity_tokens"])
                    if serving.get("kv_cache_capacity_tokens")
                    else None
                ),
                source=str(ruler_path),
            )
        )
    return cells


def find_serving_summary(
    summary: dict[str, Any], label: str, context_length: int
) -> dict[str, Any]:
    """Pick the serving metrics for a (label, context) cell from summary.json."""
    for run in summary.get("runs") or []:
        cell = run.get("cell") or {}
        if cell.get("context_length") != context_length:
            continue
        summary_payload = run.get("summary") or {}
        if summary_payload.get("label") == label:
            return summary_payload
        # Some harnesses nest the run summary under a "summary" key.
        nested = summary_payload.get("summary")
        if isinstance(nested, dict) and nested.get("label") == label:
            return nested
    return {}


def parse_budget(label: str) -> int | None:
    if not label.startswith("budget_"):
        return None
    try:
        return int(label.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def attach_retention(cells: list[RulerCell]) -> None:
    baselines: dict[int, float] = {}
    for cell in cells:
        if cell.label == FULL_CACHE_LABEL and cell.score_percent is not None:
            baselines[cell.context_length] = cell.score_percent
    for cell in cells:
        base = baselines.get(cell.context_length)
        if base and cell.score_percent is not None:
            cell.retention_percent = cell.score_percent / base * 100.0


def dedupe_ruler_cells(cells: list[RulerCell]) -> list[RulerCell]:
    """Keep the run with the most samples per task for each (label, context)."""
    best: dict[tuple[str, int], RulerCell] = {}
    for cell in cells:
        key = (cell.label, cell.context_length)
        current = best.get(key)
        if current is None or cell.samples_per_task > current.samples_per_task:
            best[key] = cell
    return list(best.values())


@dataclass
class ContextPoint:
    mode: str
    budget: int | None
    context_length: int
    residency_bytes: float
    retention_percent: float | None
    samples_per_task: int


def context_multiplier_table(cells: list[RulerCell], targets: tuple[float, ...]) -> str:
    """For each budget, the largest context with retention above each target."""
    lines = [
        "### Context multiplication at fixed KV residency",
        "",
        (
            "Residency is the retained KV footprint of one request at the cell "
            "context (budget x 128 KiB/token for eviction modes, context x 128 "
            "KiB/token for the full cache). The multiplier compares a bounded "
            "cell against the largest full-cache context that fits the same "
            "residency."
        ),
        "",
        "| Budget | Context | Residency (GiB) | RULER | Retention | Samples/task | "
        + " | ".join(f"vs full <= {int(t * 100)}% deg." for t in targets)
        + " |",
        "|---:|---:|---:|---:|---:|---:|" + "---:|" * len(targets),
    ]

    full_cells = [
        cell
        for cell in cells
        if cell.label == FULL_CACHE_LABEL and cell.score_percent is not None
    ]
    ordered = sorted(
        cells,
        key=lambda cell: (
            cell.budget or 0,
            cell.context_length,
        ),
    )
    for cell in ordered:
        if cell.budget is None:
            continue
        residency = cell.budget * 128 * 1024 if cell.budget else None
        comparisons = []
        for target in targets:
            max_full = None
            for full in full_cells:
                if full.score_percent is None or full.score_percent <= 0:
                    continue
                if cell.retention_percent is not None and (
                    cell.retention_percent < (1 - target) * 100.0
                ):
                    continue
                full_residency = full.context_length * 128 * 1024
                if residency is not None and full_residency > residency:
                    continue
                if max_full is None or full.context_length > max_full:
                    max_full = full.context_length
            comparisons.append(
                "n/a" if max_full is None else f"{cell.context_length / max_full:.1f}x"
            )
        lines.append(
            "| {budget} | {context} | {resid} | {score} | {retention} | "
            "{samples} | {comp} |".format(
                budget=cell.budget,
                context=cell.context_length,
                resid="n/a" if residency is None else f"{gib(residency):.2f}",
                score=fmt(cell.score_percent),
                retention=fmt(cell.retention_percent, "%"),
                samples=cell.samples_per_task,
                comp=" | ".join(comparisons),
            )
        )
    return "\n".join(lines)


def ruler_table(cells: list[RulerCell]) -> tuple[str, str]:
    lines = [
        "### RULER accuracy",
        "",
        (
            "Scores are the mean over 13 task splits with the official "
            "substring-match scoring. The 95% column is the Wilson interval "
            "over the aggregated samples."
        ),
        "",
        "| Context | Mode | Budget | Samples/task | RULER | 95% CI | Retention | "
        "p99 TTFT (ms) | Output tok/s | Peak KV (GiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in sorted(cells, key=lambda cell: (cell.context_length, cell.label)):
        num_tasks = max(len(cell.per_task), 1)
        total = cell.samples_per_task * num_tasks
        successes = (cell.score_percent or 0.0) / 100.0 * total
        low, high = wilson_interval(successes, total)
        lines.append(
            "| {} | {} | {} | {} | {} | +/-{:.1f} | {} | {} | {} | {} |".format(
                cell.context_length,
                cell.label,
                cell.budget if cell.budget is not None else "n/a",
                cell.samples_per_task,
                fmt(cell.score_percent),
                (high - low) * 50.0,
                fmt(cell.retention_percent, "%"),
                fmt(cell.p99_ttft_ms),
                fmt(cell.output_throughput),
                fmt(cell.peak_kv_gib),
            )
        )
    detail_lines = [
        "### Per-task RULER scores",
        "",
        "| Context | Mode | " + " | ".join(sorted(next(iter(cells)).per_task)) + " |",
        "|---:|---|" + "---:|" * len(next(iter(cells)).per_task),
    ]
    for cell in sorted(cells, key=lambda cell: (cell.context_length, cell.label)):
        row = [str(cell.context_length), cell.label]
        for task in sorted(cell.per_task):
            row.append(fmt(cell.per_task.get(task)))
        detail_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines), "\n".join(detail_lines)


def concurrency_section(
    results_root: Path, tolerance: float
) -> tuple[str, dict[str, Any]]:
    lines = [
        "### Concurrency at equal latency",
        "",
        (
            f"A bounded cell is accepted while its p99 TTFT and p99 TPOT stay "
            f"within {tolerance:.0%} of the lowest full-cache concurrency at "
            "the same context, with zero failed requests."
        ),
        "",
        "| Context | Mode | Budget | Max concurrency | Full-cache baseline | Y |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    best_multiplier = None
    best_row = None
    for path_str in sorted(
        glob.glob(str(results_root / "concurrency-*" / "capacity.json"))
    ):
        payload = load_json(Path(path_str))
        for row in payload.get("capacity") or []:
            if row.get("cache_budget_tokens") is None:
                continue
            multiplier = as_float(row.get("concurrency_multiplier"))
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    row.get("context_length"),
                    row.get("mode"),
                    row.get("cache_budget_tokens"),
                    row.get("max_latency_preserving_concurrency"),
                    row.get("full_cache_baseline_concurrency"),
                    "n/a" if multiplier is None else f"{multiplier:.2f}x",
                )
            )
            if multiplier is not None and (
                best_multiplier is None or multiplier > best_multiplier
            ):
                best_multiplier = multiplier
                best_row = dict(row)
    if best_multiplier is None:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines), {"best": best_row, "multiplier": best_multiplier}


def validation_section(results_root: Path) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for path_str in sorted(
        glob.glob(str(results_root / "**" / "validation.json"), recursive=True)
    ):
        payload = load_json(Path(path_str))
        if not payload:
            continue
        records.append(payload)
    lines = [
        "### RoPE + KV update + SDPA validation",
        "",
        (
            "Greedy generation preserves the exact token stream and per-step "
            "logprobs when compared against a reference model with the same "
            "retained KV set. This exercises logical RoPE positions, resident "
            "slot mapping into reused blocks, and FA2 SDPA over the rotated "
            "block table."
        ),
        "",
        "| Runner | Budget | Evictions | Tokens compared | Max logprob delta | "
        "Token match | Control match | Status |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                record.get("runner"),
                record.get("cache_budget_tokens"),
                record.get("evictions"),
                record.get("tokens_compared"),
                record.get("max_logprob_delta"),
                "yes" if record.get("token_match") else "no",
                "yes" if record.get("control_match") else "no",
                "pass" if record.get("passed") else "fail",
            )
        )
    if not records:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines), records


def fmt(value: Any, suffix: str = "") -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}{suffix}"
    return f"{number:.2f}{suffix}"


STYLE = {
    "background": "#ffffff",
    "axis_text": "#5f6368",
    "grid": "#dadce0",
    "title": "#202124",
    "palette": ["#4285f4", "#34a853", "#fa7b17", "#fbbc04"],
}


def capacity_series(
    cells: list[RulerCell],
) -> list[tuple[str, dict[int, float], float]]:
    """Group retention by KV budget; return per-context degradation values.

    The label reports the retained fraction of the context length, which is the
    quantity a reader cares about: a budget of 8192 against a 32768-token
    context retains 25% of the KV cache.
    """
    budgets = sorted({cell.budget for cell in cells if cell.budget is not None})
    series: list[tuple[str, dict[int, float], float]] = []
    for budget in budgets:
        values: dict[int, float] = {}
        for cell in cells:
            if cell.budget != budget or cell.retention_percent is None:
                continue
            values[cell.context_length] = 100.0 - cell.retention_percent
        if not values:
            continue
        fraction = min(budget / context for context in values)
        series.append((f"KV Cache Budget {budget}", values, fraction))
    # The full-cache baseline is the 100% capacity series.
    full_values = {
        cell.context_length: 0.0
        for cell in cells
        if cell.budget is None and cell.score_percent is not None
    }
    if full_values:
        series.append(("Full KV Cache", full_values, 1.0))
    return series


def absolute_score_series(
    cells: list[RulerCell],
) -> list[tuple[str, dict[int, float], float]]:
    """Group absolute RULER scores by retained context fraction."""
    budgets = sorted({cell.budget for cell in cells if cell.budget is not None})
    series: list[tuple[str, dict[int, float], float]] = []
    for budget in budgets:
        values = {
            cell.context_length: cell.score_percent
            for cell in cells
            if cell.budget == budget and cell.score_percent is not None
        }
        if not values:
            continue
        fraction = min(budget / context for context in values)
        series.append((f"KV Cache Budget {budget}", values, fraction))
    full_values = {
        cell.context_length: cell.score_percent
        for cell in cells
        if cell.budget is None and cell.score_percent is not None
    }
    if full_values:
        series.append(("Full KV Cache", full_values, 1.0))
    return series


def analytic_concurrency_caps(
    cells: list[RulerCell],
    context_length: int,
    block_size: int = 16,
    max_num_scheduled_tokens: int = 8192,
    max_num_seqs: int = 64,
) -> list[dict[str, Any]]:
    """Scheduler-side concurrency caps for the fixed pool at one context.

    Mirrors ``Scheduler._initialize_paged_eviction_capacity``: every running
    request reserves ``budget / block_size`` blocks plus the transient blocks a
    shared prefill-token budget can open. The full-cache cap is the pool divided
    by the context length.
    """
    pool_tokens = next((cell.pool_tokens for cell in cells if cell.pool_tokens), None)
    if not pool_tokens:
        return []
    usable_blocks = pool_tokens // block_size - 1
    full_cache_cap = max(0, (usable_blocks * block_size) // context_length)
    budgets = sorted({cell.budget for cell in cells if cell.budget is not None})
    rows: list[dict[str, Any]] = [
        {
            "label": "Full KV Cache",
            "budget": None,
            "max_concurrent": full_cache_cap,
            "multiplier": 1.0 if full_cache_cap else None,
        }
    ]
    for budget in budgets:
        budget_blocks = budget // block_size

        def required(num_running_reqs: int, budget_blocks: int = budget_blocks) -> int:
            first_blocks = min(num_running_reqs, max_num_scheduled_tokens)
            remaining_tokens = max_num_scheduled_tokens - first_blocks
            transient_blocks = first_blocks + remaining_tokens // block_size
            return num_running_reqs * budget_blocks + transient_blocks

        cap = min(max_num_seqs, usable_blocks // budget_blocks)
        while cap > 0 and required(cap) > usable_blocks:
            cap -= 1
        rows.append(
            {
                "label": f"KV Cache Budget {budget}",
                "budget": budget,
                "max_concurrent": cap,
                "multiplier": (cap / full_cache_cap) if full_cache_cap else None,
            }
        )
    return rows


def concurrency_capacity_section(
    rows: list[dict[str, Any]], context_length: int, pool_tokens: int
) -> str:
    pool_gib = pool_tokens * 128 * 1024 / (1024**3)
    lines = [
        "### Concurrency at fixed KV capacity",
        "",
        (
            f"Maximum concurrent {context_length // 1024}k-context requests in "
            f"the fixed pool ({pool_tokens:,} tokens = {pool_gib:.2f} GiB), "
            "computed from the scheduler's admission formula "
            "(`max_num_batched_tokens=8192`, `max_num_seqs=64`). Eviction "
            "reserves only the retained budget per request, so the same pool "
            "admits proportionally more users."
        ),
        "",
        "| Mode | KV/request (GiB) | Max concurrent 32k requests | Multiplier |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        budget = row["budget"]
        residency = (
            context_length * 128 * 1024 if budget is None else budget * 128 * 1024
        )
        lines.append(
            "| {} | {:.2f} | {} | {} |".format(
                row["label"],
                residency / (1024**3),
                row["max_concurrent"],
                "1.0x" if row["multiplier"] is None else f"{row['multiplier']:.1f}x",
            )
        )
    return "\n".join(lines)


def concurrency_capacity_plot(
    output_dir: Path,
    rows: list[dict[str, Any]],
    context_length: int,
    pool_tokens: int,
) -> str | None:
    """Grouped-style bar chart of concurrent users per KV mode."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return None
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    _style_axes(ax, plt)
    palette = STYLE["palette"]
    x_positions = list(range(len(rows)))
    heights = [row["max_concurrent"] for row in rows]
    top = max(heights) if heights else 1
    ax.set_ylim(0, top * 1.25)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    for index, (position, row) in enumerate(zip(x_positions, rows)):
        height = row["max_concurrent"]
        color = palette[index % len(palette)]
        rounded = FancyBboxPatch(
            (position - 0.34, 0),
            0.68,
            height,
            boxstyle="round,pad=0,rounding_size=0.04",
            linewidth=0,
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rounded)
        label = f"{height}"
        if row["multiplier"] is not None and row["multiplier"] != 1.0:
            label = f"{height}  ({row['multiplier']:.1f}x)"
        ax.text(
            position,
            height + top * 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            color=STYLE["title"],
            fontweight="bold",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [row["label"].replace("KV Cache ", "") for row in rows],
        color=STYLE["axis_text"],
    )
    ax.set_ylabel(
        f"concurrent {context_length // 1024}k requests",
        color=STYLE["axis_text"],
        fontsize=11,
    )
    pool_gib = pool_tokens * 128 * 1024 / (1024**3)
    ax.set_title(
        f"Concurrent {context_length // 1024}k-Context Users at Fixed KV Capacity",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=STYLE["title"],
        pad=34,
    )
    ax.text(
        0.0,
        1.055,
        (
            f"Maximum concurrent {context_length // 1024}k requests in the "
            f"same {pool_gib:.2f} GiB KV pool, with and without PagedEviction."
        ),
        transform=ax.transAxes,
        fontsize=11,
        color=STYLE["axis_text"],
        ha="left",
    )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "concurrency_capacity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=STYLE["background"])
    plt.close(fig)
    return str(path)


def _style_axes(ax, plt) -> None:
    ax.set_facecolor(STYLE["background"])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=STYLE["axis_text"], length=0, labelsize=10)
    ax.yaxis.grid(True, color=STYLE["grid"], linestyle="--", linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def grouped_degradation_plot(
    output_dir: Path,
    cells: list[RulerCell],
    *,
    filename: str,
    title: str,
    subtitle: str,
    ylabel: str,
    pick,
    suffix: str = "%",
    series_builder=capacity_series,
) -> str | None:
    """Grouped bars: one group per context length, one bar per capacity."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Patch
    except ImportError:
        return None

    series = series_builder(cells)
    if not series:
        return None
    contexts = sorted({context for _, values, _ in series for context in values})
    if not contexts:
        return None

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    _style_axes(ax, plt)

    n_series = len(series)
    group_width = 0.82
    bar_width = group_width / n_series
    palette = STYLE["palette"]

    x_positions = list(range(len(contexts)))
    all_values: list[float] = []
    legend_handles = []
    for index, (label, values, _) in enumerate(series):
        offsets = [
            position - group_width / 2 + bar_width * (index + 0.5)
            for position in x_positions
        ]
        heights = []
        for context in contexts:
            value = pick(values.get(context), context)
            heights.append(float("nan") if value is None else value)
        all_values.extend(value for value in heights if value == value)
        color = palette[index % len(palette)]
        for offset, height in zip(offsets, heights):
            if height != height:
                continue
            rounded = FancyBboxPatch(
                (offset - bar_width * 0.43, 0),
                bar_width * 0.86,
                height,
                boxstyle="round,pad=0,rounding_size=0.035",
                linewidth=0,
                facecolor=color,
                zorder=3,
            )
            ax.add_patch(rounded)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))

    top = max(all_values) if all_values else 1.0
    ax.set_ylim(0, top * 1.15 if top else 1.0)
    ax.set_xlim(-0.55, len(contexts) - 0.45)
    ax.set_yticks([value for value in ax.get_yticks() if 0 <= value <= top * 1.15])
    ax.set_yticklabels(
        [f"{value:.0f}{suffix}" for value in ax.get_yticks()],
        color=STYLE["axis_text"],
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{context // 1024}k" for context in contexts],
        color=STYLE["axis_text"],
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=STYLE["title"],
        pad=34,
    )
    ax.text(
        0.0,
        1.055,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        color=STYLE["axis_text"],
        ha="left",
    )
    if ylabel:
        ax.set_ylabel(ylabel, color=STYLE["axis_text"], fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    handles = legend_handles or handles
    labels = [handle.get_label() for handle in handles]
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(labels),
            frameon=False,
            fontsize=12,
            labelcolor=STYLE["axis_text"],
        )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=STYLE["background"])
    plt.close(fig)
    return str(path)


def write_plots(
    output_dir: Path, cells: list[RulerCell], targets: tuple[float, ...]
) -> list[str]:
    if importlib.util.find_spec("matplotlib") is None:
        return []
    if not cells:
        return []
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    degradation = grouped_degradation_plot(
        output_dir,
        cells,
        filename="accuracy_degradation.png",
        title="Accuracy Degradation by Context Length and KV Cache Capacity",
        subtitle=(
            "RULER accuracy degradation (%) relative to the full-cache "
            "baseline at the same context length."
        ),
        ylabel="",
        pick=lambda value, context: value,
    )
    if degradation:
        written.append(degradation)

    score_plot = grouped_degradation_plot(
        output_dir,
        cells,
        filename="ruler_score_by_capacity.png",
        title="RULER Score by Context Length and KV Cache Capacity",
        subtitle="Absolute RULER score (%) over 13 tasks.",
        ylabel="",
        pick=lambda value, context: value,
        series_builder=absolute_score_series,
    )
    if score_plot:
        written.append(score_plot)

    capacity_rows = analytic_concurrency_caps(cells, context_length=32768)
    pool_tokens = next((cell.pool_tokens for cell in cells if cell.pool_tokens), 0)
    if capacity_rows and pool_tokens:
        capacity_plot = concurrency_capacity_plot(
            output_dir, capacity_rows, 32768, pool_tokens
        )
        if capacity_plot:
            written.append(capacity_plot)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmarks/results"),
        help="Root directory containing the benchmark result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/analysis"),
    )
    parser.add_argument(
        "--relative-latency-tolerance",
        type=float,
        default=0.10,
        help="Latency tolerance used in the concurrency conclusion.",
    )
    parser.add_argument(
        "--accuracy-targets",
        default="0.10,0.20",
        help="Comma-separated relative accuracy degradation targets.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help=(
            "Recompute every RULER cell from the first N stored samples per "
            "task so all cells are compared at the same sample count."
        ),
    )
    parser.add_argument(
        "--context-lengths",
        default=None,
        help=(
            "Comma-separated context lengths to include. Defaults to every "
            "context found; set this to exclude pilot/smoke contexts."
        ),
    )
    args = parser.parse_args(argv)

    targets = tuple(
        float(part) for part in args.accuracy_targets.split(",") if part.strip()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = dedupe_ruler_cells(
        discover_ruler_cells(args.results_root, args.max_samples_per_task)
    )
    if args.context_lengths:
        allowed_contexts = {
            int(part) for part in args.context_lengths.split(",") if part.strip()
        }
        cells = [cell for cell in cells if cell.context_length in allowed_contexts]
    attach_retention(cells)

    ruler_md, per_task_md = (
        ruler_table(cells) if cells else ("No RULER results found.", "")
    )
    context_md = (
        context_multiplier_table(cells, targets) if cells else "No RULER results found."
    )
    concurrency_md, concurrency = concurrency_section(
        args.results_root, args.relative_latency_tolerance
    )
    validation_md, validation = validation_section(args.results_root)
    concurrency_rows = (
        analytic_concurrency_caps(cells, context_length=32768) if cells else []
    )
    concurrency_capacity_md = (
        concurrency_capacity_section(
            concurrency_rows,
            context_length=32768,
            pool_tokens=next(
                (cell.pool_tokens for cell in cells if cell.pool_tokens), 0
            ),
        )
        if concurrency_rows
        else "No RULER results found."
    )

    # Headline accuracy: best retention achieved at each degradation target.
    headline: dict[str, Any] = {}
    for target in targets:
        candidates = [
            cell
            for cell in cells
            if cell.budget is not None
            and cell.retention_percent is not None
            and cell.retention_percent >= (1 - target) * 100.0
        ]
        if not candidates:
            headline[f"retention_{int(target * 100)}"] = None
            continue
        best = max(candidates, key=lambda cell: cell.context_length)
        full_max = max(
            (
                full.context_length
                for full in cells
                if full.label == FULL_CACHE_LABEL
                and full.context_length * 128 * 1024 <= (best.budget or 0) * 128 * 1024
            ),
            default=None,
        )
        headline[f"retention_{int(target * 100)}"] = {
            "context_length": best.context_length,
            "budget": best.budget,
            "retention_percent": best.retention_percent,
            "score_percent": best.score_percent,
            "samples_per_task": best.samples_per_task,
            "full_cache_context_matched": full_max,
            "context_multiplier": (
                best.context_length / full_max if full_max else None
            ),
        }

    lines = [
        "# PagedEviction: measurable value with a fixed KV pool",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Headline",
        "",
    ]
    for target in targets:
        entry = headline.get(f"retention_{int(target * 100)}")
        if not entry:
            lines.append(
                f"- Retention within **{int(target * 100)}% degradation**: "
                "not reached by any tested cell."
            )
            continue
        multiplier = entry["context_multiplier"]
        lines.append(
            "- Retention within **{pct}% degradation**: budget {budget} at context "
            "{ctx} retains {ret:.1f}% ({samples} samples/task){mult}.".format(
                pct=int(target * 100),
                budget=entry["budget"],
                ctx=entry["context_length"],
                ret=entry["retention_percent"],
                samples=entry["samples_per_task"],
                mult=(
                    ""
                    if multiplier is None
                    else (
                        " = {:.1f}x the full-cache context at equal KV "
                        "residency".format(multiplier)
                    )
                ),
            )
        )
    y = concurrency.get("multiplier")
    if y is None and concurrency_rows:
        best_row = max(
            (row for row in concurrency_rows if row["budget"] is not None),
            key=lambda row: row["max_concurrent"],
        )
        lines.append(
            "- Concurrency (analytic): budget {} admits {} concurrent 32k "
            "requests vs {} full-cache ({:.1f}x) in the same pool.".format(
                best_row["budget"],
                best_row["max_concurrent"],
                concurrency_rows[0]["max_concurrent"],
                best_row["multiplier"],
            )
        )
    else:
        lines.append(
            "- Concurrency: {}.".format(
                "no sweep results found"
                if y is None
                else (
                    "up to {:.2f}x more concurrent requests at the same latency "
                    "target (context {}, budget {}, baseline {})".format(
                        y,
                        concurrency["best"].get("context_length"),
                        concurrency["best"].get("cache_budget_tokens"),
                        concurrency["best"].get("full_cache_baseline_concurrency"),
                    )
                )
            )
        )
    passed = [record for record in validation if record.get("passed")]
    lines.append(
        "- Validation: {}.".format(
            "no validation record found"
            if not validation
            else (
                "{}/{} RoPE + KV update + SDPA checks passed".format(
                    len(passed), len(validation)
                )
            )
        )
    )

    body = "\n\n".join(
        [
            ruler_md,
            per_task_md,
            context_md,
            concurrency_md,
            concurrency_capacity_md,
            validation_md,
        ]
    )
    (args.output_dir / "CONCLUSION.md").write_text(
        "\n".join(lines) + "\n\n" + body + "\n", encoding="utf-8"
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accuracy_targets": list(targets),
        "headline": headline,
        "concurrency": concurrency,
        "validation": validation,
        "ruler_cells": [
            {
                "label": cell.label,
                "budget": cell.budget,
                "context_length": cell.context_length,
                "samples_per_task": cell.samples_per_task,
                "score_percent": cell.score_percent,
                "retention_percent": cell.retention_percent,
                "per_task": cell.per_task,
                "source": cell.source,
            }
            for cell in sorted(
                cells, key=lambda cell: (cell.context_length, cell.label)
            )
        ],
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plots = write_plots(args.output_dir, cells, targets)

    print(f"conclusion: {args.output_dir / 'CONCLUSION.md'}")
    print(f"analysis:   {args.output_dir / 'analysis.json'}")
    for path in plots:
        print(f"plot:       {path}")
    for target in targets:
        print(
            "  retention <= {}%: {}".format(
                int(target * 100),
                headline.get(f"retention_{int(target * 100)}"),
            )
        )
    print(f"  concurrency multiplier: {y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
