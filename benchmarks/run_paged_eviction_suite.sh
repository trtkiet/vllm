#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the full PagedEviction experiment suite and stop on the first hard
# failure. Each stage is resumable; re-running skips completed cells.
#
# Usage:
#   benchmarks/run_paged_eviction_suite.sh [ruler|sweep|all]
#
# GPU time (L4, 100 samples/task): RULER ~23 h, sweep ~2 h.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
STAGE="${1:-all}"

RULER_GRID_ARGS=(
  --context-lengths 8192,16384,32768
  --budgets 2048,4096,8192
  --ruler-samples-per-task 50
  --max-num-batched-tokens 8192
  --results-dir benchmarks/results/paged_eviction_long_context
  --name ruler-100samples-full-grid
  --resume
)

SWEEP_ARGS=(
  --context-lengths 32768
  --budgets 4096,8192,16384
  --concurrency-levels 1,2,4,8,16
  --num-prompts 32
  --max-num-batched-tokens 8192
  --relative-latency-tolerance 0.10
  --results-dir benchmarks/results/paged_eviction_concurrency
  --name concurrency-32768
)

run_ruler() {
  echo "=== RULER grid (100 samples/task) ==="
  "$PY" benchmarks/run_paged_eviction_ruler.py "${RULER_GRID_ARGS[@]}"
}

run_sweep() {
  echo "=== Concurrency sweep (32k) ==="
  "$PY" benchmarks/run_paged_eviction_concurrency_sweep.py "${SWEEP_ARGS[@]}"
}

case "$STAGE" in
  ruler) run_ruler ;;
  sweep) run_sweep ;;
  all) run_ruler ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo "=== Analysis ==="
"$PY" benchmarks/analyze_paged_eviction_results.py \
  --results-root benchmarks/results \
  --output-dir benchmarks/results/analysis \
  --max-samples-per-task 50 \
  --context-lengths 8192,16384,32768

echo "done"
