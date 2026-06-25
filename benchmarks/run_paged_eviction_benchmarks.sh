#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: expected existing virtualenv at $VENV_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

# python benchmarks/run_paged_eviction_block_size_sweep.py
python benchmarks/run_paged_eviction_kvpress_comparison.py
