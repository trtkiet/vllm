# Reproducing the PagedEviction benchmark

This guide reproduces the measurements in
[PagedEviction: Long-Context and Concurrency Benchmark](paged_eviction.md).

The full suite takes roughly 24 hours of GPU time on one NVIDIA L4. The RULER
grid is resumable at cell and task granularity, so it can be split across
sessions.

## Prerequisites

- One CUDA GPU with at least ~20 GiB free (the recorded runs used an NVIDIA L4).
- A Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct`.
- About 20 GB of disk for the model and RULER datasets.

### Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in   # resolves for the current platform
```

The benchmark scripts additionally use `datasets`, `matplotlib`, and `tqdm`,
which the test requirements provide.

### Hugging Face access and cache

The harness loads the gated model and the public RULER datasets. Ensure the hub
cache directory exists (a missing cache directory makes `datasets` fail with
`FileNotFoundError`), then log in once:

```bash
mkdir -p ~/.cache/huggingface/hub
.venv/bin/python -c "from huggingface_hub import login; login()"
```

Prewarm the model and datasets so the timed runs do not include downloads:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', max_workers=8)
from datasets import load_dataset
for ctx in (8192, 16384, 32768):
    for task in ('niah_single_1','niah_single_2','niah_single_3',
                 'niah_multikey_1','niah_multikey_2','niah_multikey_3',
                 'niah_multivalue','niah_multiquery','vt','cwe','fwe',
                 'qa_1','qa_2'):
        load_dataset(f'lighteval/RULER-{ctx}-llama3.1-8b-chat', split=task)
print('ready')
"
```

## Run the full suite

```bash
nohup bash benchmarks/run_paged_eviction_suite.sh all \
  > /tmp/paged_eviction_suite.log 2>&1 &
```

This runs, in order:

1. **RULER grid** — 3 contexts × 4 modes = 12 cells, 50 samples/task.
2. **Analysis** — `CONCLUSION.md`, `analysis.json`, and plots.

The optional concurrency sweep (`bash benchmarks/run_paged_eviction_suite.sh
sweep`) measures latency-vs-concurrency with `vllm bench serve`; the reported
concurrency numbers instead come from the scheduler's admission formula, which
the analysis computes directly from the measured pool size.

To run a stage separately:

```bash
bash benchmarks/run_paged_eviction_suite.sh ruler
bash benchmarks/run_paged_eviction_suite.sh sweep
```

### Expected runtimes (NVIDIA L4, power-capped)

| Stage | Time |
| --- | ---: |
| RULER 8k cells (4) | ~1 h each |
| RULER 16k cells (4) | ~1 h 45 m each |
| RULER 32k cells (4) | ~2–3 h each |
| Concurrency sweep (optional) | ~4 h |

### Resuming

Every stage is resumable. Relaunching the same command skips completed cells and
continues partially finished RULER cells from the last checkpointed sample
(`<run_dir>/ruler_checkpoints/<context>_<task>.jsonl`). A cell's checkpoints are
deleted once its `ruler.json` is written.

To stop cleanly:

```bash
pkill -INT -f run_paged_eviction_ruler
pkill -f run_paged_eviction_suite
```

## Run the stages directly

### RULER grid

```bash
.venv/bin/python benchmarks/run_paged_eviction_ruler.py \
  --context-lengths 8192,16384,32768 \
  --budgets 2048,4096,8192 \
  --ruler-samples-per-task 50 \
  --max-num-batched-tokens 8192 \
  --results-dir benchmarks/results/paged_eviction_long_context \
  --name ruler-100samples-full-grid \
  --resume
```

Use `--ruler-samples-per-task 500` for the official published protocol, and
`--dry-run` to print the exact per-cell server and benchmark commands without
starting servers. Each cell launches a vLLM server with
`--paged-eviction-config '{"cache_budget_tokens": <budget>}'` (or without it for
the full-cache baseline) and then evaluates all 13 RULER task splits.

### Concurrency (analytic)

The reported concurrency numbers are computed by the analysis script from the
scheduler's admission formula and the measured pool size; no separate run is
required. An optional latency sweep is available if a measured
latency-vs-concurrency curve is wanted:

```bash
.venv/bin/python benchmarks/run_paged_eviction_concurrency_sweep.py \
  --context-lengths 32768 \
  --budgets 4096,8192,16384 \
  --concurrency-levels 1,2,4,8,16 \
  --num-prompts 32 \
  --max-num-batched-tokens 8192 \
  --relative-latency-tolerance 0.10 \
  --results-dir benchmarks/results/paged_eviction_concurrency \
  --name concurrency-32768
```

Each cell launches a fresh server and runs `vllm bench serve` at one
concurrency level, recording TTFT/TPOT percentiles, throughput, failures, and
peak KV occupancy.

### Validation test

The RoPE + KV update + SDPA oracle test runs in-process so the test can install
its deterministic FIFO scoring hook:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python -m pytest \
  tests/v1/worker/test_paged_eviction_attention_path.py -v -s
```

It covers both model runners. It currently fails on the one-block
over-eviction defect documented in the report.

### Analysis and plots

```bash
.venv/bin/python benchmarks/analyze_paged_eviction_results.py \
  --results-root benchmarks/results \
  --output-dir benchmarks/results/analysis \
  --max-samples-per-task 50 \
  --context-lengths 8192,16384,32768
```

`--max-samples-per-task` recomputes every cell from the first N stored samples
so all cells are compared at the same sample count; `--context-lengths` excludes
pilot or smoke runs from other contexts. The script writes `CONCLUSION.md`,
`analysis.json`, and the grouped-bar plots under `plots/`.

## Artifacts

| Path | Contents |
| --- | --- |
| `benchmarks/results/paged_eviction_long_context/ruler-100samples-full-grid/` | `summary.json`, `summary.csv`, `REPORT.md`, per-cell `ruler.json` and server logs |
| `benchmarks/results/analysis/` | `CONCLUSION.md`, `analysis.json`, `plots/` |
| `benchmarks/results/paged_eviction_concurrency/concurrency-32768/` | optional sweep output (`summary.json`, `summary.csv`, `capacity.json`, `REPORT.md`) |

Per-cell `ruler.json` files store every sample's prediction, references, score,
and elapsed time, so scores can be recomputed at any sample count without
re-running inference.
