# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## PagedEviction benchmarks

The `run_paged_eviction_*` scripts benchmark the PagedEviction KV-cache
feature (`vllm/config/paged_eviction.py`). They share the engine in
`run_paged_eviction_memory_bench.py`, which launches vLLM servers, scrapes
nvidia-smi and Prometheus `/metrics` samples, and writes per-run artifacts
(`commands.json`, `server.log`, `bench.json`, `metrics_samples.jsonl`,
`nvidia_smi.csv`) plus aggregate `summary.json`/`summary.csv`.

- `run_paged_eviction_memory_bench.py`: disabled/enabled serving and memory
  comparison with GPQA/GSM8K/WikiText quality checks.
- `run_paged_eviction_block_size_sweep.py`: PagedEviction across KV block
  sizes.
- `run_paged_eviction_kvpress_comparison.py`: PagedEviction vs
  transformers/KVPress at matched retained-KV budgets.
- `run_paged_eviction_ruler.py`: long-context quality (RULER) and serving
  latency over context-length/budget grids.

### Long-context RULER benchmark

`run_paged_eviction_ruler.py` measures how PagedEviction's retained-KV budget
affects long-context quality (RULER) and serving latency. For every
(context length, KV budget, repetition) cell it launches a vLLM server
(chunked prefill enabled, prefix caching disabled, FA2, bf16 KV cache, online
FP8 weights), runs a synthetic `vllm bench serve` workload at that context
length, and evaluates RULER from `lighteval/RULER-{context_length}-{model}`
(13 task splits: NIAH variants, VT, CWE, FWE, QA). Predictions are scored
with the official substring match (fraction of references found; any reference
for QA), and `REPORT.md` reports TTFT p50, output throughput, peak occupied
KV-cache memory, RULER score, and retention (score relative to the full-cache
baseline of the same context length).

Reproduce the recorded pilot (1 sample per task; noisy — use it to validate
the pipeline, not for headline numbers):

```bash
.venv/bin/python benchmarks/run_paged_eviction_ruler.py \
    --context-lengths 8192,16384,32768 \
    --budgets 2048,4096,8192 \
    --results-dir benchmarks/results/paged_eviction_long_context \
    --name ruler-8k-32k-all-tasks-repro
```

For the full published protocol, add `--ruler-samples-per-task 500`. Use
`--dry-run` to print the exact per-cell server/benchmark commands without
starting servers, and `--resume` to continue an interrupted grid. Results are
written under `benchmarks/results/` (gitignored). Requirements: Hugging Face
authentication for the gated default model (`meta-llama/Llama-3.1-8B-Instruct`)
and the `datasets`/`matplotlib` packages for RULER data and plotting.
