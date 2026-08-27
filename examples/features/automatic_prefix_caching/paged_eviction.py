# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demonstrate Automatic Prefix Caching with PagedEviction.

Run:

    .venv/bin/python \
        examples/features/automatic_prefix_caching/paged_eviction.py

To use the legacy model runner:

    VLLM_USE_V2_MODEL_RUNNER=0 .venv/bin/python \
        examples/features/automatic_prefix_caching/paged_eviction.py
"""

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
BLOCK_SIZE = 16
CACHE_BUDGET_TOKENS = 32
PROMPT_TOKEN_IDS = list(range(1, 25))


def main() -> None:
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        paged_eviction_config={
            "cache_budget_tokens": CACHE_BUDGET_TOKENS,
        },
        async_scheduling=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        disable_cascade_attn=True,
    )
    prompt = {"prompt_token_ids": PROMPT_TOKEN_IDS}

    # Populate the prefix cache without exceeding the resident KV budget.
    short_output = llm.generate(
        prompt,
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )[0]

    # Reuse the cached prompt block, then decode far enough to evict two blocks.
    long_output = llm.generate(
        prompt,
        SamplingParams(temperature=0, max_tokens=40, ignore_eos=True),
        use_tqdm=False,
    )[0]

    short_token_ids = short_output.outputs[0].token_ids
    long_token_ids = long_output.outputs[0].token_ids
    expected_cached_tokens = BLOCK_SIZE
    logical_tokens = len(PROMPT_TOKEN_IDS) + len(long_token_ids)
    expected_evictions = (logical_tokens - CACHE_BUDGET_TOKENS) // BLOCK_SIZE

    assert short_output.num_cached_tokens == 0
    assert long_output.num_cached_tokens == expected_cached_tokens
    assert short_token_ids == long_token_ids[: len(short_token_ids)]

    print("Automatic Prefix Caching + PagedEviction demo")
    print(f"  Prefix-cache hit: {long_output.num_cached_tokens} tokens")
    print(f"  Resident KV budget: {CACHE_BUDGET_TOKENS} tokens")
    print(f"  Logical sequence length: {logical_tokens} tokens")
    print(f"  Block evictions during cached request: {expected_evictions}")
    print("  Deterministic output check: passed")

    del llm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
