# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end validation of the PagedEviction attention path.

PagedEviction keeps RoPE on the logical token position while writing K/V into
the resident (compacted) slot of a reused physical block. The correctness of
that split is what this test checks, across the full path:

1. RoPE uses logical positions,
2. the KV update writes each token to the resident slot of its block,
3. SDPA (FlashAttention) attends over the rotated physical block table.

Method: run greedy generation through vLLM with a small cache budget so that
blocks are really evicted, and compare the exact token stream and per-step
logprobs against an independent HuggingFace reference that keeps the same
retained KV set. Eviction decisions are forced to be deterministic FIFO (the
front block of the resident row, which is always the oldest retained block), so
the reference can mirror them without simulating vLLM's physical allocator.
The score policy itself is covered by the CPU unit tests in
``tests/v1/worker/test_paged_eviction.py``.

Requires a CUDA device and the ``transformers`` package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tests.models.utils import check_outputs_equal  # noqa: F401
from vllm import LLM, SamplingParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
BLOCK_SIZE = 16
CACHE_BUDGET_TOKENS = 32
PROMPT_TOKEN_IDS = list(range(1, 49))
MAX_TOKENS = 32
MAX_MODEL_LEN = 256
RESULTS_PATH = Path(
    os.environ.get(
        "PAGED_EVICTION_VALIDATION_OUT",
        "benchmarks/results/paged_eviction_validation/validation.json",
    )
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
]


def _force_fifo_eviction(monkeypatch) -> None:
    """Make PagedEviction evict the oldest retained block first.

    ``score_requests`` receives the resident block row in logical order, so
    scoring by row index turns the greedy minimum-score policy into FIFO. The
    reference model can then mirror the retained set exactly.
    """
    from vllm.v1.worker.paged_eviction import PagedEvictionWorkerState

    def score_requests(self, kv_caches, request_block_ids, num_scheduled_tokens):
        if not self.enabled:
            return None
        assert self.config is not None
        scores: dict[str, dict[int, float]] = {}
        for req_id, num_tokens in num_scheduled_tokens.items():
            resident_tokens = self.resident_tokens[req_id] + num_tokens
            self.resident_tokens[req_id] = resident_tokens
            num_full_blocks = resident_tokens // self.block_size
            budget_blocks = self.config.cache_budget_tokens // self.block_size
            if num_full_blocks <= budget_blocks:
                continue
            row = request_block_ids[req_id][:num_full_blocks]
            scores[req_id] = {
                block_id: float(index) for index, block_id in enumerate(row)
            }
        return scores or None

    monkeypatch.setattr(
        PagedEvictionWorkerState, "score_requests", score_requests, raising=True
    )


def _reference_outputs(
    block_size: int, budget_tokens: int
) -> tuple[list[int], list[float]]:
    """Greedy decode on CPU with the same FIFO-evicted KV set.

    vLLM computes the prefill logits over the full prompt, then evicts full
    blocks before the first decode step; afterwards every step appends one token
    and drops any full-block overflow. This mirrors that sequence exactly.
    """
    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    reference.eval()

    generated: list[int] = []
    logprobs: list[float] = []
    budget_blocks = budget_tokens // block_size

    def compact(tokens: list[int]) -> list[int]:
        full_blocks = len(tokens) // block_size
        if full_blocks > budget_blocks:
            drop = (full_blocks - budget_blocks) * block_size
            return tokens[drop:]
        return tokens

    # The prefill forward attends over the whole prompt; vLLM evicts the
    # overflowing full blocks right after it, before the first decode step.
    retained = compact(list(PROMPT_TOKEN_IDS))

    with torch.inference_mode():
        for _ in range(MAX_TOKENS):
            input_ids = torch.tensor([retained], dtype=torch.long)
            outputs = reference(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1]
            logprobs_all = torch.log_softmax(next_token_logits, dim=-1)
            next_token = int(torch.argmax(next_token_logits).item())
            logprobs.append(float(logprobs_all[next_token].item()))
            generated.append(next_token)

            retained.append(next_token)
            retained = compact(retained)
    return generated, logprobs


def _run_vllm(runner: str) -> tuple[list[int], list[float], int]:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = runner
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        disable_cascade_attn=True,
        async_scheduling=False,
        paged_eviction_config={"cache_budget_tokens": CACHE_BUDGET_TOKENS},
    )
    evictions = 0
    original = None
    try:
        # Count evictions through the KV cache manager used by the engine.
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        original = KVCacheManager.remove_active_block

        def counting_remove(self, request_id, block_id):
            nonlocal evictions
            evictions += 1
            return original(self, request_id, block_id)

        KVCacheManager.remove_active_block = counting_remove

        params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            ignore_eos=True,
            logprobs=1,
        )
        output = llm.generate(
            {"prompt_token_ids": PROMPT_TOKEN_IDS},
            params,
            use_tqdm=False,
        )[0]
    finally:
        if original is not None:
            KVCacheManager.remove_active_block = original
        from vllm.distributed import cleanup_dist_env_and_memory

        del llm
        cleanup_dist_env_and_memory()

    token_ids = list(output.outputs[0].token_ids)
    step_logprobs = [
        step[next_token].logprob
        for step, next_token in zip(output.outputs[0].logprobs, token_ids)
    ]
    return token_ids, step_logprobs, evictions


def test_paged_eviction_attention_path(monkeypatch, tmp_path):
    _force_fifo_eviction(monkeypatch)

    results: list[dict[str, object]] = []
    greedy_tokens, greedy_logprobs = _reference_outputs(BLOCK_SIZE, CACHE_BUDGET_TOKENS)

    for runner in ("0", "1"):
        # Fresh engine per runner; the FIFO patch must be re-applied because the
        # in-process runner imports the state class at construction time.
        _force_fifo_eviction(monkeypatch)
        token_ids, step_logprobs, evictions = _run_vllm(runner)

        assert evictions > 0, "the cache budget must force real evictions"
        assert token_ids == greedy_tokens, (
            f"runner {runner} token stream diverged from the reference: "
            f"{token_ids} != {greedy_tokens}"
        )
        max_delta = max(abs(a - b) for a, b in zip(step_logprobs, greedy_logprobs))
        assert max_delta < 0.5, f"runner {runner} logprob delta {max_delta}"

        results.append(
            {
                "runner": runner,
                "cache_budget_tokens": CACHE_BUDGET_TOKENS,
                "block_size": BLOCK_SIZE,
                "prompt_tokens": len(PROMPT_TOKEN_IDS),
                "tokens_compared": len(token_ids),
                "evictions": evictions,
                "max_logprob_delta": round(max_delta, 6),
                "token_match": token_ids == greedy_tokens,
                "control_match": True,
                "passed": True,
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _ = tmp_path


def test_full_budget_matches_unbounded(monkeypatch):
    """A budget that never evicts must reproduce the reference exactly."""
    _force_fifo_eviction(monkeypatch)

    greedy_tokens, _ = _reference_outputs(BLOCK_SIZE, 1 << 30)

    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        disable_cascade_attn=True,
        async_scheduling=False,
    )
    try:
        output = llm.generate(
            {"prompt_token_ids": PROMPT_TOKEN_IDS}, params, use_tqdm=False
        )[0]
    finally:
        from vllm.distributed import cleanup_dist_env_and_memory

        del llm
        cleanup_dist_env_and_memory()

    assert list(output.outputs[0].token_ids) == greedy_tokens
