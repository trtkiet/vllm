# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import PagedEvictionConfig
from vllm.v1.worker.paged_eviction import (
    PagedEvictionWorkerState,
    compute_paged_eviction_block_scores,
)

pytestmark = pytest.mark.cpu_test


def test_block_scores_match_pytorch_reference():
    epsilon = 1e-4
    kv_caches = [
        torch.arange(48, dtype=torch.float32).reshape(3, 2, 2, 2, 2) + 1,
        torch.arange(48, dtype=torch.float32).reshape(3, 2, 2, 2, 2) + 2,
    ]
    block_ids = [0, 2]

    actual = compute_paged_eviction_block_scores(kv_caches, block_ids, epsilon)

    expected = torch.zeros(len(block_ids))
    for kv_cache in kv_caches:
        selected = kv_cache[block_ids]
        key_norms = torch.linalg.vector_norm(selected[:, 0], dim=(-2, -1))
        value_norms = torch.linalg.vector_norm(selected[:, 1], dim=(-2, -1))
        expected += (value_norms / (key_norms + epsilon)).mean(dim=-1)
    expected /= len(kv_caches)

    assert list(actual) == block_ids
    torch.testing.assert_close(
        torch.tensor(list(actual.values())),
        expected,
    )


def test_worker_scores_only_when_new_block_becomes_full():
    state = PagedEvictionWorkerState(
        PagedEvictionConfig(cache_budget_tokens=4),
        block_size=2,
    )
    state.update_from_scheduler({"request": 4}, [])
    state.set_request_blocks("request", [0, 1, 2])

    kv_cache = torch.ones(3, 2, 2, 1, 2)
    assert (
        state.score_requests(
            [kv_cache],
            {"request": [0, 1, 2]},
            {"request": 1},
        )
        is None
    )

    scores = state.score_requests(
        [kv_cache],
        {"request": [0, 1, 2]},
        {"request": 1},
    )
    assert scores is not None
    assert list(scores["request"]) == [0, 1, 2]


def test_worker_uses_finalized_block_size():
    state = PagedEvictionWorkerState(
        PagedEvictionConfig(cache_budget_tokens=16),
        block_size=8,
    )

    state.set_block_size(4)

    assert state.block_size == 4
