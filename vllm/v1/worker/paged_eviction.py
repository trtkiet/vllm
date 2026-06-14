# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

import torch

from vllm.config.paged_eviction import PagedEvictionConfig


def compute_paged_eviction_block_scores(
    kv_caches: Sequence[torch.Tensor | list[torch.Tensor]],
    block_ids: Sequence[int],
    epsilon: float,
) -> dict[int, float]:
    """Compute mean token ||V||2 / (||K||2 + epsilon) per physical block."""
    if not block_ids:
        return {}

    unique_caches: list[torch.Tensor] = []
    seen_ptrs: set[int] = set()
    for kv_cache in kv_caches:
        if not isinstance(kv_cache, torch.Tensor):
            raise ValueError("PagedEviction only supports attention KV tensors.")
        if kv_cache.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(kv_cache.data_ptr())
        if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
            raise ValueError(
                "PagedEviction requires KV tensors with shape "
                "[num_blocks, 2, block_size, num_kv_heads, head_size]."
            )
        unique_caches.append(kv_cache)

    if not unique_caches:
        raise ValueError("PagedEviction requires at least one attention KV tensor.")

    block_ids_tensor = torch.tensor(
        block_ids,
        dtype=torch.long,
        device=unique_caches[0].device,
    )
    scores = torch.zeros(
        len(block_ids),
        dtype=torch.float32,
        device=unique_caches[0].device,
    )
    for kv_cache in unique_caches:
        selected = kv_cache.index_select(0, block_ids_tensor).float()
        keys = selected[:, 0]
        values = selected[:, 1]
        key_norms = torch.linalg.vector_norm(keys, dim=(-2, -1))
        value_norms = torch.linalg.vector_norm(values, dim=(-2, -1))
        scores += (value_norms / (key_norms + epsilon)).mean(dim=-1)
    scores /= len(unique_caches)
    return dict(zip(block_ids, scores.cpu().tolist()))


def validate_paged_eviction_attention_groups(
    config: PagedEvictionConfig | None,
    attn_groups: Iterable[Iterable[object]],
) -> None:
    if config is None or not config.enabled:
        return
    for groups in attn_groups:
        for group in groups:
            backend = getattr(group, "backend", None)
            if backend is None or backend.get_name() != "FLASH_ATTN":
                raise ValueError(
                    "PagedEviction currently requires the FlashAttention backend."
                )


class PagedEvictionWorkerState:
    def __init__(self, config: PagedEvictionConfig | None, block_size: int):
        self.config = config
        self.block_size = block_size
        self.resident_tokens: dict[str, int] = {}
        self.block_scores: dict[int, float] = {}
        self.request_block_ids: dict[str, list[int]] = {}

    @property
    def enabled(self) -> bool:
        return self.config is not None and self.config.enabled

    def set_block_size(self, block_size: int) -> None:
        if self.enabled:
            assert self.config is not None
            if self.config.cache_budget_tokens % block_size != 0:
                raise ValueError(
                    "PagedEviction cache_budget_tokens must align with the "
                    "worker KV block size."
                )
        self.block_size = block_size

    def update_from_scheduler(
        self,
        resident_tokens: dict[str, int] | None,
        finished_req_ids: Iterable[str],
    ) -> None:
        if not self.enabled:
            return
        for req_id in finished_req_ids:
            self.resident_tokens.pop(req_id, None)
            self.request_block_ids.pop(req_id, None)
        if resident_tokens:
            self.resident_tokens.update(resident_tokens)

    def invalidate_blocks(self, block_ids: Iterable[int]) -> None:
        if not self.enabled:
            return
        for block_id in block_ids:
            self.block_scores.pop(block_id, None)

    def set_request_blocks(self, req_id: str, block_ids: Sequence[int]) -> None:
        if self.enabled:
            self.request_block_ids[req_id] = list(block_ids)

    def append_request_blocks(self, req_id: str, block_ids: Sequence[int]) -> None:
        if self.enabled:
            self.request_block_ids.setdefault(req_id, []).extend(block_ids)

    def score_requests(
        self,
        kv_caches: Sequence[torch.Tensor | list[torch.Tensor]],
        request_block_ids: dict[str, list[int]],
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, dict[int, float]] | None:
        if not self.enabled:
            return None

        assert self.config is not None
        requests_to_score: dict[str, list[int]] = {}
        missing_block_ids: list[int] = []
        missing_seen: set[int] = set()
        for req_id, num_tokens in num_scheduled_tokens.items():
            resident_tokens = self.resident_tokens[req_id] + num_tokens
            self.resident_tokens[req_id] = resident_tokens
            if (
                resident_tokens <= self.config.cache_budget_tokens
                or resident_tokens % self.block_size != 0
            ):
                continue

            block_ids = request_block_ids[req_id]
            requests_to_score[req_id] = block_ids
            for block_id in block_ids:
                if block_id not in self.block_scores and block_id not in missing_seen:
                    missing_seen.add(block_id)
                    missing_block_ids.append(block_id)

        self.block_scores.update(
            compute_paged_eviction_block_scores(
                kv_caches,
                missing_block_ids,
                self.config.score_epsilon,
            )
        )
        if not requests_to_score:
            return None
        return {
            req_id: {block_id: self.block_scores[block_id] for block_id in block_ids}
            for req_id, block_ids in requests_to_score.items()
        }
