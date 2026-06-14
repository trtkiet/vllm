# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config
class PagedEvictionConfig:
    """Configuration for decode-only block-wise KV cache eviction."""

    cache_budget_tokens: int = Field(gt=0)
    """Maximum number of full resident KV tokens per request."""

    enabled: bool = True
    """Whether PagedEviction is enabled."""

    score_epsilon: float = Field(default=1e-6, gt=0)
    """Numerical stabilizer for the key-norm denominator."""

    def compute_hash(self) -> str:
        factors = (
            self.enabled,
            self.cache_budget_tokens,
            self.score_epsilon,
        )
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
