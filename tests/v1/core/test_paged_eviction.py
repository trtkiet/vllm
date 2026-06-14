# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import PagedEvictionConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import (
    PagedEvictionRequestState,
    Scheduler,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def make_manager(block_size: int = 4) -> KVCacheManager:
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=64,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=False,
    )


def make_request() -> Request:
    return Request(
        request_id="request",
        prompt_token_ids=list(range(8)),
        sampling_params=SamplingParams(max_tokens=32),
        pooling_params=None,
    )


def make_validation_config() -> VllmConfig:
    config = object.__new__(VllmConfig)
    config.paged_eviction_config = PagedEvictionConfig(cache_budget_tokens=16)
    config.device_config = SimpleNamespace(device_type="cuda")
    config.model_config = SimpleNamespace(
        runner_type="generate",
        is_encoder_decoder=False,
        is_multimodal_model=False,
        is_hybrid=False,
        use_mla=False,
        enforce_eager=True,
        disable_cascade_attn=True,
        dtype=torch.float16,
    )
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    config.scheduler_config = SimpleNamespace(
        async_scheduling=False,
        enable_chunked_prefill=False,
        scheduler_cls=None,
    )
    config.speculative_config = None
    config.diffusion_config = None
    config.cache_config = SimpleNamespace(
        enable_prefix_caching=False,
        cache_dtype="auto",
        sliding_window=None,
        block_size=16,
    )
    config.attention_config = SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN)
    config.kv_transfer_config = None
    config.ec_transfer_config = None
    return config


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("device_config.device_type", "cpu", "requires a CUDA device"),
        ("model_config.runner_type", "pooling", "generative models"),
        ("model_config.is_encoder_decoder", True, "decoder-only"),
        ("model_config.is_multimodal_model", True, "multimodal"),
        ("model_config.is_hybrid", True, "hybrid attention"),
        ("model_config.use_mla", True, "MLA"),
        ("model_config.enforce_eager", False, "eager execution"),
        ("model_config.disable_cascade_attn", False, "cascade attention"),
        ("parallel_config.tensor_parallel_size", 2, "TP=PP=DP=PCP=DCP=1"),
        ("scheduler_config.async_scheduling", True, "synchronous scheduling"),
        ("scheduler_config.enable_chunked_prefill", True, "chunked prefill"),
        ("scheduler_config.scheduler_cls", "custom", "default scheduler"),
        ("speculative_config", object(), "speculative decoding"),
        ("diffusion_config", object(), "diffusion models"),
        ("cache_config.enable_prefix_caching", True, "prefix caching"),
        ("cache_config.cache_dtype", "fp8", "float16 or bfloat16"),
        ("model_config.dtype", torch.float32, "float16 or bfloat16"),
        ("cache_config.sliding_window", 128, "full attention"),
        ("paged_eviction_config.cache_budget_tokens", 15, "block-aligned"),
        ("attention_config.backend", AttentionBackendEnum.FLASHINFER, "FlashAttention"),
        (
            "kv_transfer_config",
            SimpleNamespace(kv_connector="connector"),
            "KV connectors",
        ),
        ("ec_transfer_config", object(), "EC connectors"),
    ],
)
def test_unsupported_configurations_fail_validation(path, value, message):
    config = make_validation_config()
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)

    with pytest.raises(ValueError, match=message):
        config._validate_paged_eviction_config()


def test_supported_configuration_passes_validation():
    make_validation_config()._validate_paged_eviction_config()


def test_scheduler_rejects_multiple_kv_cache_groups():
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )
    scheduler = object.__new__(Scheduler)
    scheduler.paged_eviction_config = PagedEvictionConfig(cache_budget_tokens=4)
    scheduler.kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec),
            KVCacheGroupSpec(layer_names=["layer.1"], kv_cache_spec=spec),
        ],
    )

    with pytest.raises(ValueError, match="exactly one full-attention"):
        scheduler._validate_paged_eviction_kv_cache()


def test_resident_allocation_does_not_recreate_evicted_logical_capacity():
    manager = make_manager()
    request = make_request()

    initial = manager.allocate_slots(
        request,
        num_new_tokens=8,
        num_required_tokens=8,
    )
    assert initial is not None
    initial_ids = manager.get_block_ids(request.request_id)[0]
    assert len(initial_ids) == 2

    evicted_block_id = initial_ids[0]
    manager.remove_active_block(request.request_id, evicted_block_id)
    request.num_computed_tokens = 8

    appended = manager.allocate_slots(
        request,
        num_new_tokens=1,
        num_required_tokens=5,
    )
    assert appended is not None
    assert appended.get_block_ids()[0] == [evicted_block_id]

    block_ids = manager.get_block_ids(request.request_id)[0]
    assert block_ids == [initial_ids[1], evicted_block_id]
    assert len(block_ids) == len(set(block_ids))


def test_remove_active_block_frees_exactly_one_block():
    manager = make_manager()
    request = make_request()
    assert manager.allocate_slots(request, 8, num_required_tokens=8) is not None

    block_ids = manager.get_block_ids(request.request_id)[0]
    num_free_before = manager.block_pool.get_num_free_blocks()
    manager.remove_active_block(request.request_id, block_ids[0])

    assert manager.block_pool.get_num_free_blocks() == num_free_before + 1
    assert manager.get_block_ids(request.request_id)[0] == block_ids[1:]


def test_remove_active_block_rejects_shared_block():
    manager = make_manager()
    request = make_request()
    assert manager.allocate_slots(request, 8, num_required_tokens=8) is not None

    block = manager.get_blocks(request.request_id).blocks[0][0]
    block.ref_cnt += 1
    with pytest.raises(ValueError, match="shared KV cache block"):
        manager.remove_active_block(request.request_id, block.block_id)


def test_scheduler_repeated_eviction_preserves_logical_progress_and_order():
    manager = make_manager(block_size=2)
    request = make_request()
    assert manager.allocate_slots(request, 6, num_required_tokens=6) is not None
    request.num_computed_tokens = 10

    scheduler = object.__new__(Scheduler)
    scheduler.block_size = 2
    scheduler.kv_cache_manager = manager
    scheduler.paged_eviction_config = PagedEvictionConfig(cache_budget_tokens=4)
    scheduler.paged_eviction_states = {
        request.request_id: PagedEvictionRequestState(resident_tokens=4)
    }
    scheduler.requests = {request.request_id: request}

    def evict(scores: dict[int, float]) -> None:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.num_scheduled_tokens = {request.request_id: 2}
        model_runner_output = ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            paged_eviction_scores={request.request_id: scores},
        )
        scheduler._update_paged_eviction(
            scheduler_output,
            model_runner_output,
        )

    first_row = manager.get_block_ids(request.request_id)[0]
    logical_tokens = request.num_computed_tokens
    evict({first_row[0]: 2.0, first_row[1]: 1.0, first_row[2]: 3.0})
    assert manager.get_block_ids(request.request_id)[0] == [
        first_row[0],
        first_row[2],
    ]
    assert scheduler.paged_eviction_states[request.request_id].block_table_dirty
    assert request.num_computed_tokens == logical_tokens

    request.num_computed_tokens += 2
    assert manager.allocate_slots(request, 2, num_required_tokens=6) is not None
    second_row = manager.get_block_ids(request.request_id)[0]
    assert second_row == [first_row[0], first_row[2], first_row[1]]

    logical_tokens = request.num_computed_tokens
    evict({second_row[0]: 0.0, second_row[1]: 2.0, second_row[2]: 1.0})
    final_row = manager.get_block_ids(request.request_id)[0]
    assert final_row == second_row[1:]
    assert len(final_row) == len(set(final_row))
    assert request.num_computed_tokens == logical_tokens


def test_prompt_must_fit_block_aligned_budget():
    scheduler = object.__new__(Scheduler)
    scheduler.block_size = 4
    scheduler.paged_eviction_config = PagedEvictionConfig(cache_budget_tokens=4)
    scheduler.paged_eviction_states = {}
    scheduler.requests = {}

    with pytest.raises(ValueError, match="prompt requires 8 block-aligned"):
        scheduler.add_request(make_request())


def test_disabled_state_is_safe_for_partially_constructed_scheduler():
    scheduler = object.__new__(Scheduler)

    assert not scheduler.paged_eviction_enabled


def test_preemption_rejects_logical_sequence_larger_than_budget():
    scheduler = object.__new__(Scheduler)
    scheduler.block_size = 4
    scheduler.paged_eviction_config = PagedEvictionConfig(cache_budget_tokens=8)
    scheduler.paged_eviction_states = {
        "request": PagedEvictionRequestState(resident_tokens=8)
    }
    scheduler.kv_cache_manager = Mock()
    scheduler.encoder_cache_manager = Mock()
    scheduler._inflight_prefills = set()

    request = make_request()
    request.append_output_token_ids(1)
    request.status = RequestStatus.RUNNING

    with pytest.raises(RuntimeError, match="Prefill compression is not supported"):
        scheduler._preempt_request(request, 0.0)

    scheduler.kv_cache_manager.free.assert_not_called()
