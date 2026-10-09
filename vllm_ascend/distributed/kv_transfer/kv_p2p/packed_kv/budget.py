# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Staging buffer budget calculation and configuration."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import ARENA_ALIGNMENT, DEFAULT_PAGE_SIZE

MIB = 1024 * 1024
KIB = 1024
DEFAULT_ARENA_CAPACITY_MIB = 32
DEFAULT_CHUNK_CAPACITY_MIB = 16
DEFAULT_MAX_CONCURRENT_CHUNKS = 2
DEFAULT_MIN_DIRECT_SIZE = 1 * MIB
DEFAULT_V1_MAX_PROMPT_TOKENS = 8192


@dataclass(frozen=True)
class StagingConfig:
    enabled: bool = False
    arena_capacity: int = DEFAULT_ARENA_CAPACITY_MIB * MIB
    page_size: int = DEFAULT_PAGE_SIZE
    chunk_capacity: int = DEFAULT_CHUNK_CAPACITY_MIB * MIB
    max_concurrent_chunks: int = DEFAULT_MAX_CONCURRENT_CHUNKS
    min_direct_size: int = DEFAULT_MIN_DIRECT_SIZE
    v1_max_prompt_tokens: int = DEFAULT_V1_MAX_PROMPT_TOKENS
    alignment: int = ARENA_ALIGNMENT

    def __post_init__(self) -> None:
        if self.chunk_capacity < self.min_direct_size:
            raise ValueError(
                "staging chunk capacity must be greater than or equal to "
                "the direct-transfer threshold: "
                f"chunk_capacity={self.chunk_capacity}, min_direct_size={self.min_direct_size}"
            )
        if self.v1_max_prompt_tokens < 0:
            raise ValueError(
                "VLLM_ASCEND_STAGING_V1_MAX_PROMPT_TOKENS must be non-negative, "
                f"got {self.v1_max_prompt_tokens}"
            )


def staging_config_from_env() -> StagingConfig:
    """Read staging configuration from environment variables."""
    enabled = envs.VLLM_ASCEND_STAGING_ENABLED
    capacity_mib = envs.VLLM_ASCEND_STAGING_CAPACITY_MIB
    page_size_kib = envs.VLLM_ASCEND_STAGING_PAGE_SIZE_KIB
    chunk_capacity_mib = envs.VLLM_ASCEND_STAGING_CHUNK_CAPACITY_MIB
    max_concurrent_chunks = envs.VLLM_ASCEND_STAGING_MAX_CONCURRENT_CHUNKS
    min_direct_size = envs.VLLM_ASCEND_STAGING_MIN_DIRECT_SIZE
    v1_max_prompt_tokens = envs.VLLM_ASCEND_STAGING_V1_MAX_PROMPT_TOKENS
    if page_size_kib <= 0 or (page_size_kib & (page_size_kib - 1)) != 0:
        raise ValueError(f"VLLM_ASCEND_STAGING_PAGE_SIZE_KIB must be a positive power of two, got {page_size_kib}")
    if capacity_mib <= 0:
        raise ValueError(f"VLLM_ASCEND_STAGING_CAPACITY_MIB must be positive, got {capacity_mib}")
    if chunk_capacity_mib <= 0:
        raise ValueError(f"VLLM_ASCEND_STAGING_CHUNK_CAPACITY_MIB must be positive, got {chunk_capacity_mib}")
    if max_concurrent_chunks <= 0:
        raise ValueError(f"VLLM_ASCEND_STAGING_MAX_CONCURRENT_CHUNKS must be positive, got {max_concurrent_chunks}")
    arena_capacity = capacity_mib * MIB
    chunk_capacity = chunk_capacity_mib * MIB
    if chunk_capacity > arena_capacity:
        raise ValueError("staging chunk capacity cannot exceed the per-worker staging arena capacity")
    if chunk_capacity < min_direct_size:
        raise ValueError(
            "staging chunk capacity must be greater than or equal to the "
            "direct-transfer threshold (VLLM_ASCEND_STAGING_CHUNK_CAPACITY_MIB "
            "and VLLM_ASCEND_STAGING_MIN_DIRECT_SIZE): "
            f"chunk_capacity={chunk_capacity}, min_direct_size={min_direct_size}"
        )
    if v1_max_prompt_tokens < 0:
        raise ValueError(
            "VLLM_ASCEND_STAGING_V1_MAX_PROMPT_TOKENS must be non-negative, "
            f"got {v1_max_prompt_tokens}"
        )
    return StagingConfig(
        enabled=enabled,
        arena_capacity=arena_capacity,
        page_size=page_size_kib * KIB,
        chunk_capacity=chunk_capacity,
        max_concurrent_chunks=max_concurrent_chunks,
        min_direct_size=min_direct_size,
        v1_max_prompt_tokens=v1_max_prompt_tokens,
    )


def compute_staging_reservation(config: StagingConfig) -> int:
    """Compute total HBM bytes to reserve for staging buffers.

    Returns 0 if staging is disabled.  Otherwise returns the allocation
    size including alignment overhead::

        arena_bytes = ceil(arena_capacity / page_size) * page_size
        reservation = arena_bytes + alignment - 1   (for initial alignment)
    """
    if not config.enabled:
        return 0
    arena_bytes = ((config.arena_capacity + config.page_size - 1) // config.page_size) * config.page_size
    return arena_bytes + config.alignment - 1


__all__ = [
    "StagingConfig",
    "compute_staging_reservation",
    "staging_config_from_env",
]
