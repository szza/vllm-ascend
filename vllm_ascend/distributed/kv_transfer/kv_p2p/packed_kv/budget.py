# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Staging buffer budget calculation and configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import POOL_ALIGNMENT

DEFAULT_NUM_SLOTS = 2
DEFAULT_SLOT_CAPACITY_MIB = 16
MIB = 1024 * 1024
DEFAULT_MIN_DIRECT_SIZE = 1 * MIB


@dataclass(frozen=True)
class StagingConfig:
    enabled: bool = False
    num_slots: int = DEFAULT_NUM_SLOTS
    slot_capacity: int = DEFAULT_SLOT_CAPACITY_MIB * MIB
    min_direct_size: int = DEFAULT_MIN_DIRECT_SIZE
    alignment: int = POOL_ALIGNMENT


def staging_config_from_env() -> StagingConfig:
    """Read staging configuration from environment variables."""
    enabled = os.getenv("VLLM_ASCEND_STAGING_ENABLED", "0") == "1"
    num_slots = int(os.getenv("VLLM_ASCEND_STAGING_NUM_SLOTS", str(DEFAULT_NUM_SLOTS)))
    slot_mib = int(os.getenv("VLLM_ASCEND_STAGING_SLOT_CAPACITY_MIB", str(DEFAULT_SLOT_CAPACITY_MIB)))
    min_direct_size = int(os.getenv("VLLM_ASCEND_STAGING_MIN_DIRECT_SIZE", str(DEFAULT_MIN_DIRECT_SIZE)))
    return StagingConfig(
        enabled=enabled,
        num_slots=num_slots,
        slot_capacity=slot_mib * MIB,
        min_direct_size=min_direct_size,
    )


def compute_staging_reservation(config: StagingConfig) -> int:
    """Compute total HBM bytes to reserve for staging buffers.

    Returns 0 if staging is disabled.  Otherwise returns the allocation
    size including alignment overhead::

        slot_stride = ceil(slot_capacity / alignment) * alignment
        pool_bytes  = num_slots * slot_stride
        reservation = pool_bytes + alignment - 1   (for initial alignment)
    """
    if not config.enabled:
        return 0
    alignment = config.alignment
    slot_stride = ((config.slot_capacity + alignment - 1) // alignment) * alignment
    pool_bytes = config.num_slots * slot_stride
    return pool_bytes + alignment - 1


__all__ = [
    "StagingConfig",
    "compute_staging_reservation",
    "staging_config_from_env",
]
