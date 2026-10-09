# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator


def test_allocates_payload_sized_page_extents_and_coalesces() -> None:
    allocator = StagingAllocator(capacity_bytes=1024, page_size=64, alignment=64, device="cpu")
    first = allocator.allocate(65)
    second = allocator.allocate(128)
    assert first is not None and second is not None
    assert first.allocated_bytes == 128
    assert second.offset == 128
    assert allocator.free_bytes == 768

    allocator.release(first)
    allocator.release(second)
    assert allocator.free_bytes == 1024
    assert allocator.free_extent_count == 1
    assert allocator.allocate(1024) is not None


def test_extent_view_is_bounded_to_requested_payload() -> None:
    allocator = StagingAllocator(capacity_bytes=256, page_size=64, alignment=64, device="cpu")
    lease = allocator.allocate(90)
    assert lease is not None
    view = allocator.view(lease)
    assert view.numel() == 90
    view.copy_(torch.arange(90, dtype=torch.int8))
    assert allocator.address(lease) == allocator.base_ptr + lease.offset


def test_arena_exhaustion_and_fragmentation_return_none() -> None:
    allocator = StagingAllocator(capacity_bytes=256, page_size=64, alignment=64, device="cpu")
    leases = [allocator.allocate(64) for _ in range(4)]
    assert all(lease is not None for lease in leases)
    assert allocator.allocate(1) is None

    allocator.release(leases[0])
    allocator.release(leases[2])
    assert allocator.free_bytes == 128
    assert allocator.allocate(128) is None


def test_stale_and_duplicate_release_are_rejected() -> None:
    allocator = StagingAllocator(capacity_bytes=128, page_size=64, alignment=64, device="cpu")
    lease = allocator.allocate(64)
    assert lease is not None
    allocator.release(lease)
    with pytest.raises(ValueError, match="stale"):
        allocator.release(lease)


def test_constructor_validates_page_size() -> None:
    with pytest.raises(ValueError, match="power of two"):
        StagingAllocator(capacity_bytes=1024, page_size=96, device="cpu")
