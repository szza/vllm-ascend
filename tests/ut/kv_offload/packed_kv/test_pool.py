# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import (
    SlotState,
    StagingPool,
)

# =====================================================================
# Slot management
# =====================================================================


class TestSlotManagement:
    def test_pool_slot_count(self) -> None:
        pool = StagingPool(num_slots=3, slot_capacity=256, alignment=64, device="cpu")
        assert pool.num_slots == 3

    def test_acquire_returns_free_slot(self) -> None:
        pool = StagingPool(num_slots=2, slot_capacity=256, alignment=64, device="cpu")
        slot = pool.acquire()
        assert slot is not None
        assert slot.state == SlotState.ACQUIRED
        assert slot.slot_id == 0

    def test_acquire_exhausted_returns_none(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        pool.acquire()
        assert pool.acquire() is None

    def test_release_increments_generation(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        slot = pool.acquire()
        assert slot is not None
        assert slot.generation == 0
        pool.release(slot.slot_id)
        assert pool.get_slot(0).generation == 1

    def test_release_makes_slot_reacquirable(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        slot = pool.acquire()
        assert slot is not None
        pool.release(slot.slot_id)
        slot2 = pool.acquire()
        assert slot2 is not None
        assert slot2.slot_id == 0
        assert slot2.generation == 1

    def test_double_release_raises(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        slot = pool.acquire()
        assert slot is not None
        pool.release(slot.slot_id)
        with pytest.raises(ValueError, match="already FREE"):
            pool.release(slot.slot_id)


# =====================================================================
# Alignment
# =====================================================================


class TestAlignment:
    def test_base_ptr_aligned(self) -> None:
        alignment = 64
        pool = StagingPool(num_slots=2, slot_capacity=256, alignment=alignment, device="cpu")
        assert pool.base_ptr % alignment == 0

    def test_base_ptr_aligned_large(self) -> None:
        alignment = 4096
        pool = StagingPool(num_slots=2, slot_capacity=8192, alignment=alignment, device="cpu")
        assert pool.base_ptr % alignment == 0

    def test_slot_stride_aligned(self) -> None:
        alignment = 64
        pool = StagingPool(num_slots=2, slot_capacity=100, alignment=alignment, device="cpu")
        assert pool.slot_stride % alignment == 0
        assert pool.slot_stride >= 100

    def test_slots_non_overlapping(self) -> None:
        pool = StagingPool(num_slots=3, slot_capacity=256, alignment=64, device="cpu")
        ranges = []
        for i in range(pool.num_slots):
            slot = pool.get_slot(i)
            ranges.append((slot.offset, slot.offset + slot.capacity))
        ranges.sort()
        for i in range(1, len(ranges)):
            assert ranges[i][0] >= ranges[i - 1][1]

    def test_pool_fits_in_allocation(self) -> None:
        pool = StagingPool(num_slots=3, slot_capacity=256, alignment=64, device="cpu")
        for i in range(pool.num_slots):
            slot = pool.get_slot(i)
            assert slot.offset + slot.capacity <= pool.total_bytes


# =====================================================================
# Registration (mock engine)
# =====================================================================


class TestRegistration:
    def test_register_calls_engine(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        engine = MagicMock()
        engine.register_memory.return_value = 0
        pool.register(engine)
        engine.register_memory.assert_called_once_with(pool.base_ptr, pool.total_bytes)

    def test_unregister_calls_engine(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        engine = MagicMock()
        engine.register_memory.return_value = 0
        pool.register(engine)
        pool.unregister(engine)
        engine.unregister_memory.assert_called_once_with(pool.base_ptr)

    def test_register_failure_raises(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        engine = MagicMock()
        engine.register_memory.return_value = -1
        with pytest.raises(RuntimeError, match="registration failed"):
            pool.register(engine)

    def test_close_unregisters(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        engine = MagicMock()
        engine.register_memory.return_value = 0
        pool.register(engine)
        pool.close()
        engine.unregister_memory.assert_called_once()

    def test_close_idempotent(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        engine = MagicMock()
        engine.register_memory.return_value = 0
        pool.register(engine)
        pool.close()
        pool.close()
        assert engine.unregister_memory.call_count == 1

    def test_close_without_register(self) -> None:
        pool = StagingPool(num_slots=1, slot_capacity=256, alignment=64, device="cpu")
        pool.close()


# =====================================================================
# Slot views (CPU tensor)
# =====================================================================


class TestSlotView:
    def test_slot_view_shape(self) -> None:
        cap = 256
        pool = StagingPool(num_slots=2, slot_capacity=cap, alignment=64, device="cpu")
        view = pool.slot_view(0)
        assert view.shape == (cap,)
        assert view.dtype == torch.int8

    def test_slot_view_writable(self) -> None:
        pool = StagingPool(num_slots=2, slot_capacity=256, alignment=64, device="cpu")
        view = pool.slot_view(0)
        view.fill_(42)
        assert view[0].item() == 42

    def test_slot_views_independent(self) -> None:
        pool = StagingPool(num_slots=2, slot_capacity=256, alignment=64, device="cpu")
        v0 = pool.slot_view(0)
        v1 = pool.slot_view(1)
        v0.fill_(1)
        v1.fill_(2)
        assert v0[0].item() == 1
        assert v1[0].item() == 2

    def test_slot_ptr(self) -> None:
        pool = StagingPool(num_slots=2, slot_capacity=256, alignment=64, device="cpu")
        assert pool.slot_ptr(0) == pool.base_ptr
        assert pool.slot_ptr(1) == pool.base_ptr + pool.slot_stride


# =====================================================================
# Constructor validation
# =====================================================================


class TestConstructorValidation:
    def test_zero_slots_raises(self) -> None:
        with pytest.raises(ValueError, match="num_slots"):
            StagingPool(num_slots=0, slot_capacity=256, alignment=64, device="cpu")

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="slot_capacity"):
            StagingPool(num_slots=1, slot_capacity=0, alignment=64, device="cpu")

    def test_non_power_of_two_alignment_raises(self) -> None:
        with pytest.raises(ValueError, match="alignment"):
            StagingPool(num_slots=1, slot_capacity=256, alignment=48, device="cpu")
