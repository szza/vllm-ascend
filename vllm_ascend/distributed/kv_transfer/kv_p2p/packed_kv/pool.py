# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""HBM staging pool for gather/scatter of fragmented KV blocks."""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Any

import torch

POOL_ALIGNMENT = 2 * 1024 * 1024  # 2 MiB


class SlotState(enum.IntEnum):
    FREE = 0
    ACQUIRED = 1


@dataclass
class StagingSlot:
    slot_id: int
    offset: int
    capacity: int
    generation: int = 0
    state: SlotState = SlotState.FREE


class StagingPool:
    """Fixed-capacity staging buffer pool for RDMA gather/scatter.

    Allocates a contiguous HBM region via ``torch.empty``, aligns the base
    address, and carves it into equal-sized slots.  Each slot can be
    independently acquired and released.  The pool registers itself with a
    Mooncake ``TransferEngine`` independently of the KV-cache registration
    path (``GlobalTE.register_buffer`` is one-shot and cannot be reused).
    """

    def __init__(
        self,
        num_slots: int = 2,
        slot_capacity: int = 16 * 1024 * 1024,
        alignment: int = POOL_ALIGNMENT,
        device: str = "npu",
    ) -> None:
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if slot_capacity <= 0:
            raise ValueError(f"slot_capacity must be positive, got {slot_capacity}")
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be a positive power of two, got {alignment}")

        self._alignment = alignment
        self._slot_stride = ((slot_capacity + alignment - 1) // alignment) * alignment
        self._pool_bytes = num_slots * self._slot_stride
        self._slot_capacity = slot_capacity

        raw_bytes = self._pool_bytes + alignment - 1
        self._raw_tensor = torch.empty(raw_bytes, dtype=torch.int8, device=device)

        raw_ptr = self._raw_tensor.data_ptr()
        aligned_ptr = ((raw_ptr + alignment - 1) // alignment) * alignment
        self._align_offset = aligned_ptr - raw_ptr
        self._pool_view = self._raw_tensor.narrow(0, self._align_offset, self._pool_bytes)

        self._slots = [
            StagingSlot(
                slot_id=i,
                offset=i * self._slot_stride,
                capacity=slot_capacity,
            )
            for i in range(num_slots)
        ]
        # Slot views are immutable byte ranges within the pool.  Cache them
        # once so the hot gather/scatter path does not recreate an NPU view
        # with ``narrow`` for every chunk.
        self._slot_views = [
            self._pool_view.narrow(0, slot.offset, slot.capacity) for slot in self._slots
        ]
        self._registered = False
        self._engine: Any = None
        # Connector transfers can call acquire/release from multiple worker
        # threads.  Keep the critical section limited to slot state changes;
        # copy and RDMA operations happen after the lock is released.
        self._slot_lock = threading.Lock()

    # -- properties --------------------------------------------------------

    @property
    def num_slots(self) -> int:
        return len(self._slots)

    @property
    def slot_stride(self) -> int:
        return self._slot_stride

    @property
    def total_bytes(self) -> int:
        return self._pool_bytes

    @property
    def slot_capacity(self) -> int:
        """Usable payload bytes in one slot."""
        return self._slot_capacity

    @property
    def base_ptr(self) -> int:
        return self._pool_view.data_ptr()

    # -- slot management ---------------------------------------------------

    def acquire(self) -> StagingSlot | None:
        with self._slot_lock:
            for slot in self._slots:
                if slot.state == SlotState.FREE:
                    slot.state = SlotState.ACQUIRED
                    return slot
            return None

    def release(self, slot_id: int) -> None:
        with self._slot_lock:
            slot = self._slots[slot_id]
            if slot.state == SlotState.FREE:
                raise ValueError(f"Slot {slot_id} is already FREE")
            slot.state = SlotState.FREE
            slot.generation += 1

    def slot_view(self, slot_id: int) -> torch.Tensor:
        return self._slot_views[slot_id]

    def slot_ptr(self, slot_id: int) -> int:
        slot = self._slots[slot_id]
        return self.base_ptr + slot.offset

    def get_slot(self, slot_id: int) -> StagingSlot:
        return self._slots[slot_id]

    # -- registration ------------------------------------------------------

    def register(self, engine: Any) -> None:
        ret = engine.register_memory(self.base_ptr, self._pool_bytes)
        if ret != 0:
            raise RuntimeError(f"Staging pool registration failed with ret={ret}")
        self._registered = True
        self._engine = engine

    def unregister(self, engine: Any) -> None:
        if not self._registered:
            return
        engine.unregister_memory(self.base_ptr)
        self._registered = False
        self._engine = None

    def close(self) -> None:
        if self._registered and self._engine is not None:
            self.unregister(self._engine)


__all__ = [
    "POOL_ALIGNMENT",
    "SlotState",
    "StagingPool",
    "StagingSlot",
]
