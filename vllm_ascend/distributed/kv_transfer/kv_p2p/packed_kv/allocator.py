# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Page-based HBM allocator for Mooncake staging transfers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_PAGE_SIZE = 256 * 1024
ARENA_ALIGNMENT = 2 * 1024 * 1024


@dataclass(frozen=True)
class StagingLease:
    lease_id: int
    offset: int
    payload_bytes: int
    allocated_bytes: int
    page_count: int


class StagingAllocator:
    """Allocate page-aligned contiguous extents from a registered HBM arena."""

    def __init__(
        self,
        capacity_bytes: int,
        page_size: int = DEFAULT_PAGE_SIZE,
        alignment: int = ARENA_ALIGNMENT,
        device: str = "npu",
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be positive, got {capacity_bytes}")
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError(f"page_size must be a positive power of two, got {page_size}")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"alignment must be a positive power of two, got {alignment}")

        self._page_size = page_size
        self._alignment = alignment
        self._page_count = (capacity_bytes + page_size - 1) // page_size
        self._capacity_bytes = self._page_count * page_size
        raw_bytes = self._capacity_bytes + alignment - 1
        self._raw_tensor = torch.empty(raw_bytes, dtype=torch.int8, device=device)
        raw_ptr = self._raw_tensor.data_ptr()
        aligned_ptr = ((raw_ptr + alignment - 1) // alignment) * alignment
        self._align_offset = aligned_ptr - raw_ptr
        self._arena = self._raw_tensor.narrow(0, self._align_offset, self._capacity_bytes)

        self._free_extents: list[tuple[int, int]] = [(0, self._page_count)]
        self._active: dict[int, StagingLease] = {}
        self._next_lease_id = 1
        self._lock = threading.Lock()
        self._registered = False
        self._engine: Any = None

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def total_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def base_ptr(self) -> int:
        return self._arena.data_ptr()

    @property
    def device(self) -> torch.device:
        return self._arena.device

    @property
    def free_bytes(self) -> int:
        with self._lock:
            return sum(pages for _, pages in self._free_extents) * self._page_size

    @property
    def active_bytes(self) -> int:
        with self._lock:
            return sum(lease.allocated_bytes for lease in self._active.values())

    @property
    def active_leases(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def free_extent_count(self) -> int:
        with self._lock:
            return len(self._free_extents)

    @property
    def arena_view(self) -> torch.Tensor:
        return self._arena

    def allocate(self, payload_bytes: int) -> StagingLease | None:
        if payload_bytes <= 0:
            raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")
        page_count = (payload_bytes + self._page_size - 1) // self._page_size
        with self._lock:
            for index, (start_page, extent_pages) in enumerate(self._free_extents):
                if extent_pages < page_count:
                    continue
                if extent_pages == page_count:
                    self._free_extents.pop(index)
                else:
                    self._free_extents[index] = (start_page + page_count, extent_pages - page_count)
                lease_id = self._next_lease_id
                self._next_lease_id += 1
                lease = StagingLease(
                    lease_id=lease_id,
                    offset=start_page * self._page_size,
                    payload_bytes=payload_bytes,
                    allocated_bytes=page_count * self._page_size,
                    page_count=page_count,
                )
                self._active[lease_id] = lease
                return lease
        return None

    def release(self, lease: StagingLease) -> None:
        with self._lock:
            active = self._active.get(lease.lease_id)
            if active != lease:
                raise ValueError(f"Unknown or stale staging lease {lease.lease_id}")
            del self._active[lease.lease_id]
            start_page = lease.offset // self._page_size
            self._free_extents.append((start_page, lease.page_count))
            self._free_extents.sort()
            merged: list[tuple[int, int]] = []
            for extent_start, extent_pages in self._free_extents:
                if merged and merged[-1][0] + merged[-1][1] == extent_start:
                    prev_start, prev_pages = merged[-1]
                    merged[-1] = (prev_start, prev_pages + extent_pages)
                else:
                    merged.append((extent_start, extent_pages))
            self._free_extents = merged

    def address(self, lease: StagingLease) -> int:
        self._validate_active(lease)
        return self.base_ptr + lease.offset

    def view(self, lease: StagingLease) -> torch.Tensor:
        self._validate_active(lease)
        return self._arena.narrow(0, lease.offset, lease.payload_bytes)

    def _validate_active(self, lease: StagingLease) -> None:
        with self._lock:
            if self._active.get(lease.lease_id) != lease:
                raise ValueError(f"Unknown or stale staging lease {lease.lease_id}")

    def register(self, engine: Any) -> None:
        ret = engine.register_memory(self.base_ptr, self._capacity_bytes)
        if ret != 0:
            raise RuntimeError(f"Staging arena registration failed with ret={ret}")
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


__all__ = ["ARENA_ALIGNMENT", "DEFAULT_PAGE_SIZE", "StagingAllocator", "StagingLease"]
