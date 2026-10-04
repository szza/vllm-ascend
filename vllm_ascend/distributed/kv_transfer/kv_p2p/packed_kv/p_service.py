# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Prefill-side staging service.

Handles ``PREPARE_READ`` requests from the decode side: gathers
fragmented KV cache blocks into a contiguous staging slot and returns
the slot address via ``PACK_READY``.  On ``READ_ACK`` the slot is
released for reuse.

The service is transport-agnostic — callers invoke methods directly
(in-process tests) or via ZMQ (connector integration in Batch 4e).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    pack_into_staging,
    pack_into_staging_multi,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PackReadyMsg,
    PrepareReadMsg,
    ReadAckMsg,
    StagingErrorMsg,
)

logger = logging.getLogger(__name__)


@dataclass
class _ActiveSlot:
    """Tracks a slot that has been gathered and exposed for RDMA read."""

    slot_id: int
    transfer_id: str
    chunk_id: int
    ready: threading.Event = field(default_factory=threading.Event, repr=False)
    error: str | None = None


@dataclass
class PrefillStagingService:
    """P-side handler for staging gather/release lifecycle.

    Parameters
    ----------
    pool : StagingPool
        The P-side staging pool (HBM buffer registered with TE).
    kv_tensor : torch.Tensor
        Flat view of the P-side KV cache (all layers/components).
    kv_base_addr : int
        ``kv_tensor.data_ptr()`` — base address for offset computation.
    """

    pool: StagingPool
    kv_tensor: torch.Tensor
    kv_base_addr: int
    kv_regions: list[tuple[int, torch.Tensor]] = field(default_factory=list, init=True)
    _active_slots: dict[tuple[str, int], _ActiveSlot] = field(default_factory=dict, init=False)
    _active_slots_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _validate_prepare_read(self, msg: PrepareReadMsg) -> str | None:
        if msg.total_bytes <= 0:
            return "payload must be positive"
        if msg.total_bytes > self.pool.slot_capacity:
            return (
                f"payload exceeds staging slot capacity: payload={msg.total_bytes}, "
                f"capacity={self.pool.slot_capacity}"
            )

        regions = self.kv_regions
        if not regions and self.kv_tensor.numel() > 0:
            regions = [(self.kv_base_addr, self.kv_tensor.view(-1).view(torch.int8))]
        if not regions:
            return "no KV regions registered"

        max_end = 0
        for src_offset, packed_offset, nbytes in msg.gather_entries:
            if nbytes <= 0:
                return f"gather size must be positive: nbytes={nbytes}"
            if packed_offset < 0 or packed_offset + nbytes > msg.total_bytes:
                return (
                    f"packed range out of bounds: offset={packed_offset}, nbytes={nbytes}, "
                    f"payload={msg.total_bytes}"
                )
            max_end = max(max_end, packed_offset + nbytes)

            if not any(
                base <= src_offset
                and src_offset + nbytes <= base + tensor.numel() * tensor.element_size()
                for base, tensor in regions
            ):
                return f"source range is outside registered KV regions: addr=0x{src_offset:x}, nbytes={nbytes}"

        if max_end != msg.total_bytes:
            return f"payload does not match gather ranges: max_end={max_end}, payload={msg.total_bytes}"
        return None

    def handle_prepare_read(self, msg: PrepareReadMsg) -> PackReadyMsg | StagingErrorMsg:
        """Gather KV data into a staging slot and return its address.

        Returns ``PackReadyMsg`` on success, ``StagingErrorMsg`` if no
        slot is available.
        """
        key = (msg.transfer_id, msg.chunk_id)

        error = self._validate_prepare_read(msg)
        if error is not None:
            return StagingErrorMsg(
                transfer_id=msg.transfer_id,
                chunk_id=msg.chunk_id,
                code=3,
                reason=error,
            )

        with self._active_slots_lock:
            existing = self._active_slots.get(key)
            if existing is not None:
                slot = None
            else:
                slot = self.pool.acquire()
                if slot is None:
                    return StagingErrorMsg(
                        transfer_id=msg.transfer_id,
                        chunk_id=msg.chunk_id,
                        code=1,
                        reason="no staging slot available",
                    )
                existing = _ActiveSlot(
                    slot_id=slot.slot_id,
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                )
                self._active_slots[key] = existing

        if slot is None:
            existing.ready.wait()
            if existing.error is not None:
                return StagingErrorMsg(
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                    code=4,
                    reason=existing.error,
                )
            return PackReadyMsg(
                transfer_id=msg.transfer_id,
                chunk_id=msg.chunk_id,
                slot_addr=self.pool.slot_ptr(existing.slot_id),
                payload_bytes=msg.total_bytes,
            )

        gather_entries = [
            GatherEntry(src_offset=src, packed_offset=off, nbytes=n) for src, off, n in msg.gather_entries
        ]

        staging_view = self.pool.slot_view(slot.slot_id)
        t_gather = time.perf_counter()
        try:
            if self.kv_regions:
                pack_into_staging_multi(self.kv_regions, staging_view, gather_entries)
            else:
                pack_into_staging(self.kv_tensor, self.kv_base_addr, staging_view, gather_entries)
        except Exception:
            with self._active_slots_lock:
                active = self._active_slots.pop(key, None)
            if active is not None:
                active.error = "gather failed"
                active.ready.set()
                self.pool.release(active.slot_id)
            raise

        gather_ms = (time.perf_counter() - t_gather) * 1000

        existing.ready.set()

        slot_addr = self.pool.slot_ptr(slot.slot_id)
        logger.debug(
            "P gather done: transfer=%s chunk=%d slot=%d addr=0x%x bytes=%d gather_ms=%.2f",
            msg.transfer_id,
            msg.chunk_id,
            slot.slot_id,
            slot_addr,
            msg.total_bytes,
            gather_ms,
        )
        return PackReadyMsg(
            transfer_id=msg.transfer_id,
            chunk_id=msg.chunk_id,
            slot_addr=slot_addr,
            payload_bytes=msg.total_bytes,
            gather_ms=gather_ms,
        )

    def handle_read_ack(self, msg: ReadAckMsg) -> bool:
        """Release the staging slot after D confirms RDMA + scatter done.

        Returns True if the slot was found and released, False if the
        transfer_id/chunk_id was unknown (idempotent).
        """
        key = (msg.transfer_id, msg.chunk_id)
        with self._active_slots_lock:
            active = self._active_slots.get(key)
        if active is not None:
            active.ready.wait()
        with self._active_slots_lock:
            active = self._active_slots.pop(key, None)
        if active is None:
            return False
        self.pool.release(active.slot_id)
        logger.debug(
            "P slot released: transfer=%s chunk=%d slot=%d success=%s",
            msg.transfer_id,
            msg.chunk_id,
            active.slot_id,
            msg.success,
        )
        return True

    @property
    def active_slot_count(self) -> int:
        with self._active_slots_lock:
            return len(self._active_slots)


__all__ = [
    "PrefillStagingService",
]
