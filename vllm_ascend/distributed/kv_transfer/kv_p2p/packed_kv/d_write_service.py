# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Decode-side staging service for WRITE mode.

Handles ``PREPARE_WRITE`` requests from the prefill side: allocates a
staging slot and returns its address via ``WRITE_READY``.  On
``WRITE_DONE`` the data is scattered from the staging slot into the
KV cache and the slot is released.

Mirror of ``PrefillStagingService`` (which handles READ mode on P side).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    unpack_from_staging,
    unpack_from_staging_multi,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    ScatterEntry,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PrepareWriteMsg,
    StagingErrorMsg,
    WriteDoneMsg,
    WriteReadyMsg,
)

logger = logging.getLogger(__name__)


@dataclass
class _PendingSlot:
    slot_id: int
    transfer_id: str
    chunk_id: int
    scatter_entries: list[ScatterEntry]


@dataclass
class DecodeWriteService:
    """D-side handler for WRITE-mode staging lifecycle.

    Parameters
    ----------
    pool : StagingPool
        D-side staging pool (HBM buffer registered with TE).
    kv_tensor : torch.Tensor
        Flat view of the D-side KV cache.
    kv_base_addr : int
        ``kv_tensor.data_ptr()`` — base address for offset computation.
    kv_regions : list[tuple[int, torch.Tensor]]
        Multi-region KV cache: sorted list of (base_addr, flat_tensor).
    """

    pool: StagingPool
    kv_tensor: torch.Tensor
    kv_base_addr: int
    kv_regions: list[tuple[int, torch.Tensor]] = field(default_factory=list, init=True)
    _pending: dict[tuple[str, int], _PendingSlot] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def handle_prepare_write(self, msg: PrepareWriteMsg) -> WriteReadyMsg | StagingErrorMsg:
        """Allocate a staging slot for an incoming RDMA WRITE.

        Returns ``WriteReadyMsg`` on success, ``StagingErrorMsg`` if no
        slot is available or validation fails.
        """
        key = (msg.transfer_id, msg.chunk_id)

        error = self._validate(msg)
        if error is not None:
            return StagingErrorMsg(
                transfer_id=msg.transfer_id,
                chunk_id=msg.chunk_id,
                code=3,
                reason=error,
            )

        with self._lock:
            existing = self._pending.get(key)
            if existing is not None:
                return WriteReadyMsg(
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                    slot_addr=self.pool.slot_ptr(existing.slot_id),
                    payload_bytes=msg.total_bytes,
                )

            slot = self.pool.acquire()
            if slot is None:
                return StagingErrorMsg(
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                    code=1,
                    reason="no staging slot available",
                )

            scatter_entries = [
                ScatterEntry(dst_offset=dst, packed_offset=off, nbytes=n)
                for dst, off, n in msg.scatter_entries
            ]
            self._pending[key] = _PendingSlot(
                slot_id=slot.slot_id,
                transfer_id=msg.transfer_id,
                chunk_id=msg.chunk_id,
                scatter_entries=scatter_entries,
            )

        slot_addr = self.pool.slot_ptr(slot.slot_id)
        logger.debug(
            "D slot allocated: transfer=%s chunk=%d slot=%d addr=0x%x bytes=%d",
            msg.transfer_id,
            msg.chunk_id,
            slot.slot_id,
            slot_addr,
            msg.total_bytes,
        )
        return WriteReadyMsg(
            transfer_id=msg.transfer_id,
            chunk_id=msg.chunk_id,
            slot_addr=slot_addr,
            payload_bytes=msg.total_bytes,
        )

    def handle_write_done(self, msg: WriteDoneMsg) -> bool:
        """Scatter data from staging slot to KV cache and release.

        Returns True if the slot was found and processed, False if
        the transfer_id/chunk_id was unknown (idempotent).
        """
        key = (msg.transfer_id, msg.chunk_id)
        with self._lock:
            pending = self._pending.pop(key, None)
        if pending is None:
            return False

        if msg.success:
            staging_view = self.pool.slot_view(pending.slot_id)
            try:
                if self.kv_regions:
                    unpack_from_staging_multi(
                        staging_view,
                        self.kv_regions,
                        pending.scatter_entries,
                    )
                else:
                    unpack_from_staging(
                        staging_view,
                        self.kv_tensor,
                        self.kv_base_addr,
                        pending.scatter_entries,
                    )
            except Exception:
                self.pool.release(pending.slot_id)
                raise

        self.pool.release(pending.slot_id)
        logger.debug(
            "D scatter done: transfer=%s chunk=%d slot=%d success=%s",
            msg.transfer_id,
            msg.chunk_id,
            pending.slot_id,
            msg.success,
        )
        return True

    def _validate(self, msg: PrepareWriteMsg) -> str | None:
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
        for dst_offset, packed_offset, nbytes in msg.scatter_entries:
            if nbytes <= 0:
                return f"scatter size must be positive: nbytes={nbytes}"
            if packed_offset < 0 or packed_offset + nbytes > msg.total_bytes:
                return (
                    f"packed range out of bounds: offset={packed_offset}, nbytes={nbytes}, "
                    f"payload={msg.total_bytes}"
                )
            max_end = max(max_end, packed_offset + nbytes)

            if not any(
                base <= dst_offset
                and dst_offset + nbytes <= base + tensor.numel() * tensor.element_size()
                for base, tensor in regions
            ):
                return f"destination range is outside registered KV regions: addr=0x{dst_offset:x}, nbytes={nbytes}"

        if max_end != msg.total_bytes:
            return f"payload does not match scatter ranges: max_end={max_end}, payload={msg.total_bytes}"
        return None

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


__all__ = [
    "DecodeWriteService",
]
