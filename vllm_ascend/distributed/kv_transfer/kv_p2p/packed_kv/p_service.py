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

import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable

import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    pack_into_staging,
    pack_into_staging_multi,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator, StagingLease
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    STAGING_ERR_ARENA_EXHAUSTED,
    PackReadyMsg,
    PackReadyBatchItem,
    PackReadyBatchMsg,
    PrepareReadBatchMsg,
    PrepareReadMsg,
    StagingErrorMsg,
    ReadAckMsg,
)

@dataclass
class _ActiveLease:
    """Tracks a gathered extent exposed for RDMA read."""

    lease: StagingLease
    transfer_id: str
    chunk_id: int
    ready: threading.Event = field(default_factory=threading.Event, repr=False)
    error: str | None = None


@dataclass
class PrefillStagingService:
    """P-side handler for staging gather/release lifecycle.

    Parameters
    ----------
    allocator : StagingAllocator
        The P-side staging arena registered with TE.
    kv_tensor : torch.Tensor
        Flat view of the P-side KV cache (all layers/components).
    kv_base_addr : int
        ``kv_tensor.data_ptr()`` — base address for offset computation.
    """

    allocator: StagingAllocator
    kv_tensor: torch.Tensor
    kv_base_addr: int
    kv_regions: list[tuple[int, torch.Tensor]] = field(default_factory=list, init=True)
    max_concurrent_chunks: int = 1
    on_transfer_staged: Callable[[str], None] | None = None
    _active_leases: dict[tuple[str, int], _ActiveLease] = field(default_factory=dict, init=False)
    _active_leases_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _staged_chunks: dict[str, set[int]] = field(default_factory=dict, init=False, repr=False)
    _staged_expected: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _staged_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _record_staged_chunk(self, msg: PrepareReadMsg) -> None:
        """Notify the scheduler path when an all-staging request is gathered."""
        if not msg.release_source_when_ready or msg.expected_chunks <= 0:
            return
        callback = None
        with self._staged_lock:
            expected = self._staged_expected.setdefault(msg.transfer_id, msg.expected_chunks)
            if expected != msg.expected_chunks:
                logger.warning(
                    "P staging source-release chunk count changed: transfer=%s expected=%d got=%d",
                    msg.transfer_id,
                    expected,
                    msg.expected_chunks,
                )
                return
            gathered = self._staged_chunks.setdefault(msg.transfer_id, set())
            if msg.chunk_id in gathered:
                return
            gathered.add(msg.chunk_id)
            if len(gathered) == expected:
                self._staged_chunks.pop(msg.transfer_id, None)
                self._staged_expected.pop(msg.transfer_id, None)
                callback = self.on_transfer_staged
        if callback is not None:
            logger.info(
                "P staging source transfer complete after gather: transfer=%s chunks=%d",
                msg.transfer_id,
                msg.expected_chunks,
            )
            try:
                callback(msg.transfer_id)
            except Exception:
                logger.exception("P staging source completion callback failed: transfer=%s", msg.transfer_id)

    def _validate_prepare_read(self, msg: PrepareReadMsg) -> str | None:
        if msg.total_bytes <= 0:
            return "payload must be positive"
        if msg.total_bytes > self.allocator.capacity_bytes:
            return f"payload exceeds staging arena capacity: payload={msg.total_bytes}, capacity={self.allocator.capacity_bytes}"

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
        handler_started_at = time.perf_counter()
        key = (msg.transfer_id, msg.chunk_id)

        validation_started_at = time.perf_counter()
        error = self._validate_prepare_read(msg)
        validation_finished_at = time.perf_counter()
        if error is not None:
            return StagingErrorMsg(
                transfer_id=msg.transfer_id,
                chunk_id=msg.chunk_id,
                code=3,
                reason=error,
            )

        allocation_started_at = time.perf_counter()
        is_owner = False
        with self._active_leases_lock:
            existing = self._active_leases.get(key)
            if existing is None:
                lease = self.allocator.allocate(msg.total_bytes)
                if lease is None:
                    logger.info(
                        "P staging arena has no contiguous extent: transfer=%s chunk=%d "
                        "bytes=%d free_bytes=%d capacity_bytes=%d",
                        msg.transfer_id,
                        msg.chunk_id,
                        msg.total_bytes,
                        self.allocator.free_bytes,
                        self.allocator.capacity_bytes,
                    )
                    return StagingErrorMsg(
                        transfer_id=msg.transfer_id,
                        chunk_id=msg.chunk_id,
                        code=STAGING_ERR_ARENA_EXHAUSTED,
                        reason="staging arena has no contiguous extent available",
                    )
                existing = _ActiveLease(
                    lease=lease,
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                )
                self._active_leases[key] = existing
                is_owner = True
            elif existing.lease.payload_bytes != msg.total_bytes:
                return StagingErrorMsg(
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                    code=3,
                    reason="duplicate prepare changed payload size",
                )
        allocation_finished_at = time.perf_counter()

        if not is_owner:
            wait_started_at = time.perf_counter()
            existing.ready.wait()
            wait_finished_at = time.perf_counter()
            if existing.error is not None:
                return StagingErrorMsg(
                    transfer_id=msg.transfer_id,
                    chunk_id=msg.chunk_id,
                    code=4,
                    reason=existing.error,
                )
            logger.info(
                "P handle_prepare timing: transfer=%s chunk=%d outcome=duplicate "
                "validate_ms=%.2f slot_ms=%.2f duplicate_wait_ms=%.2f "
                "handler_total_ms=%.2f",
                msg.transfer_id,
                msg.chunk_id,
                (validation_finished_at - validation_started_at) * 1000,
                (allocation_finished_at - allocation_started_at) * 1000,
                (wait_finished_at - wait_started_at) * 1000,
                (time.perf_counter() - handler_started_at) * 1000,
            )
            return PackReadyMsg(
                transfer_id=msg.transfer_id,
                chunk_id=msg.chunk_id,
                lease_id=existing.lease.lease_id,
                staging_addr=self.allocator.address(existing.lease),
                payload_bytes=msg.total_bytes,
            )

        lease = existing.lease

        entry_started_at = time.perf_counter()
        gather_entries = [
            GatherEntry(src_offset=src, packed_offset=off, nbytes=n) for src, off, n in msg.gather_entries
        ]
        entry_finished_at = time.perf_counter()

        slot_view_started_at = time.perf_counter()
        staging_view = self.allocator.view(lease)
        slot_view_finished_at = time.perf_counter()
        t_gather = time.perf_counter()
        gather_timing: dict[str, float] = {}
        try:
            if self.kv_regions:
                pack_into_staging_multi(self.kv_regions, staging_view, gather_entries, timing=gather_timing)
            else:
                pack_into_staging(
                    self.kv_tensor,
                    self.kv_base_addr,
                    staging_view,
                    gather_entries,
                    timing=gather_timing,
                )
        except Exception:
            with self._active_leases_lock:
                active = self._active_leases.pop(key, None)
            if active is not None:
                active.error = "gather failed"
                active.ready.set()
                self.allocator.release(active.lease)
            raise

        gather_ms = (time.perf_counter() - t_gather) * 1000
        pack_finished_at = time.perf_counter()

        logger.info(
            "P gather breakdown: transfer=%s chunk=%d entries=%d bytes=%d total_ms=%.2f "
            "descriptor_ms=%.2f descriptor_list_ms=%.2f descriptor_tensor_ms=%.2f "
            "descriptor_prepare_ms=%.2f descriptor_alloc_ms=%.2f "
            "descriptor_fill_ms=%.2f "
            "swap_blocks_batch_ms=%.2f synchronize_ms=%.2f copy_stream_wait_ms=%.2f "
            "slice_gather_ms=%.2f pack_total_ms=%.2f",
            msg.transfer_id,
            msg.chunk_id,
            len(gather_entries),
            msg.total_bytes,
            gather_ms,
            gather_timing.get("descriptor_ms", 0.0),
            gather_timing.get("descriptor_list_ms", 0.0),
            gather_timing.get("descriptor_tensor_ms", 0.0),
            gather_timing.get("descriptor_prepare_ms", 0.0),
            gather_timing.get("descriptor_alloc_ms", 0.0),
            gather_timing.get("descriptor_fill_ms", 0.0),
            gather_timing.get("swap_blocks_batch_ms", 0.0),
            gather_timing.get("synchronize_ms", 0.0),
            gather_timing.get("copy_stream_wait_ms", 0.0),
            gather_timing.get("slice_gather_ms", 0.0),
            gather_timing.get(
                "pack_into_staging_multi_ms",
                gather_timing.get("pack_into_staging_ms", 0.0),
            ),
        )

        existing.ready.set()
        self._record_staged_chunk(msg)

        staging_addr = self.allocator.address(lease)
        response_preparation_finished_at = time.perf_counter()
        logger.info(
            "P handle_prepare timing: transfer=%s chunk=%d outcome=ready "
            "validate_ms=%.2f slot_ms=%.2f entry_build_ms=%.2f slot_view_ms=%.2f "
            "pack_ms=%.2f post_pack_ms=%.2f handler_total_ms=%.2f",
            msg.transfer_id,
            msg.chunk_id,
            (validation_finished_at - validation_started_at) * 1000,
            (allocation_finished_at - allocation_started_at) * 1000,
            (entry_finished_at - entry_started_at) * 1000,
            (slot_view_finished_at - slot_view_started_at) * 1000,
            (pack_finished_at - t_gather) * 1000,
            (response_preparation_finished_at - pack_finished_at) * 1000,
            (response_preparation_finished_at - handler_started_at) * 1000,
        )
        logger.debug(
            "P gather done: transfer=%s chunk=%d lease=%d addr=0x%x bytes=%d gather_ms=%.2f",
            msg.transfer_id,
            msg.chunk_id,
            lease.lease_id,
            staging_addr,
            msg.total_bytes,
            gather_ms,
        )
        return PackReadyMsg(
            transfer_id=msg.transfer_id,
            chunk_id=msg.chunk_id,
            lease_id=lease.lease_id,
            staging_addr=staging_addr,
            payload_bytes=msg.total_bytes,
            gather_ms=gather_ms,
        )

    def handle_read_ack(self, msg: ReadAckMsg) -> bool:
        """Release the staging slot after D confirms the RDMA read is done.

        Returns True if the slot was found and released, False if the
        transfer_id/chunk_id was unknown (idempotent).
        """
        key = (msg.transfer_id, msg.chunk_id)
        with self._active_leases_lock:
            active = self._active_leases.get(key)
            if active is not None and active.lease.lease_id != msg.lease_id:
                return False
        if active is not None:
            active.ready.wait()
        with self._active_leases_lock:
            active = self._active_leases.get(key)
            if active is not None and active.lease.lease_id != msg.lease_id:
                return False
            active = self._active_leases.pop(key, None)
        if active is None:
            return False
        self.allocator.release(active.lease)
        logger.debug(
            "P staging extent released: transfer=%s chunk=%d lease=%d success=%s",
            msg.transfer_id,
            msg.chunk_id,
            active.lease.lease_id,
            msg.success,
        )
        return True

    def handle_prepare_read_batch(self, msg: PrepareReadBatchMsg) -> PackReadyBatchMsg:
        """Gather a batch synchronously for direct callers and fallbacks.

        The ZMQ listener fans batch items out to its gather executor.  This
        method remains the transport-independent implementation used by local
        callers and tests.
        """
        results: list[PackReadyBatchItem] = []
        for item in msg.chunks:
            response = self.handle_prepare_read(
                PrepareReadMsg(
                    transfer_id=msg.transfer_id,
                    chunk_id=item.chunk_id,
                    gather_entries=item.gather_entries,
                    total_bytes=item.total_bytes,
                    release_source_when_ready=msg.release_source_when_ready,
                    expected_chunks=msg.expected_chunks,
                )
            )
            if isinstance(response, StagingErrorMsg):
                results.append(
                    PackReadyBatchItem(
                        chunk_id=item.chunk_id,
                        success=False,
                        error_code=response.code,
                        error=response.reason,
                    )
                )
            else:
                results.append(
                    PackReadyBatchItem(
                        chunk_id=item.chunk_id,
                        success=True,
                        lease_id=response.lease_id,
                        staging_addr=response.staging_addr,
                        payload_bytes=response.payload_bytes,
                        gather_ms=response.gather_ms,
                    )
                )
        return PackReadyBatchMsg(transfer_id=msg.transfer_id, results=results)

    @property
    def active_lease_count(self) -> int:
        with self._active_leases_lock:
            return len(self._active_leases)


__all__ = [
    "PrefillStagingService",
]
