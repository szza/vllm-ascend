# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Prefill-side staging service.

Handles batched READ prepare requests from the decode side: gathers
one whole window into staging extents and returns ``PACK_READY_BATCH``.
On ``READ_ACK_BATCH`` the extents are released for reuse.

The service is transport-agnostic — callers invoke methods directly
(in-process tests) or via ZMQ (connector integration in Batch 4e).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    pack_into_staging_multi_batch,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator, StagingLease
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    STAGING_ERR_ARENA_EXHAUSTED,
    PackReadyBatchItem,
    PackReadyBatchMsg,
    PrepareReadBatchMsg,
    PrepareReadMsg,
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
    _kv_ready_event: Any = field(default=None, init=False, repr=False)

    def record_kv_ready(self, event: Any) -> None:
        """Publish the event recorded when the producing forward completed."""
        self._kv_ready_event = event

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
        """Reserve and gather one batch, or reserve nothing.

        A batch is one staging window. Partial allocation is not exposed to
        the caller: either every chunk gets a lease and is gathered, or no
        lease from this attempt remains active.
        """
        if not msg.chunks:
            return PackReadyBatchMsg(transfer_id=msg.transfer_id, results=[])

        prepares = [self._prepare_from_batch_item(msg, item) for item in msg.chunks]
        validation_errors = [self._validate_prepare_read(prepare) for prepare in prepares]
        if any(error is not None for error in validation_errors):
            return self._failed_batch(
                msg,
                code=3,
                errors=validation_errors,
                default_error="batch rejected before allocation",
            )

        with self._active_leases_lock:
            keys = [(msg.transfer_id, item.chunk_id) for item in msg.chunks]
            existing = [self._active_leases.get(key) for key in keys]
            if any(active is not None for active in existing) and not all(active is not None for active in existing):
                return self._failed_batch(msg, code=3, default_error="partial duplicate prepare")
            if all(active is not None for active in existing):
                mismatched = [
                    active is not None and active.lease.payload_bytes != item.total_bytes
                    for active, item in zip(existing, msg.chunks)
                ]
                if any(mismatched):
                    return self._failed_batch(msg, code=3, default_error="duplicate prepare changed payload size")
                reused = existing
                owned: list[_ActiveLease] = []
            else:
                leases = self.allocator.try_allocate_all([item.total_bytes for item in msg.chunks])
                if leases is None:
                    logger.info(
                        "P staging batch arena exhausted: transfer=%s chunks=%d bytes=%d "
                        "free_bytes=%d capacity_bytes=%d",
                        msg.transfer_id,
                        len(msg.chunks),
                        sum(item.total_bytes for item in msg.chunks),
                        self.allocator.free_bytes,
                        self.allocator.capacity_bytes,
                    )
                    return self._failed_batch(
                        msg,
                        code=STAGING_ERR_ARENA_EXHAUSTED,
                        default_error="staging arena has no contiguous extent for batch window",
                    )
                owned = []
                for key, item, lease in zip(keys, msg.chunks, leases):
                    active = _ActiveLease(lease=lease, transfer_id=msg.transfer_id, chunk_id=item.chunk_id)
                    self._active_leases[key] = active
                    owned.append(active)
                reused = []

        if reused:
            results: list[PackReadyBatchItem] = []
            for prepare, active in zip(prepares, reused):
                assert active is not None
                active.ready.wait()
                if active.error is not None:
                    results.append(
                        PackReadyBatchItem(
                            chunk_id=prepare.chunk_id,
                            success=False,
                            error_code=4,
                            error=active.error,
                        )
                    )
                    continue
                results.append(
                    PackReadyBatchItem(
                        chunk_id=prepare.chunk_id,
                        success=True,
                        lease_id=active.lease.lease_id,
                        staging_addr=self.allocator.address(active.lease),
                        payload_bytes=prepare.total_bytes,
                    )
                )
            return PackReadyBatchMsg(transfer_id=msg.transfer_id, results=results)

        results = []
        try:
            window_gather_ms, gather_timing = self._gather_window(prepares, owned)
            for index, (prepare, active) in enumerate(zip(prepares, owned)):
                results.append(
                    PackReadyBatchItem(
                        chunk_id=prepare.chunk_id,
                        success=True,
                        lease_id=active.lease.lease_id,
                        staging_addr=self.allocator.address(active.lease),
                        payload_bytes=prepare.total_bytes,
                        # Window time lives on the first chunk so callers that
                        # sum per-chunk gather_ms do not multiply it.
                        gather_ms=window_gather_ms if index == 0 else 0.0,
                    )
                )
            self._log_window_gather(prepares[0].transfer_id, prepares, window_gather_ms, gather_timing)
            for prepare in prepares:
                self._record_staged_chunk(prepare)
        except Exception as exc:
            logger.exception("P staging batch gather failed: transfer=%s", msg.transfer_id)
            self._release_owned_batch(msg.transfer_id, owned, error=f"gather failed: {exc}")
            return self._failed_batch(msg, code=4, default_error=f"gather failed: {exc}")
        finally:
            for active in owned:
                active.ready.set()
        return PackReadyBatchMsg(transfer_id=msg.transfer_id, results=results)

    def _prepare_from_batch_item(self, msg: PrepareReadBatchMsg, item: Any) -> PrepareReadMsg:
        return PrepareReadMsg(
            transfer_id=msg.transfer_id,
            chunk_id=item.chunk_id,
            gather_entries=item.gather_entries,
            total_bytes=item.total_bytes,
            release_source_when_ready=msg.release_source_when_ready,
            expected_chunks=msg.expected_chunks,
        )

    def _gather_window(
        self,
        prepares: list[PrepareReadMsg],
        owned: list[_ActiveLease],
    ) -> tuple[float, dict[str, float]]:
        regions = self.kv_regions
        if not regions:
            regions = [(self.kv_base_addr, self.kv_tensor)]
        staging_views = []
        gather_batches = []
        for prepare, active in zip(prepares, owned):
            staging_views.append(self.allocator.view(active.lease))
            gather_batches.append(
                [GatherEntry(src_offset=src, packed_offset=off, nbytes=n) for src, off, n in prepare.gather_entries]
            )
        gather_timing: dict[str, float] = {}
        started_at = time.perf_counter()
        pack_into_staging_multi_batch(
            regions,
            staging_views,
            gather_batches,
            timing=gather_timing,
            ready_event=self._kv_ready_event,
        )
        return (time.perf_counter() - started_at) * 1000, gather_timing

    def _log_window_gather(
        self,
        transfer_id: str,
        prepares: list[PrepareReadMsg],
        gather_ms: float,
        gather_timing: dict[str, float],
    ) -> None:
        logger.info(
            "P gather batch breakdown: transfer=%s chunks=%d entries=%d bytes=%d total_ms=%.2f "
            "descriptor_ms=%.2f descriptor_list_ms=%.2f descriptor_tensor_ms=%.2f "
            "descriptor_prepare_ms=%.2f descriptor_alloc_ms=%.2f "
            "descriptor_fill_ms=%.2f "
            "swap_blocks_batch_ms=%.2f synchronize_ms=%.2f copy_stream_wait_ms=%.2f "
            "slice_gather_ms=%.2f pack_total_ms=%.2f kv_ready=%s",
            transfer_id,
            len(prepares),
            sum(len(prepare.gather_entries) for prepare in prepares),
            sum(prepare.total_bytes for prepare in prepares),
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
            gather_timing.get("pack_into_staging_multi_batch_ms", 0.0),
            "event" if self._kv_ready_event is not None else "none",
        )

    def _failed_batch(
        self,
        msg: PrepareReadBatchMsg,
        *,
        code: int,
        default_error: str,
        errors: list[str | None] | None = None,
    ) -> PackReadyBatchMsg:
        results = []
        for index, item in enumerate(msg.chunks):
            error = errors[index] if errors is not None and errors[index] is not None else default_error
            results.append(
                PackReadyBatchItem(
                    chunk_id=item.chunk_id,
                    success=False,
                    error_code=code,
                    error=error,
                )
            )
        return PackReadyBatchMsg(transfer_id=msg.transfer_id, results=results)

    def _release_owned_batch(self, transfer_id: str, owned: list[_ActiveLease], *, error: str) -> None:
        with self._active_leases_lock:
            for active in owned:
                active.error = error
                current = self._active_leases.get((transfer_id, active.chunk_id))
                if current is active:
                    self._active_leases.pop((transfer_id, active.chunk_id), None)
        for active in owned:
            self.allocator.release(active.lease)

    @property
    def active_lease_count(self) -> int:
        with self._active_leases_lock:
            return len(self._active_leases)


__all__ = [
    "PrefillStagingService",
]
