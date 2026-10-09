# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Decode-side staging coordinator.

Orchestrates the full READY → READ → SCATTER → ACK cycle for a
``TransferPlan``:

1. **Direct runs** are transferred via the TE. When packed chunks are also
   present and a batch callback is available, their descriptors are submitted
   with the first staging window.
2. **Packed chunks** go through the staging protocol:
   acquire D slot → PREPARE_READ → PACK_READY → RDMA read (one large
   entry) → READ_ACK (release P lease) → scatter from D slot → release D slot.

The coordinator is transport-agnostic: the actual RDMA read and
P-service communication are supplied via callbacks so the module can
be tested in-process without ZMQ or a real TransferEngine.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    unpack_from_staging,
    unpack_from_staging_multi,
    unpack_from_staging_multi_batch,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    PackedChunk,
    TransferPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator, StagingLease
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    STAGING_ERR_ARENA_EXHAUSTED,
    PackReadyBatchMsg,
    PackReadyMsg,
    PrepareReadBatchItem,
    PrepareReadBatchMsg,
    PrepareReadMsg,
    ReadAckBatchItem,
    ReadAckBatchMsg,
    ReadAckMsg,
    StagingErrorMsg,
)

DirectTransferFn = Callable[[list[int], list[int], list[int]], int]
"""(src_addrs, dst_addrs, lengths) → return code (0 = success)."""

PrepareReadFn = Callable[[PrepareReadMsg], Union[PackReadyMsg, StagingErrorMsg]]  # noqa: UP007
"""Send PREPARE_READ to P, get PACK_READY or error back."""

PrepareReadBatchFn = Callable[[PrepareReadBatchMsg], PackReadyBatchMsg]
"""Send a batch PREPARE_READ to P and get all gather results back."""

RdmaReadFn = Callable[[int, int, int], int]
"""(local_dst_addr, remote_src_addr, nbytes) → return code."""

RdmaReadBatchFn = Callable[[list[int], list[int], list[int]], int]
"""(local_dst_addrs, remote_src_addrs, lengths) → return code."""


@dataclass(frozen=True)
class _ReadDescriptor:
    """One Mooncake READ descriptor in D-local/P-remote address order."""

    local_dst: int
    remote_src: int
    nbytes: int
    kind: str
    chunk_id: int | None = None

SendAckFn = Callable[[ReadAckMsg], None]
"""Send READ_ACK to P after scatter is done."""

SendAckBatchFn = Callable[[ReadAckBatchMsg], None]
"""Send completion status for a batch of chunks to P."""


@dataclass
class StagedTransferResult:
    """Outcome of a staged transfer execution."""

    success: bool
    direct_bytes: int = 0
    packed_bytes: int = 0
    direct_entries: int = 0
    packed_entries: int = 0
    chunks_completed: int = 0
    error: str | None = None


@dataclass
class _WindowOutcome:
    """Result of one staging window, including a direct-RDMA fallback."""

    error: str | None = None
    packed_bytes: int = 0
    packed_entries: int = 0
    chunks_completed: int = 0
    direct_completed: bool = False
    fallback_rest: bool = False
    direct_bytes: int = 0
    direct_entries: int = 0


class DecodeStagingCoordinator:
    """D-side orchestrator for staged KV cache transfers.

    Parameters
    ----------
    allocator : StagingAllocator
        D-side staging arena for receiving packed data.
    dst_tensor : torch.Tensor
        Flat view of D-side KV cache.
    dst_base_addr : int
        ``dst_tensor.data_ptr()``.
    """

    def __init__(
        self,
        allocator: StagingAllocator,
        dst_tensor: torch.Tensor,
        dst_base_addr: int,
        dst_regions: list[tuple[int, torch.Tensor]] | None = None,
        chunk_capacity: int | None = None,
        max_concurrent_chunks: int = 1,
    ) -> None:
        self.allocator = allocator
        self.dst_tensor = dst_tensor
        self.dst_base_addr = dst_base_addr
        self.dst_regions = dst_regions or []
        self.chunk_capacity = chunk_capacity or allocator.capacity_bytes
        self.max_concurrent_chunks = max_concurrent_chunks

    def execute(
        self,
        plan: TransferPlan,
        transfer_id: str,
        *,
        direct_transfer: DirectTransferFn | None = None,
        prepare_read: PrepareReadFn,
        rdma_read: RdmaReadFn,
        send_ack: SendAckFn,
        prepare_read_batch: PrepareReadBatchFn | None = None,
        send_ack_batch: SendAckBatchFn | None = None,
        rdma_read_batch: RdmaReadBatchFn | None = None,
        merged_rdma_read: RdmaReadBatchFn | None = None,
        batch_window_size: int | None = None,
        release_source_when_ready: bool = False,
        expected_packed_chunks: int = 0,
    ) -> StagedTransferResult:
        """Execute a full TransferPlan.

        Parameters
        ----------
        plan : TransferPlan
            Output of ``TransferPlanner.plan()``.
        transfer_id : str
            Unique identifier for this transfer (used in protocol messages).
        direct_transfer : callable, optional
            Batch RDMA read for direct runs.  If None, direct runs are
            skipped (caller handles them separately).
        prepare_read : callable
            Sends PREPARE_READ to P, returns PACK_READY or error.
        rdma_read : callable
            RDMA read from P slot to D slot (single large entry).
        send_ack : callable
            Sends READ_ACK to P after the RDMA read completes. Scatter does
            not use the P extent, so the P lease is not held across it.
        prepare_read_batch : callable, optional
            Batched PREPARE_READ callback. Used for multi-chunk windows and
            for a mixed DirectRun + single-chunk window when available.
        send_ack_batch : callable, optional
            Batched READ_ACK callback.
        rdma_read_batch : callable, optional
            Batched RDMA callback. If omitted, the batch protocol falls back
            to one RDMA call per chunk.
        merged_rdma_read : callable, optional
            Batched RDMA callback used when DirectRun and staging descriptors
            are submitted together. If omitted, ``rdma_read_batch`` is used.
        batch_window_size : int, optional
            Maximum number of chunks in one batch. Defaults to one chunk.
        """
        result = StagedTransferResult(success=True)

        direct_descriptors = self._build_direct_descriptors(plan)
        merged_read = merged_rdma_read or rdma_read_batch
        direct_pending = bool(direct_descriptors and plan.packed_chunks and merged_read is not None)

        # Preserve the legacy direct-first behavior when a merged callback is
        # unavailable. The merged path defers DirectRun until PACK_READY
        # supplies the staging source address.
        if plan.direct_runs and direct_transfer is not None and not direct_pending:
            ret = self._execute_direct_runs(plan, direct_transfer)
            if ret != 0:
                result.success = False
                result.error = f"direct transfer failed: ret={ret}"
                return result
            result.direct_bytes = plan.total_direct_bytes
            result.direct_entries = len(plan.direct_runs)

        use_batch = (
            len(plan.packed_chunks) > 1
            and prepare_read_batch is not None
            and send_ack_batch is not None
        )
        if direct_pending and len(plan.packed_chunks) == 1:
            use_batch = prepare_read_batch is not None and send_ack_batch is not None
        if use_batch:
            window_size = batch_window_size or 1
            if window_size <= 0:
                result.success = False
                result.error = f"invalid batch window size: {window_size}"
                return result
            packed_batches = (
                plan.packed_chunks[start : start + window_size]
                for start in range(0, len(plan.packed_chunks), window_size)
            )
        else:
            packed_batches = ((chunk,) for chunk in plan.packed_chunks)

        fallback_rest = False
        direct_accounted = result.direct_entries > 0
        for packed_batch in packed_batches:
            pending_descriptors = direct_descriptors if direct_pending else ()
            if fallback_rest:
                outcome = self._arena_fallback_outcome(
                    packed_batch,
                    direct_transfer,
                    error="previous window found no contiguous staging extent; remaining chunks use direct RDMA",
                )
            elif use_batch:
                outcome = self._execute_batch(
                    packed_batch,
                    transfer_id=transfer_id,
                    prepare_read_batch=prepare_read_batch,
                    rdma_read=rdma_read,
                    rdma_read_batch=merged_read if direct_pending else rdma_read_batch,
                    send_ack_batch=send_ack_batch,
                    direct_descriptors=pending_descriptors,
                    release_source_when_ready=release_source_when_ready,
                    expected_packed_chunks=expected_packed_chunks,
                    direct_transfer=direct_transfer,
                )
            else:
                outcome = self._execute_chunk(
                    packed_batch[0],
                    transfer_id=transfer_id,
                    prepare_read=prepare_read,
                    rdma_read=rdma_read,
                    send_ack=send_ack,
                    rdma_read_batch=merged_read if direct_pending else None,
                    direct_descriptors=pending_descriptors,
                    release_source_when_ready=release_source_when_ready,
                    expected_packed_chunks=expected_packed_chunks,
                    direct_transfer=direct_transfer,
                )
            if outcome.error is not None:
                result.success = False
                result.error = outcome.error
                return result
            result.packed_bytes += outcome.packed_bytes
            result.packed_entries += outcome.packed_entries
            result.chunks_completed += outcome.chunks_completed
            result.direct_bytes += outcome.direct_bytes
            result.direct_entries += outcome.direct_entries
            if outcome.direct_completed and not direct_accounted:
                result.direct_bytes += plan.total_direct_bytes
                result.direct_entries += len(plan.direct_runs)
                direct_accounted = True
            if outcome.direct_completed or outcome.fallback_rest:
                direct_pending = False
            if outcome.fallback_rest:
                fallback_rest = True

        return result

    def _execute_direct_runs(self, plan: TransferPlan, direct_transfer: DirectTransferFn) -> int:
        descriptors = self._build_direct_descriptors(plan)
        src_addrs = [descriptor.remote_src for descriptor in descriptors]
        dst_addrs = [descriptor.local_dst for descriptor in descriptors]
        lengths = [descriptor.nbytes for descriptor in descriptors]
        return direct_transfer(src_addrs, dst_addrs, lengths)

    @staticmethod
    def _build_direct_descriptors(plan: TransferPlan) -> tuple[_ReadDescriptor, ...]:
        return tuple(
            _ReadDescriptor(
                local_dst=run.dst_offset,
                remote_src=run.src_offset,
                nbytes=run.nbytes,
                kind="direct",
            )
            for run in plan.direct_runs
        )

    def _direct_fallback_chunks(
        self,
        chunks: tuple[PackedChunk, ...] | list[PackedChunk],
        direct_transfer: DirectTransferFn | None,
        extra_descriptors: tuple[_ReadDescriptor, ...] = (),
    ) -> str | None:
        """RDMA packed fragments directly from P KV into D KV."""
        if direct_transfer is None:
            return "direct fallback unavailable"
        src_addrs = [descriptor.remote_src for descriptor in extra_descriptors]
        dst_addrs = [descriptor.local_dst for descriptor in extra_descriptors]
        lengths = [descriptor.nbytes for descriptor in extra_descriptors]
        for chunk in chunks:
            if len(chunk.gather_entries) != len(chunk.scatter_entries):
                return f"chunk {chunk.chunk_id} gather/scatter entry count mismatch"
            for gather, scatter in zip(chunk.gather_entries, chunk.scatter_entries):
                if gather.nbytes != scatter.nbytes:
                    return f"chunk {chunk.chunk_id} gather/scatter size mismatch"
                src_addrs.append(gather.src_offset)
                dst_addrs.append(scatter.dst_offset)
                lengths.append(gather.nbytes)
        if not lengths:
            return None
        logger.info(
            "D staging direct fallback: chunks=%d descriptors=%d bytes=%d",
            len(chunks),
            len(lengths),
            sum(lengths),
        )
        ret = direct_transfer(src_addrs, dst_addrs, lengths)
        if ret != 0:
            return f"direct fallback RDMA failed: ret={ret}"
        return None

    def _arena_fallback_outcome(
        self,
        chunks: tuple[PackedChunk, ...] | list[PackedChunk],
        direct_transfer: DirectTransferFn | None,
        extra_descriptors: tuple[_ReadDescriptor, ...] = (),
        error: str = "staging arena has no contiguous extent",
    ) -> "_WindowOutcome":
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        logger.info(
            "D staging arena has no contiguous extent: chunks=%s free_bytes=%d capacity_bytes=%d "
            "direct_fallback=%s reason=%s",
            chunk_ids,
            self.allocator.free_bytes,
            self.allocator.capacity_bytes,
            direct_transfer is not None,
            error,
        )
        if direct_transfer is None:
            return _WindowOutcome(error=error)
        fallback_error = self._direct_fallback_chunks(chunks, direct_transfer, extra_descriptors)
        if fallback_error is not None:
            return _WindowOutcome(error=fallback_error)
        return _WindowOutcome(
            fallback_rest=True,
            direct_completed=bool(extra_descriptors),
            direct_bytes=sum(chunk.payload_bytes for chunk in chunks),
            direct_entries=sum(len(chunk.gather_entries) for chunk in chunks),
        )

    def _execute_chunk(
        self,
        chunk: PackedChunk,
        transfer_id: str,
        prepare_read: PrepareReadFn,
        rdma_read: RdmaReadFn,
        send_ack: SendAckFn,
        rdma_read_batch: RdmaReadBatchFn | None = None,
        direct_descriptors: tuple[_ReadDescriptor, ...] = (),
        release_source_when_ready: bool = False,
        expected_packed_chunks: int = 0,
        direct_transfer: DirectTransferFn | None = None,
    ) -> "_WindowOutcome":
        """Execute one packed chunk. Arena exhaustion falls back to direct RDMA."""
        if chunk.payload_bytes > self.allocator.capacity_bytes:
            return self._arena_fallback_outcome(
                (chunk,),
                direct_transfer,
                direct_descriptors,
                error=(
                    f"chunk {chunk.chunk_id} exceeds D staging arena capacity: "
                    f"payload={chunk.payload_bytes}, capacity={self.allocator.capacity_bytes}"
                ),
            )

        t0 = time.perf_counter()
        lease = self.allocator.allocate(chunk.payload_bytes)
        if lease is None:
            return self._arena_fallback_outcome(
                (chunk,),
                direct_transfer,
                direct_descriptors,
                error=f"no contiguous D staging extent for chunk {chunk.chunk_id}",
            )
        t_acquire = time.perf_counter()

        try:
            prepare_msg = PrepareReadMsg(
                transfer_id=transfer_id,
                chunk_id=chunk.chunk_id,
                gather_entries=[(g.src_offset, g.packed_offset, g.nbytes) for g in chunk.gather_entries],
                total_bytes=chunk.payload_bytes,
                release_source_when_ready=release_source_when_ready,
                expected_chunks=expected_packed_chunks,
            )

            response = prepare_read(prepare_msg)
            t_prepare = time.perf_counter()

            if isinstance(response, StagingErrorMsg):
                if response.code == STAGING_ERR_ARENA_EXHAUSTED:
                    return self._arena_fallback_outcome(
                        (chunk,),
                        direct_transfer,
                        direct_descriptors,
                        error=f"P rejected PREPARE_READ for chunk {chunk.chunk_id}: {response.reason}",
                    )
                return _WindowOutcome(
                    error=f"P rejected PREPARE_READ for chunk {chunk.chunk_id}: {response.reason}"
                )
            if response.payload_bytes != chunk.payload_bytes:
                self._send_abort(send_ack, transfer_id, chunk.chunk_id, response.lease_id)
                return _WindowOutcome(
                    error=(
                        f"P returned wrong payload size for chunk {chunk.chunk_id}: "
                        f"expected={chunk.payload_bytes}, got={response.payload_bytes}"
                    )
                )

            d_staging_addr = self.allocator.address(lease)
            staging_descriptor = _ReadDescriptor(
                local_dst=d_staging_addr,
                remote_src=response.staging_addr,
                nbytes=chunk.payload_bytes,
                kind="staging",
                chunk_id=chunk.chunk_id,
            )
            if direct_descriptors:
                if rdma_read_batch is None:
                    self._send_abort(send_ack, transfer_id, chunk.chunk_id, response.lease_id)
                    return _WindowOutcome(
                        error=f"merged RDMA callback unavailable for chunk {chunk.chunk_id}"
                    )
                descriptors = (*direct_descriptors, staging_descriptor)
                logger.info(
                    "D merged RDMA batch start: transfer_id=%s direct_entries=%d packed_chunks=1 "
                    "total_descriptors=%d total_bytes=%d",
                    transfer_id,
                    len(direct_descriptors),
                    len(descriptors),
                    sum(descriptor.nbytes for descriptor in descriptors),
                )
                ret = rdma_read_batch(
                    [descriptor.local_dst for descriptor in descriptors],
                    [descriptor.remote_src for descriptor in descriptors],
                    [descriptor.nbytes for descriptor in descriptors],
                )
            else:
                ret = rdma_read(d_staging_addr, response.staging_addr, chunk.payload_bytes)
            t_rdma = time.perf_counter()

            if ret != 0:
                self._send_abort(send_ack, transfer_id, chunk.chunk_id, response.lease_id)
                return _WindowOutcome(error=f"RDMA read failed for chunk {chunk.chunk_id}: ret={ret}")

            # The P lease protects the remote extent only. Release it before
            # scatter so arena occupancy does not include the D-side copy.
            send_ack(
                ReadAckMsg(
                    transfer_id=transfer_id,
                    chunk_id=chunk.chunk_id,
                    lease_id=response.lease_id,
                    success=True,
                )
            )
            t_ack = time.perf_counter()

            try:
                staging_view = self.allocator.view(lease)
                scatter_timing: dict[str, float] = {}
                if self.dst_regions:
                    unpack_from_staging_multi(
                        staging_view,
                        self.dst_regions,
                        chunk.scatter_entries,
                        timing=scatter_timing,
                    )
                else:
                    unpack_from_staging(
                        staging_view,
                        self.dst_tensor,
                        self.dst_base_addr,
                        chunk.scatter_entries,
                        timing=scatter_timing,
                    )
            except Exception:
                logger.exception(
                    "D scatter failed after P lease release: transfer=%s chunk=%d",
                    transfer_id,
                    chunk.chunk_id,
                )
                raise
            t_scatter = time.perf_counter()

            logger.info(
                "D scatter breakdown: transfer=%s chunk=%d bytes=%d total_ms=%.2f "
                "descriptor_ms=%.2f descriptor_list_ms=%.2f descriptor_tensor_ms=%.2f "
                "descriptor_prepare_ms=%.2f descriptor_alloc_ms=%.2f "
                "descriptor_fill_ms=%.2f "
                "swap_blocks_batch_ms=%.2f synchronize_ms=%.2f copy_stream_wait_ms=%.2f "
                "slice_scatter_ms=%.2f unpack_total_ms=%.2f",
                transfer_id,
                chunk.chunk_id,
                chunk.payload_bytes,
                (t_scatter - t_rdma) * 1000,
                scatter_timing.get("descriptor_ms", 0.0),
                scatter_timing.get("descriptor_list_ms", 0.0),
                scatter_timing.get("descriptor_tensor_ms", 0.0),
                scatter_timing.get("descriptor_prepare_ms", 0.0),
                scatter_timing.get("descriptor_alloc_ms", 0.0),
                scatter_timing.get("descriptor_fill_ms", 0.0),
                scatter_timing.get("swap_blocks_batch_ms", 0.0),
                scatter_timing.get("synchronize_ms", 0.0),
                scatter_timing.get("copy_stream_wait_ms", 0.0),
                scatter_timing.get("slice_scatter_ms", 0.0),
                scatter_timing.get(
                    "unpack_from_staging_multi_ms",
                    scatter_timing.get("unpack_from_staging_ms", 0.0),
                ),
            )

            p_gather_ms = getattr(response, 'gather_ms', 0.0)
            logger.info(
                "D chunk timing: transfer=%s chunk=%d bytes=%d | "
                "acquire=%.2fms prepare=%.2fms(p_gather=%.2fms) "
                "rdma=%.2fms scatter=%.2fms ack=%.2fms total=%.2fms",
                transfer_id,
                chunk.chunk_id,
                chunk.payload_bytes,
                (t_acquire - t0) * 1000,
                (t_prepare - t_acquire) * 1000,
                p_gather_ms,
                (t_rdma - t_prepare) * 1000,
                (t_scatter - t_ack) * 1000,
                (t_ack - t_rdma) * 1000,
                (t_scatter - t0) * 1000,
            )
            logger.info(
                "D staging window timing: transfer_id=%s chunks=1 entries=%d bytes=%d "
                "acquire_ms=%.2f prepare_ms=%.2f rdma_ms=%.2f scatter_ms=%.2f "
                "ack_ms=%.2f total_ms=%.2f success=true",
                transfer_id,
                len(chunk.scatter_entries),
                chunk.payload_bytes,
                (t_acquire - t0) * 1000,
                (t_prepare - t_acquire) * 1000,
                (t_rdma - t_prepare) * 1000,
                (t_scatter - t_ack) * 1000,
                (t_ack - t_rdma) * 1000,
                (t_scatter - t0) * 1000,
            )
        finally:
            self.allocator.release(lease)

        return _WindowOutcome(
            packed_bytes=chunk.payload_bytes,
            packed_entries=len(chunk.scatter_entries),
            chunks_completed=1,
            direct_completed=bool(direct_descriptors),
        )

    def _execute_batch(
        self,
        chunks: tuple[PackedChunk, ...],
        transfer_id: str,
        prepare_read_batch: PrepareReadBatchFn,
        rdma_read: RdmaReadFn,
        rdma_read_batch: RdmaReadBatchFn | None,
        send_ack_batch: SendAckBatchFn,
        direct_descriptors: tuple[_ReadDescriptor, ...] = (),
        release_source_when_ready: bool = False,
        expected_packed_chunks: int = 0,
        direct_transfer: DirectTransferFn | None = None,
    ) -> "_WindowOutcome":
        """Execute one batch window with one RDMA submission and scatter."""
        t0 = time.perf_counter()
        leases: list[StagingLease] = []
        for chunk in chunks:
            if chunk.payload_bytes > self.allocator.capacity_bytes:
                return self._arena_fallback_outcome(
                    chunks,
                    direct_transfer,
                    direct_descriptors,
                    error=(
                        f"chunk {chunk.chunk_id} exceeds D staging arena capacity: "
                        f"payload={chunk.payload_bytes}, capacity={self.allocator.capacity_bytes}"
                    ),
                )
        for chunk in chunks:
            lease = self.allocator.allocate(chunk.payload_bytes)
            if lease is None:
                for acquired in leases:
                    self.allocator.release(acquired)
                return self._arena_fallback_outcome(
                    chunks,
                    direct_transfer,
                    direct_descriptors,
                    error="no contiguous D staging extent for batch window",
                )
            leases.append(lease)
        t_acquire = time.perf_counter()

        ack_items: list[ReadAckBatchItem] = []
        packed_bytes = 0
        packed_entries = 0
        chunks_completed = 0
        first_error: str | None = None
        ready_by_chunk = {}
        scatter_timings: dict[int, tuple[float, float]] = {}
        t_prepare = t_acquire
        t_rdma = t_prepare
        t_ack = t_rdma
        ack_ms = 0.0
        scatter_total_ms = 0.0
        direct_completed = False
        ack_sent = False
        arena_failed: list[PackedChunk] = []
        fallback_rest = False
        fallback_direct_bytes = 0
        fallback_direct_entries = 0
        try:
            logger.info(
                "D staging batch prepare start: transfer_id=%s chunks=%d bytes=%d",
                transfer_id,
                len(chunks),
                sum(chunk.payload_bytes for chunk in chunks),
            )
            response = prepare_read_batch(
                PrepareReadBatchMsg(
                    transfer_id=transfer_id,
                    chunks=[
                        PrepareReadBatchItem(
                            chunk_id=chunk.chunk_id,
                            gather_entries=[
                                (g.src_offset, g.packed_offset, g.nbytes) for g in chunk.gather_entries
                            ],
                            total_bytes=chunk.payload_bytes,
                        )
                        for chunk in chunks
                    ],
                    release_source_when_ready=release_source_when_ready,
                    expected_chunks=expected_packed_chunks,
                )
            )
            t_prepare = time.perf_counter()
            if response.transfer_id != transfer_id:
                first_error = (
                    f"P returned wrong transfer id for batch: "
                    f"expected={transfer_id}, got={response.transfer_id}"
                )
                logger.error(
                    "D staging batch prepare transfer mismatch: transfer_id=%s response_transfer_id=%s",
                    transfer_id,
                    response.transfer_id,
                )
            else:
                ready_by_chunk = {item.chunk_id: item for item in response.results}
                logger.info(
                    "D staging batch ready received: transfer_id=%s results=%d success=%d failed=%d",
                    transfer_id,
                    len(response.results),
                    sum(item.success for item in response.results),
                    sum(not item.success for item in response.results),
                )
                ready_chunks: list[tuple[PackedChunk, StagingLease, Any]] = []
                for chunk, lease in zip(chunks, leases):
                    ready = ready_by_chunk.get(chunk.chunk_id)
                    if ready is None:
                        first_error = f"P returned no result for chunk {chunk.chunk_id}"
                        logger.error(
                            "D staging batch missing ready: transfer_id=%s chunk_id=%d",
                            transfer_id,
                            chunk.chunk_id,
                        )
                        break
                    if not ready.success:
                        if (
                            ready.error_code == STAGING_ERR_ARENA_EXHAUSTED
                            and direct_transfer is not None
                        ):
                            arena_failed.append(chunk)
                            logger.info(
                                "D staging batch arena exhausted: transfer_id=%s chunk_id=%d",
                                transfer_id,
                                chunk.chunk_id,
                            )
                            continue
                        first_error = f"P rejected PREPARE_READ for chunk {chunk.chunk_id}: {ready.error}"
                        logger.error(
                            "D staging batch chunk rejected: transfer_id=%s chunk_id=%d code=%d reason=%s",
                            transfer_id,
                            chunk.chunk_id,
                            ready.error_code,
                            ready.error,
                        )
                        break
                    if ready.payload_bytes != chunk.payload_bytes:
                        first_error = (
                            f"P returned wrong payload size for chunk {chunk.chunk_id}: "
                            f"expected={chunk.payload_bytes}, got={ready.payload_bytes}"
                        )
                        logger.error(
                            "D staging batch payload mismatch: transfer_id=%s chunk_id=%d expected=%d got=%d",
                            transfer_id,
                            chunk.chunk_id,
                            chunk.payload_bytes,
                            ready.payload_bytes,
                        )
                        if ready.lease_id:
                            ack_items.append(
                                ReadAckBatchItem(chunk_id=chunk.chunk_id, lease_id=ready.lease_id, success=False)
                            )
                        break

                    ready_chunks.append((chunk, lease, ready))

                if first_error is None and ready_chunks:
                    staging_descriptors = [
                        _ReadDescriptor(
                            local_dst=self.allocator.address(slot),
                            remote_src=ready.staging_addr,
                            nbytes=chunk.payload_bytes,
                            kind="staging",
                            chunk_id=chunk.chunk_id,
                        )
                        for chunk, slot, ready in ready_chunks
                    ]
                    descriptors = (*direct_descriptors, *staging_descriptors)
                    d_addrs = [descriptor.local_dst for descriptor in descriptors]
                    p_addrs = [descriptor.remote_src for descriptor in descriptors]
                    lengths = [descriptor.nbytes for descriptor in descriptors]
                    logger.info(
                        "D staging batch RDMA start: transfer_id=%s direct_entries=%d chunks=%d "
                        "total_descriptors=%d bytes=%d",
                        transfer_id,
                        len(direct_descriptors),
                        len(staging_descriptors),
                        len(descriptors),
                        sum(lengths),
                    )
                    if direct_descriptors and rdma_read_batch is None:
                        ret = -1
                        first_error = "merged RDMA callback unavailable"
                    elif rdma_read_batch is not None:
                        ret = rdma_read_batch(d_addrs, p_addrs, lengths)
                    else:
                        ret = 0
                        for d_addr, p_addr, nbytes in zip(
                            [descriptor.local_dst for descriptor in staging_descriptors],
                            [descriptor.remote_src for descriptor in staging_descriptors],
                            [descriptor.nbytes for descriptor in staging_descriptors],
                        ):
                            ret = rdma_read(d_addr, p_addr, nbytes)
                            if ret != 0:
                                break
                    t_rdma = time.perf_counter()
                    if ret != 0:
                        first_error = f"RDMA batch read failed: ret={ret}"
                        logger.error("D staging batch RDMA failed: transfer_id=%s ret=%d", transfer_id, ret)
                    else:
                        direct_completed = bool(direct_descriptors)
                        logger.info(
                            "D staging batch RDMA done: transfer_id=%s staging_chunks=%d "
                            "total_descriptors=%d",
                            transfer_id,
                            len(staging_descriptors),
                            len(lengths),
                        )
                        for chunk, _, ready in ready_chunks:
                            ack_items.append(
                                ReadAckBatchItem(
                                    chunk_id=chunk.chunk_id,
                                    lease_id=ready.lease_id,
                                    success=True,
                                )
                            )
                        if ack_items:
                            logger.info(
                                "D staging batch ACK send: transfer_id=%s chunks=%d success=%d failed=%d",
                                transfer_id,
                                len(ack_items),
                                sum(item.success for item in ack_items),
                                sum(not item.success for item in ack_items),
                            )
                            t_ack_start = time.perf_counter()
                            send_ack_batch(ReadAckBatchMsg(transfer_id=transfer_id, results=ack_items))
                            t_ack = time.perf_counter()
                            ack_ms = (t_ack - t_ack_start) * 1000
                            ack_sent = True
                        scatter_timing: dict[str, float] = {}
                        scatter_start = time.perf_counter()
                        try:
                            staging_views = [self.allocator.view(lease) for _, lease, _ in ready_chunks]
                            scatter_entry_batches = [chunk.scatter_entries for chunk, _, _ in ready_chunks]
                            dst_regions = self.dst_regions or [(self.dst_base_addr, self.dst_tensor)]
                            unpack_from_staging_multi_batch(
                                staging_views,
                                dst_regions,
                                scatter_entry_batches,
                                timing=scatter_timing,
                            )
                        except Exception as exc:
                            first_error = f"batch scatter failed: {exc}"
                            logger.exception("D staging batch scatter failed: transfer_id=%s", transfer_id)
                        scatter_end = time.perf_counter()
                        scatter_total_ms = (scatter_end - scatter_start) * 1000
                        logger.info(
                            "D scatter breakdown: transfer=%s chunks=%d entries=%d bytes=%d total_ms=%.2f "
                            "descriptor_ms=%.2f descriptor_list_ms=%.2f descriptor_tensor_ms=%.2f "
                            "descriptor_prepare_ms=%.2f descriptor_alloc_ms=%.2f "
                            "descriptor_fill_ms=%.2f "
                            "swap_blocks_batch_ms=%.2f synchronize_ms=%.2f copy_stream_wait_ms=%.2f "
                            "slice_scatter_ms=%.2f unpack_total_ms=%.2f",
                            transfer_id,
                            len(ready_chunks),
                            sum(len(chunk.scatter_entries) for chunk, _, _ in ready_chunks),
                            sum(chunk.payload_bytes for chunk, _, _ in ready_chunks),
                            scatter_total_ms,
                            scatter_timing.get("descriptor_ms", 0.0),
                            scatter_timing.get("descriptor_list_ms", 0.0),
                            scatter_timing.get("descriptor_tensor_ms", 0.0),
                            scatter_timing.get("descriptor_prepare_ms", 0.0),
                            scatter_timing.get("descriptor_alloc_ms", 0.0),
                            scatter_timing.get("descriptor_fill_ms", 0.0),
                            scatter_timing.get("swap_blocks_batch_ms", 0.0),
                            scatter_timing.get("synchronize_ms", 0.0),
                            scatter_timing.get("copy_stream_wait_ms", 0.0),
                            scatter_timing.get("slice_scatter_ms", 0.0),
                            scatter_timing.get(
                                "unpack_from_staging_multi_batch_ms",
                                scatter_timing.get("unpack_from_staging_multi_ms", 0.0),
                            ),
                        )
                        for chunk, _, ready in ready_chunks:
                            scatter_timings[chunk.chunk_id] = (scatter_start, scatter_end)
                            if first_error is not None:
                                break
                            packed_bytes += chunk.payload_bytes
                            packed_entries += len(chunk.scatter_entries)
                            chunks_completed += 1

                if first_error is None and arena_failed:
                    extra = () if ready_chunks else direct_descriptors
                    fallback_error = self._direct_fallback_chunks(
                        arena_failed, direct_transfer, extra
                    )
                    if fallback_error is not None:
                        first_error = fallback_error
                    else:
                        fallback_rest = True
                        fallback_direct_bytes = sum(chunk.payload_bytes for chunk in arena_failed)
                        fallback_direct_entries = sum(
                            len(chunk.gather_entries) for chunk in arena_failed
                        )
                        if extra:
                            direct_completed = True

                if first_error is not None:
                    acked_ids = {item.chunk_id for item in ack_items}
                    for chunk in chunks:
                        ready = ready_by_chunk.get(chunk.chunk_id)
                        if ready is not None and ready.success and chunk.chunk_id not in acked_ids:
                            ack_items.append(
                                ReadAckBatchItem(
                                    chunk_id=chunk.chunk_id,
                                    lease_id=ready.lease_id,
                                    success=False,
                                )
                            )

            # Release every P-side slot that returned PackReady, including
            # chunks after the first error that were gathered successfully.
            ready_ids = {chunk_id for chunk_id, ready in ready_by_chunk.items() if ready.success}
            acked_ids = {item.chunk_id for item in ack_items}
            for chunk in chunks:
                if chunk.chunk_id in ready_ids and chunk.chunk_id not in acked_ids:
                    ready = ready_by_chunk[chunk.chunk_id]
                    ack_items.append(
                        ReadAckBatchItem(chunk_id=chunk.chunk_id, lease_id=ready.lease_id, success=False)
                    )
            if ack_items and not ack_sent:
                logger.info(
                    "D staging batch ACK send: transfer_id=%s chunks=%d success=%d failed=%d",
                    transfer_id,
                    len(ack_items),
                    sum(item.success for item in ack_items),
                    sum(not item.success for item in ack_items),
                )
                t_ack_start = time.perf_counter()
                send_ack_batch(ReadAckBatchMsg(transfer_id=transfer_id, results=ack_items))
                t_ack = time.perf_counter()
                ack_ms = (t_ack - t_ack_start) * 1000
            if ack_items:
                ack_by_chunk = {item.chunk_id: item for item in ack_items}
                for chunk in chunks:
                    ready = ready_by_chunk.get(chunk.chunk_id)
                    ack_item = ack_by_chunk.get(chunk.chunk_id)
                    p_gather_ms = getattr(ready, "gather_ms", 0.0) if ready is not None else 0.0
                    scatter_start, scatter_end = scatter_timings.get(chunk.chunk_id, (t_rdma, t_rdma))
                    scatter_ms = (scatter_end - scatter_start) * 1000 if ack_item and ack_item.success else 0.0
                    logger.info(
                        "D chunk timing: transfer=%s chunk=%d bytes=%d | "
                        "acquire=%.2fms prepare=%.2fms(p_gather=%.2fms) "
                        "rdma=%.2fms scatter=%.2fms ack=%.2fms total=%.2fms",
                        transfer_id,
                        chunk.chunk_id,
                        chunk.payload_bytes,
                        (t_acquire - t0) * 1000,
                        (t_prepare - t_acquire) * 1000,
                        p_gather_ms,
                        (t_rdma - t_prepare) * 1000,
                        scatter_ms,
                        ack_ms,
                        (max(t_ack, scatter_end) - t0) * 1000,
                    )
            logger.info(
                "D staging window timing: transfer_id=%s chunks=%d entries=%d bytes=%d "
                "acquire_ms=%.2f prepare_ms=%.2f rdma_ms=%.2f scatter_ms=%.2f "
                "ack_ms=%.2f total_ms=%.2f success=%s",
                transfer_id,
                len(chunks),
                sum(len(chunk.scatter_entries) for chunk in chunks),
                sum(chunk.payload_bytes for chunk in chunks),
                (t_acquire - t0) * 1000,
                (t_prepare - t_acquire) * 1000,
                (t_rdma - t_prepare) * 1000,
                scatter_total_ms,
                ack_ms,
                (time.perf_counter() - t0) * 1000,
                first_error is None,
            )
            return _WindowOutcome(
                error=first_error,
                packed_bytes=packed_bytes,
                packed_entries=packed_entries,
                chunks_completed=chunks_completed,
                direct_completed=direct_completed,
                fallback_rest=fallback_rest,
                direct_bytes=fallback_direct_bytes,
                direct_entries=fallback_direct_entries,
            )
        finally:
            for lease in leases:
                self.allocator.release(lease)

    @staticmethod
    def _send_abort(send_ack: SendAckFn, transfer_id: str, chunk_id: int, lease_id: int) -> None:
        """Release the P lease after a D-side RDMA failure.

        Scatter failures do not call this. A successful READ_ACK has already
        released the P lease, and a second ACK must not be required to finish
        the D-side copy.

        The P service treats both success and abort ACKs as terminal for the
        reserved slot.  A failed control message must not hide the original
        transfer failure, so it is logged and swallowed here.
        """
        try:
            send_ack(ReadAckMsg(transfer_id=transfer_id, chunk_id=chunk_id, lease_id=lease_id, success=False))
        except Exception:
            logger.exception(
                "Failed to send abort ACK: transfer=%s chunk=%d",
                transfer_id,
                chunk_id,
            )


__all__ = [
    "DecodeStagingCoordinator",
    "StagedTransferResult",
]
