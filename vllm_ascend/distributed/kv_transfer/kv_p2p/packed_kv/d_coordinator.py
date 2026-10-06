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
   entry) → scatter from D slot → READ_ACK → release D slot.

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
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
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


class DecodeStagingCoordinator:
    """D-side orchestrator for staged KV cache transfers.

    Parameters
    ----------
    pool : StagingPool
        D-side staging pool for receiving packed data.
    dst_tensor : torch.Tensor
        Flat view of D-side KV cache.
    dst_base_addr : int
        ``dst_tensor.data_ptr()``.
    """

    def __init__(
        self,
        pool: StagingPool,
        dst_tensor: torch.Tensor,
        dst_base_addr: int,
        dst_regions: list[tuple[int, torch.Tensor]] | None = None,
    ) -> None:
        self.pool = pool
        self.dst_tensor = dst_tensor
        self.dst_base_addr = dst_base_addr
        self.dst_regions = dst_regions or []

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
            Sends READ_ACK to P after scatter.
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
            Maximum number of chunks in one batch. Defaults to the number of
            D-side staging slots.
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
            window_size = batch_window_size or self.pool.num_slots
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

        for packed_batch in packed_batches:
            if use_batch:
                batch_result = self._execute_batch(
                    packed_batch,
                    transfer_id=transfer_id,
                    prepare_read_batch=prepare_read_batch,
                    rdma_read=rdma_read,
                    rdma_read_batch=merged_read if direct_pending else rdma_read_batch,
                    send_ack_batch=send_ack_batch,
                    direct_descriptors=direct_descriptors if direct_pending else (),
                )
                err, packed_bytes, packed_entries, chunks_completed, direct_completed = batch_result
                result.packed_bytes += packed_bytes
                result.packed_entries += packed_entries
                result.chunks_completed += chunks_completed
                if err is not None:
                    result.success = False
                    result.error = err
                    return result
                if direct_completed:
                    result.direct_bytes = plan.total_direct_bytes
                    result.direct_entries = len(plan.direct_runs)
                    direct_pending = False
                continue

            chunk = packed_batch[0]
            err = self._execute_chunk(
                chunk,
                transfer_id=transfer_id,
                prepare_read=prepare_read,
                rdma_read=rdma_read,
                send_ack=send_ack,
                rdma_read_batch=merged_read if direct_pending else None,
                direct_descriptors=direct_descriptors if direct_pending else (),
            )
            if err is not None:
                result.success = False
                result.error = err
                return result
            result.packed_bytes += chunk.payload_bytes
            result.packed_entries += len(chunk.scatter_entries)
            result.chunks_completed += 1
            if direct_pending:
                result.direct_bytes = plan.total_direct_bytes
                result.direct_entries = len(plan.direct_runs)
                direct_pending = False

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

    def _execute_chunk(
        self,
        chunk: PackedChunk,
        transfer_id: str,
        prepare_read: PrepareReadFn,
        rdma_read: RdmaReadFn,
        send_ack: SendAckFn,
        rdma_read_batch: RdmaReadBatchFn | None = None,
        direct_descriptors: tuple[_ReadDescriptor, ...] = (),
    ) -> str | None:
        """Execute one packed chunk. Returns error string or None."""
        if chunk.payload_bytes > self.pool.slot_capacity:
            return (
                f"chunk {chunk.chunk_id} exceeds D staging slot capacity: "
                f"payload={chunk.payload_bytes}, capacity={self.pool.slot_capacity}"
            )

        t0 = time.perf_counter()
        slot = self.pool.acquire()
        if slot is None:
            return f"no D staging slot for chunk {chunk.chunk_id}"
        t_acquire = time.perf_counter()

        try:
            prepare_msg = PrepareReadMsg(
                transfer_id=transfer_id,
                chunk_id=chunk.chunk_id,
                gather_entries=[(g.src_offset, g.packed_offset, g.nbytes) for g in chunk.gather_entries],
                total_bytes=chunk.payload_bytes,
            )

            response = prepare_read(prepare_msg)
            t_prepare = time.perf_counter()

            if isinstance(response, StagingErrorMsg):
                return f"P rejected PREPARE_READ for chunk {chunk.chunk_id}: {response.reason}"
            if response.payload_bytes != chunk.payload_bytes:
                self._send_abort(send_ack, transfer_id, chunk.chunk_id)
                return (
                    f"P returned wrong payload size for chunk {chunk.chunk_id}: "
                    f"expected={chunk.payload_bytes}, got={response.payload_bytes}"
                )

            d_slot_addr = self.pool.slot_ptr(slot.slot_id)
            staging_descriptor = _ReadDescriptor(
                local_dst=d_slot_addr,
                remote_src=response.slot_addr,
                nbytes=chunk.payload_bytes,
                kind="staging",
                chunk_id=chunk.chunk_id,
            )
            if direct_descriptors:
                if rdma_read_batch is None:
                    self._send_abort(send_ack, transfer_id, chunk.chunk_id)
                    return f"merged RDMA callback unavailable for chunk {chunk.chunk_id}"
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
                ret = rdma_read(d_slot_addr, response.slot_addr, chunk.payload_bytes)
            t_rdma = time.perf_counter()

            if ret != 0:
                self._send_abort(send_ack, transfer_id, chunk.chunk_id)
                return f"RDMA read failed for chunk {chunk.chunk_id}: ret={ret}"

            try:
                staging_view = self.pool.slot_view(slot.slot_id)
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
                self._send_abort(send_ack, transfer_id, chunk.chunk_id)
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

            send_ack(ReadAckMsg(transfer_id=transfer_id, chunk_id=chunk.chunk_id, success=True))
            t_ack = time.perf_counter()

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
                (t_scatter - t_rdma) * 1000,
                (t_ack - t_scatter) * 1000,
                (t_ack - t0) * 1000,
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
                (t_scatter - t_rdma) * 1000,
                (t_ack - t_scatter) * 1000,
                (t_ack - t0) * 1000,
            )
        finally:
            self.pool.release(slot.slot_id)

        return None

    def _execute_batch(
        self,
        chunks: tuple[PackedChunk, ...],
        transfer_id: str,
        prepare_read_batch: PrepareReadBatchFn,
        rdma_read: RdmaReadFn,
        rdma_read_batch: RdmaReadBatchFn | None,
        send_ack_batch: SendAckBatchFn,
        direct_descriptors: tuple[_ReadDescriptor, ...] = (),
    ) -> tuple[str | None, int, int, int, bool]:
        """Execute one batch window with one RDMA submission and scatter."""
        t0 = time.perf_counter()
        slots = []
        for chunk in chunks:
            if chunk.payload_bytes > self.pool.slot_capacity:
                return (
                    f"chunk {chunk.chunk_id} exceeds D staging slot capacity: "
                    f"payload={chunk.payload_bytes}, capacity={self.pool.slot_capacity}",
                    0,
                    0,
                    0,
                    False,
                )
        for chunk in chunks:
            slot = self.pool.acquire()
            if slot is None:
                for acquired in slots:
                    self.pool.release(acquired.slot_id)
                return f"no D staging slot for batch window", 0, 0, 0, False
            slots.append(slot)
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
                ready_chunks: list[tuple[PackedChunk, Any, Any]] = []
                for chunk, slot in zip(chunks, slots):
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
                        ack_items.append(ReadAckBatchItem(chunk_id=chunk.chunk_id, success=False))
                        break

                    ready_chunks.append((chunk, slot, ready))

                if first_error is None and ready_chunks:
                    staging_descriptors = [
                        _ReadDescriptor(
                            local_dst=self.pool.slot_ptr(slot.slot_id),
                            remote_src=ready.slot_addr,
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
                        scatter_timing: dict[str, float] = {}
                        scatter_start = time.perf_counter()
                        try:
                            staging_views = [self.pool.slot_view(slot.slot_id) for _, slot, _ in ready_chunks]
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
                        for chunk, _, _ in ready_chunks:
                            scatter_timings[chunk.chunk_id] = (scatter_start, scatter_end)
                            if first_error is not None:
                                break
                            ack_items.append(ReadAckBatchItem(chunk_id=chunk.chunk_id, success=True))
                            packed_bytes += chunk.payload_bytes
                            packed_entries += len(chunk.scatter_entries)
                            chunks_completed += 1

                if first_error is not None:
                    acked_ids = {item.chunk_id for item in ack_items}
                    for chunk in chunks:
                        if chunk.chunk_id in ready_by_chunk and chunk.chunk_id not in acked_ids:
                            ack_items.append(ReadAckBatchItem(chunk_id=chunk.chunk_id, success=False))

            # Release every P-side slot that returned PackReady, including
            # chunks after the first error that were gathered successfully.
            ready_ids = set(ready_by_chunk)
            acked_ids = {item.chunk_id for item in ack_items}
            for chunk in chunks:
                if chunk.chunk_id in ready_ids and chunk.chunk_id not in acked_ids:
                    ack_items.append(ReadAckBatchItem(chunk_id=chunk.chunk_id, success=False))
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
                        (t_ack - t0) * 1000,
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
                (t_ack - t0) * 1000,
                first_error is None,
            )
            return first_error, packed_bytes, packed_entries, chunks_completed, direct_completed
        finally:
            for slot in slots:
                self.pool.release(slot.slot_id)

    @staticmethod
    def _send_abort(send_ack: SendAckFn, transfer_id: str, chunk_id: int) -> None:
        """Release the P slot after a D-side RDMA/scatter failure.

        The P service treats both success and abort ACKs as terminal for the
        reserved slot.  A failed control message must not hide the original
        transfer failure, so it is logged and swallowed here.
        """
        try:
            send_ack(ReadAckMsg(transfer_id=transfer_id, chunk_id=chunk_id, success=False))
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
