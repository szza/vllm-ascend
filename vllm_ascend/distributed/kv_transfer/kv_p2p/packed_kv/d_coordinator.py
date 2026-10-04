# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Decode-side staging coordinator.

Orchestrates the full READY → READ → SCATTER → ACK cycle for a
``TransferPlan``:

1. **Direct runs** are transferred via the TE as before (already large
   enough to trigger HCCS+RoCE dual-link splitting).
2. **Packed chunks** go through the staging protocol:
   acquire D slot → PREPARE_READ → PACK_READY → RDMA read (one large
   entry) → scatter from D slot → READ_ACK → release D slot.

The coordinator is transport-agnostic: the actual RDMA read and
P-service communication are supplied via callbacks so the module can
be tested in-process without ZMQ or a real TransferEngine.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    unpack_from_staging,
    unpack_from_staging_multi,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    PackedChunk,
    TransferPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PackReadyMsg,
    PrepareReadMsg,
    ReadAckMsg,
    StagingErrorMsg,
)

logger = logging.getLogger(__name__)

DirectTransferFn = Callable[[list[int], list[int], list[int]], int]
"""(src_addrs, dst_addrs, lengths) → return code (0 = success)."""

PrepareReadFn = Callable[[PrepareReadMsg], Union[PackReadyMsg, StagingErrorMsg]]  # noqa: UP007
"""Send PREPARE_READ to P, get PACK_READY or error back."""

RdmaReadFn = Callable[[int, int, int], int]
"""(local_dst_addr, remote_src_addr, nbytes) → return code."""

SendAckFn = Callable[[ReadAckMsg], None]
"""Send READ_ACK to P after scatter is done."""


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
        """
        result = StagedTransferResult(success=True)

        if plan.direct_runs and direct_transfer is not None:
            ret = self._execute_direct_runs(plan, direct_transfer)
            if ret != 0:
                result.success = False
                result.error = f"direct transfer failed: ret={ret}"
                return result
            result.direct_bytes = plan.total_direct_bytes
            result.direct_entries = len(plan.direct_runs)

        for chunk in plan.packed_chunks:
            err = self._execute_chunk(
                chunk,
                transfer_id=transfer_id,
                prepare_read=prepare_read,
                rdma_read=rdma_read,
                send_ack=send_ack,
            )
            if err is not None:
                result.success = False
                result.error = err
                return result
            result.packed_bytes += chunk.payload_bytes
            result.packed_entries += len(chunk.scatter_entries)
            result.chunks_completed += 1

        return result

    def _execute_direct_runs(self, plan: TransferPlan, direct_transfer: DirectTransferFn) -> int:
        src_addrs = [dr.src_offset for dr in plan.direct_runs]
        dst_addrs = [dr.dst_offset for dr in plan.direct_runs]
        lengths = [dr.nbytes for dr in plan.direct_runs]
        return direct_transfer(src_addrs, dst_addrs, lengths)

    def _execute_chunk(
        self,
        chunk: PackedChunk,
        transfer_id: str,
        prepare_read: PrepareReadFn,
        rdma_read: RdmaReadFn,
        send_ack: SendAckFn,
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
            ret = rdma_read(d_slot_addr, response.slot_addr, chunk.payload_bytes)
            t_rdma = time.perf_counter()

            if ret != 0:
                self._send_abort(send_ack, transfer_id, chunk.chunk_id)
                return f"RDMA read failed for chunk {chunk.chunk_id}: ret={ret}"

            try:
                staging_view = self.pool.slot_view(slot.slot_id)
                if self.dst_regions:
                    unpack_from_staging_multi(
                        staging_view,
                        self.dst_regions,
                        chunk.scatter_entries,
                    )
                else:
                    unpack_from_staging(
                        staging_view,
                        self.dst_tensor,
                        self.dst_base_addr,
                        chunk.scatter_entries,
                    )
            except Exception:
                self._send_abort(send_ack, transfer_id, chunk.chunk_id)
                raise
            t_scatter = time.perf_counter()

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
        finally:
            self.pool.release(slot.slot_id)

        return None

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
