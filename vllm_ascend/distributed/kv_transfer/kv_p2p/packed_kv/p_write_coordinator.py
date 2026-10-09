# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Prefill-side staging coordinator for WRITE mode.

Orchestrates the full GATHER → PREPARE_WRITE → RDMA WRITE → WRITE_DONE
cycle for a ``TransferPlan``:

1. **Direct runs** are transferred via the TE as before (already large
   enough to trigger HCCS+RoCE dual-link splitting).
2. **Packed chunks** go through the staging protocol:
   P acquires local slot → gathers into it → PREPARE_WRITE → WRITE_READY
   (D slot addr) → RDMA WRITE (P slot → D slot) → WRITE_DONE → P releases.

Mirror of ``DecodeStagingCoordinator`` (which orchestrates READ mode
on D side).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Union

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    pack_into_staging,
    pack_into_staging_multi,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.d_coordinator import (
    StagedTransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
    PackedChunk,
    TransferPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PrepareWriteMsg,
    StagingErrorMsg,
    WriteDoneMsg,
    WriteReadyMsg,
)

logger = logging.getLogger(__name__)

DirectWriteFn = Callable[[list[int], list[int], list[int]], int]
"""(src_addrs, dst_addrs, lengths) → return code (0 = success)."""

PrepareWriteFn = Callable[[PrepareWriteMsg], Union[WriteReadyMsg, StagingErrorMsg]]  # noqa: UP007
"""Send PREPARE_WRITE to D, get WRITE_READY or error back."""

RdmaWriteFn = Callable[[int, int, int], int]
"""(local_src_addr, remote_dst_addr, nbytes) → return code."""

SendWriteDoneFn = Callable[[WriteDoneMsg], None]
"""Send WRITE_DONE to D after RDMA WRITE completes."""


class PrefillWriteCoordinator:
    """P-side orchestrator for staged KV cache WRITE transfers.

    Parameters
    ----------
    allocator : StagingAllocator
        P-side staging arena for gathering fragmented data.
    src_tensor : torch.Tensor
        Flat view of P-side KV cache.
    src_base_addr : int
        ``src_tensor.data_ptr()``.
    src_regions : list[tuple[int, torch.Tensor]] | None
        Multi-region KV cache: sorted list of (base_addr, flat_tensor).
    """

    def __init__(
        self,
        allocator: StagingAllocator,
        src_tensor: torch.Tensor,
        src_base_addr: int,
        src_regions: list[tuple[int, torch.Tensor]] | None = None,
        chunk_capacity: int | None = None,
    ) -> None:
        self.allocator = allocator
        self.src_tensor = src_tensor
        self.src_base_addr = src_base_addr
        self.src_regions = src_regions or []
        self.chunk_capacity = chunk_capacity or allocator.capacity_bytes

    def execute(
        self,
        plan: TransferPlan,
        transfer_id: str,
        *,
        direct_write: DirectWriteFn | None = None,
        prepare_write: PrepareWriteFn,
        rdma_write: RdmaWriteFn,
        send_write_done: SendWriteDoneFn,
    ) -> StagedTransferResult:
        """Execute a full TransferPlan via WRITE mode.

        Parameters
        ----------
        plan : TransferPlan
            Output of ``TransferPlanner.plan()``.
        transfer_id : str
            Unique identifier for this transfer.
        direct_write : callable, optional
            Batch RDMA write for direct runs.  If None, direct runs are
            skipped (caller handles them separately).
        prepare_write : callable
            Sends PREPARE_WRITE to D, returns WRITE_READY or error.
        rdma_write : callable
            RDMA write from P slot to D slot (single large entry).
        send_write_done : callable
            Sends WRITE_DONE to D after RDMA WRITE.
        """
        result = StagedTransferResult(success=True)

        if plan.direct_runs and direct_write is not None:
            ret = self._execute_direct_runs(plan, direct_write)
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
                prepare_write=prepare_write,
                rdma_write=rdma_write,
                send_write_done=send_write_done,
            )
            if err is not None:
                result.success = False
                result.error = err
                return result
            result.packed_bytes += chunk.payload_bytes
            result.packed_entries += len(chunk.gather_entries)
            result.chunks_completed += 1

        return result

    def _execute_direct_runs(self, plan: TransferPlan, direct_write: DirectWriteFn) -> int:
        src_addrs = [dr.src_offset for dr in plan.direct_runs]
        dst_addrs = [dr.dst_offset for dr in plan.direct_runs]
        lengths = [dr.nbytes for dr in plan.direct_runs]
        return direct_write(src_addrs, dst_addrs, lengths)

    def _execute_chunk(
        self,
        chunk: PackedChunk,
        transfer_id: str,
        prepare_write: PrepareWriteFn,
        rdma_write: RdmaWriteFn,
        send_write_done: SendWriteDoneFn,
    ) -> str | None:
        """Execute one packed chunk via WRITE mode. Returns error string or None."""
        if chunk.payload_bytes > self.allocator.capacity_bytes:
            return (
                f"chunk {chunk.chunk_id} exceeds P staging arena capacity: "
                f"payload={chunk.payload_bytes}, capacity={self.allocator.capacity_bytes}"
            )
        lease = self.allocator.allocate(chunk.payload_bytes)
        if lease is None:
            return f"no contiguous P staging extent for chunk {chunk.chunk_id}"

        try:
            staging_view = self.allocator.view(lease)
            try:
                if self.src_regions:
                    pack_into_staging_multi(self.src_regions, staging_view, chunk.gather_entries)
                else:
                    pack_into_staging(
                        self.src_tensor,
                        self.src_base_addr,
                        staging_view,
                        chunk.gather_entries,
                    )
            except Exception:
                raise

            prepare_msg = PrepareWriteMsg(
                transfer_id=transfer_id,
                chunk_id=chunk.chunk_id,
                scatter_entries=[
                    (s.dst_offset, s.packed_offset, s.nbytes) for s in chunk.scatter_entries
                ],
                total_bytes=chunk.payload_bytes,
            )

            response = prepare_write(prepare_msg)
            if isinstance(response, StagingErrorMsg):
                return f"D rejected PREPARE_WRITE for chunk {chunk.chunk_id}: {response.reason}"
            if response.payload_bytes != chunk.payload_bytes:
                self._send_abort(send_write_done, transfer_id, chunk.chunk_id, response.lease_id)
                return (
                    f"D returned wrong payload size for chunk {chunk.chunk_id}: "
                    f"expected={chunk.payload_bytes}, got={response.payload_bytes}"
                )

            p_staging_addr = self.allocator.address(lease)
            ret = rdma_write(p_staging_addr, response.staging_addr, chunk.payload_bytes)
            if ret != 0:
                self._send_abort(send_write_done, transfer_id, chunk.chunk_id, response.lease_id)
                return f"RDMA write failed for chunk {chunk.chunk_id}: ret={ret}"

            send_write_done(
                WriteDoneMsg(
                    transfer_id=transfer_id,
                    chunk_id=chunk.chunk_id,
                    lease_id=response.lease_id,
                    success=True,
                )
            )

            logger.debug(
                "P chunk done: transfer=%s chunk=%d bytes=%d",
                transfer_id,
                chunk.chunk_id,
                chunk.payload_bytes,
            )
        finally:
            self.allocator.release(lease)

        return None

    @staticmethod
    def _send_abort(send_write_done: SendWriteDoneFn, transfer_id: str, chunk_id: int, lease_id: int) -> None:
        """Notify D of failure so it can release its lease."""
        try:
            send_write_done(
                WriteDoneMsg(transfer_id=transfer_id, chunk_id=chunk_id, lease_id=lease_id, success=False)
            )
        except Exception:
            logger.exception(
                "Failed to send abort WRITE_DONE: transfer=%s chunk=%d",
                transfer_id,
                chunk_id,
            )


__all__ = [
    "PrefillWriteCoordinator",
]
