# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Gather/scatter copy backend for staging pool.

Provides two paths:
- **Batch DMA** (NPU): packs all entries into three descriptor tensors and
  issues a single ``swap_blocks_batch`` call with ``DIRECTION_D2D=2``.
  This dispatches to ``aclrtMemcpyBatchAsync`` (CANN 8.5+) or a
  per-entry ``aclrtMemcpyAsync`` loop on older CANN.
- **Tensor slice copy** (fallback): pure-torch ``Tensor.copy_()`` loop.
  Used when ``_C_ascend`` is not available (CPU tests, non-NPU devices).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
    ScatterEntry,
    TransferPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool

DIRECTION_D2D = 2

_has_batch_dma: bool | None = None


def _synchronize_batch_copy() -> None:
    """Wait for the async NPU DMA issued by ``swap_blocks_batch``.

    The Ascend operator is asynchronous with respect to the Python caller.
    Staging protocol messages must only be sent after the copy is visible to
    the peer, so synchronize the current stream before returning from a batch
    gather or scatter.

    TODO: replace the full-stream synchronization with a per-copy NPU event.
    The event can be recorded after ``swap_blocks_batch`` and waited on by the
    control path, preserving overlap with unrelated work on the same device.
    """
    npu = getattr(torch, "npu", None)
    if npu is not None:
        npu.current_stream().synchronize()


def _check_batch_dma() -> bool:
    global _has_batch_dma
    if _has_batch_dma is None:
        try:
            torch.ops._C_ascend.swap_blocks_batch  # noqa: B018
            _has_batch_dma = True
        except AttributeError:
            _has_batch_dma = False
    return _has_batch_dma


def _batch_gather(
    staging_base_addr: int,
    gather_entries: Sequence[GatherEntry],
) -> None:
    n = len(gather_entries)
    src_ptrs = torch.empty(n, dtype=torch.int64)
    dst_ptrs = torch.empty(n, dtype=torch.int64)
    sizes = torch.empty(n, dtype=torch.int64)
    for i, g in enumerate(gather_entries):
        src_ptrs[i] = g.src_offset
        dst_ptrs[i] = staging_base_addr + g.packed_offset
        sizes[i] = g.nbytes
    torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
    _synchronize_batch_copy()


def _batch_scatter(
    staging_base_addr: int,
    scatter_entries: Sequence[ScatterEntry],
) -> None:
    n = len(scatter_entries)
    src_ptrs = torch.empty(n, dtype=torch.int64)
    dst_ptrs = torch.empty(n, dtype=torch.int64)
    sizes = torch.empty(n, dtype=torch.int64)
    for i, s in enumerate(scatter_entries):
        src_ptrs[i] = staging_base_addr + s.packed_offset
        dst_ptrs[i] = s.dst_offset
        sizes[i] = s.nbytes
    torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
    _synchronize_batch_copy()


def _slice_gather(
    src: torch.Tensor,
    src_base_addr: int,
    staging: torch.Tensor,
    gather_entries: Sequence[GatherEntry],
) -> None:
    src_flat = src.view(-1)
    stg_flat = staging.view(-1)
    src_len = src_flat.shape[0]
    stg_len = stg_flat.shape[0]
    for g in gather_entries:
        rel = g.src_offset - src_base_addr
        if rel < 0 or rel + g.nbytes > src_len:
            raise ValueError(f"Gather src out of bounds: offset={rel}, nbytes={g.nbytes}, src_len={src_len}")
        if g.packed_offset < 0 or g.packed_offset + g.nbytes > stg_len:
            raise ValueError(
                f"Gather staging out of bounds: packed_offset={g.packed_offset}, nbytes={g.nbytes}, stg_len={stg_len}"
            )
        stg_flat[g.packed_offset : g.packed_offset + g.nbytes].copy_(src_flat[rel : rel + g.nbytes])


def _slice_scatter(
    staging: torch.Tensor,
    dst: torch.Tensor,
    dst_base_addr: int,
    scatter_entries: Sequence[ScatterEntry],
) -> None:
    stg_flat = staging.view(-1)
    dst_flat = dst.view(-1)
    stg_len = stg_flat.shape[0]
    dst_len = dst_flat.shape[0]
    for s in scatter_entries:
        rel = s.dst_offset - dst_base_addr
        if rel < 0 or rel + s.nbytes > dst_len:
            raise ValueError(f"Scatter dst out of bounds: offset={rel}, nbytes={s.nbytes}, dst_len={dst_len}")
        if s.packed_offset < 0 or s.packed_offset + s.nbytes > stg_len:
            raise ValueError(
                f"Scatter staging out of bounds: packed_offset={s.packed_offset}, nbytes={s.nbytes}, stg_len={stg_len}"
            )
        dst_flat[rel : rel + s.nbytes].copy_(stg_flat[s.packed_offset : s.packed_offset + s.nbytes])


def _slice_gather_multi(
    regions: Sequence[tuple[int, torch.Tensor]],
    staging: torch.Tensor,
    gather_entries: Sequence[GatherEntry],
) -> None:
    bases = [r[0] for r in regions]
    stg_flat = staging.view(-1)
    stg_len = stg_flat.shape[0]
    for g in gather_entries:
        idx = bisect.bisect_right(bases, g.src_offset) - 1
        if idx < 0:
            raise ValueError(f"Gather src address 0x{g.src_offset:x} below all regions")
        base, tensor = regions[idx]
        src_flat = tensor.view(-1)
        rel = g.src_offset - base
        if rel < 0 or rel + g.nbytes > src_flat.shape[0]:
            raise ValueError(
                f"Gather src out of bounds: region_base=0x{base:x}, "
                f"offset={rel}, nbytes={g.nbytes}, region_len={src_flat.shape[0]}"
            )
        if g.packed_offset < 0 or g.packed_offset + g.nbytes > stg_len:
            raise ValueError(
                f"Gather staging out of bounds: packed_offset={g.packed_offset}, nbytes={g.nbytes}, stg_len={stg_len}"
            )
        stg_flat[g.packed_offset : g.packed_offset + g.nbytes].copy_(src_flat[rel : rel + g.nbytes])


def _slice_scatter_multi(
    staging: torch.Tensor,
    regions: Sequence[tuple[int, torch.Tensor]],
    scatter_entries: Sequence[ScatterEntry],
) -> None:
    bases = [r[0] for r in regions]
    stg_flat = staging.view(-1)
    stg_len = stg_flat.shape[0]
    for s in scatter_entries:
        idx = bisect.bisect_right(bases, s.dst_offset) - 1
        if idx < 0:
            raise ValueError(f"Scatter dst address 0x{s.dst_offset:x} below all regions")
        base, tensor = regions[idx]
        dst_flat = tensor.view(-1)
        rel = s.dst_offset - base
        if rel < 0 or rel + s.nbytes > dst_flat.shape[0]:
            raise ValueError(
                f"Scatter dst out of bounds: region_base=0x{base:x}, "
                f"offset={rel}, nbytes={s.nbytes}, region_len={dst_flat.shape[0]}"
            )
        if s.packed_offset < 0 or s.packed_offset + s.nbytes > stg_len:
            raise ValueError(
                f"Scatter staging out of bounds: packed_offset={s.packed_offset}, nbytes={s.nbytes}, stg_len={stg_len}"
            )
        dst_flat[rel : rel + s.nbytes].copy_(stg_flat[s.packed_offset : s.packed_offset + s.nbytes])


def pack_into_staging(
    src: torch.Tensor,
    src_base_addr: int,
    staging: torch.Tensor,
    gather_entries: Sequence[GatherEntry],
) -> None:
    """Gather from KV cache regions into a contiguous staging buffer."""
    if not gather_entries:
        return
    if _check_batch_dma() and getattr(src, "is_npu", False):
        _batch_gather(staging.data_ptr(), gather_entries)
    else:
        _slice_gather(src, src_base_addr, staging, gather_entries)


def unpack_from_staging(
    staging: torch.Tensor,
    dst: torch.Tensor,
    dst_base_addr: int,
    scatter_entries: Sequence[ScatterEntry],
) -> None:
    """Scatter from staging buffer to KV cache regions."""
    if not scatter_entries:
        return
    if _check_batch_dma() and getattr(dst, "is_npu", False):
        _batch_scatter(staging.data_ptr(), scatter_entries)
    else:
        _slice_scatter(staging, dst, dst_base_addr, scatter_entries)


def pack_into_staging_multi(
    regions: Sequence[tuple[int, torch.Tensor]],
    staging: torch.Tensor,
    gather_entries: Sequence[GatherEntry],
) -> None:
    """Gather from multiple KV cache tensors into a contiguous staging buffer.

    ``regions`` is a list of ``(base_addr, tensor)`` pairs sorted by
    ``base_addr``.  Each ``GatherEntry.src_offset`` is matched to the
    region whose ``base_addr`` is closest-below via binary search.
    """
    if not gather_entries:
        return
    if _check_batch_dma() and getattr(staging, "is_npu", False):
        _batch_gather(staging.data_ptr(), gather_entries)
    else:
        _slice_gather_multi(regions, staging, gather_entries)


def unpack_from_staging_multi(
    staging: torch.Tensor,
    regions: Sequence[tuple[int, torch.Tensor]],
    scatter_entries: Sequence[ScatterEntry],
) -> None:
    """Scatter from staging buffer to multiple KV cache tensors.

    ``regions`` is a list of ``(base_addr, tensor)`` pairs sorted by
    ``base_addr``.  Each ``ScatterEntry.dst_offset`` is matched to the
    region whose ``base_addr`` is closest-below via binary search.
    """
    if not scatter_entries:
        return
    if _check_batch_dma() and getattr(staging, "is_npu", False):
        _batch_scatter(staging.data_ptr(), scatter_entries)
    else:
        _slice_scatter_multi(staging, regions, scatter_entries)


def execute_plan_on_tensors(
    plan: TransferPlan,
    src: torch.Tensor,
    src_base_addr: int,
    dst: torch.Tensor,
    dst_base_addr: int,
    staging_pool: StagingPool,
) -> None:
    """Execute a full TransferPlan using tensor copies on a single device.

    Direct runs copy straight from src to dst.  Packed chunks go through
    the staging pool: acquire slot -> gather -> (future: RDMA) -> scatter
    -> release slot.  Used for single-device byte-consistency testing.
    """
    use_dma = _check_batch_dma() and getattr(src, "is_npu", False)

    if plan.direct_runs:
        if use_dma:
            n = len(plan.direct_runs)
            src_ptrs = torch.empty(n, dtype=torch.int64)
            dst_ptrs = torch.empty(n, dtype=torch.int64)
            sizes = torch.empty(n, dtype=torch.int64)
            for i, dr in enumerate(plan.direct_runs):
                src_ptrs[i] = dr.src_offset
                dst_ptrs[i] = dr.dst_offset
                sizes[i] = dr.nbytes
            torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
        else:
            src_flat = src.view(-1)
            dst_flat = dst.view(-1)
            for dr in plan.direct_runs:
                s_rel = dr.src_offset - src_base_addr
                d_rel = dr.dst_offset - dst_base_addr
                dst_flat[d_rel : d_rel + dr.nbytes].copy_(src_flat[s_rel : s_rel + dr.nbytes])

    for chunk in plan.packed_chunks:
        slot = staging_pool.acquire()
        if slot is None:
            raise RuntimeError("No staging slot available")

        staging_view = staging_pool.slot_view(slot.slot_id)
        pack_into_staging(src, src_base_addr, staging_view, chunk.gather_entries)
        unpack_from_staging(staging_view, dst, dst_base_addr, chunk.scatter_entries)

        staging_pool.release(slot.slot_id)


def execute_plan_on_tensors_multi(
    plan: TransferPlan,
    src_regions: Sequence[tuple[int, torch.Tensor]],
    dst_regions: Sequence[tuple[int, torch.Tensor]],
    staging_pool: StagingPool,
) -> None:
    """Execute a full TransferPlan across multiple source/destination tensors.

    Multi-region variant of ``execute_plan_on_tensors``.  Each region is a
    ``(base_addr, tensor)`` pair sorted by ``base_addr``.  Direct runs use
    binary search to locate the correct tensor per entry; packed chunks use
    ``pack_into_staging_multi`` / ``unpack_from_staging_multi``.
    """
    first_tensor = src_regions[0][1] if src_regions else (dst_regions[0][1] if dst_regions else None)
    use_dma = _check_batch_dma() and first_tensor is not None and getattr(first_tensor, "is_npu", False)

    if plan.direct_runs:
        if use_dma:
            n = len(plan.direct_runs)
            src_ptrs = torch.empty(n, dtype=torch.int64)
            dst_ptrs = torch.empty(n, dtype=torch.int64)
            sizes = torch.empty(n, dtype=torch.int64)
            for i, dr in enumerate(plan.direct_runs):
                src_ptrs[i] = dr.src_offset
                dst_ptrs[i] = dr.dst_offset
                sizes[i] = dr.nbytes
            torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
        else:
            src_bases = [r[0] for r in src_regions]
            dst_bases = [r[0] for r in dst_regions]
            for dr in plan.direct_runs:
                si = bisect.bisect_right(src_bases, dr.src_offset) - 1
                di = bisect.bisect_right(dst_bases, dr.dst_offset) - 1
                s_base, s_tensor = src_regions[si]
                d_base, d_tensor = dst_regions[di]
                s_rel = dr.src_offset - s_base
                d_rel = dr.dst_offset - d_base
                d_tensor.view(-1)[d_rel : d_rel + dr.nbytes].copy_(s_tensor.view(-1)[s_rel : s_rel + dr.nbytes])

    for chunk in plan.packed_chunks:
        slot = staging_pool.acquire()
        if slot is None:
            raise RuntimeError("No staging slot available")

        staging_view = staging_pool.slot_view(slot.slot_id)
        pack_into_staging_multi(src_regions, staging_view, chunk.gather_entries)
        unpack_from_staging_multi(staging_view, dst_regions, chunk.scatter_entries)

        staging_pool.release(slot.slot_id)


__all__ = [
    "execute_plan_on_tensors",
    "execute_plan_on_tensors_multi",
    "pack_into_staging",
    "pack_into_staging_multi",
    "unpack_from_staging",
    "unpack_from_staging_multi",
]
