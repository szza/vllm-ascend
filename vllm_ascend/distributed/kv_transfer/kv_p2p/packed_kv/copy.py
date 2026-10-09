# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Gather/scatter copy backend for staging extents.

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
import time
from collections.abc import Sequence

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
    ScatterEntry,
    TransferPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator

DIRECTION_D2D = 2

_has_batch_dma: bool | None = None
_batch_copy_streams = {}


def _get_batch_copy_stream():
    """Return the stream used by packed-KV batch copies for this NPU device."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        return None

    device_idx = int(npu.current_device())
    stream = _batch_copy_streams.get(device_idx)
    if stream is None:
        stream = npu.Stream()
        _batch_copy_streams[device_idx] = stream
    return stream


def _run_batch_copy(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    timing: dict[str, float] | None = None,
    *,
    wait_current_stream: bool = False,
) -> None:
    """Submit and wait for one batch copy without synchronizing the default stream.

    The Ascend operator is asynchronous with respect to the Python caller.
    Staging protocol messages must only be sent after the copy is visible to
    the peer, so wait for a completion event before returning from a batch
    gather or scatter.  The event is recorded on a dedicated stream, which
    avoids waiting for unrelated work queued on the current/default stream.
    """
    npu = getattr(torch, "npu", None)
    if npu is None:
        torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
        return

    copy_stream = _get_batch_copy_stream()
    if copy_stream is None:
        torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
        return

    t_swap = time.perf_counter()
    current_stream = npu.current_stream() if wait_current_stream else None
    with torch.npu.stream(copy_stream):
        if current_stream is not None:
            # Gather reads KV data produced by the model stream.  Preserve
            # that dependency without synchronizing the host on the stream.
            copy_stream.wait_stream(current_stream)
        torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
        copy_done = torch.npu.Event()
        copy_done.record(copy_stream)
    if timing is not None:
        timing["swap_blocks_batch_ms"] = (time.perf_counter() - t_swap) * 1000

    t_wait = time.perf_counter()
    copy_done.synchronize()
    wait_ms = (time.perf_counter() - t_wait) * 1000
    if timing is not None:
        # Keep the old key so existing log parsers continue to work.  It now
        # measures only this batch's event, rather than a full current-stream
        # synchronization.
        timing["synchronize_ms"] = wait_ms
        timing["copy_stream_wait_ms"] = wait_ms


def _check_batch_dma() -> bool:
    global _has_batch_dma
    if _has_batch_dma is None:
        try:
            torch.ops._C_ascend.swap_blocks_batch  # noqa: B018
            _has_batch_dma = True
        except AttributeError:
            _has_batch_dma = False
    return _has_batch_dma


def _build_cpu_descriptor_tensors(
    src_addrs: Sequence[int],
    dst_addrs: Sequence[int],
    sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build host descriptor arrays in bulk for ``swap_blocks_batch``."""
    if not (len(src_addrs) == len(dst_addrs) == len(sizes)):
        raise ValueError(
            "descriptor arrays must have equal lengths: "
            f"src={len(src_addrs)}, dst={len(dst_addrs)}, sizes={len(sizes)}"
        )
    return (
        torch.tensor(src_addrs, dtype=torch.int64, device="cpu"),
        torch.tensor(dst_addrs, dtype=torch.int64, device="cpu"),
        torch.tensor(sizes, dtype=torch.int64, device="cpu"),
    )


def _batch_gather(
    staging_base_addr: int,
    gather_entries: Sequence[GatherEntry],
    timing: dict[str, float] | None = None,
) -> None:
    t_descriptor = time.perf_counter()
    src_addrs = [g.src_offset for g in gather_entries]
    dst_addrs = [staging_base_addr + g.packed_offset for g in gather_entries]
    sizes = [g.nbytes for g in gather_entries]
    t_descriptor_prepare = time.perf_counter()
    src_ptrs, dst_ptrs, sizes = _build_cpu_descriptor_tensors(src_addrs, dst_addrs, sizes)
    t_descriptor_alloc = time.perf_counter()
    if timing is not None:
        timing["descriptor_list_ms"] = (t_descriptor_prepare - t_descriptor) * 1000
        timing["descriptor_tensor_ms"] = (t_descriptor_alloc - t_descriptor_prepare) * 1000
        timing["descriptor_prepare_ms"] = (t_descriptor_prepare - t_descriptor) * 1000
        timing["descriptor_alloc_ms"] = (t_descriptor_alloc - t_descriptor_prepare) * 1000
        # Keep the legacy key, but define it as tensor construction only.
        timing["descriptor_fill_ms"] = (t_descriptor_alloc - t_descriptor_prepare) * 1000
        timing["descriptor_ms"] = (time.perf_counter() - t_descriptor) * 1000

    _run_batch_copy(src_ptrs, dst_ptrs, sizes, timing, wait_current_stream=True)


def _batch_scatter(
    staging_base_addr: int,
    scatter_entries: Sequence[ScatterEntry],
    timing: dict[str, float] | None = None,
) -> None:
    _batch_scatter_multi(((staging_base_addr, scatter_entries),), timing)


def _batch_scatter_multi(
    scatter_batches: Sequence[tuple[int, Sequence[ScatterEntry]]],
    timing: dict[str, float] | None = None,
) -> None:
    """Submit scatter entries from multiple staging slots in one DMA batch."""
    t_descriptor = time.perf_counter()
    src_addrs: list[int] = []
    dst_addrs: list[int] = []
    size_values: list[int] = []
    for staging_base_addr, scatter_entries in scatter_batches:
        for s in scatter_entries:
            src_addrs.append(staging_base_addr + s.packed_offset)
            dst_addrs.append(s.dst_offset)
            size_values.append(s.nbytes)
    t_descriptor_prepare = time.perf_counter()
    src_ptrs, dst_ptrs, sizes = _build_cpu_descriptor_tensors(src_addrs, dst_addrs, size_values)
    t_descriptor_alloc = time.perf_counter()

    if timing is not None:
        timing["descriptor_list_ms"] = (t_descriptor_prepare - t_descriptor) * 1000
        timing["descriptor_tensor_ms"] = (t_descriptor_alloc - t_descriptor_prepare) * 1000
        timing["descriptor_prepare_ms"] = (t_descriptor_prepare - t_descriptor) * 1000
        timing["descriptor_alloc_ms"] = (t_descriptor_alloc - t_descriptor_prepare) * 1000
        timing["descriptor_fill_ms"] = (t_descriptor_alloc - t_descriptor_prepare) * 1000
        timing["descriptor_ms"] = (time.perf_counter() - t_descriptor) * 1000

    _run_batch_copy(src_ptrs, dst_ptrs, sizes, timing)


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
    timing: dict[str, float] | None = None,
) -> None:
    """Gather from KV cache regions into a contiguous staging buffer."""
    if not gather_entries:
        return
    t_total = time.perf_counter()
    if _check_batch_dma() and getattr(src, "is_npu", False):
        _batch_gather(staging.data_ptr(), gather_entries, timing)
    else:
        t_slice = time.perf_counter()
        _slice_gather(src, src_base_addr, staging, gather_entries)
        if timing is not None:
            timing["slice_gather_ms"] = (time.perf_counter() - t_slice) * 1000
    if timing is not None:
        timing["pack_into_staging_ms"] = (time.perf_counter() - t_total) * 1000


def unpack_from_staging(
    staging: torch.Tensor,
    dst: torch.Tensor,
    dst_base_addr: int,
    scatter_entries: Sequence[ScatterEntry],
    timing: dict[str, float] | None = None,
) -> None:
    """Scatter from staging buffer to KV cache regions."""
    if not scatter_entries:
        return
    t_total = time.perf_counter()
    if _check_batch_dma() and getattr(dst, "is_npu", False):
        _batch_scatter(staging.data_ptr(), scatter_entries, timing)
    else:
        t_slice = time.perf_counter()
        _slice_scatter(staging, dst, dst_base_addr, scatter_entries)
        if timing is not None:
            timing["slice_scatter_ms"] = (time.perf_counter() - t_slice) * 1000
    if timing is not None:
        timing["unpack_from_staging_ms"] = (time.perf_counter() - t_total) * 1000


def pack_into_staging_multi(
    regions: Sequence[tuple[int, torch.Tensor]],
    staging: torch.Tensor,
    gather_entries: Sequence[GatherEntry],
    timing: dict[str, float] | None = None,
) -> None:
    """Gather from multiple KV cache tensors into a contiguous staging buffer.

    ``regions`` is a list of ``(base_addr, tensor)`` pairs sorted by
    ``base_addr``.  Each ``GatherEntry.src_offset`` is matched to the
    region whose ``base_addr`` is closest-below via binary search.
    """
    if not gather_entries:
        return
    t_total = time.perf_counter()
    if _check_batch_dma() and getattr(staging, "is_npu", False):
        _batch_gather(staging.data_ptr(), gather_entries, timing)
    else:
        t_slice = time.perf_counter()
        _slice_gather_multi(regions, staging, gather_entries)
        if timing is not None:
            timing["slice_gather_ms"] = (time.perf_counter() - t_slice) * 1000
    if timing is not None:
        timing["pack_into_staging_multi_ms"] = (time.perf_counter() - t_total) * 1000


def unpack_from_staging_multi(
    staging: torch.Tensor,
    regions: Sequence[tuple[int, torch.Tensor]],
    scatter_entries: Sequence[ScatterEntry],
    timing: dict[str, float] | None = None,
) -> None:
    """Scatter from staging buffer to multiple KV cache tensors.

    ``regions`` is a list of ``(base_addr, tensor)`` pairs sorted by
    ``base_addr``.  Each ``ScatterEntry.dst_offset`` is matched to the
    region whose ``base_addr`` is closest-below via binary search.
    """
    if not scatter_entries:
        return
    t_total = time.perf_counter()
    if _check_batch_dma() and getattr(staging, "is_npu", False):
        _batch_scatter(staging.data_ptr(), scatter_entries, timing)
    else:
        t_slice = time.perf_counter()
        _slice_scatter_multi(staging, regions, scatter_entries)
        if timing is not None:
            timing["slice_scatter_ms"] = (time.perf_counter() - t_slice) * 1000
    if timing is not None:
        timing["unpack_from_staging_multi_ms"] = (time.perf_counter() - t_total) * 1000


def unpack_from_staging_multi_batch(
    staging_views: Sequence[torch.Tensor],
    regions: Sequence[tuple[int, torch.Tensor]],
    scatter_entry_batches: Sequence[Sequence[ScatterEntry]],
    timing: dict[str, float] | None = None,
) -> None:
    """Scatter multiple staging slots into KV regions with one DMA batch."""
    if len(staging_views) != len(scatter_entry_batches):
        raise ValueError(
            "staging_views and scatter_entry_batches must have the same length: "
            f"views={len(staging_views)}, batches={len(scatter_entry_batches)}"
        )
    if not staging_views or not any(scatter_entry_batches):
        return

    t_total = time.perf_counter()
    use_dma = _check_batch_dma() and all(getattr(view, "is_npu", False) for view in staging_views)
    if use_dma:
        _batch_scatter_multi(
            tuple(
                (staging.data_ptr(), scatter_entries)
                for staging, scatter_entries in zip(staging_views, scatter_entry_batches)
                if scatter_entries
            ),
            timing,
        )
    else:
        t_slice = time.perf_counter()
        for staging, scatter_entries in zip(staging_views, scatter_entry_batches):
            if scatter_entries:
                _slice_scatter_multi(staging, regions, scatter_entries)
        if timing is not None:
            timing["slice_scatter_ms"] = (time.perf_counter() - t_slice) * 1000
    if timing is not None:
        timing["unpack_from_staging_multi_batch_ms"] = (time.perf_counter() - t_total) * 1000


def execute_plan_on_tensors(
    plan: TransferPlan,
    src: torch.Tensor,
    src_base_addr: int,
    dst: torch.Tensor,
    dst_base_addr: int,
    allocator: StagingAllocator,
) -> None:
    """Execute a full TransferPlan using tensor copies on a single device.

    Direct runs copy straight from src to dst.  Packed chunks go through
    the staging allocator: allocate extent -> gather -> scatter -> release.
    Used for single-device byte-consistency testing.
    """
    use_dma = _check_batch_dma() and getattr(src, "is_npu", False)

    if plan.direct_runs:
        if use_dma:
            src_ptrs, dst_ptrs, sizes = _build_cpu_descriptor_tensors(
                [dr.src_offset for dr in plan.direct_runs],
                [dr.dst_offset for dr in plan.direct_runs],
                [dr.nbytes for dr in plan.direct_runs],
            )
            torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, DIRECTION_D2D)
        else:
            src_flat = src.view(-1)
            dst_flat = dst.view(-1)
            for dr in plan.direct_runs:
                s_rel = dr.src_offset - src_base_addr
                d_rel = dr.dst_offset - dst_base_addr
                dst_flat[d_rel : d_rel + dr.nbytes].copy_(src_flat[s_rel : s_rel + dr.nbytes])

    for chunk in plan.packed_chunks:
        lease = allocator.allocate(chunk.payload_bytes)
        if lease is None:
            raise RuntimeError("No staging extent available")
        try:
            staging_view = allocator.view(lease)
            pack_into_staging(src, src_base_addr, staging_view, chunk.gather_entries)
            unpack_from_staging(staging_view, dst, dst_base_addr, chunk.scatter_entries)
        finally:
            allocator.release(lease)


def execute_plan_on_tensors_multi(
    plan: TransferPlan,
    src_regions: Sequence[tuple[int, torch.Tensor]],
    dst_regions: Sequence[tuple[int, torch.Tensor]],
    allocator: StagingAllocator,
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
            src_ptrs, dst_ptrs, sizes = _build_cpu_descriptor_tensors(
                [dr.src_offset for dr in plan.direct_runs],
                [dr.dst_offset for dr in plan.direct_runs],
                [dr.nbytes for dr in plan.direct_runs],
            )
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
        lease = allocator.allocate(chunk.payload_bytes)
        if lease is None:
            raise RuntimeError("No staging extent available")
        try:
            staging_view = allocator.view(lease)
            pack_into_staging_multi(src_regions, staging_view, chunk.gather_entries)
            unpack_from_staging_multi(staging_view, dst_regions, chunk.scatter_entries)
        finally:
            allocator.release(lease)


__all__ = [
    "execute_plan_on_tensors",
    "execute_plan_on_tensors_multi",
    "pack_into_staging",
    "pack_into_staging_multi",
    "unpack_from_staging",
    "unpack_from_staging_multi",
    "unpack_from_staging_multi_batch",
]
