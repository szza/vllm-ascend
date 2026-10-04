# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Adapter between MooncakeConnectorV1 transfer metadata and the planner.

Two entry points:

* ``spans_from_flat_entries`` — wraps the connector's already-computed
  ``(src_list, dst_list, length_list)`` as ``CopySpan`` objects so the
  planner can classify and pack them.

* ``staged_transfer`` — end-to-end helper: take a ``TransferPlan``,
  execute direct runs via the TE, and for packed chunks do the full
  PREPARE_READ → PACK_READY → READ → SCATTER → ACK cycle.
  (Batch 4c/4d; currently only the span conversion is implemented.)
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    CopySpan,
    TransferPlan,
    TransferPlanner,
)


def spans_from_flat_entries(
    src_list: Sequence[int],
    dst_list: Sequence[int],
    length_list: Sequence[int],
    request_id: str,
    group_id: int = 0,
    layer_idx: int = 0,
) -> list[CopySpan]:
    """Convert the connector's flat transfer descriptor lists to CopySpans.

    The connector's ``_transfer_kv_cache_all_groups`` builds flat
    ``src_list / dst_list / length_list`` arrays across all groups, layers,
    and components for a single request.  This function wraps each entry as a
    ``CopySpan`` suitable for ``TransferPlanner.plan()``.

    Each entry gets a unique ``component_idx`` so the planner will not
    attempt to merge entries that the connector has already grouped (via
    ``group_concurrent_contiguous`` / ``split_if_not_byte_contiguous``).
    The planner's ``_merge_contiguous`` only merges spans with identical
    metadata AND byte-contiguous addresses, so these component indices are a
    safe no-op when addresses are non-contiguous.
    """
    if not (len(src_list) == len(dst_list) == len(length_list)):
        raise ValueError(f"List length mismatch: src={len(src_list)}, dst={len(dst_list)}, length={len(length_list)}")
    return [
        CopySpan(
            request_id=request_id,
            group_id=group_id,
            layer_idx=layer_idx,
            component_idx=i,
            src_offset=src,
            dst_offset=dst,
            nbytes=nbytes,
        )
        for i, (src, dst, nbytes) in enumerate(zip(src_list, dst_list, length_list))
    ]


def plan_from_flat_entries(
    src_list: Sequence[int],
    dst_list: Sequence[int],
    length_list: Sequence[int],
    request_id: str,
    peer_session: str,
    min_direct_size: int = 1024 * 1024,
    chunk_capacity: int = 16 * 1024 * 1024,
) -> TransferPlan:
    """One-shot: convert flat entries to CopySpans, plan, and return.

    Convenience wrapper combining ``spans_from_flat_entries`` with
    ``TransferPlanner.plan()`` for the common case of a single request's
    transfer descriptors.
    """
    spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id)
    planner = TransferPlanner(
        min_direct_size=min_direct_size,
        chunk_capacity=chunk_capacity,
    )
    return planner.plan(spans, peer_session=peer_session)


__all__ = [
    "plan_from_flat_entries",
    "spans_from_flat_entries",
]
