# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Pure-CPU transfer planner that packs fragmented KV block mappings."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024  # 16 MiB
DEFAULT_MIN_DIRECT_SIZE = int(
    os.getenv("VLLM_ASCEND_STAGING_MIN_DIRECT_SIZE", 1 * 1024 * 1024)
)
MAX_DESCRIPTORS_PER_CHUNK = 65536
PACK_ALIGNMENT = 64


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopySpan:
    request_id: str
    group_id: int
    layer_idx: int
    component_idx: int
    src_offset: int
    dst_offset: int
    nbytes: int


@dataclass(frozen=True)
class GatherEntry:
    src_offset: int
    packed_offset: int
    nbytes: int


@dataclass(frozen=True)
class ScatterEntry:
    packed_offset: int
    dst_offset: int
    nbytes: int


@dataclass(frozen=True)
class PackedChunk:
    chunk_id: int
    peer_session: str
    payload_bytes: int
    request_ids: frozenset[str]
    gather_entries: tuple[GatherEntry, ...]
    scatter_entries: tuple[ScatterEntry, ...]


@dataclass(frozen=True)
class DirectRun:
    src_offset: int
    dst_offset: int
    nbytes: int
    request_ids: frozenset[str]


@dataclass(frozen=True)
class TransferPlan:
    direct_runs: tuple[DirectRun, ...]
    packed_chunks: tuple[PackedChunk, ...]
    total_direct_bytes: int
    total_packed_bytes: int
    original_entry_count: int
    final_entry_count: int


# ---------------------------------------------------------------------------
# Helper: block mapping -> CopySpan
# ---------------------------------------------------------------------------


def spans_from_block_mapping(
    local_block_ids: list[int],
    remote_block_ids: list[int],
    src_base: int,
    dst_base: int,
    block_len: int,
    src_block_stride: int,
    dst_block_stride: int,
    request_id: str,
    group_id: int,
    layer_idx: int,
    component_idx: int,
    tp_offset: int = 0,
    tp_num_pulls: int = 1,
) -> list[CopySpan]:
    """Convert block ID mappings to byte-level CopySpan descriptors.

    Address formula from mooncake_connector.py:1012-1018:
      inner_block_len = block_len // tp_num_pulls
      src = src_base + block_id * src_block_stride + tp_offset * inner_block_len
      dst = dst_base + block_id * dst_block_stride
    """
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError(f"Block ID list length mismatch: local={len(local_block_ids)}, remote={len(remote_block_ids)}")
    if not local_block_ids:
        return []

    inner_block_len = block_len // tp_num_pulls
    tp_src_shift = tp_offset * inner_block_len

    return [
        CopySpan(
            request_id=request_id,
            group_id=group_id,
            layer_idx=layer_idx,
            component_idx=component_idx,
            src_offset=src_base + local_bid * src_block_stride + tp_src_shift,
            dst_offset=dst_base + remote_bid * dst_block_stride,
            nbytes=inner_block_len,
        )
        for local_bid, remote_bid in zip(local_block_ids, remote_block_ids)
    ]


# ---------------------------------------------------------------------------
# TransferPlanner
# ---------------------------------------------------------------------------


class TransferPlanner:
    def __init__(
        self,
        chunk_capacity: int = DEFAULT_CHUNK_BYTES,
        min_direct_size: int = DEFAULT_MIN_DIRECT_SIZE,
    ) -> None:
        if chunk_capacity <= 0:
            raise ValueError(f"chunk_capacity must be positive, got {chunk_capacity}")
        if min_direct_size <= 0:
            raise ValueError(f"min_direct_size must be positive, got {min_direct_size}")
        self._chunk_capacity = chunk_capacity
        self._min_direct_size = min_direct_size

    def plan(
        self,
        spans: list[CopySpan],
        peer_session: str,
    ) -> TransferPlan:
        if not spans:
            return TransferPlan(
                direct_runs=(),
                packed_chunks=(),
                total_direct_bytes=0,
                total_packed_bytes=0,
                original_entry_count=0,
                final_entry_count=0,
            )

        original_count = len(spans)
        merged = self._merge_contiguous(spans)
        direct_runs, fragments = self._classify(merged)
        packed_chunks = self._pack_fragments(fragments, peer_session)

        total_direct = sum(d.nbytes for d in direct_runs)
        total_packed = sum(c.payload_bytes for c in packed_chunks)
        final_count = len(direct_runs) + len(packed_chunks)

        return TransferPlan(
            direct_runs=tuple(direct_runs),
            packed_chunks=tuple(packed_chunks),
            total_direct_bytes=total_direct,
            total_packed_bytes=total_packed,
            original_entry_count=original_count,
            final_entry_count=final_count,
        )

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _merge_contiguous(spans: list[CopySpan]) -> list[CopySpan]:
        partitions: dict[tuple[int, int, int], list[CopySpan]] = defaultdict(list)
        for s in spans:
            partitions[(s.group_id, s.layer_idx, s.component_idx)].append(s)

        merged: list[CopySpan] = []
        for partition in partitions.values():
            partition.sort(key=lambda s: s.src_offset)
            cur = partition[0]
            for nxt in partition[1:]:
                if (
                    cur.src_offset + cur.nbytes == nxt.src_offset
                    and cur.dst_offset + cur.nbytes == nxt.dst_offset
                    and cur.request_id == nxt.request_id
                ):
                    cur = CopySpan(
                        request_id=cur.request_id,
                        group_id=cur.group_id,
                        layer_idx=cur.layer_idx,
                        component_idx=cur.component_idx,
                        src_offset=cur.src_offset,
                        dst_offset=cur.dst_offset,
                        nbytes=cur.nbytes + nxt.nbytes,
                    )
                else:
                    merged.append(cur)
                    cur = nxt
            merged.append(cur)
        return merged

    def _classify(self, merged: list[CopySpan]) -> tuple[list[DirectRun], list[CopySpan]]:
        direct_runs: list[DirectRun] = []
        fragments: list[CopySpan] = []
        for s in merged:
            if s.nbytes >= self._min_direct_size:
                direct_runs.append(
                    DirectRun(
                        src_offset=s.src_offset,
                        dst_offset=s.dst_offset,
                        nbytes=s.nbytes,
                        request_ids=frozenset({s.request_id}),
                    )
                )
            else:
                fragments.append(s)
        return direct_runs, fragments

    def _pack_fragments(
        self,
        fragments: list[CopySpan],
        peer_session: str,
    ) -> list[PackedChunk]:
        if not fragments:
            return []

        oversized = next((f for f in fragments if f.nbytes > self._chunk_capacity), None)
        if oversized is not None:
            raise ValueError(
                "Packed fragment exceeds staging slot capacity: "
                f"fragment_bytes={oversized.nbytes}, chunk_capacity={self._chunk_capacity}"
            )

        chunks: list[_ChunkBuilder] = [_ChunkBuilder(0, peer_session)]
        for frag in fragments:
            cur = chunks[-1]
            if cur.payload + frag.nbytes > self._chunk_capacity:
                cur = _ChunkBuilder(len(chunks), peer_session)
                chunks.append(cur)
            cur.add(frag)

        # Tail optimisation: if the last chunk is too small and the
        # second-to-last has room, redistribute tail spans.
        if len(chunks) >= 2 and chunks[-1].payload < self._min_direct_size:
            tail = chunks.pop()
            prev = chunks[-1]
            for frag, gather, scatter in zip(tail.frags, tail.gather, tail.scatter):
                if prev.payload + frag.nbytes <= self._chunk_capacity:
                    prev.add(frag)
                else:
                    tail2 = _ChunkBuilder(len(chunks), peer_session)
                    tail2.add(frag)
                    chunks.append(tail2)
                    prev = tail2

        return [c.build() for c in chunks]


class _ChunkBuilder:
    __slots__ = (
        "chunk_id",
        "peer_session",
        "payload",
        "request_ids",
        "frags",
        "gather",
        "scatter",
    )

    def __init__(self, chunk_id: int, peer_session: str) -> None:
        self.chunk_id = chunk_id
        self.peer_session = peer_session
        self.payload = 0
        self.request_ids: set[str] = set()
        self.frags: list[CopySpan] = []
        self.gather: list[GatherEntry] = []
        self.scatter: list[ScatterEntry] = []

    def add(self, frag: CopySpan) -> None:
        offset = self.payload
        self.gather.append(GatherEntry(src_offset=frag.src_offset, packed_offset=offset, nbytes=frag.nbytes))
        self.scatter.append(ScatterEntry(packed_offset=offset, dst_offset=frag.dst_offset, nbytes=frag.nbytes))
        self.payload += frag.nbytes
        self.request_ids.add(frag.request_id)
        self.frags.append(frag)

    def build(self) -> PackedChunk:
        return PackedChunk(
            chunk_id=self.chunk_id,
            peer_session=self.peer_session,
            payload_bytes=self.payload,
            request_ids=frozenset(self.request_ids),
            gather_entries=tuple(self.gather),
            scatter_entries=tuple(self.scatter),
        )


__all__ = [
    "CopySpan",
    "DirectRun",
    "GatherEntry",
    "PackedChunk",
    "ScatterEntry",
    "TransferPlan",
    "TransferPlanner",
    "spans_from_block_mapping",
]
