# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    CopySpan,
    TransferPlan,
    TransferPlanner,
    spans_from_block_mapping,
)

BLOCK_LEN = 4096
STRIDE = BLOCK_LEN
PEER = "peer-0"
MIB = 1024 * 1024


# =====================================================================
# spans_from_block_mapping
# =====================================================================


class TestSpansFromBlockMapping:
    def test_ascending_blocks(self) -> None:
        spans = spans_from_block_mapping(
            local_block_ids=[0, 1, 2],
            remote_block_ids=[10, 11, 12],
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        assert len(spans) == 3
        assert spans[0].src_offset == 0
        assert spans[0].dst_offset == 10 * STRIDE
        assert spans[1].src_offset == 1 * STRIDE
        assert spans[2].src_offset == 2 * STRIDE
        assert all(s.nbytes == BLOCK_LEN for s in spans)

    def test_descending_blocks(self) -> None:
        spans = spans_from_block_mapping(
            local_block_ids=[2, 1, 0],
            remote_block_ids=[12, 11, 10],
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        assert len(spans) == 3
        assert spans[0].src_offset == 2 * STRIDE
        assert spans[1].src_offset == 1 * STRIDE
        assert spans[2].src_offset == 0

    def test_empty_blocks(self) -> None:
        spans = spans_from_block_mapping(
            local_block_ids=[],
            remote_block_ids=[],
            src_base=1000,
            dst_base=2000,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        assert spans == []

    def test_block_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            spans_from_block_mapping(
                local_block_ids=[0, 1],
                remote_block_ids=[10],
                src_base=0,
                dst_base=0,
                block_len=BLOCK_LEN,
                src_block_stride=STRIDE,
                dst_block_stride=STRIDE,
                request_id="r1",
                group_id=0,
                layer_idx=0,
                component_idx=0,
            )

    def test_tp_offset(self) -> None:
        tp_num_pulls = 4
        inner = BLOCK_LEN // tp_num_pulls
        spans = spans_from_block_mapping(
            local_block_ids=[5],
            remote_block_ids=[20],
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
            tp_offset=2,
            tp_num_pulls=tp_num_pulls,
        )
        assert len(spans) == 1
        assert spans[0].src_offset == 5 * STRIDE + 2 * inner
        assert spans[0].dst_offset == 20 * STRIDE
        assert spans[0].nbytes == inner

    def test_stride_differs_from_len(self) -> None:
        big_stride = BLOCK_LEN * 2
        spans = spans_from_block_mapping(
            local_block_ids=[0, 1],
            remote_block_ids=[3, 4],
            src_base=100,
            dst_base=200,
            block_len=BLOCK_LEN,
            src_block_stride=big_stride,
            dst_block_stride=big_stride,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        assert spans[0].src_offset == 100
        assert spans[1].src_offset == 100 + big_stride
        assert spans[0].dst_offset == 200 + 3 * big_stride


# =====================================================================
# _merge_contiguous
# =====================================================================


class TestMergeContiguous:
    def _make_span(
        self,
        src: int,
        dst: int,
        nbytes: int,
        request_id: str = "r1",
        group_id: int = 0,
        layer_idx: int = 0,
        component_idx: int = 0,
    ) -> CopySpan:
        return CopySpan(
            request_id=request_id,
            group_id=group_id,
            layer_idx=layer_idx,
            component_idx=component_idx,
            src_offset=src,
            dst_offset=dst,
            nbytes=nbytes,
        )

    def test_merge_ascending_both_sides(self) -> None:
        spans = [
            self._make_span(0, 1000, 100),
            self._make_span(100, 1100, 100),
            self._make_span(200, 1200, 100),
        ]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 1
        assert merged[0].src_offset == 0
        assert merged[0].dst_offset == 1000
        assert merged[0].nbytes == 300

    def test_merge_descending_both_sides(self) -> None:
        spans = [
            self._make_span(200, 1200, 100),
            self._make_span(100, 1100, 100),
            self._make_span(0, 1000, 100),
        ]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 1
        assert merged[0].src_offset == 0
        assert merged[0].nbytes == 300

    def test_merge_one_side_gap(self) -> None:
        spans = [
            self._make_span(0, 1000, 100),
            self._make_span(100, 1300, 100),
        ]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 2

    def test_merge_different_groups(self) -> None:
        spans = [
            self._make_span(0, 1000, 100, group_id=0),
            self._make_span(100, 1100, 100, group_id=1),
        ]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 2

    def test_merge_different_components(self) -> None:
        spans = [
            self._make_span(0, 1000, 100, component_idx=0),
            self._make_span(100, 1100, 100, component_idx=1),
        ]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 2

    def test_merge_single_span(self) -> None:
        spans = [self._make_span(0, 1000, 100)]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 1
        assert merged[0].nbytes == 100

    def test_merge_with_stride_gap(self) -> None:
        spans = [
            self._make_span(0, 1000, 100),
            self._make_span(200, 1200, 100),
        ]
        merged = TransferPlanner._merge_contiguous(spans)
        assert len(merged) == 2


# =====================================================================
# _classify
# =====================================================================


class TestClassify:
    def _span(self, nbytes: int) -> CopySpan:
        return CopySpan("r1", 0, 0, 0, 0, 0, nbytes)

    def test_large_span_is_direct(self) -> None:
        planner = TransferPlanner(min_direct_size=MIB)
        direct, frags = planner._classify([self._span(2 * MIB)])
        assert len(direct) == 1
        assert len(frags) == 0

    def test_small_span_is_fragment(self) -> None:
        planner = TransferPlanner(min_direct_size=MIB)
        direct, frags = planner._classify([self._span(512)])
        assert len(direct) == 0
        assert len(frags) == 1

    def test_threshold_boundary(self) -> None:
        planner = TransferPlanner(min_direct_size=MIB)
        direct, frags = planner._classify([self._span(MIB)])
        assert len(direct) == 1
        assert len(frags) == 0


# =====================================================================
# _pack_fragments
# =====================================================================


class TestPackFragments:
    def _frag(
        self,
        src: int,
        dst: int,
        nbytes: int,
        request_id: str = "r1",
    ) -> CopySpan:
        return CopySpan(request_id, 0, 0, 0, src, dst, nbytes)

    def test_single_chunk(self) -> None:
        planner = TransferPlanner(chunk_capacity=1024, min_direct_size=512)
        frags = [self._frag(0, 100, 200), self._frag(300, 400, 200)]
        chunks = planner._pack_fragments(frags, PEER)
        assert len(chunks) == 1
        assert chunks[0].payload_bytes == 400

    def test_multi_chunk_split(self) -> None:
        planner = TransferPlanner(chunk_capacity=300, min_direct_size=100)
        frags = [self._frag(i * 100, i * 200, 100) for i in range(5)]
        chunks = planner._pack_fragments(frags, PEER)
        assert len(chunks) >= 2
        total = sum(c.payload_bytes for c in chunks)
        assert total == 500

    def test_chunk_offsets_sequential(self) -> None:
        planner = TransferPlanner(chunk_capacity=4096, min_direct_size=100)
        frags = [self._frag(i * 50, i * 80, 50) for i in range(10)]
        chunks = planner._pack_fragments(frags, PEER)
        for chunk in chunks:
            offsets = [g.packed_offset for g in chunk.gather_entries]
            for i in range(1, len(offsets)):
                prev_end = offsets[i - 1] + chunk.gather_entries[i - 1].nbytes
                assert offsets[i] == prev_end

    def test_gather_scatter_symmetric(self) -> None:
        planner = TransferPlanner(chunk_capacity=4096, min_direct_size=100)
        frags = [self._frag(i * 50, i * 80, 50) for i in range(5)]
        chunks = planner._pack_fragments(frags, PEER)
        for chunk in chunks:
            assert len(chunk.gather_entries) == len(chunk.scatter_entries)
            for g, s in zip(chunk.gather_entries, chunk.scatter_entries):
                assert g.packed_offset == s.packed_offset
                assert g.nbytes == s.nbytes

    def test_tail_redistribution(self) -> None:
        planner = TransferPlanner(chunk_capacity=300, min_direct_size=200)
        frags = [
            self._frag(0, 100, 250),
            self._frag(300, 400, 30),
        ]
        chunks = planner._pack_fragments(frags, PEER)
        assert len(chunks) == 1
        assert chunks[0].payload_bytes == 280

    def test_request_ids_tracked(self) -> None:
        planner = TransferPlanner(chunk_capacity=4096, min_direct_size=100)
        frags = [
            self._frag(0, 100, 50, request_id="r1"),
            self._frag(50, 200, 50, request_id="r2"),
        ]
        chunks = planner._pack_fragments(frags, PEER)
        assert len(chunks) == 1
        assert chunks[0].request_ids == frozenset({"r1", "r2"})


# =====================================================================
# plan() end-to-end
# =====================================================================


class TestPlanEndToEnd:
    def test_all_contiguous_all_direct(self) -> None:
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        spans = spans_from_block_mapping(
            local_block_ids=list(range(256)),
            remote_block_ids=list(range(256)),
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        plan = planner.plan(spans, PEER)
        assert len(plan.packed_chunks) == 0
        assert len(plan.direct_runs) >= 1
        assert plan.total_direct_bytes == 256 * BLOCK_LEN
        assert plan.total_packed_bytes == 0

    def test_all_fragmented_all_packed(self) -> None:
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        rng = random.Random(42)
        local_ids = list(range(64))
        remote_ids = list(range(64))
        rng.shuffle(local_ids)
        rng.shuffle(remote_ids)
        spans = spans_from_block_mapping(
            local_block_ids=local_ids,
            remote_block_ids=remote_ids,
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        plan = planner.plan(spans, PEER)
        assert plan.total_packed_bytes > 0
        assert plan.total_direct_bytes + plan.total_packed_bytes == 64 * BLOCK_LEN

    def test_mixed_direct_and_packed(self) -> None:
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        contiguous_count = MIB // BLOCK_LEN
        local_ids = list(range(contiguous_count))
        remote_ids = list(range(contiguous_count))
        rng = random.Random(99)
        scattered_local = list(range(contiguous_count, contiguous_count + 10))
        scattered_remote = list(range(contiguous_count, contiguous_count + 10))
        rng.shuffle(scattered_local)
        rng.shuffle(scattered_remote)
        all_local = local_ids + scattered_local
        all_remote = remote_ids + scattered_remote
        spans = spans_from_block_mapping(
            local_block_ids=all_local,
            remote_block_ids=all_remote,
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        plan = planner.plan(spans, PEER)
        assert len(plan.direct_runs) >= 1
        total = plan.total_direct_bytes + plan.total_packed_bytes
        assert total == len(all_local) * BLOCK_LEN

    def test_random_mapping_correctness(self) -> None:
        rng = random.Random(123)
        n_blocks = 32
        block_len = 128
        src_base = 0
        dst_base = n_blocks * block_len

        local_ids = list(range(n_blocks))
        remote_ids = list(range(n_blocks))
        rng.shuffle(local_ids)
        rng.shuffle(remote_ids)

        src_mem = bytearray(rng.getrandbits(8) for _ in range(n_blocks * block_len))
        dst_direct = bytearray(n_blocks * block_len)
        dst_staged = bytearray(n_blocks * block_len)

        for local_bid, remote_bid in zip(local_ids, remote_ids):
            s = src_base + local_bid * block_len
            d = remote_bid * block_len
            dst_direct[d : d + block_len] = src_mem[s : s + block_len]

        spans = spans_from_block_mapping(
            local_block_ids=local_ids,
            remote_block_ids=remote_ids,
            src_base=src_base,
            dst_base=dst_base,
            block_len=block_len,
            src_block_stride=block_len,
            dst_block_stride=block_len,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        planner = TransferPlanner(chunk_capacity=512, min_direct_size=256)
        plan = planner.plan(spans, PEER)

        for dr in plan.direct_runs:
            s = dr.src_offset
            d = dr.dst_offset - dst_base
            dst_staged[d : d + dr.nbytes] = src_mem[s : s + dr.nbytes]

        for chunk in plan.packed_chunks:
            staging = bytearray(chunk.payload_bytes)
            for g in chunk.gather_entries:
                staging[g.packed_offset : g.packed_offset + g.nbytes] = src_mem[g.src_offset : g.src_offset + g.nbytes]
            for sc in chunk.scatter_entries:
                d = sc.dst_offset - dst_base
                dst_staged[d : d + sc.nbytes] = staging[sc.packed_offset : sc.packed_offset + sc.nbytes]

        assert dst_staged == dst_direct

    def test_multi_request_same_peer(self) -> None:
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        spans_r1 = spans_from_block_mapping(
            local_block_ids=[0],
            remote_block_ids=[10],
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        spans_r2 = spans_from_block_mapping(
            local_block_ids=[5],
            remote_block_ids=[20],
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r2",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        plan = planner.plan(spans_r1 + spans_r2, PEER)
        all_request_ids: set[str] = set()
        for chunk in plan.packed_chunks:
            all_request_ids.update(chunk.request_ids)
        for dr in plan.direct_runs:
            all_request_ids.update(dr.request_ids)
        assert all_request_ids == {"r1", "r2"}

    def test_no_target_overlap(self) -> None:
        rng = random.Random(77)
        n = 40
        local_ids = list(range(n))
        remote_ids = list(range(n))
        rng.shuffle(local_ids)
        rng.shuffle(remote_ids)
        spans = spans_from_block_mapping(
            local_block_ids=local_ids,
            remote_block_ids=remote_ids,
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        plan = planner.plan(spans, PEER)

        dst_ranges: list[tuple[int, int]] = []
        for dr in plan.direct_runs:
            dst_ranges.append((dr.dst_offset, dr.dst_offset + dr.nbytes))
        for chunk in plan.packed_chunks:
            for sc in chunk.scatter_entries:
                dst_ranges.append((sc.dst_offset, sc.dst_offset + sc.nbytes))
        dst_ranges.sort()
        for i in range(1, len(dst_ranges)):
            assert dst_ranges[i][0] >= dst_ranges[i - 1][1]

    def test_coverage_complete(self) -> None:
        rng = random.Random(55)
        n = 20
        local_ids = list(range(n))
        remote_ids = list(range(n))
        rng.shuffle(local_ids)
        rng.shuffle(remote_ids)
        spans = spans_from_block_mapping(
            local_block_ids=local_ids,
            remote_block_ids=remote_ids,
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        plan = planner.plan(spans, PEER)

        total = plan.total_direct_bytes + plan.total_packed_bytes
        assert total == n * BLOCK_LEN


# =====================================================================
# Boundary tests
# =====================================================================


class TestBoundary:
    def _spans_of_total(self, total_bytes: int, per_span: int = 64) -> list[CopySpan]:
        count = total_bytes // per_span
        return [CopySpan("r1", 0, 0, 0, i * per_span * 3, i * per_span * 5, per_span) for i in range(count)]

    def test_capacity_minus_one(self) -> None:
        cap = 1024
        planner = TransferPlanner(chunk_capacity=cap, min_direct_size=cap + 1)
        frags = self._spans_of_total(cap - 1, per_span=63)
        plan = planner.plan(frags, PEER)
        assert len(plan.packed_chunks) == 1

    def test_capacity_exact(self) -> None:
        cap = 1024
        planner = TransferPlanner(chunk_capacity=cap, min_direct_size=cap + 1)
        frags = self._spans_of_total(cap, per_span=64)
        plan = planner.plan(frags, PEER)
        assert len(plan.packed_chunks) == 1
        assert plan.packed_chunks[0].payload_bytes == cap

    def test_capacity_plus_one(self) -> None:
        cap = 640
        per = 64
        count = cap // per + 1
        planner = TransferPlanner(chunk_capacity=cap, min_direct_size=cap + 1)
        spans = [CopySpan("r1", 0, 0, 0, i * per * 3, i * per * 5, per) for i in range(count)]
        plan = planner.plan(spans, PEER)
        assert len(plan.packed_chunks) == 2

    def test_zero_spans(self) -> None:
        planner = TransferPlanner()
        plan = planner.plan([], PEER)
        assert plan == TransferPlan(
            direct_runs=(),
            packed_chunks=(),
            total_direct_bytes=0,
            total_packed_bytes=0,
            original_entry_count=0,
            final_entry_count=0,
        )

    def test_single_block_single_span(self) -> None:
        planner = TransferPlanner(chunk_capacity=16 * MIB, min_direct_size=MIB)
        spans = spans_from_block_mapping(
            local_block_ids=[7],
            remote_block_ids=[3],
            src_base=0,
            dst_base=0,
            block_len=BLOCK_LEN,
            src_block_stride=STRIDE,
            dst_block_stride=STRIDE,
            request_id="r1",
            group_id=0,
            layer_idx=0,
            component_idx=0,
        )
        plan = planner.plan(spans, PEER)
        assert plan.original_entry_count == 1
        total = plan.total_direct_bytes + plan.total_packed_bytes
        assert total == BLOCK_LEN
