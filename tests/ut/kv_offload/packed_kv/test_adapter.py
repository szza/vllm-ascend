# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.adapter import (
    plan_from_flat_entries,
    spans_from_flat_entries,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    execute_plan_on_tensors,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    TransferPlanner,
    spans_from_block_mapping,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool

_MIB = 1024 * 1024


# =====================================================================
# spans_from_flat_entries
# =====================================================================


class TestSpansFromFlatEntries:
    def test_basic_conversion(self) -> None:
        src = [1000, 2000, 3000]
        dst = [5000, 6000, 7000]
        lengths = [64, 128, 256]
        spans = spans_from_flat_entries(src, dst, lengths, request_id="r0")
        assert len(spans) == 3
        assert spans[0].src_offset == 1000
        assert spans[0].dst_offset == 5000
        assert spans[0].nbytes == 64
        assert spans[0].request_id == "r0"
        assert spans[0].component_idx == 0
        assert spans[1].component_idx == 1
        assert spans[2].component_idx == 2

    def test_empty_lists(self) -> None:
        spans = spans_from_flat_entries([], [], [], request_id="r0")
        assert spans == []

    def test_single_entry(self) -> None:
        spans = spans_from_flat_entries([100], [200], [50], request_id="r1")
        assert len(spans) == 1
        assert spans[0].src_offset == 100
        assert spans[0].dst_offset == 200
        assert spans[0].nbytes == 50

    def test_custom_group_layer(self) -> None:
        spans = spans_from_flat_entries(
            [100],
            [200],
            [50],
            request_id="r2",
            group_id=3,
            layer_idx=7,
        )
        assert spans[0].group_id == 3
        assert spans[0].layer_idx == 7

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            spans_from_flat_entries([1, 2], [3], [4, 5], request_id="r0")

    def test_unique_component_idx(self) -> None:
        """Each entry gets a unique component_idx to prevent false merges."""
        n = 100
        spans = spans_from_flat_entries(
            list(range(0, n * 100, 100)),
            list(range(0, n * 200, 200)),
            [64] * n,
            request_id="r0",
        )
        indices = [s.component_idx for s in spans]
        assert indices == list(range(n))


# =====================================================================
# plan_from_flat_entries
# =====================================================================


class TestPlanFromFlatEntries:
    def test_all_small_entries_packed(self) -> None:
        """Entries smaller than min_direct_size go into packed chunks."""
        n = 10
        src = [i * 1000 for i in range(n)]
        dst = [i * 2000 for i in range(n)]
        lengths = [512] * n
        plan = plan_from_flat_entries(
            src,
            dst,
            lengths,
            request_id="r0",
            peer_session="p0",
            min_direct_size=_MIB,
            chunk_capacity=_MIB,
        )
        assert len(plan.packed_chunks) > 0
        assert plan.total_packed_bytes == 512 * n

    def test_all_large_entries_direct(self) -> None:
        """Entries >= min_direct_size become direct runs."""
        n = 3
        entry_size = 2 * _MIB
        src = [i * entry_size * 2 for i in range(n)]
        dst = [i * entry_size * 2 for i in range(n)]
        lengths = [entry_size] * n
        plan = plan_from_flat_entries(
            src,
            dst,
            lengths,
            request_id="r0",
            peer_session="p0",
            min_direct_size=_MIB,
        )
        assert len(plan.direct_runs) == n
        assert len(plan.packed_chunks) == 0
        assert plan.total_direct_bytes == entry_size * n

    def test_mixed_entries(self) -> None:
        """Mix of large and small entries."""
        plan = plan_from_flat_entries(
            [0, 10_000_000, 20_000_000],
            [100_000_000, 110_000_000, 120_000_000],
            [2 * _MIB, 512, 2 * _MIB],
            request_id="r0",
            peer_session="p0",
            min_direct_size=_MIB,
        )
        assert len(plan.direct_runs) == 2
        assert len(plan.packed_chunks) == 1

    def test_empty_entries(self) -> None:
        plan = plan_from_flat_entries([], [], [], request_id="r0", peer_session="p0")
        assert len(plan.direct_runs) == 0
        assert len(plan.packed_chunks) == 0


# =====================================================================
# Adapter ↔ Planner equivalence
# =====================================================================


class TestAdapterPlannerEquivalence:
    def test_adapter_spans_match_block_mapping_spans(self) -> None:
        """The adapter's flat-entry spans produce the same addresses as
        spans_from_block_mapping when given the same block mapping."""
        num_blocks = 16
        block_len = 512
        tp_num_pulls = 2
        inner_block_len = block_len // tp_num_pulls
        src_base = 0x1000_0000
        dst_base = 0x2000_0000

        rng = random.Random(99)
        n = 5
        src_ids = rng.sample(range(num_blocks), n)
        dst_ids = rng.sample(range(num_blocks), n)

        # Method 1: spans_from_block_mapping (planner API)
        planner_spans = spans_from_block_mapping(
            local_block_ids=src_ids,
            remote_block_ids=dst_ids,
            src_base=src_base,
            dst_base=dst_base,
            block_len=block_len,
            src_block_stride=block_len,
            dst_block_stride=block_len,
            request_id="r0",
            group_id=0,
            layer_idx=0,
            component_idx=0,
            tp_offset=0,
            tp_num_pulls=tp_num_pulls,
        )

        # Method 2: compute flat lists as connector does, then adapter
        src_list = []
        dst_list = []
        length_list = []
        for s_id, d_id in zip(src_ids, dst_ids):
            src_list.append(src_base + s_id * block_len + 0 * inner_block_len)
            dst_list.append(dst_base + d_id * block_len)
            length_list.append(inner_block_len)
        adapter_spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")

        # Both should produce the same (src_offset, dst_offset, nbytes) tuples
        planner_addrs = sorted((s.src_offset, s.dst_offset, s.nbytes) for s in planner_spans)
        adapter_addrs = sorted((s.src_offset, s.dst_offset, s.nbytes) for s in adapter_spans)
        assert planner_addrs == adapter_addrs


# =====================================================================
# End-to-end byte correctness via adapter
# =====================================================================


class TestAdapterEndToEnd:
    def test_flat_entries_to_plan_to_copy(self) -> None:
        """Full pipeline: flat entries → adapter → planner → execute →
        byte-exact result, simulating what the connector would do."""
        rng = random.Random(42)
        num_blocks = 32
        block_len = 512
        tp_num_pulls = 1
        inner_block_len = block_len // tp_num_pulls

        src = torch.randint(0, 127, (num_blocks * block_len,), dtype=torch.int8)
        sentinel = 0x55
        dst = torch.full((num_blocks * block_len,), sentinel, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()

        # Simulate what the connector does: build flat lists from block mapping
        n = 8
        src_ids = rng.sample(range(num_blocks), n)
        dst_ids = rng.sample(range(num_blocks), n)

        src_list = []
        dst_list = []
        length_list = []
        for s_id, d_id in zip(src_ids, dst_ids):
            src_list.append(src_base + s_id * block_len)
            dst_list.append(dst_base + d_id * block_len)
            length_list.append(inner_block_len)

        # Adapter → Planner → Execute
        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=2048, chunk_capacity=8192)
        plan = planner.plan(spans, peer_session="p0")
        pool = StagingPool(num_slots=4, slot_capacity=8192, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)

        # Verify byte correctness
        expected = torch.full((num_blocks * block_len,), sentinel, dtype=torch.int8)
        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - src_base
            d_rel = d_addr - dst_base
            expected[d_rel : d_rel + length] = src[s_rel : s_rel + length]

        assert torch.equal(dst, expected)

    def test_multi_request_flat_entries(self) -> None:
        """Multiple requests' flat entries combined into a single plan."""
        rng = random.Random(77)
        num_blocks = 64
        block_len = 256

        src = torch.randint(0, 127, (num_blocks * block_len,), dtype=torch.int8)
        dst = torch.zeros(num_blocks * block_len, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()

        all_spans = []
        all_entries: list[tuple[int, int, int]] = []

        for req_id in range(3):
            n = rng.randint(3, 8)
            src_ids = rng.sample(range(num_blocks), n)
            dst_ids = rng.sample(range(num_blocks), n)

            src_list = [src_base + s_id * block_len for s_id in src_ids]
            dst_list = [dst_base + d_id * block_len for d_id in dst_ids]
            length_list = [block_len] * n

            spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id=str(req_id))
            all_spans.extend(spans)
            all_entries.extend(zip(src_list, dst_list, length_list))

        planner = TransferPlanner(min_direct_size=1024, chunk_capacity=4096)
        plan = planner.plan(all_spans, peer_session="p0")
        pool = StagingPool(num_slots=4, slot_capacity=4096, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)

        # Verify
        expected = torch.zeros_like(dst)
        for s_addr, d_addr, length in all_entries:
            s_rel = s_addr - src_base
            d_rel = d_addr - dst_base
            expected[d_rel : d_rel + length] = src[s_rel : s_rel + length]

        assert torch.equal(dst, expected)

    def test_sentinel_preserved_with_adapter(self) -> None:
        """Non-transferred regions keep their sentinel value."""
        num_blocks = 16
        block_len = 512
        sentinel = 99

        src = torch.randint(0, 127, (num_blocks * block_len,), dtype=torch.int8)
        dst = torch.full((num_blocks * block_len,), sentinel, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()

        # Transfer only blocks 2,5 → 3,7
        src_list = [
            src_base + 2 * block_len,
            src_base + 5 * block_len,
        ]
        dst_list = [
            dst_base + 3 * block_len,
            dst_base + 7 * block_len,
        ]
        length_list = [block_len, block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=_MIB, chunk_capacity=_MIB)
        plan = planner.plan(spans, peer_session="p0")
        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)

        # Check transferred blocks match
        assert dst[3 * block_len : 4 * block_len].tolist() == src[2 * block_len : 3 * block_len].tolist()
        assert dst[7 * block_len : 8 * block_len].tolist() == src[5 * block_len : 6 * block_len].tolist()

        # Check non-transferred blocks still have sentinel
        for b in range(num_blocks):
            if b not in (3, 7):
                block_data = dst[b * block_len : (b + 1) * block_len].tolist()
                assert all(v == sentinel for v in block_data), f"Block {b} was modified"

    def test_plan_from_flat_entries_convenience(self) -> None:
        """plan_from_flat_entries produces a usable plan."""
        plan = plan_from_flat_entries(
            src_list=[0, 1000, 2000],
            dst_list=[5000, 6000, 7000],
            length_list=[512, 512, 512],
            request_id="r0",
            peer_session="p0",
            min_direct_size=_MIB,
            chunk_capacity=_MIB,
        )
        assert plan.original_entry_count == 3
        assert plan.total_packed_bytes == 512 * 3
        assert len(plan.packed_chunks) > 0

    def test_entry_count_reduction(self) -> None:
        """The planner packs many small entries into fewer chunks."""
        n = 50
        src_list = [i * 1000 for i in range(n)]
        dst_list = [i * 2000 for i in range(n)]
        length_list = [256] * n

        plan = plan_from_flat_entries(
            src_list,
            dst_list,
            length_list,
            request_id="r0",
            peer_session="p0",
            min_direct_size=_MIB,
            chunk_capacity=_MIB,
        )
        assert plan.original_entry_count == n
        assert plan.final_entry_count < n
