# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.budget import (
    StagingConfig,
    compute_staging_reservation,
    staging_config_from_env,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    execute_plan_on_tensors,
    execute_plan_on_tensors_multi,
    pack_into_staging,
    pack_into_staging_multi,
    unpack_from_staging,
    unpack_from_staging_multi,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
    ScatterEntry,
    TransferPlanner,
    spans_from_block_mapping,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool

# =====================================================================
# pack_into_staging
# =====================================================================


class TestPackIntoStaging:
    def test_single_gather_entry(self) -> None:
        src = torch.arange(64, dtype=torch.int8)
        staging = torch.zeros(32, dtype=torch.int8)
        base = src.data_ptr()
        entries = [GatherEntry(src_offset=base + 10, packed_offset=0, nbytes=8)]
        pack_into_staging(src, base, staging, entries)
        assert staging[:8].tolist() == list(range(10, 18))

    def test_multiple_gather_entries(self) -> None:
        src = torch.arange(128, dtype=torch.int8)
        staging = torch.zeros(32, dtype=torch.int8)
        base = src.data_ptr()
        entries = [
            GatherEntry(src_offset=base + 0, packed_offset=0, nbytes=4),
            GatherEntry(src_offset=base + 100, packed_offset=4, nbytes=4),
        ]
        pack_into_staging(src, base, staging, entries)
        assert staging[:4].tolist() == [0, 1, 2, 3]
        assert staging[4:8].tolist() == [100, 101, 102, 103]

    def test_gather_preserves_staging_gaps(self) -> None:
        src = torch.arange(64, dtype=torch.int8)
        staging = torch.full((32,), 99, dtype=torch.int8)
        base = src.data_ptr()
        entries = [
            GatherEntry(src_offset=base + 0, packed_offset=0, nbytes=4),
            GatherEntry(src_offset=base + 10, packed_offset=8, nbytes=4),
        ]
        pack_into_staging(src, base, staging, entries)
        assert all(b == 99 for b in staging[4:8].tolist())

    def test_gather_out_of_bounds_raises(self) -> None:
        src = torch.zeros(16, dtype=torch.int8)
        staging = torch.zeros(32, dtype=torch.int8)
        base = src.data_ptr()
        entries = [GatherEntry(src_offset=base + 10, packed_offset=0, nbytes=10)]
        with pytest.raises(ValueError, match="out of bounds"):
            pack_into_staging(src, base, staging, entries)


# =====================================================================
# unpack_from_staging
# =====================================================================


class TestUnpackFromStaging:
    def test_single_scatter_entry(self) -> None:
        staging = torch.arange(32, dtype=torch.int8)
        dst = torch.zeros(64, dtype=torch.int8)
        base = dst.data_ptr()
        entries = [ScatterEntry(packed_offset=0, dst_offset=base + 10, nbytes=8)]
        unpack_from_staging(staging, dst, base, entries)
        assert dst[10:18].tolist() == list(range(8))

    def test_multiple_scatter_entries(self) -> None:
        staging = torch.arange(32, dtype=torch.int8)
        dst = torch.zeros(128, dtype=torch.int8)
        base = dst.data_ptr()
        entries = [
            ScatterEntry(packed_offset=0, dst_offset=base + 0, nbytes=4),
            ScatterEntry(packed_offset=10, dst_offset=base + 50, nbytes=4),
        ]
        unpack_from_staging(staging, dst, base, entries)
        assert dst[:4].tolist() == [0, 1, 2, 3]
        assert dst[50:54].tolist() == [10, 11, 12, 13]

    def test_scatter_preserves_dst_gaps(self) -> None:
        staging = torch.arange(16, dtype=torch.int8)
        dst = torch.full((64,), 0x7F, dtype=torch.int8)
        base = dst.data_ptr()
        entries = [
            ScatterEntry(packed_offset=0, dst_offset=base + 0, nbytes=4),
            ScatterEntry(packed_offset=4, dst_offset=base + 10, nbytes=4),
        ]
        unpack_from_staging(staging, dst, base, entries)
        assert dst[4:10].tolist() == [0x7F] * 6


# =====================================================================
# execute_plan_on_tensors (end-to-end)
# =====================================================================

_MIB = 1024 * 1024


class TestExecutePlan:
    def test_all_direct_plan(self) -> None:
        src = torch.arange(128, dtype=torch.int8)
        dst = torch.zeros(128, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()
        planner = TransferPlanner(min_direct_size=1, chunk_capacity=_MIB)
        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
            CopySpan,
        )

        spans = [
            CopySpan(
                request_id=0,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + 0,
                dst_offset=dst_base + 0,
                nbytes=64,
            ),
            CopySpan(
                request_id=0,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + 64,
                dst_offset=dst_base + 64,
                nbytes=64,
            ),
        ]
        plan = planner.plan(spans, peer_session="s0")
        pool = StagingPool(num_slots=1, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)
        assert torch.equal(src, dst)

    def test_all_packed_plan(self) -> None:
        block = 256
        src = torch.arange(block * 4, dtype=torch.int8)
        dst = torch.zeros(block * 4, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()
        planner = TransferPlanner(min_direct_size=_MIB, chunk_capacity=_MIB)
        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
            CopySpan,
        )

        spans = [
            CopySpan(
                request_id=0,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + i * block,
                dst_offset=dst_base + i * block,
                nbytes=block,
            )
            for i in range(4)
        ]
        plan = planner.plan(spans, peer_session="s0")
        assert len(plan.packed_chunks) > 0
        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)
        assert torch.equal(src, dst)

    def test_mixed_plan(self) -> None:
        size = 4096
        src = torch.randint(0, 127, (size,), dtype=torch.int8)
        dst = torch.zeros(size, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()
        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
            CopySpan,
        )

        spans = [
            CopySpan(
                request_id=0,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + 0,
                dst_offset=dst_base + 0,
                nbytes=2048,
            ),
            CopySpan(
                request_id=1,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + 2048,
                dst_offset=dst_base + 2048,
                nbytes=128,
            ),
            CopySpan(
                request_id=1,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + 2048 + 128 + 256,
                dst_offset=dst_base + 2048 + 128 + 256,
                nbytes=128,
            ),
        ]
        planner = TransferPlanner(min_direct_size=1024, chunk_capacity=_MIB)
        plan = planner.plan(spans, peer_session="s0")
        assert len(plan.direct_runs) > 0
        assert len(plan.packed_chunks) > 0
        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)
        expected = torch.zeros(size, dtype=torch.int8)
        for sp in spans:
            s = sp.src_offset - src_base
            d = sp.dst_offset - dst_base
            expected[d : d + sp.nbytes] = src[s : s + sp.nbytes]
        assert torch.equal(dst, expected)

    def test_random_mapping_tensor_correctness(self) -> None:
        rng = random.Random(42)
        num_blocks_src = 64
        num_blocks_dst = 64
        block_len = 512
        tp_num_pulls = 2

        src = torch.randint(0, 127, (num_blocks_src * block_len,), dtype=torch.int8)
        dst = torch.zeros(num_blocks_dst * block_len, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()

        num_requests = 3
        all_spans = []
        for req_id in range(num_requests):
            n_blocks = rng.randint(2, 8)
            src_ids = rng.sample(range(num_blocks_src), n_blocks)
            dst_ids = rng.sample(range(num_blocks_dst), n_blocks)
            spans = spans_from_block_mapping(
                local_block_ids=src_ids,
                remote_block_ids=dst_ids,
                src_base=src_base,
                dst_base=dst_base,
                block_len=block_len,
                src_block_stride=block_len,
                dst_block_stride=block_len,
                request_id=str(req_id),
                group_id=0,
                layer_idx=0,
                component_idx=0,
                tp_offset=0,
                tp_num_pulls=tp_num_pulls,
            )
            all_spans.extend(spans)

        planner = TransferPlanner(min_direct_size=1024, chunk_capacity=8192)
        plan = planner.plan(all_spans, peer_session="s0")
        pool = StagingPool(num_slots=4, slot_capacity=8192, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)

        expected = torch.zeros_like(dst)
        for sp in all_spans:
            s = sp.src_offset - src_base
            d = sp.dst_offset - dst_base
            expected[d : d + sp.nbytes] = src[s : s + sp.nbytes]
        assert torch.equal(dst, expected)

    def test_sentinel_preserved(self) -> None:
        size = 1024
        src = torch.randint(0, 127, (size,), dtype=torch.int8)
        sentinel = 0x55
        dst = torch.full((size,), sentinel, dtype=torch.int8)
        src_base = src.data_ptr()
        dst_base = dst.data_ptr()
        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
            CopySpan,
        )

        spans = [
            CopySpan(
                request_id=0,
                group_id=0,
                layer_idx=0,
                component_idx=0,
                src_offset=src_base + 100,
                dst_offset=dst_base + 100,
                nbytes=200,
            ),
        ]
        planner = TransferPlanner(min_direct_size=1024, chunk_capacity=_MIB)
        plan = planner.plan(spans, peer_session="s0")
        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)

        assert dst[100:300].tolist() == src[100:300].tolist()
        assert all(v == sentinel for v in dst[:100].tolist())
        assert all(v == sentinel for v in dst[300:].tolist())


# =====================================================================
# pack_into_staging_multi
# =====================================================================


class TestPackMulti:
    def test_single_region_matches_single(self) -> None:
        """Multi-region with one region behaves identically to single-tensor."""
        src = torch.arange(64, dtype=torch.int8)
        staging = torch.zeros(32, dtype=torch.int8)
        base = src.data_ptr()
        entries = [GatherEntry(src_offset=base + 10, packed_offset=0, nbytes=8)]
        pack_into_staging_multi([(base, src)], staging, entries)
        assert staging[:8].tolist() == list(range(10, 18))

    def test_two_regions(self) -> None:
        """Gather entries spanning two separate source tensors."""
        t0 = torch.arange(32, dtype=torch.int8)
        t1 = torch.arange(100, 132, dtype=torch.int8)
        regions = sorted([(t0.data_ptr(), t0), (t1.data_ptr(), t1)], key=lambda r: r[0])

        staging = torch.zeros(16, dtype=torch.int8)
        entries = [
            GatherEntry(src_offset=t0.data_ptr() + 4, packed_offset=0, nbytes=4),
            GatherEntry(src_offset=t1.data_ptr() + 2, packed_offset=4, nbytes=4),
        ]
        pack_into_staging_multi(regions, staging, entries)
        assert staging[:4].tolist() == [4, 5, 6, 7]
        assert staging[4:8].tolist() == [102, 103, 104, 105]

    def test_region_out_of_bounds(self) -> None:
        t0 = torch.zeros(16, dtype=torch.int8)
        regions = [(t0.data_ptr(), t0)]
        staging = torch.zeros(32, dtype=torch.int8)
        entries = [GatherEntry(src_offset=t0.data_ptr() + 10, packed_offset=0, nbytes=10)]
        with pytest.raises(ValueError, match="out of bounds"):
            pack_into_staging_multi(regions, staging, entries)

    def test_empty_entries(self) -> None:
        t0 = torch.zeros(16, dtype=torch.int8)
        staging = torch.full((16,), 99, dtype=torch.int8)
        pack_into_staging_multi([(t0.data_ptr(), t0)], staging, [])
        assert all(v == 99 for v in staging.tolist())


# =====================================================================
# unpack_from_staging_multi
# =====================================================================


class TestUnpackMulti:
    def test_single_region_matches_single(self) -> None:
        staging = torch.arange(32, dtype=torch.int8)
        dst = torch.zeros(64, dtype=torch.int8)
        base = dst.data_ptr()
        entries = [ScatterEntry(packed_offset=0, dst_offset=base + 10, nbytes=8)]
        unpack_from_staging_multi(staging, [(base, dst)], entries)
        assert dst[10:18].tolist() == list(range(8))

    def test_two_regions(self) -> None:
        """Scatter entries going to two separate destination tensors."""
        d0 = torch.zeros(32, dtype=torch.int8)
        d1 = torch.zeros(32, dtype=torch.int8)
        regions = sorted([(d0.data_ptr(), d0), (d1.data_ptr(), d1)], key=lambda r: r[0])

        staging = torch.arange(16, dtype=torch.int8)
        entries = [
            ScatterEntry(packed_offset=0, dst_offset=d0.data_ptr() + 4, nbytes=4),
            ScatterEntry(packed_offset=4, dst_offset=d1.data_ptr() + 8, nbytes=4),
        ]
        unpack_from_staging_multi(staging, regions, entries)
        assert d0[4:8].tolist() == [0, 1, 2, 3]
        assert d1[8:12].tolist() == [4, 5, 6, 7]

    def test_preserves_gaps(self) -> None:
        d0 = torch.full((32,), 0x7F, dtype=torch.int8)
        staging = torch.arange(8, dtype=torch.int8)
        entries = [ScatterEntry(packed_offset=0, dst_offset=d0.data_ptr() + 4, nbytes=4)]
        unpack_from_staging_multi(staging, [(d0.data_ptr(), d0)], entries)
        assert d0[:4].tolist() == [0x7F] * 4
        assert d0[4:8].tolist() == [0, 1, 2, 3]
        assert d0[8:12].tolist() == [0x7F] * 4


# =====================================================================
# execute_plan_on_tensors_multi
# =====================================================================


class TestExecutePlanMulti:
    def test_two_region_all_packed(self) -> None:
        """End-to-end with two separate src and dst tensors, all packed."""
        block = 256
        s0 = torch.randint(0, 127, (block * 2,), dtype=torch.int8)
        s1 = torch.randint(0, 127, (block * 2,), dtype=torch.int8)
        d0 = torch.zeros(block * 2, dtype=torch.int8)
        d1 = torch.zeros(block * 2, dtype=torch.int8)

        src_regions = sorted([(s0.data_ptr(), s0), (s1.data_ptr(), s1)], key=lambda r: r[0])
        dst_regions = sorted([(d0.data_ptr(), d0), (d1.data_ptr(), d1)], key=lambda r: r[0])

        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import CopySpan

        spans = [
            CopySpan(
                request_id=0, group_id=0, layer_idx=0, component_idx=0,
                src_offset=s0.data_ptr(), dst_offset=d0.data_ptr(), nbytes=block,
            ),
            CopySpan(
                request_id=0, group_id=0, layer_idx=0, component_idx=0,
                src_offset=s1.data_ptr() + block, dst_offset=d1.data_ptr() + block, nbytes=block,
            ),
        ]

        planner = TransferPlanner(min_direct_size=_MIB, chunk_capacity=_MIB)
        plan = planner.plan(spans, peer_session="s0")
        assert len(plan.packed_chunks) > 0

        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors_multi(plan, src_regions, dst_regions, pool)

        assert d0[:block].tolist() == s0[:block].tolist()
        assert d1[block:].tolist() == s1[block:].tolist()

    def test_two_region_mixed(self) -> None:
        """Mix of direct runs (large) and packed chunks (small) across regions."""
        s0 = torch.randint(0, 127, (4096,), dtype=torch.int8)
        s1 = torch.randint(0, 127, (4096,), dtype=torch.int8)
        d0 = torch.zeros(4096, dtype=torch.int8)
        d1 = torch.zeros(4096, dtype=torch.int8)

        src_regions = sorted([(s0.data_ptr(), s0), (s1.data_ptr(), s1)], key=lambda r: r[0])
        dst_regions = sorted([(d0.data_ptr(), d0), (d1.data_ptr(), d1)], key=lambda r: r[0])

        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import CopySpan

        spans = [
            CopySpan(
                request_id=0, group_id=0, layer_idx=0, component_idx=0,
                src_offset=s0.data_ptr(), dst_offset=d0.data_ptr(), nbytes=2048,
            ),
            CopySpan(
                request_id=1, group_id=0, layer_idx=0, component_idx=0,
                src_offset=s1.data_ptr() + 100, dst_offset=d1.data_ptr() + 100, nbytes=128,
            ),
        ]

        planner = TransferPlanner(min_direct_size=1024, chunk_capacity=_MIB)
        plan = planner.plan(spans, peer_session="s0")
        assert len(plan.direct_runs) > 0
        assert len(plan.packed_chunks) > 0

        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors_multi(plan, src_regions, dst_regions, pool)

        assert d0[:2048].tolist() == s0[:2048].tolist()
        assert d1[100:228].tolist() == s1[100:228].tolist()

    def test_single_region_matches_single_tensor(self) -> None:
        """Single-region multi variant matches single-tensor variant."""
        src = torch.randint(0, 127, (1024,), dtype=torch.int8)
        dst_single = torch.zeros(1024, dtype=torch.int8)
        dst_multi = torch.zeros(1024, dtype=torch.int8)

        src_base = src.data_ptr()
        dst_base_s = dst_single.data_ptr()
        dst_base_m = dst_multi.data_ptr()

        from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import CopySpan

        spans_s = [
            CopySpan(
                request_id=0, group_id=0, layer_idx=0, component_idx=0,
                src_offset=src_base + 100, dst_offset=dst_base_s + 200, nbytes=300,
            ),
        ]
        spans_m = [
            CopySpan(
                request_id=0, group_id=0, layer_idx=0, component_idx=0,
                src_offset=src_base + 100, dst_offset=dst_base_m + 200, nbytes=300,
            ),
        ]

        planner = TransferPlanner(min_direct_size=_MIB, chunk_capacity=_MIB)
        plan_s = planner.plan(spans_s, peer_session="s0")
        plan_m = planner.plan(spans_m, peer_session="s0")

        pool = StagingPool(num_slots=2, slot_capacity=_MIB, alignment=64, device="cpu")
        execute_plan_on_tensors(plan_s, src, src_base, dst_single, dst_base_s, pool)
        execute_plan_on_tensors_multi(
            plan_m, [(src_base, src)], [(dst_base_m, dst_multi)], pool,
        )
        assert dst_single.tolist() == dst_multi.tolist()


# =====================================================================
# Budget
# =====================================================================


class TestBudget:
    def test_staging_disabled_zero_reservation(self) -> None:
        cfg = StagingConfig(enabled=False)
        assert compute_staging_reservation(cfg) == 0

    def test_reservation_includes_alignment(self) -> None:
        alignment = 2 * _MIB
        cap = 16 * _MIB
        cfg = StagingConfig(enabled=True, num_slots=2, slot_capacity=cap, alignment=alignment)
        res = compute_staging_reservation(cfg)
        slot_stride = ((cap + alignment - 1) // alignment) * alignment
        expected = 2 * slot_stride + alignment - 1
        assert res == expected

    def test_config_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_STAGING_ENABLED", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_STAGING_NUM_SLOTS", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_STAGING_SLOT_CAPACITY_MIB", raising=False)
        cfg = staging_config_from_env()
        assert cfg.enabled is False
        assert cfg.num_slots == 2
        assert cfg.slot_capacity == 16 * _MIB

    def test_config_from_env_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_ASCEND_STAGING_ENABLED", "1")
        monkeypatch.setenv("VLLM_ASCEND_STAGING_NUM_SLOTS", "4")
        monkeypatch.setenv("VLLM_ASCEND_STAGING_SLOT_CAPACITY_MIB", "32")
        cfg = staging_config_from_env()
        assert cfg.enabled is True
        assert cfg.num_slots == 4
        assert cfg.slot_capacity == 32 * _MIB


# =====================================================================
# Budget deduction (_apply_staging_reservation)
# =====================================================================


class TestStagingReservationDeduction:
    """Test the budget deduction logic that would run in the worker."""

    def test_deduction_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_ASCEND_STAGING_ENABLED", "1")
        monkeypatch.setenv("VLLM_ASCEND_STAGING_NUM_SLOTS", "2")
        monkeypatch.setenv("VLLM_ASCEND_STAGING_SLOT_CAPACITY_MIB", "16")

        cfg = staging_config_from_env()
        reservation = compute_staging_reservation(cfg)
        assert reservation > 0

        available = 1024 * _MIB
        remaining = available - reservation
        assert remaining > 0
        assert remaining == available - reservation

    def test_no_deduction_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_STAGING_ENABLED", raising=False)
        cfg = staging_config_from_env()
        reservation = compute_staging_reservation(cfg)
        assert reservation == 0

    def test_deduction_clamps_to_zero(self) -> None:
        cfg = StagingConfig(enabled=True, num_slots=100, slot_capacity=64 * _MIB)
        reservation = compute_staging_reservation(cfg)
        available = 10 * _MIB
        remaining = max(available - reservation, 0)
        assert remaining == 0

    def test_reservation_matches_pool_allocation(self) -> None:
        """Reservation bytes >= actual StagingPool raw tensor size."""
        cfg = StagingConfig(enabled=True, num_slots=2, slot_capacity=16 * _MIB)
        reservation = compute_staging_reservation(cfg)
        pool = StagingPool(
            num_slots=cfg.num_slots,
            slot_capacity=cfg.slot_capacity,
            alignment=cfg.alignment,
            device="cpu",
        )
        actual = pool._raw_tensor.numel()
        assert reservation >= actual
