# SPDX-License-Identifier: Apache-2.0
"""Integration tests: D-coordinator + P-service with simulated RDMA.

The "RDMA" step is a plain memcpy from P's staging slot to D's staging
slot, which is exactly what the real TransferEngine does (just over the
network).  This lets us verify the full protocol flow on CPU.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.adapter import (
    spans_from_flat_entries,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    execute_plan_on_tensors,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.d_coordinator import (
    DecodeStagingCoordinator,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.p_service import (
    PrefillStagingService,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    TransferPlanner,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PackReadyBatchMsg,
    PackReadyMsg,
    PrepareReadBatchItem,
    PrepareReadBatchMsg,
    PrepareReadMsg,
    ReadAckBatchMsg,
    ReadAckMsg,
    StagingErrorMsg,
)


def _make_rdma_read_fn(p_pool: StagingPool, d_pool: StagingPool):
    """Create a simulated RDMA read that copies between staging pools.

    In production, this is ``batch_transfer_sync_read(D_slot, P_slot, size)``
    over HCCS+RoCE.  Here we simulate it with a tensor memcpy.
    """
    p_buf = p_pool._pool_view
    d_buf = d_pool._pool_view
    p_base = p_pool.base_ptr
    d_base = d_pool.base_ptr

    def rdma_read(d_addr: int, p_addr: int, nbytes: int) -> int:
        p_off = p_addr - p_base
        d_off = d_addr - d_base
        d_buf.view(-1)[d_off : d_off + nbytes].copy_(p_buf.view(-1)[p_off : p_off + nbytes])
        return 0

    return rdma_read


def _make_direct_transfer_fn(src_tensor: torch.Tensor, src_base: int, dst_tensor: torch.Tensor, dst_base: int):
    """Simulated direct RDMA for large runs (no staging)."""

    def direct_transfer(src_addrs: list[int], dst_addrs: list[int], lengths: list[int]) -> int:
        src_flat = src_tensor.view(-1)
        dst_flat = dst_tensor.view(-1)
        for s, d, n in zip(src_addrs, dst_addrs, lengths):
            s_rel = s - src_base
            d_rel = d - dst_base
            dst_flat[d_rel : d_rel + n].copy_(src_flat[s_rel : s_rel + n])
        return 0

    return direct_transfer


_MIB = 1024 * 1024


class _Fixture:
    """Shared setup for P+D staged transfer tests."""

    def __init__(
        self,
        num_blocks: int = 32,
        block_len: int = 512,
        num_slots: int = 2,
        slot_capacity: int = 8192,
        min_direct_size: int = 2048,
        chunk_capacity: int = 8192,
        sentinel: int = 99,
    ):
        self.num_blocks = num_blocks
        self.block_len = block_len
        self.sentinel = sentinel
        self.min_direct_size = min_direct_size
        self.chunk_capacity = chunk_capacity

        self.p_kv = torch.randint(0, 127, (num_blocks * block_len,), dtype=torch.int8)
        self.d_kv = torch.full((num_blocks * block_len,), sentinel, dtype=torch.int8)
        self.p_base = self.p_kv.data_ptr()
        self.d_base = self.d_kv.data_ptr()

        self.p_pool = StagingPool(num_slots=num_slots, slot_capacity=slot_capacity, alignment=64, device="cpu")
        self.d_pool = StagingPool(num_slots=num_slots, slot_capacity=slot_capacity, alignment=64, device="cpu")

        self.p_service = PrefillStagingService(
            pool=self.p_pool,
            kv_tensor=self.p_kv,
            kv_base_addr=self.p_base,
        )

        self.d_coordinator = DecodeStagingCoordinator(
            pool=self.d_pool,
            dst_tensor=self.d_kv,
            dst_base_addr=self.d_base,
        )

        self.rdma_read = _make_rdma_read_fn(self.p_pool, self.d_pool)
        self.direct_transfer = _make_direct_transfer_fn(self.p_kv, self.p_base, self.d_kv, self.d_base)

        self.ack_log: list[ReadAckMsg] = []

    def prepare_read_fn(self, msg: PrepareReadMsg) -> PackReadyMsg | StagingErrorMsg:
        return self.p_service.handle_prepare_read(msg)

    def send_ack_fn(self, msg: ReadAckMsg) -> None:
        self.p_service.handle_read_ack(msg)
        self.ack_log.append(msg)


# =====================================================================
# Full P+D staged transfer
# =====================================================================


class TestStagedTransfer:
    def test_single_chunk_byte_exact(self) -> None:
        """Single packed chunk: P gather → RDMA → D scatter → byte match."""
        f = _Fixture()
        rng = random.Random(42)
        n = 4
        src_ids = rng.sample(range(f.num_blocks), n)
        dst_ids = rng.sample(range(f.num_blocks), n)

        src_list = [f.p_base + s * f.block_len for s in src_ids]
        dst_list = [f.d_base + d * f.block_len for d in dst_ids]
        length_list = [f.block_len] * n

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-001",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        assert result.chunks_completed > 0

        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_multi_chunk_serial(self) -> None:
        """Multiple chunks processed serially with 1-slot pools."""
        f = _Fixture(num_slots=1, slot_capacity=2048, chunk_capacity=2048)
        rng = random.Random(77)
        n = 8
        src_ids = rng.sample(range(f.num_blocks), n)
        dst_ids = rng.sample(range(f.num_blocks), n)

        src_list = [f.p_base + s * f.block_len for s in src_ids]
        dst_list = [f.d_base + d * f.block_len for d in dst_ids]
        length_list = [f.block_len] * n

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        assert len(plan.packed_chunks) > 1

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-002",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        assert result.chunks_completed == len(plan.packed_chunks)

        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_multi_chunk_batch_uses_one_rdma_submission_per_window(self) -> None:
        """Batch prepare and batch RDMA preserve data and window boundaries."""
        f = _Fixture(num_slots=2, slot_capacity=2048, chunk_capacity=2048)
        rng = random.Random(78)
        n = 8
        src_ids = rng.sample(range(f.num_blocks), n)
        dst_ids = rng.sample(range(f.num_blocks), n)

        src_list = [f.p_base + s * f.block_len for s in src_ids]
        dst_list = [f.d_base + d * f.block_len for d in dst_ids]
        length_list = [f.block_len] * n
        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r-batch")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")
        assert len(plan.packed_chunks) > 1

        rdma_batch_calls: list[int] = []

        def rdma_read_batch(d_addrs: list[int], p_addrs: list[int], lengths: list[int]) -> int:
            rdma_batch_calls.append(len(lengths))
            for d_addr, p_addr, nbytes in zip(d_addrs, p_addrs, lengths):
                assert f.rdma_read(d_addr, p_addr, nbytes) == 0
            return 0

        def send_ack_batch(msg: ReadAckBatchMsg) -> None:
            for item in msg.results:
                assert f.p_service.handle_read_ack(
                    ReadAckMsg(transfer_id=msg.transfer_id, chunk_id=item.chunk_id, success=item.success)
                )

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-batch",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
            prepare_read_batch=f.p_service.handle_prepare_read_batch,
            send_ack_batch=send_ack_batch,
            rdma_read_batch=rdma_read_batch,
        )

        assert result.success
        assert result.chunks_completed == len(plan.packed_chunks)
        assert rdma_batch_calls == [2] * ((len(plan.packed_chunks) + 1) // 2)
        assert f.p_service.active_slot_count == 0
        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_sentinel_preserved(self) -> None:
        """Non-transferred blocks keep sentinel value."""
        f = _Fixture()
        src_list = [f.p_base + 2 * f.block_len, f.p_base + 5 * f.block_len]
        dst_list = [f.d_base + 3 * f.block_len, f.d_base + 7 * f.block_len]
        length_list = [f.block_len, f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-003",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        for b in range(f.num_blocks):
            block_data = f.d_kv[b * f.block_len : (b + 1) * f.block_len].tolist()
            if b in (3, 7):
                continue
            assert all(v == f.sentinel for v in block_data), f"Block {b} was modified"

    def test_mixed_direct_and_packed(self) -> None:
        """Large runs go direct, small fragments go through staging."""
        f = _Fixture(
            num_blocks=64,
            block_len=1024,
            slot_capacity=16384,
            min_direct_size=2048,
            chunk_capacity=16384,
        )

        src_list = []
        dst_list = []
        length_list = []

        # 2 large runs of 4 blocks each (>= min_direct_size=2048)
        # Use non-overlapping dst regions: blocks 0-3 and 4-7
        for i in range(2):
            src_list.append(f.p_base + (i * 4) * f.block_len)
            dst_list.append(f.d_base + (i * 4) * f.block_len)
            length_list.append(4 * f.block_len)

        # 10 small 1-block entries (< min_direct_size=2048)
        # Use dst blocks 20-29 to avoid overlap
        for i in range(10):
            src_list.append(f.p_base + (10 + i) * f.block_len)
            dst_list.append(f.d_base + (20 + i) * f.block_len)
            length_list.append(f.block_len)

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        assert len(plan.direct_runs) > 0
        assert len(plan.packed_chunks) > 0

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-004",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        assert result.direct_entries > 0
        assert result.packed_entries > 0

        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_ack_releases_p_slots(self) -> None:
        """After transfer, all P slots are released via READ_ACK."""
        f = _Fixture()
        src_list = [f.p_base + i * f.block_len for i in range(4)]
        dst_list = [f.d_base + i * f.block_len for i in range(4)]
        length_list = [f.block_len] * 4

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-005",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        assert f.p_service.active_slot_count == 0
        assert len(f.ack_log) == len(plan.packed_chunks)


# =====================================================================
# Staged vs execute_plan_on_tensors equivalence
# =====================================================================


class TestStagedEquivalence:
    def test_staged_matches_local_copy(self) -> None:
        """Staged P+D transfer produces byte-identical results to
        execute_plan_on_tensors (single-device reference)."""
        rng = random.Random(42)
        num_blocks = 32
        block_len = 512

        p_kv = torch.randint(0, 127, (num_blocks * block_len,), dtype=torch.int8)

        n = 8
        src_ids = rng.sample(range(num_blocks), n)
        dst_ids = rng.sample(range(num_blocks), n)

        src_list = [p_kv.data_ptr() + s * block_len for s in src_ids]
        length_list = [block_len] * n

        # --- Reference: execute_plan_on_tensors (single device) ---
        ref_dst = torch.zeros(num_blocks * block_len, dtype=torch.int8)
        ref_base = ref_dst.data_ptr()
        ref_dst_list = [ref_base + d * block_len for d in dst_ids]

        spans_ref = spans_from_flat_entries(src_list, ref_dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=2048, chunk_capacity=8192)
        plan_ref = planner.plan(spans_ref, peer_session="p0")
        ref_pool = StagingPool(num_slots=4, slot_capacity=8192, alignment=64, device="cpu")
        execute_plan_on_tensors(plan_ref, p_kv, p_kv.data_ptr(), ref_dst, ref_base, ref_pool)

        # --- Staged: P+D with simulated RDMA ---
        stg_dst = torch.zeros(num_blocks * block_len, dtype=torch.int8)
        stg_base = stg_dst.data_ptr()
        stg_dst_list = [stg_base + d * block_len for d in dst_ids]

        spans_stg = spans_from_flat_entries(src_list, stg_dst_list, length_list, request_id="r0")
        plan_stg = planner.plan(spans_stg, peer_session="p0")

        p_pool = StagingPool(num_slots=4, slot_capacity=8192, alignment=64, device="cpu")
        d_pool = StagingPool(num_slots=4, slot_capacity=8192, alignment=64, device="cpu")

        p_svc = PrefillStagingService(pool=p_pool, kv_tensor=p_kv, kv_base_addr=p_kv.data_ptr())
        d_coord = DecodeStagingCoordinator(pool=d_pool, dst_tensor=stg_dst, dst_base_addr=stg_base)
        rdma_fn = _make_rdma_read_fn(p_pool, d_pool)
        direct_fn = _make_direct_transfer_fn(p_kv, p_kv.data_ptr(), stg_dst, stg_base)

        result = d_coord.execute(
            plan_stg,
            transfer_id="eq-test",
            direct_transfer=direct_fn,
            prepare_read=p_svc.handle_prepare_read,
            rdma_read=rdma_fn,
            send_ack=lambda msg: p_svc.handle_read_ack(msg),
        )

        assert result.success
        assert torch.equal(stg_dst, ref_dst)


# =====================================================================
# Failure scenarios
# =====================================================================


class TestFailureHandling:
    def test_p_slot_exhausted(self) -> None:
        """P has no slots → coordinator returns error, no false success."""
        f = _Fixture(num_slots=1)
        f.p_pool.acquire()

        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-fail-p",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert not result.success
        assert "rejected" in result.error.lower() or "no staging slot" in result.error.lower()

    def test_d_slot_exhausted(self) -> None:
        """D has no slots → coordinator returns error."""
        f = _Fixture(num_slots=1)
        f.d_pool.acquire()

        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-fail-d",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert not result.success
        assert "no D staging slot" in result.error

    def test_rdma_failure(self) -> None:
        """RDMA read returns non-zero → coordinator reports failure."""
        f = _Fixture()
        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        def failing_rdma(d_addr: int, p_addr: int, nbytes: int) -> int:
            return -1

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-fail-rdma",
            direct_transfer=f.direct_transfer,
            prepare_read=f.prepare_read_fn,
            rdma_read=failing_rdma,
            send_ack=f.send_ack_fn,
        )

        assert not result.success
        assert "RDMA" in result.error
        assert len(f.ack_log) == 1
        assert f.ack_log[0].success is False
        assert f.p_service.active_slot_count == 0

    def test_direct_transfer_failure(self) -> None:
        """Direct transfer returns non-zero → failure reported."""
        f = _Fixture(
            num_blocks=64,
            block_len=4096,
            min_direct_size=1,
            chunk_capacity=_MIB,
        )
        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [4096]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=1, chunk_capacity=_MIB)
        plan = planner.plan(spans, peer_session="p0")

        assert len(plan.direct_runs) > 0

        def failing_direct(src: list[int], dst: list[int], lens: list[int]) -> int:
            return -1

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-fail-direct",
            direct_transfer=failing_direct,
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert not result.success
        assert "direct" in result.error.lower()


# =====================================================================
# P-service unit tests
# =====================================================================


class TestPrefillService:
    def test_prepare_read_batch_returns_per_chunk_results(self) -> None:
        f = _Fixture(num_slots=2)
        msg = PrepareReadBatchMsg(
            transfer_id="tx-batch-service",
            chunks=[
                PrepareReadBatchItem(
                    chunk_id=0,
                    gather_entries=[(f.p_base, 0, f.block_len)],
                    total_bytes=f.block_len,
                ),
                PrepareReadBatchItem(
                    chunk_id=1,
                    gather_entries=[(f.p_base + f.block_len, 0, f.block_len)],
                    total_bytes=f.block_len,
                ),
            ],
        )

        response = f.p_service.handle_prepare_read_batch(msg)

        assert isinstance(response, PackReadyBatchMsg)
        assert [item.chunk_id for item in response.results] == [0, 1]
        assert all(item.success for item in response.results)
        assert f.p_service.active_slot_count == 2
        for item in response.results:
            assert f.p_service.handle_read_ack(
                ReadAckMsg(transfer_id=msg.transfer_id, chunk_id=item.chunk_id)
            )
        assert f.p_service.active_slot_count == 0

    def test_idempotent_prepare_read(self) -> None:
        """Duplicate PREPARE_READ returns same slot (no double alloc)."""
        f = _Fixture()
        msg = PrepareReadMsg(
            transfer_id="tx-idem",
            chunk_id=0,
            gather_entries=[(f.p_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )

        r1 = f.p_service.handle_prepare_read(msg)
        r2 = f.p_service.handle_prepare_read(msg)

        assert isinstance(r1, PackReadyMsg)
        assert isinstance(r2, PackReadyMsg)
        assert r1.slot_addr == r2.slot_addr
        assert f.p_service.active_slot_count == 1

    def test_read_ack_idempotent(self) -> None:
        """Duplicate READ_ACK is a no-op (returns False)."""
        f = _Fixture()
        msg = PrepareReadMsg(
            transfer_id="tx-ack-idem",
            chunk_id=0,
            gather_entries=[(f.p_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )
        f.p_service.handle_prepare_read(msg)

        ack = ReadAckMsg(transfer_id="tx-ack-idem", chunk_id=0)
        assert f.p_service.handle_read_ack(ack) is True
        assert f.p_service.handle_read_ack(ack) is False

    def test_unknown_ack(self) -> None:
        """ACK for unknown transfer returns False."""
        f = _Fixture()
        ack = ReadAckMsg(transfer_id="unknown", chunk_id=99)
        assert f.p_service.handle_read_ack(ack) is False

    def test_prepare_rejects_out_of_range_source(self) -> None:
        f = _Fixture()
        msg = PrepareReadMsg(
            transfer_id="tx-invalid-src",
            chunk_id=0,
            gather_entries=[(f.p_base - 1, 0, f.block_len)],
            total_bytes=f.block_len,
        )

        response = f.p_service.handle_prepare_read(msg)

        assert isinstance(response, StagingErrorMsg)
        assert "outside" in response.reason
        assert f.p_service.active_slot_count == 0
        assert f.p_pool.acquire() is not None

    def test_gather_exception_releases_slot(self, monkeypatch) -> None:
        f = _Fixture()

        def fail_gather(*args, **kwargs):
            raise RuntimeError("gather failed")

        monkeypatch.setattr(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.p_service.pack_into_staging",
            fail_gather,
        )
        msg = PrepareReadMsg(
            transfer_id="tx-gather-fail",
            chunk_id=0,
            gather_entries=[(f.p_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )

        with pytest.raises(RuntimeError, match="gather failed"):
            f.p_service.handle_prepare_read(msg)

        assert f.p_service.active_slot_count == 0
        slot = f.p_pool.acquire()
        assert slot is not None
        f.p_pool.release(slot.slot_id)

    def test_concurrent_duplicate_prepare_reuses_one_slot(self) -> None:
        f = _Fixture(num_slots=2)
        msg = PrepareReadMsg(
            transfer_id="tx-concurrent",
            chunk_id=0,
            gather_entries=[(f.p_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda _: f.p_service.handle_prepare_read(msg), range(2)))

        assert all(isinstance(response, PackReadyMsg) for response in responses)
        assert responses[0].slot_addr == responses[1].slot_addr
        assert f.p_service.active_slot_count == 1


# =====================================================================
# Multi-region P+D staged transfer
# =====================================================================


def _make_rdma_read_fn_multi(p_pool: StagingPool, d_pool: StagingPool):
    """Simulated RDMA read between multi-region staging pools."""
    p_buf = p_pool._pool_view
    d_buf = d_pool._pool_view
    p_base = p_pool.base_ptr
    d_base = d_pool.base_ptr

    def rdma_read(d_addr: int, p_addr: int, nbytes: int) -> int:
        p_off = p_addr - p_base
        d_off = d_addr - d_base
        d_buf.view(-1)[d_off : d_off + nbytes].copy_(p_buf.view(-1)[p_off : p_off + nbytes])
        return 0

    return rdma_read


class _MultiRegionFixture:
    """Setup with multiple separate KV tensors (simulates per-layer allocation)."""

    def __init__(
        self,
        num_tensors: int = 3,
        tensor_size: int = 2048,
        num_slots: int = 2,
        slot_capacity: int = 8192,
        min_direct_size: int = 2048,
        chunk_capacity: int = 8192,
        sentinel: int = 99,
    ):
        self.num_tensors = num_tensors
        self.tensor_size = tensor_size
        self.sentinel = sentinel
        self.min_direct_size = min_direct_size
        self.chunk_capacity = chunk_capacity

        self.p_tensors = [torch.randint(0, 127, (tensor_size,), dtype=torch.int8) for _ in range(num_tensors)]
        self.d_tensors = [torch.full((tensor_size,), sentinel, dtype=torch.int8) for _ in range(num_tensors)]

        self.p_regions = sorted([(t.data_ptr(), t) for t in self.p_tensors], key=lambda r: r[0])
        self.d_regions = sorted([(t.data_ptr(), t) for t in self.d_tensors], key=lambda r: r[0])

        self.p_pool = StagingPool(num_slots=num_slots, slot_capacity=slot_capacity, alignment=64, device="cpu")
        self.d_pool = StagingPool(num_slots=num_slots, slot_capacity=slot_capacity, alignment=64, device="cpu")

        dummy = torch.empty(0, dtype=torch.int8)
        self.p_service = PrefillStagingService(
            pool=self.p_pool,
            kv_tensor=dummy,
            kv_base_addr=0,
            kv_regions=self.p_regions,
        )
        self.d_coordinator = DecodeStagingCoordinator(
            pool=self.d_pool,
            dst_tensor=dummy,
            dst_base_addr=0,
            dst_regions=self.d_regions,
        )

        self.rdma_read = _make_rdma_read_fn_multi(self.p_pool, self.d_pool)
        self.ack_log: list[ReadAckMsg] = []

    def prepare_read_fn(self, msg: PrepareReadMsg) -> PackReadyMsg | StagingErrorMsg:
        return self.p_service.handle_prepare_read(msg)

    def send_ack_fn(self, msg: ReadAckMsg) -> None:
        self.p_service.handle_read_ack(msg)
        self.ack_log.append(msg)


class TestMultiRegionStagedTransfer:
    def test_cross_tensor_transfer(self) -> None:
        """Transfer entries from different P tensors to different D tensors."""
        f = _MultiRegionFixture(num_tensors=3, tensor_size=2048)

        src_list = []
        dst_list = []
        length_list = []
        block_len = 256
        for i in range(f.num_tensors):
            src_list.append(f.p_tensors[i].data_ptr() + 0)
            dst_list.append(f.d_tensors[i].data_ptr() + 0)
            length_list.append(block_len)

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-multi-001",
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        for i in range(f.num_tensors):
            assert f.d_tensors[i][:block_len].tolist() == f.p_tensors[i][:block_len].tolist()
            assert all(v == f.sentinel for v in f.d_tensors[i][block_len:].tolist())

    def test_multi_region_byte_exact(self) -> None:
        """Multiple entries per tensor, verify byte-exact transfer."""
        f = _MultiRegionFixture(num_tensors=2, tensor_size=4096)

        src_list = []
        dst_list = []
        length_list = []
        block_len = 512
        for i in range(f.num_tensors):
            for j in range(3):
                offset = j * block_len
                src_list.append(f.p_tensors[i].data_ptr() + offset)
                dst_list.append(f.d_tensors[i].data_ptr() + offset)
                length_list.append(block_len)

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-multi-002",
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        for s, d, n in zip(src_list, dst_list, length_list):
            for i, (base, tensor) in enumerate(f.d_regions):
                if base <= d < base + f.tensor_size:
                    d_rel = d - base
                    break
            for i, (base, tensor) in enumerate(f.p_regions):
                if base <= s < base + f.tensor_size:
                    s_rel = s - base
                    break
            dst_data = f.d_tensors[i][d_rel : d_rel + n].tolist()
            src_data = f.p_tensors[i][s_rel : s_rel + n].tolist()
            assert dst_data == src_data

    def test_multi_region_ack_releases_slots(self) -> None:
        """All P slots released after multi-region transfer."""
        f = _MultiRegionFixture(num_tensors=2, tensor_size=2048)

        src_list = [f.p_tensors[0].data_ptr(), f.p_tensors[1].data_ptr()]
        dst_list = [f.d_tensors[0].data_ptr(), f.d_tensors[1].data_ptr()]
        length_list = [256, 256]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.d_coordinator.execute(
            plan,
            transfer_id="tx-multi-003",
            prepare_read=f.prepare_read_fn,
            rdma_read=f.rdma_read,
            send_ack=f.send_ack_fn,
        )

        assert result.success
        assert f.p_service.active_slot_count == 0
