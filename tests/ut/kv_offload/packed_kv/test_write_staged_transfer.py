# SPDX-License-Identifier: Apache-2.0
"""Integration tests: P-write-coordinator + D-write-service with simulated RDMA.

Mirror of ``test_staged_transfer.py`` for WRITE mode.  The "RDMA" step
is a plain memcpy from P's staging slot to D's staging slot.
"""

from __future__ import annotations

import random

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.adapter import (
    spans_from_flat_entries,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.d_write_service import (
    DecodeWriteService,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.p_write_coordinator import (
    PrefillWriteCoordinator,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    TransferPlanner,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.allocator import StagingAllocator
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PrepareWriteMsg,
    StagingErrorMsg,
    WriteDoneMsg,
    WriteReadyMsg,
)


def _make_rdma_write_fn(p_pool: StagingAllocator, d_pool: StagingAllocator):
    """Simulated RDMA WRITE: copy from P staging slot to D staging slot."""
    p_buf = p_pool.arena_view
    d_buf = d_pool.arena_view
    p_base = p_pool.base_ptr
    d_base = d_pool.base_ptr

    def rdma_write(p_addr: int, d_addr: int, nbytes: int) -> int:
        p_off = p_addr - p_base
        d_off = d_addr - d_base
        d_buf.view(-1)[d_off : d_off + nbytes].copy_(p_buf.view(-1)[p_off : p_off + nbytes])
        return 0

    return rdma_write


def _make_direct_write_fn(src_tensor: torch.Tensor, src_base: int, dst_tensor: torch.Tensor, dst_base: int):
    """Simulated direct RDMA WRITE for large runs (no staging)."""

    def direct_write(src_addrs: list[int], dst_addrs: list[int], lengths: list[int]) -> int:
        src_flat = src_tensor.view(-1)
        dst_flat = dst_tensor.view(-1)
        for s, d, n in zip(src_addrs, dst_addrs, lengths):
            s_rel = s - src_base
            d_rel = d - dst_base
            dst_flat[d_rel : d_rel + n].copy_(src_flat[s_rel : s_rel + n])
        return 0

    return direct_write


class _WriteFixture:
    """Shared setup for P-write-coordinator + D-write-service tests."""

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

        self.p_pool = StagingAllocator(capacity_bytes=num_slots * slot_capacity, page_size=64, alignment=64, device="cpu")
        self.d_pool = StagingAllocator(capacity_bytes=num_slots * slot_capacity, page_size=64, alignment=64, device="cpu")

        self.p_coordinator = PrefillWriteCoordinator(
            allocator=self.p_pool,
            src_tensor=self.p_kv,
            src_base_addr=self.p_base,
        )

        self.d_service = DecodeWriteService(
            allocator=self.d_pool,
            kv_tensor=self.d_kv,
            kv_base_addr=self.d_base,
        )

        self.rdma_write = _make_rdma_write_fn(self.p_pool, self.d_pool)
        self.direct_write = _make_direct_write_fn(self.p_kv, self.p_base, self.d_kv, self.d_base)

        self.write_done_log: list[WriteDoneMsg] = []

    def prepare_write_fn(self, msg: PrepareWriteMsg) -> WriteReadyMsg | StagingErrorMsg:
        return self.d_service.handle_prepare_write(msg)

    def send_write_done_fn(self, msg: WriteDoneMsg) -> None:
        self.d_service.handle_write_done(msg)
        self.write_done_log.append(msg)


# =====================================================================
# Full P+D WRITE-mode staged transfer
# =====================================================================


class TestWriteStagedTransfer:
    def test_single_chunk_byte_exact(self) -> None:
        f = _WriteFixture()
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

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w001",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        assert result.chunks_completed > 0

        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_multi_chunk_serial(self) -> None:
        f = _WriteFixture(num_slots=1, slot_capacity=2048, chunk_capacity=2048)
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

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w002",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        assert result.chunks_completed == len(plan.packed_chunks)

        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_sentinel_preserved(self) -> None:
        f = _WriteFixture()
        src_list = [f.p_base + 2 * f.block_len, f.p_base + 5 * f.block_len]
        dst_list = [f.d_base + 3 * f.block_len, f.d_base + 7 * f.block_len]
        length_list = [f.block_len, f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w003",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        for b in range(f.num_blocks):
            block_data = f.d_kv[b * f.block_len : (b + 1) * f.block_len].tolist()
            if b in (3, 7):
                continue
            assert all(v == f.sentinel for v in block_data), f"Block {b} was modified"

    def test_mixed_direct_and_packed(self) -> None:
        f = _WriteFixture(
            num_blocks=64,
            block_len=1024,
            slot_capacity=16384,
            min_direct_size=2048,
            chunk_capacity=16384,
        )

        src_list = []
        dst_list = []
        length_list = []

        for i in range(2):
            src_list.append(f.p_base + (i * 4) * f.block_len)
            dst_list.append(f.d_base + (i * 4) * f.block_len)
            length_list.append(4 * f.block_len)

        for i in range(10):
            src_list.append(f.p_base + (10 + i) * f.block_len)
            dst_list.append(f.d_base + (20 + i) * f.block_len)
            length_list.append(f.block_len)

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        assert len(plan.direct_runs) > 0
        assert len(plan.packed_chunks) > 0

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w004",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        assert result.direct_entries > 0
        assert result.packed_entries > 0

        for s_addr, d_addr, length in zip(src_list, dst_list, length_list):
            s_rel = s_addr - f.p_base
            d_rel = d_addr - f.d_base
            assert f.d_kv[d_rel : d_rel + length].tolist() == f.p_kv[s_rel : s_rel + length].tolist()

    def test_write_done_releases_d_slots(self) -> None:
        f = _WriteFixture()
        src_list = [f.p_base + i * f.block_len for i in range(4)]
        dst_list = [f.d_base + i * f.block_len for i in range(4)]
        length_list = [f.block_len] * 4

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w005",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        assert f.d_service.pending_count == 0
        assert len(f.write_done_log) == len(plan.packed_chunks)

    def test_empty_plan(self) -> None:
        f = _WriteFixture()
        spans = spans_from_flat_entries([], [], [], request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-empty",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        assert result.chunks_completed == 0
        assert result.direct_entries == 0


# =====================================================================
# Failure scenarios
# =====================================================================


class TestWriteFailureHandling:
    def test_d_slot_exhausted(self) -> None:
        f = _WriteFixture(num_slots=1)
        f.d_pool.allocate(8192)

        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-fail-d",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert not result.success
        assert "rejected" in result.error.lower() or "no staging slot" in result.error.lower()

    def test_p_slot_exhausted(self) -> None:
        f = _WriteFixture(num_slots=1)
        f.p_pool.allocate(8192)

        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-fail-p",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert not result.success
        assert "no contiguous P staging extent" in result.error

    def test_rdma_write_failure(self) -> None:
        f = _WriteFixture()
        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [f.block_len]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        def failing_rdma(p_addr: int, d_addr: int, nbytes: int) -> int:
            return -1

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-fail-rdma",
            direct_write=f.direct_write,
            prepare_write=f.prepare_write_fn,
            rdma_write=failing_rdma,
            send_write_done=f.send_write_done_fn,
        )

        assert not result.success
        assert "RDMA" in result.error
        assert len(f.write_done_log) == 1
        assert f.write_done_log[0].success is False
        assert f.d_service.pending_count == 0

    def test_direct_write_failure(self) -> None:
        f = _WriteFixture(
            num_blocks=64,
            block_len=4096,
            min_direct_size=1,
            chunk_capacity=1024 * 1024,
        )
        src_list = [f.p_base]
        dst_list = [f.d_base]
        length_list = [4096]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=1, chunk_capacity=1024 * 1024)
        plan = planner.plan(spans, peer_session="p0")

        assert len(plan.direct_runs) > 0

        def failing_direct(src: list[int], dst: list[int], lens: list[int]) -> int:
            return -1

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-fail-direct",
            direct_write=failing_direct,
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert not result.success
        assert "direct" in result.error.lower()


# =====================================================================
# D-write-service unit tests
# =====================================================================


class TestDecodeWriteService:
    def test_idempotent_prepare_write(self) -> None:
        f = _WriteFixture()
        msg = PrepareWriteMsg(
            transfer_id="tx-idem",
            chunk_id=0,
            scatter_entries=[(f.d_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )

        r1 = f.d_service.handle_prepare_write(msg)
        r2 = f.d_service.handle_prepare_write(msg)

        assert isinstance(r1, WriteReadyMsg)
        assert isinstance(r2, WriteReadyMsg)
        assert r1.staging_addr == r2.staging_addr
        assert f.d_service.pending_count == 1

    def test_write_done_idempotent(self) -> None:
        f = _WriteFixture()
        msg = PrepareWriteMsg(
            transfer_id="tx-done-idem",
            chunk_id=0,
            scatter_entries=[(f.d_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )
        ready = f.d_service.handle_prepare_write(msg)
        done = WriteDoneMsg(transfer_id="tx-done-idem", chunk_id=0, lease_id=ready.lease_id)
        assert f.d_service.handle_write_done(done) is True
        assert f.d_service.handle_write_done(done) is False

    def test_unknown_write_done(self) -> None:
        f = _WriteFixture()
        done = WriteDoneMsg(transfer_id="unknown", chunk_id=99, lease_id=1)
        assert f.d_service.handle_write_done(done) is False

    def test_prepare_rejects_out_of_range_dest(self) -> None:
        f = _WriteFixture()
        msg = PrepareWriteMsg(
            transfer_id="tx-invalid-dst",
            chunk_id=0,
            scatter_entries=[(f.d_base - 1, 0, f.block_len)],
            total_bytes=f.block_len,
        )

        response = f.d_service.handle_prepare_write(msg)

        assert isinstance(response, StagingErrorMsg)
        assert "outside" in response.reason
        assert f.d_service.pending_count == 0
        assert f.d_pool.allocate(8192) is not None

    def test_abort_write_done_releases_slot(self) -> None:
        f = _WriteFixture()
        msg = PrepareWriteMsg(
            transfer_id="tx-abort",
            chunk_id=0,
            scatter_entries=[(f.d_base, 0, f.block_len)],
            total_bytes=f.block_len,
        )
        ready = f.d_service.handle_prepare_write(msg)
        done = WriteDoneMsg(transfer_id="tx-abort", chunk_id=0, lease_id=ready.lease_id, success=False)
        assert f.d_service.handle_write_done(done) is True
        assert f.d_service.pending_count == 0
        lease = f.d_pool.allocate(8192)
        assert lease is not None
        f.d_pool.release(lease)


# =====================================================================
# Multi-region WRITE-mode staged transfer
# =====================================================================


class _MultiRegionWriteFixture:
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

        self.p_pool = StagingAllocator(capacity_bytes=num_slots * slot_capacity, page_size=64, alignment=64, device="cpu")
        self.d_pool = StagingAllocator(capacity_bytes=num_slots * slot_capacity, page_size=64, alignment=64, device="cpu")

        dummy = torch.empty(0, dtype=torch.int8)
        self.p_coordinator = PrefillWriteCoordinator(
            allocator=self.p_pool,
            src_tensor=dummy,
            src_base_addr=0,
            src_regions=self.p_regions,
        )
        self.d_service = DecodeWriteService(
            allocator=self.d_pool,
            kv_tensor=dummy,
            kv_base_addr=0,
            kv_regions=self.d_regions,
        )

        self.rdma_write = _make_rdma_write_fn(self.p_pool, self.d_pool)
        self.write_done_log: list[WriteDoneMsg] = []

    def prepare_write_fn(self, msg: PrepareWriteMsg) -> WriteReadyMsg | StagingErrorMsg:
        return self.d_service.handle_prepare_write(msg)

    def send_write_done_fn(self, msg: WriteDoneMsg) -> None:
        self.d_service.handle_write_done(msg)
        self.write_done_log.append(msg)


class TestMultiRegionWriteStagedTransfer:
    def test_cross_tensor_transfer(self) -> None:
        f = _MultiRegionWriteFixture(num_tensors=3, tensor_size=2048)

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

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-multi-001",
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        for i in range(f.num_tensors):
            assert f.d_tensors[i][:block_len].tolist() == f.p_tensors[i][:block_len].tolist()
            assert all(v == f.sentinel for v in f.d_tensors[i][block_len:].tolist())

    def test_multi_region_byte_exact(self) -> None:
        f = _MultiRegionWriteFixture(num_tensors=2, tensor_size=4096)

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

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-multi-002",
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        for s, d, n in zip(src_list, dst_list, length_list):
            for base, tensor in f.d_regions:
                if base <= d < base + f.tensor_size:
                    d_rel = d - base
                    d_data = tensor[d_rel : d_rel + n].tolist()
                    break
            for base, tensor in f.p_regions:
                if base <= s < base + f.tensor_size:
                    s_rel = s - base
                    s_data = tensor[s_rel : s_rel + n].tolist()
                    break
            assert d_data == s_data

    def test_multi_region_releases_d_slots(self) -> None:
        f = _MultiRegionWriteFixture(num_tensors=2, tensor_size=2048)

        src_list = [f.p_tensors[0].data_ptr(), f.p_tensors[1].data_ptr()]
        dst_list = [f.d_tensors[0].data_ptr(), f.d_tensors[1].data_ptr()]
        length_list = [256, 256]

        spans = spans_from_flat_entries(src_list, dst_list, length_list, request_id="r0")
        planner = TransferPlanner(min_direct_size=f.min_direct_size, chunk_capacity=f.chunk_capacity)
        plan = planner.plan(spans, peer_session="p0")

        result = f.p_coordinator.execute(
            plan,
            transfer_id="tx-w-multi-003",
            prepare_write=f.prepare_write_fn,
            rdma_write=f.rdma_write,
            send_write_done=f.send_write_done_fn,
        )

        assert result.success
        assert f.d_service.pending_count == 0
