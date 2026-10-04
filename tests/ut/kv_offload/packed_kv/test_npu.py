#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NPU-specific tests for packed_kv: HBM allocation + batch DMA path.

Requires torch_npu and NPU hardware. Run inside the container on npu-227:
  export LD_LIBRARY_PATH="/sharedata/zimoliu/miniforge3/envs/aisbench_eval/lib:${LD_LIBRARY_PATH:-}"
  cd /sharedata/szza/vllm-ascend
  python3 tests/ut/kv_offload/packed_kv/test_npu.py
"""
import importlib.util
import os
import random
import sys
import types

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

# Stub vllm_ascend package tree so we load from source, not site-packages
pkg_tree = [
    "vllm_ascend",
    "vllm_ascend.distributed",
    "vllm_ascend.distributed.kv_transfer",
    "vllm_ascend.distributed.kv_transfer.kv_p2p",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv",
]
for pkg in pkg_tree:
    parts = pkg.split(".")
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(os.getcwd(), *parts)]
    m.__package__ = pkg
    sys.modules[pkg] = m
    parent = ".".join(parts[:-1])
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], parts[-1], m)

_packed_kv_pkg = sys.modules["vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv"]
_base_dir = "vllm_ascend/distributed/kv_transfer/kv_p2p/packed_kv"
for mod_name in ["planner", "pool", "copy", "budget"]:
    fqn = f"vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.{mod_name}"
    spec = importlib.util.spec_from_file_location(fqn, os.path.join(_base_dir, f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    setattr(_packed_kv_pkg, mod_name, mod)

import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.copy import (
    _check_batch_dma,
    execute_plan_on_tensors,
    pack_into_staging,
    unpack_from_staging,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import (
    GatherEntry,
    ScatterEntry,
    TransferPlanner,
    spans_from_block_mapping,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.pool import StagingPool

DEVICE = "npu:0"


def test_pool_npu_allocation():
    """StagingPool allocates aligned HBM on NPU."""
    alignment = 2 * 1024 * 1024
    pool = StagingPool(num_slots=2, slot_capacity=1024 * 1024, alignment=alignment, device=DEVICE)
    assert pool.base_ptr % alignment == 0
    v = pool.slot_view(0)
    assert v.device.type == "npu"
    v.fill_(42)
    assert v[0].item() == 42
    print("  PASS: pool_npu_allocation")


def test_batch_dma_available():
    """swap_blocks_batch should be available with _C_ascend."""
    has_dma = _check_batch_dma()
    print(f"  INFO: _check_batch_dma() = {has_dma}")
    try:
        torch.ops._C_ascend.swap_blocks_batch  # noqa: B018
        print("  PASS: swap_blocks_batch op exists")
    except AttributeError:
        print("  SKIP: _C_ascend not loaded (vllm_ascend not compiled), fallback path will be used")


def test_gather_scatter_npu():
    """Gather/scatter on NPU tensors (whichever path is active)."""
    src = torch.arange(256, dtype=torch.int8, device="cpu").to(DEVICE)
    staging = torch.zeros(64, dtype=torch.int8, device=DEVICE)
    src_base = src.data_ptr()
    stg_base = staging.data_ptr()

    gather_entries = [
        GatherEntry(src_offset=src_base + 10, packed_offset=0, nbytes=8),
        GatherEntry(src_offset=src_base + 100, packed_offset=8, nbytes=8),
    ]
    pack_into_staging(src, src_base, staging, gather_entries)

    dst = torch.zeros(256, dtype=torch.int8, device=DEVICE)
    dst_base = dst.data_ptr()
    scatter_entries = [
        ScatterEntry(packed_offset=0, dst_offset=dst_base + 10, nbytes=8),
        ScatterEntry(packed_offset=8, dst_offset=dst_base + 100, nbytes=8),
    ]
    unpack_from_staging(staging, dst, dst_base, scatter_entries)

    torch.npu.synchronize()

    expected_10_18 = list(range(10, 18))
    expected_100_108 = list(range(100, 108))
    actual_10_18 = dst[10:18].cpu().tolist()
    actual_100_108 = dst[100:108].cpu().tolist()
    assert actual_10_18 == expected_10_18, f"gather/scatter mismatch at [10:18]: {actual_10_18} != {expected_10_18}"
    assert actual_100_108 == expected_100_108, f"gather/scatter mismatch at [100:108]: {actual_100_108} != {expected_100_108}"
    # Untouched regions should be zero
    assert dst[0:10].cpu().sum().item() == 0
    print("  PASS: gather_scatter_npu")


def test_execute_plan_npu():
    """Full plan execution on NPU with random block mapping."""
    rng = random.Random(42)
    num_blocks = 64
    block_len = 512
    tp_num_pulls = 2

    src = torch.randint(0, 127, (num_blocks * block_len,), dtype=torch.int8, device=DEVICE)
    dst = torch.zeros(num_blocks * block_len, dtype=torch.int8, device=DEVICE)
    src_base = src.data_ptr()
    dst_base = dst.data_ptr()

    all_spans = []
    for req_id in range(3):
        n_blocks = rng.randint(2, 8)
        src_ids = rng.sample(range(num_blocks), n_blocks)
        dst_ids = rng.sample(range(num_blocks), n_blocks)
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
    pool = StagingPool(num_slots=4, slot_capacity=8192, alignment=64, device=DEVICE)
    execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)

    torch.npu.synchronize()

    # Verify byte-for-byte correctness
    expected = torch.zeros_like(dst)
    for sp in all_spans:
        s = sp.src_offset - src_base
        d = sp.dst_offset - dst_base
        expected[d : d + sp.nbytes] = src[s : s + sp.nbytes]

    match = torch.equal(dst.cpu(), expected.cpu())
    assert match, "NPU execute_plan byte mismatch!"
    print(f"  PASS: execute_plan_npu (direct={len(plan.direct_runs)}, packed={len(plan.packed_chunks)}, spans={len(all_spans)})")


def test_sentinel_preserved_npu():
    """Untransferred regions on NPU remain untouched."""
    size = 4096
    sentinel = 85  # 0x55
    src = torch.randint(0, 127, (size,), dtype=torch.int8, device=DEVICE)
    dst = torch.full((size,), sentinel, dtype=torch.int8, device=DEVICE)
    src_base = src.data_ptr()
    dst_base = dst.data_ptr()

    from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.planner import CopySpan

    spans = [
        CopySpan(request_id="0", group_id=0, layer_idx=0, component_idx=0,
                 src_offset=src_base + 500, dst_offset=dst_base + 500, nbytes=1000),
    ]
    planner = TransferPlanner(min_direct_size=2048, chunk_capacity=16384)
    plan = planner.plan(spans, peer_session="s0")
    pool = StagingPool(num_slots=2, slot_capacity=16384, alignment=64, device=DEVICE)
    execute_plan_on_tensors(plan, src, src_base, dst, dst_base, pool)
    torch.npu.synchronize()

    dst_cpu = dst.cpu()
    assert dst_cpu[500:1500].tolist() == src[500:1500].cpu().tolist()
    assert all(v == sentinel for v in dst_cpu[:500].tolist())
    assert all(v == sentinel for v in dst_cpu[1500:].tolist())
    print("  PASS: sentinel_preserved_npu")


if __name__ == "__main__":
    print(f"torch: {torch.__version__}, torch_npu: {torch_npu.__version__}")
    print(f"NPU available: {torch.npu.is_available()}, count: {torch.npu.device_count()}")
    torch.npu.set_device(0)
    print(f"Using device: {DEVICE}\n")

    tests = [
        test_pool_npu_allocation,
        test_batch_dma_available,
        test_gather_scatter_npu,
        test_execute_plan_npu,
        test_sentinel_preserved_npu,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {len(tests) - failed}/{len(tests)} passed, {failed} failed")
    sys.exit(1 if failed else 0)
