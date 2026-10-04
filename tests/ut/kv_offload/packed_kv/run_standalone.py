#!/usr/bin/env python3
"""Standalone test runner for packed_kv modules.

Stubs the vllm_ascend package tree so pool.py, copy.py, budget.py, and
planner.py can be imported without vllm/torch_npu, then runs all test
modules via pytest.
"""

import importlib.util
import os
import sys
import types

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

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

_modules = ["planner", "pool", "copy", "budget", "adapter", "protocol", "p_service", "d_coordinator"]
for mod_name in _modules:
    fqn = f"vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.{mod_name}"
    spec = importlib.util.spec_from_file_location(fqn, os.path.join(_base_dir, f"{mod_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    setattr(_packed_kv_pkg, mod_name, mod)

if __name__ == "__main__":
    import pytest

    test_dir = os.path.join("tests", "ut", "kv_offload", "packed_kv")
    sys.exit(
        pytest.main(
            [
                "-sv",
                "--noconftest",
                os.path.join(test_dir, "test_planner.py"),
                os.path.join(test_dir, "test_pool.py"),
                os.path.join(test_dir, "test_copy.py"),
                os.path.join(test_dir, "test_protocol.py"),
                os.path.join(test_dir, "test_adapter.py"),
                os.path.join(test_dir, "test_staged_transfer.py"),
            ]
        )
    )
