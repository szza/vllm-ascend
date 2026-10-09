#!/usr/bin/env python3
"""Standalone test runner for packed_kv modules without the vLLM runtime."""

import importlib.util
import logging
import os
import sys
import types

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

vllm = types.ModuleType("vllm")
vllm.__path__ = []
vllm_logger = types.ModuleType("vllm.logger")
vllm_logger.logger = logging.getLogger("vllm")
sys.modules["vllm"] = vllm
sys.modules["vllm.logger"] = vllm_logger

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
envs_spec = importlib.util.spec_from_file_location("vllm_ascend.envs", "vllm_ascend/envs.py")
envs = importlib.util.module_from_spec(envs_spec)
sys.modules["vllm_ascend.envs"] = envs
envs_spec.loader.exec_module(envs)
sys.modules["vllm_ascend"].envs = envs
_base_dir = "vllm_ascend/distributed/kv_transfer/kv_p2p/packed_kv"

_modules = [
    "planner",
    "allocator",
    "copy",
    "budget",
    "adapter",
    "protocol",
    "p_service",
    "d_coordinator",
    "d_write_service",
    "p_write_coordinator",
]
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
                os.path.join(test_dir, "test_allocator.py"),
                os.path.join(test_dir, "test_copy.py"),
                os.path.join(test_dir, "test_protocol.py"),
                os.path.join(test_dir, "test_adapter.py"),
                os.path.join(test_dir, "test_staged_transfer.py"),
                os.path.join(test_dir, "test_write_staged_transfer.py"),
            ]
        )
    )
