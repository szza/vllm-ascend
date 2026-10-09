# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Staging transfer control protocol messages.

Defines the message types exchanged between coordinator and service
for the packed KV transfer protocol.

READ mode (v1 connector — D initiates):
  D → P: ``PrepareReadMsg``  — request P to gather into a staging extent
  P → D: ``PackReadyMsg``    — gather done, extent address and lease returned
  D → P: ``ReadAckMsg``      — RDMA complete, P may release the lease

WRITE mode (layerwise connector — P initiates):
  P → D: ``PrepareWriteMsg`` — request D to allocate a staging extent
  D → P: ``WriteReadyMsg``   — extent allocated, address and lease returned
  P → D: ``WriteDoneMsg``    — RDMA WRITE complete, D may scatter + release

Serialization uses ``msgspec`` (consistent with the existing connector).
"""

from __future__ import annotations

import enum
from typing import Any

import msgspec


class StagingMsgType(enum.IntEnum):
    PREPARE_READ = 1
    PACK_READY = 2
    READ_ACK = 3
    CAPABILITY = 4
    ERROR = 5
    PREPARE_WRITE = 6
    WRITE_READY = 7
    WRITE_DONE = 8
    PREPARE_READ_BATCH = 9
    PACK_READY_BATCH = 10
    READ_ACK_BATCH = 11


class PrepareReadMsg(msgspec.Struct, array_like=True):
    """D → P: request gather into a P-side staging extent.

    ``gather_entries`` is a list of (src_offset, packed_offset, nbytes)
    tuples describing which source KV regions to pack.
    """

    transfer_id: str
    chunk_id: int
    gather_entries: list[tuple[int, int, int]]
    total_bytes: int
    release_source_when_ready: bool = False
    expected_chunks: int = 0


class PackReadyMsg(msgspec.Struct, array_like=True):
    """P -> D: gather complete, staging extent exposed for RDMA read."""

    transfer_id: str
    chunk_id: int
    lease_id: int
    staging_addr: int
    payload_bytes: int
    gather_ms: float = 0.0


class ReadAckMsg(msgspec.Struct, array_like=True):
    """D -> P: chunk finished or aborted, P may release its lease."""

    transfer_id: str
    chunk_id: int
    lease_id: int
    success: bool = True


class PrepareReadBatchItem(msgspec.Struct, array_like=True):
    """One chunk in a batched READ prepare request."""

    chunk_id: int
    gather_entries: list[tuple[int, int, int]]
    total_bytes: int


class PrepareReadBatchMsg(msgspec.Struct, array_like=True):
    """D -> P: request gathers for multiple chunks."""

    transfer_id: str
    chunks: list[PrepareReadBatchItem]
    release_source_when_ready: bool = False
    expected_chunks: int = 0


class PackReadyBatchItem(msgspec.Struct, array_like=True):
    """Result for one chunk in a batched READ prepare response."""

    chunk_id: int
    success: bool
    lease_id: int = 0
    staging_addr: int = 0
    payload_bytes: int = 0
    gather_ms: float = 0.0
    error_code: int = 0
    error: str = ""


class PackReadyBatchMsg(msgspec.Struct, array_like=True):
    """P -> D: gather results for multiple chunks."""

    transfer_id: str
    results: list[PackReadyBatchItem]


class ReadAckBatchItem(msgspec.Struct, array_like=True):
    """Completion status for one chunk in a batched READ ack."""

    chunk_id: int
    lease_id: int
    success: bool = True


class ReadAckBatchMsg(msgspec.Struct, array_like=True):
    """D -> P: release multiple READ staging leases."""

    transfer_id: str
    results: list[ReadAckBatchItem]


class StagingCapabilityMsg(msgspec.Struct, array_like=True):
    """Exchanged during handshake to advertise staging support."""

    supported: bool
    max_chunk_bytes: int = 0
    arena_capacity_bytes: int = 0
    page_size: int = 0
    protocol_version: int = 2


# No contiguous staging extent. Callers may fall back to direct RDMA.
STAGING_ERR_ARENA_EXHAUSTED = 1


class StagingErrorMsg(msgspec.Struct, array_like=True):
    """Error response for any staging request."""

    transfer_id: str
    chunk_id: int
    code: int
    reason: str


class PrepareWriteMsg(msgspec.Struct, array_like=True):
    """P -> D: request D to allocate a staging extent for RDMA WRITE.

    ``scatter_entries`` is a list of (dst_offset, packed_offset, nbytes)
    tuples describing where D should scatter data after the WRITE.
    """

    transfer_id: str
    chunk_id: int
    scatter_entries: list[tuple[int, int, int]]
    total_bytes: int


class WriteReadyMsg(msgspec.Struct, array_like=True):
    """D -> P: staging extent allocated, address returned for RDMA WRITE."""

    transfer_id: str
    chunk_id: int
    lease_id: int
    staging_addr: int
    payload_bytes: int


class WriteDoneMsg(msgspec.Struct, array_like=True):
    """P -> D: RDMA WRITE complete, D may scatter and release the lease."""

    transfer_id: str
    chunk_id: int
    lease_id: int
    success: bool = True


_MSG_TYPE_MAP: dict[StagingMsgType, type[msgspec.Struct]] = {
    StagingMsgType.PREPARE_READ: PrepareReadMsg,
    StagingMsgType.PACK_READY: PackReadyMsg,
    StagingMsgType.READ_ACK: ReadAckMsg,
    StagingMsgType.PREPARE_READ_BATCH: PrepareReadBatchMsg,
    StagingMsgType.PACK_READY_BATCH: PackReadyBatchMsg,
    StagingMsgType.READ_ACK_BATCH: ReadAckBatchMsg,
    StagingMsgType.CAPABILITY: StagingCapabilityMsg,
    StagingMsgType.ERROR: StagingErrorMsg,
    StagingMsgType.PREPARE_WRITE: PrepareWriteMsg,
    StagingMsgType.WRITE_READY: WriteReadyMsg,
    StagingMsgType.WRITE_DONE: WriteDoneMsg,
}

_TYPE_MSG_MAP: dict[type[msgspec.Struct], StagingMsgType] = {v: k for k, v in _MSG_TYPE_MAP.items()}

_encoder = msgspec.msgpack.Encoder()
_decoders: dict[StagingMsgType, msgspec.msgpack.Decoder[Any]] = {
    mt: msgspec.msgpack.Decoder(cls) for mt, cls in _MSG_TYPE_MAP.items()
}


def encode_msg(msg: msgspec.Struct) -> bytes:
    """Encode a staging message to bytes with a 1-byte type prefix."""
    msg_type = _TYPE_MSG_MAP.get(type(msg))
    if msg_type is None:
        raise ValueError(f"Unknown message type: {type(msg)}")
    return bytes([msg_type]) + _encoder.encode(msg)


def decode_msg(data: bytes) -> msgspec.Struct:
    """Decode a staging message from bytes (1-byte type prefix + msgpack body)."""
    if len(data) < 2:
        raise ValueError(f"Message too short: {len(data)} bytes")
    msg_type_val = data[0]
    try:
        msg_type = StagingMsgType(msg_type_val)
    except ValueError:
        raise ValueError(f"Unknown message type byte: {msg_type_val}") from None
    decoder = _decoders[msg_type]
    return decoder.decode(data[1:])


def msg_type_of(data: bytes) -> StagingMsgType:
    """Peek at the message type without decoding the body."""
    if not data:
        raise ValueError("Empty message")
    return StagingMsgType(data[0])


__all__ = [
    "StagingMsgType",
    "PrepareReadMsg",
    "PackReadyMsg",
    "ReadAckMsg",
    "PrepareReadBatchItem",
    "PrepareReadBatchMsg",
    "PackReadyBatchItem",
    "PackReadyBatchMsg",
    "ReadAckBatchItem",
    "ReadAckBatchMsg",
    "PrepareWriteMsg",
    "WriteReadyMsg",
    "WriteDoneMsg",
    "StagingCapabilityMsg",
    "StagingErrorMsg",
    "encode_msg",
    "decode_msg",
    "msg_type_of",
]
