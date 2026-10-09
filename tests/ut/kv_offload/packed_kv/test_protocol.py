# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.packed_kv.protocol import (
    PackReadyBatchItem,
    PackReadyBatchMsg,
    PackReadyMsg,
    PrepareReadBatchItem,
    PrepareReadBatchMsg,
    PrepareReadMsg,
    PrepareWriteMsg,
    ReadAckBatchItem,
    ReadAckBatchMsg,
    ReadAckMsg,
    StagingCapabilityMsg,
    StagingErrorMsg,
    StagingMsgType,
    WriteDoneMsg,
    WriteReadyMsg,
    decode_msg,
    encode_msg,
    msg_type_of,
)

# =====================================================================
# Encode / decode roundtrip
# =====================================================================


class TestEncodeDecodeRoundtrip:
    def test_prepare_read_batch_roundtrip(self) -> None:
        msg = PrepareReadBatchMsg(
            transfer_id="tx-batch",
            chunks=[
                PrepareReadBatchItem(
                    chunk_id=0,
                    gather_entries=[(100, 0, 64)],
                    total_bytes=64,
                ),
                PrepareReadBatchItem(
                    chunk_id=1,
                    gather_entries=[(200, 0, 128)],
                    total_bytes=128,
                ),
            ],
            release_source_when_ready=True,
            expected_chunks=2,
        )
        decoded = decode_msg(encode_msg(msg))
        assert isinstance(decoded, PrepareReadBatchMsg)
        assert decoded.transfer_id == "tx-batch"
        assert [item.chunk_id for item in decoded.chunks] == [0, 1]
        assert decoded.release_source_when_ready is True
        assert decoded.expected_chunks == 2

    def test_pack_ready_batch_roundtrip(self) -> None:
        msg = PackReadyBatchMsg(
            transfer_id="tx-batch-ready",
            results=[
                PackReadyBatchItem(
                    chunk_id=0,
                    success=True,
                    lease_id=10,
                    staging_addr=0x1000,
                    payload_bytes=64,
                ),
                PackReadyBatchItem(
                    chunk_id=1,
                    success=False,
                    error_code=1,
                    error="no slot",
                ),
            ],
        )
        decoded = decode_msg(encode_msg(msg))
        assert isinstance(decoded, PackReadyBatchMsg)
        assert decoded.results[0].staging_addr == 0x1000
        assert decoded.results[1].error == "no slot"

    def test_read_ack_batch_roundtrip(self) -> None:
        msg = ReadAckBatchMsg(
            transfer_id="tx-ack-batch",
            results=[ReadAckBatchItem(chunk_id=0, lease_id=10), ReadAckBatchItem(chunk_id=1, lease_id=11, success=False)],
        )
        decoded = decode_msg(encode_msg(msg))
        assert isinstance(decoded, ReadAckBatchMsg)
        assert [item.success for item in decoded.results] == [True, False]

    def test_prepare_read_roundtrip(self) -> None:
        msg = PrepareReadMsg(
            transfer_id="tx-001",
            chunk_id=0,
            gather_entries=[(100, 0, 64), (200, 64, 128)],
            total_bytes=192,
            release_source_when_ready=True,
            expected_chunks=3,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, PrepareReadMsg)
        assert decoded.transfer_id == "tx-001"
        assert decoded.chunk_id == 0
        assert decoded.gather_entries == [(100, 0, 64), (200, 64, 128)]
        assert decoded.total_bytes == 192
        assert decoded.release_source_when_ready is True
        assert decoded.expected_chunks == 3

    def test_pack_ready_roundtrip(self) -> None:
        msg = PackReadyMsg(
            transfer_id="tx-002",
            chunk_id=3,
            lease_id=42,
            staging_addr=0xDEADBEEF,
            payload_bytes=1024,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, PackReadyMsg)
        assert decoded.transfer_id == "tx-002"
        assert decoded.chunk_id == 3
        assert decoded.lease_id == 42
        assert decoded.staging_addr == 0xDEADBEEF
        assert decoded.payload_bytes == 1024

    def test_read_ack_roundtrip(self) -> None:
        msg = ReadAckMsg(transfer_id="tx-003", chunk_id=5, lease_id=50)
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, ReadAckMsg)
        assert decoded.transfer_id == "tx-003"
        assert decoded.chunk_id == 5
        assert decoded.success is True

    def test_failed_read_ack_roundtrip(self) -> None:
        msg = ReadAckMsg(transfer_id="tx-003-fail", chunk_id=6, lease_id=60, success=False)
        decoded = decode_msg(encode_msg(msg))
        assert isinstance(decoded, ReadAckMsg)
        assert decoded.success is False

    def test_capability_roundtrip(self) -> None:
        msg = StagingCapabilityMsg(
            supported=True,
            max_chunk_bytes=16 * 1024 * 1024,
            arena_capacity_bytes=32 * 1024 * 1024,
            page_size=256 * 1024,
            protocol_version=2,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, StagingCapabilityMsg)
        assert decoded.supported is True
        assert decoded.max_chunk_bytes == 16 * 1024 * 1024
        assert decoded.arena_capacity_bytes == 32 * 1024 * 1024
        assert decoded.page_size == 256 * 1024
        assert decoded.protocol_version == 2

    def test_capability_defaults(self) -> None:
        msg = StagingCapabilityMsg(supported=False)
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, StagingCapabilityMsg)
        assert decoded.supported is False
        assert decoded.max_chunk_bytes == 0
        assert decoded.arena_capacity_bytes == 0
        assert decoded.protocol_version == 2

    def test_prepare_write_roundtrip(self) -> None:
        msg = PrepareWriteMsg(
            transfer_id="tx-w001",
            chunk_id=2,
            scatter_entries=[(300, 0, 128), (500, 128, 256)],
            total_bytes=384,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, PrepareWriteMsg)
        assert decoded.transfer_id == "tx-w001"
        assert decoded.chunk_id == 2
        assert decoded.scatter_entries == [(300, 0, 128), (500, 128, 256)]
        assert decoded.total_bytes == 384

    def test_write_ready_roundtrip(self) -> None:
        msg = WriteReadyMsg(
            transfer_id="tx-w002",
            chunk_id=1,
            lease_id=12,
            staging_addr=0xCAFEBABE,
            payload_bytes=2048,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, WriteReadyMsg)
        assert decoded.transfer_id == "tx-w002"
        assert decoded.chunk_id == 1
        assert decoded.lease_id == 12
        assert decoded.staging_addr == 0xCAFEBABE
        assert decoded.payload_bytes == 2048

    def test_write_done_roundtrip(self) -> None:
        msg = WriteDoneMsg(transfer_id="tx-w003", chunk_id=4, lease_id=44)
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, WriteDoneMsg)
        assert decoded.transfer_id == "tx-w003"
        assert decoded.chunk_id == 4
        assert decoded.success is True

    def test_failed_write_done_roundtrip(self) -> None:
        msg = WriteDoneMsg(transfer_id="tx-w003-fail", chunk_id=5, lease_id=55, success=False)
        decoded = decode_msg(encode_msg(msg))
        assert isinstance(decoded, WriteDoneMsg)
        assert decoded.success is False

    def test_error_roundtrip(self) -> None:
        msg = StagingErrorMsg(
            transfer_id="tx-004",
            chunk_id=1,
            code=42,
            reason="slot unavailable",
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, StagingErrorMsg)
        assert decoded.transfer_id == "tx-004"
        assert decoded.code == 42
        assert decoded.reason == "slot unavailable"


# =====================================================================
# msg_type_of (peek without decoding)
# =====================================================================


class TestMsgTypeOf:
    def test_peek_prepare_read(self) -> None:
        data = encode_msg(PrepareReadMsg("t", 0, [], 0))
        assert msg_type_of(data) == StagingMsgType.PREPARE_READ

    def test_peek_pack_ready(self) -> None:
        data = encode_msg(PackReadyMsg("t", 0, 1, 0, 0))
        assert msg_type_of(data) == StagingMsgType.PACK_READY

    def test_peek_read_ack(self) -> None:
        data = encode_msg(ReadAckMsg("t", 0, 1))
        assert msg_type_of(data) == StagingMsgType.READ_ACK

    def test_peek_capability(self) -> None:
        data = encode_msg(StagingCapabilityMsg(supported=True))
        assert msg_type_of(data) == StagingMsgType.CAPABILITY

    def test_peek_error(self) -> None:
        data = encode_msg(StagingErrorMsg("t", 0, 0, ""))
        assert msg_type_of(data) == StagingMsgType.ERROR

    def test_peek_prepare_write(self) -> None:
        data = encode_msg(PrepareWriteMsg("t", 0, [], 0))
        assert msg_type_of(data) == StagingMsgType.PREPARE_WRITE

    def test_peek_write_ready(self) -> None:
        data = encode_msg(WriteReadyMsg("t", 0, 1, 0, 0))
        assert msg_type_of(data) == StagingMsgType.WRITE_READY

    def test_peek_write_done(self) -> None:
        data = encode_msg(WriteDoneMsg("t", 0, 1))
        assert msg_type_of(data) == StagingMsgType.WRITE_DONE


# =====================================================================
# Error handling
# =====================================================================


class TestErrorHandling:
    def test_decode_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            decode_msg(b"")

    def test_decode_single_byte_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            decode_msg(b"\x01")

    def test_decode_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown message type byte"):
            decode_msg(b"\xff\x00")

    def test_msg_type_of_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            msg_type_of(b"")

    def test_encode_unknown_type_raises(self) -> None:
        class FakeMsg:
            pass

        with pytest.raises(ValueError, match="Unknown message type"):
            encode_msg(FakeMsg())  # type: ignore[arg-type]


# =====================================================================
# Wire format properties
# =====================================================================


class TestWireFormat:
    def test_type_prefix_is_single_byte(self) -> None:
        data = encode_msg(ReadAckMsg("t", 0, 1))
        assert data[0] == StagingMsgType.READ_ACK

    def test_large_gather_entries(self) -> None:
        entries = [(i * 1000, i * 64, 64) for i in range(10000)]
        msg = PrepareReadMsg(
            transfer_id="big",
            chunk_id=0,
            gather_entries=entries,
            total_bytes=64 * 10000,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, PrepareReadMsg)
        assert len(decoded.gather_entries) == 10000
        assert decoded.gather_entries[9999] == (9999000, 9999 * 64, 64)

    def test_large_address_values(self) -> None:
        addr = 0x7FFF_FFFF_FFFF_0000
        msg = PackReadyMsg(
            transfer_id="addr-test",
            chunk_id=0,
            lease_id=9,
            staging_addr=addr,
            payload_bytes=16 * 1024 * 1024,
        )
        data = encode_msg(msg)
        decoded = decode_msg(data)
        assert isinstance(decoded, PackReadyMsg)
        assert decoded.staging_addr == addr
