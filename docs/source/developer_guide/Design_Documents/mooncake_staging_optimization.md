# Mooncake Staging Optimization Proposal

This document records a design proposal for reducing control-plane round trips
in the `MooncakeConnectorV1` staging path. It is based on the current P/D
protocol and is intentionally written as a proposal; it does not imply that
the protocol changes described here are already implemented.

The proposal keeps worker-level metadata separate from request-level transfer
state. This distinction is important because `GET_META_MSG` is cacheable,
while staging slots and packed ranges are allocated for one request and must be
released after that request completes.

## Current Protocol

For one D worker, one P worker, and one staging batch, the current control
sequence is:

```text
Metadata miss:
    D -> P  GET_META_MSG
    D <- P  MooncakeAgentMetadata
    D -> P  PREPARE_READ_BATCH
    D <- P  PACK_READY_BATCH
    D -> P  READ_ACK_BATCH
    D <- P  ACK
    D -> P  DONE_RECVING_MSG
    D <- P  ACK

Metadata hit:
    D -> P  PREPARE_READ_BATCH
    D <- P  PACK_READY_BATCH
    D -> P  READ_ACK_BATCH
    D <- P  ACK
    D -> P  DONE_RECVING_MSG
    D <- P  ACK
```

The request/response pair is one ZeroMQ round trip. The raw `ACK` is the
response for the ZeroMQ request; it is not the operation that releases data.

- `READ_ACK_BATCH` releases the P-side staging slots.
- `DONE_RECVING_MSG` completes the request-level transfer accounting and allows
  P to release the original KV-cache blocks.

The current implementation is in:

- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
  (`_get_remote_metadata`, `_transfer_via_staging`, and
  `_send_done_recv_signal`).
- `vllm_ascend/distributed/kv_transfer/kv_p2p/packed_kv/p_service.py` for
  staging slot ownership and release.

If the request is split into `N` staging windows because of the finite staging
pool, the current round-trip count is approximately:

```text
metadata miss: 2N + 2
metadata hit:  2N + 1
```

The exact wall-clock time can be lower when different P worker endpoints are
processed in parallel, but the number of control messages remains the same.

## Keep Metadata Static

`MooncakeAgentMetadata` describes one P worker and can be cached by D using:

```text
remote_engine_id + remote_handshake_port
```

It contains the worker's `te_rpc_port`, KV-cache base addresses, block strides,
KV group layout, and related static information. It must not contain a
request-specific staging slot address.

A staging slot is different:

```text
request-scoped
chunk-scoped
allocated dynamically
valid only while a lease is active
released after READ_ACK
```

Putting a slot address into `MooncakeAgentMetadata` would make the metadata
cache hold stale or reused addresses. The static metadata response should
remain independently cacheable.

## Put Dynamic Staging State in Request Parameters

The request-level source of truth is `kv_transfer_params`. P creates this data
after prefill and includes the P-side `remote_block_ids`, request ID, engine
identity, and transfer layout information. D stores it in `ReqMeta` and later
derives per-worker pull tasks.

The proposed extension is an optional request-level field:

```text
kv_transfer_params:
    remote_request_id
    remote_engine_id
    remote_host / remote_port
    remote_block_ids
    group_pulls / parallel-layout fields
    prepared_staging_manifest: optional
```

The proxy should treat `prepared_staging_manifest` as opaque request data and
forward it to D. It should not cache or reinterpret the manifest.

## Prepared Staging Manifest

When P can prepare data before D starts the pull, the manifest can contain:

```text
PreparedStagingChunk:
    transfer_id
    chunk_id
    p_worker_handshake_port
    te_rpc_port
    slot_addr or opaque slot token
    payload_bytes
    lease_id / generation
    packed segment mapping
```

The segment mapping must preserve more than a logical block ID. A single
logical block can expand across KV groups, layers, K/V components, TP offsets,
and byte ranges. A usable manifest therefore needs enough information for D to
scatter the packed payload into its local KV blocks, for example:

```text
segment:
    group_id
    layer_idx
    component_idx
    remote_block_id
    packed_offset
    nbytes
    local destination descriptor
```

The `lease_id` prevents an old request or duplicate ACK from releasing a slot
that has already been reused. The raw address may be carried inside the local
Mooncake deployment, but an opaque token is safer for proxy transport and
allows P to validate ownership before exposing the address to D.

## Preferred Data Flow

The preferred flow is:

```text
P worker
    KV write becomes visible
    └── gather eligible fragments into a registered staging slot
        └── create a prepared staging manifest

P scheduler / proxy
    └── forward kv_transfer_params + prepared_staging_manifest

D scheduler / worker
    ├── use GET_META only for static metadata when needed
    ├── use the manifest for P staging slots
    ├── Mooncake RDMA read
    ├── scatter into D KV cache
    └── final READ_ACK_BATCH(finalize_request=true)

P worker
    ├── release staging slots
    ├── update request completion accounting
    └── release original P KV blocks after all P/D shards complete
```

This removes the separate `PREPARE_READ_BATCH` request when P has already
prepared the staging chunks. `GET_META` remains a static metadata handshake,
and can be performed independently or proactively.

## Parallel Layout Constraint

P cannot always prepare a D-specific packed layout when it creates the initial
`kv_transfer_params`. At that point P may not know which D TP/CP rank will be
selected, or what local block layout D will use. This is especially relevant
for unequal TP, CP, HMA, and replicated-K paths.

There are two valid strategies:

1. **Canonical P-side layout**

   P prepares a layout independent of the selected D rank. D uses the manifest
   to select the relevant segments and scatter them locally. This can consume
   more staging memory or transfer extra data.

2. **D-specific prepare intent**

   D sends a logical pull plan containing `group_pulls`, P block IDs, TP/CP
   offsets, and local destination descriptors. P converts that plan into a
   staging manifest. This preserves the current transfer size but still needs
   one control exchange unless the intent is sent before the D request begins.

The second strategy is more memory efficient; the first has the best chance of
removing the prepare RTT completely. The implementation should select one
explicitly instead of assuming that a block ID alone determines the packed
layout.

## Combining Completion Messages

For a staging-enabled request, the final `READ_ACK_BATCH` can carry a request
completion marker:

```text
ReadAckBatch:
    transfer_id
    completed_chunks
    finalize_request
    remote_request_id
    remote_port_send_num
```

P handles the message in this order:

1. Release the acknowledged staging slots.
2. Update the per-port completion count.
3. If all P/D pull tasks have completed, finish the request tracker.
4. Release the original P KV blocks.
5. Return one ZeroMQ `ACK`.

`finalize_request` is only valid on the last batch. If the transfer has no
packed chunks, there is no `READ_ACK_BATCH`, so the existing
`DONE_RECVING_MSG` path remains necessary. If multiple P worker endpoints are
involved, `remote_port_send_num` or an equivalent aggregate must be retained.

## RTT Reduction

For one endpoint and one staging window:

| Path | Current | With prepared manifest and final ACK merge | Saved |
| --- | ---: | ---: | ---: |
| Metadata miss | 4 RTT | 2 RTT | 2 |
| Metadata hit | 3 RTT | 1 RTT | 2 |

The optimized sequence is:

```text
Mooncake RDMA read from prepared P slot
    -> D scatter
    -> READ_ACK_BATCH(finalize_request=true)
```

The table assumes that a metadata miss still requires one static `GET_META`
round trip. If the prepared manifest also supplies the endpoint information
needed for a staging-only transfer, D can use the manifest without fetching
metadata first; that specialized case can reduce the metadata-miss path to one
RTT as well. Mixed direct/staging transfers still need the static metadata for
the direct descriptors.

If P cannot prepare the manifest until it receives a D-specific pull plan, the
prepare exchange remains necessary. In that case, merging only the completion
messages saves one RTT:

```text
metadata miss: 4 -> 3 RTT
metadata hit:  3 -> 2 RTT
```

For `N` staging windows, the first design saves up to two RTTs overall, while
the completion-only design saves one RTT. Parallel requests to distinct P
worker ports reduce elapsed time toward the maximum endpoint latency, but do
not reduce the message count.

## Slot Lifetime and Failure Handling

Prepared staging changes the lifetime of a staging slot from “after
`PREPARE_READ`” to “after P prepares the request.” The implementation needs:

- a bounded prepared-manifest cache;
- a lease ID or generation per slot;
- timeout reclamation for requests that never reach D;
- cancellation handling when the request is aborted;
- idempotent final ACK processing;
- cleanup when P or D restarts.

The P-side staging pool is already a registered Mooncake memory region, but its
slot allocator is finite. A prefetch design must not hold slots indefinitely
while waiting for D scheduling.

## Compatibility and Rollout

The change should be introduced incrementally:

1. Keep `GET_META_MSG` and `MooncakeAgentMetadata` unchanged.
2. Add an optional `prepared_staging_manifest` to request transfer parameters.
3. Add a protocol/version capability so old D workers fall back to
   `PREPARE_READ_BATCH`.
4. Merge request finalization into the last `READ_ACK_BATCH` only when both
   sides advertise support.
5. Retain `DONE_RECVING_MSG` for direct-only and legacy transfers.

The fallback path must remain idempotent because a D retry can arrive after P
has already released a staging slot or completed request accounting.

## Validation Plan

The implementation should validate:

- metadata cache hit and miss behavior;
- one and multiple staging windows;
- multiple P worker endpoints and unequal TP/CP layouts;
- duplicate and late ACKs;
- D cancellation before the first RDMA read;
- P/D restart and lease timeout reclamation;
- direct-only, staging-only, and mixed direct/staging requests;
- RTT count and elapsed time separately.

The most useful metrics are:

```text
metadata_rtt_count
prepare_rtt_count
read_ack_rtt_count
done_rtt_count
prepared_slot_wait_ms
gather_ms
rdma_ms
slot_lease_ms
```

The primary optimization target is the number of control-plane RTTs. Pipelining
different P endpoints is a separate elapsed-latency optimization and should be
measured independently.

## All-or-Nothing Staging Window

This proposal changes how a staging window reacts when the arena cannot hold
every chunk. It is not implemented. It is separate from the prepared-manifest
proposal above: that proposal removes a prepare round trip when P can pack
ahead of time; this one removes a wasted acknowledgement round trip when the
window cannot be packed at all.

`chunk_capacity` is a maximum, not a fixed slot. A 16 MiB packed payload is one
chunk whose `payload_bytes` is 16 MiB. `StagingAllocator.allocate` rounds that
payload up to the page size and either returns a lease for the whole payload or
returns nothing. It does not shrink a 64 MiB chunk into the leftover 15 MiB.

The mixed path is in the batch window, not in the allocator. D sends every
chunk in the window in one `PREPARE_READ_BATCH`. P gathers the chunks that fit
and returns `STAGING_ERR_ARENA_EXHAUSTED` for the rest. D then RDMA-reads the
successful leases and sends the failed chunks through Direct. A 7-chunk window
that only fits the first chunk still uses both control round trips:

```text
D -> P  PREPARE_READ_BATCH          all chunks in the window
D <- P  PACK_READY_BATCH            some ready, some exhausted
        RDMA read of the ready chunks, then scatter
D -> P  READ_ACK_BATCH              release only the granted leases
D <- P  ACK
        Direct RDMA for the exhausted chunks
```

That mix keeps the small-descriptor Direct transfer and still pays for one
gather plus `READ_ACK_BATCH`. On the 2026-10-09 C96 run, 25 of 27 short-arena
transfers staged 1 of 7 chunks and sent the other 6 through Direct.

The window should be all staging or all Direct. Before gather, P checks that
every chunk in the batch has a contiguous extent of its actual size. If yes, P
reserves all of them and gathers. If any chunk does not fit, P reserves none
and returns one batch-exhausted response. D then sends the whole window through
Direct. The next window is judged again after earlier leases are released; one
short window does not forbid staging for the rest of the request.

D applies the same rule to its own arena before sending `PREPARE_READ_BATCH`.
If D cannot hold every chunk, it does not send the prepare message.

```text
Window fits:
    D -> P  PREPARE_READ_BATCH
    D <- P  PACK_READY_BATCH       includes gather time
            one staging RDMA, then scatter
    D -> P  READ_ACK_BATCH
    D <- P  ACK

P cannot fit the window:
    D -> P  PREPARE_READ_BATCH
    D <- P  batch exhausted        no gather, no lease
            one Direct RDMA for the whole window

D cannot fit the window:
            no staging ZeroMQ message
            one Direct RDMA for the whole window
```

The success path stays at two staging round trips. `READ_ACK_BATCH` is still
required because P holds the leases until D has finished the read. A separate
capacity-probe message should not be added; it would cost an extra round trip
on the success path.

The failure path drops `READ_ACK_BATCH` / `ACK`, because there is no lease to
release. When the rejection comes from P, `PREPARE_READ_BATCH` remains, but its
wait no longer includes gather. When D rejects locally, both staging round
trips are skipped. Large entries already classified as Direct by
`min_direct_size` are unchanged; this rule only removes the partial packed
window.
