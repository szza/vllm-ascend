# P-Side Architecture with MooncakeConnectorV1

This document describes the P (prefill) side of vLLM Ascend disaggregated
prefill when the KV connector is `MooncakeConnectorV1`. It focuses on the
process topology inside one P instance and on the metadata that allows a D
(decode) instance to pull the required KV blocks.

The implementation references in this document are:

- `vllm_ascend/patch/platform/patch_multiproc_executor.py` for local worker
  process creation and scheduler/worker queues.
- `vllm_ascend/worker/worker.py` for HCCL process-group initialization.
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` for
  Mooncake endpoint registration, metadata exchange, block mapping, and KV
  transfer.

## P Instance Topology

One P instance is one vLLM engine. The engine process owns the scheduler and
creates one local worker process for each rank in its parallel configuration.
The local world size is:

```text
world_size = tensor_parallel_size * pipeline_parallel_size
             * prefill_context_parallel_size
```

For a single-node P instance with `tensor_parallel_size=16`,
`pipeline_parallel_size=1`, and `prefill_context_parallel_size=1`, the
topology is:

```text
vllm serve (API / engine process)
└── EngineCore / scheduler
    ├── Worker rank 0  (TP0, NPU0)
    ├── Worker rank 1  (TP1, NPU1)
    ├── ...
    └── Worker rank 15 (TP15, NPU15)
```

Each worker owns a model shard and its own KV-cache allocation on its NPU.
Therefore, in the 16-card example, “16 workers” means 16 logical TP ranks,
not 16 threads inside one worker. A data-parallel deployment can create more
than one such engine group; the formula above applies to each local engine
group.

## Process and Communication Planes

The P instance has three distinct communication planes:

```text
                         control / scheduling
 EngineCore ─────────────────────────────────────┐
     │                                            │
     │ shared-memory MessageQueue + response MQ   │
     ▼                                            │
 TP0 worker ───┐                                  │
 TP1 worker ───┼── HCCL process groups ── NPU ranks
 ...           │
 TP15 worker ──┘
     │
     └── Mooncake Transfer Engine endpoint per worker
             (registered NPU KV-cache buffers)
                         ▲
                         │ KV data plane
                    D-side workers
```

### Scheduler and worker control plane

`AscendMultiprocExecutor` starts one child process per local rank. Scheduler
outputs and worker responses use vLLM's shared-memory `MessageQueue`; pipe
handles are used for readiness and parent-death signaling. This shared memory
is for control messages and result metadata. The model weights and NPU KV
cache are not copied into ordinary host shared memory.

### NPU rank communication

Each worker initializes the distributed environment with the `hccl` backend.
HCCL is used for tensor/model-parallel operations between NPU ranks. HCCL is
separate from the Mooncake P2P path used to exchange KV cache between P and D
instances.

### Mooncake KV data plane

Each worker creates or obtains a Mooncake Transfer Engine, registers the KV
cache buffers, and starts the relevant send/receive thread. A worker's
registered metadata describes the device memory ranges and layout needed by a
remote worker to address its KV blocks directly.

## Worker-Level Mooncake Endpoints

The P engine has one logical `engine_id`, while every worker/rank exposes its
own control and data-plane endpoint values.

| Field | Scope | Purpose |
| --- | --- | --- |
| `engine_id` | P engine | Stable logical namespace for the whole P instance. D uses it to keep metadata and port mappings from different P engines separate. |
| `remote_port` | P engine | Base handshake port carried in request metadata. Worker-specific handshake ports are derived from this base and the rank/device offset. |
| `handshake_port` | P worker/rank | ZeroMQ-style control endpoint used for metadata requests and completion notifications. |
| `te_rpc_port` | P worker/rank | Mooncake Transfer Engine RPC endpoint used by the data transfer operation. It is obtained from `engine.get_rpc_port()`. |
| `remote_host` | P endpoint | Host or Pod address used to reach the worker endpoint. |

The worker computes its handshake port from the configured KV base port and
its data-parallel, tensor-parallel, pipeline, and prefill-context rank. The
Transfer Engine RPC port is assigned by the Transfer Engine and is reported in
the worker metadata. Consequently, a P Pod with 16 TP workers has 16 rank
contexts and normally 16 worker-level Mooncake endpoints, even though all
workers belong to the same `engine_id`.

## Metadata Exchanged with D

After registering its KV buffers, a P worker publishes a
`MooncakeAgentMetadata` record. It contains:

- the engine-level `engine_id`;
- the worker's `te_rpc_port` and `handshake_port`;
- the KV-cache group and layer mapping;
- base addresses, lengths, strides, and scaling information for registered KV
  buffers;
- block size and the number of available blocks;
- the worker's reachable IP address.

D first uses the control endpoint (`remote_host` plus the selected
`handshake_port`) to request this metadata. It caches the result under:

```text
remote_engine_id -> remote_handshake_port ->
    KV addresses / strides / block scale / te_rpc_port
```

This explains why `remote_engine_id` is present even when `remote_host` and
ports are already available: an engine identifier prevents metadata from
different P instances from being mixed and is the key used by the connector's
remote metadata and port maps.

## How D Selects P Blocks to Pull

The P scheduler finishes prefill and returns KV transfer parameters alongside
the request. The parameters include:

```text
remote_engine_id
remote_host
remote_port                 # P handshake-port base
remote_request_id
remote_block_ids             # P-side logical block IDs
remote_pcp_size / remote_dcp_size / remote_ptp_size
num_prompt_blocks / remote_block_size
```

The D scheduler allocates local destination blocks and stores the request in
its connector metadata. During metadata construction, the connector derives a
per-shard pull plan:

```text
for each D-local KV shard:
    choose one or more P handshake ports
    compute the matching P block-id slice
    compute the D local destination block-id slice
    create a GroupPull description for each KV group
```

The mapping accounts for tensor-parallel and context-parallel layouts. For
unequal P/D TP sizes, one D rank can pull the corresponding KV-head shard from
multiple P ranks. Prefix-cache hits are skipped, and only the external block
range is scheduled for transfer. The resulting lists are paired by position:

```text
remote_handshake_port_list = [[P-port-A], [P-port-B], ...]
remote_block_ids_list       = [[P-blocks], [P-blocks], ...]
local_block_ids_list        = [[D-blocks], [D-blocks], ...]
```

The connector then submits one receive task per selected P endpoint. A receive
task carries both the remote block IDs and the local destination IDs, so the
worker does not scan the P cache or infer block ownership from the request
text. It reads exactly the registered remote addresses corresponding to the
listed block IDs and writes them into the allocated D blocks.

## P-to-D Pull Sequence

1. The P scheduler completes prefill and keeps the request's P KV blocks alive
   until the transfer is acknowledged.
2. The P side returns `remote_engine_id`, the P host, the base port, the
   request ID, and the P block IDs in `kv_transfer_params`.
3. The D scheduler allocates destination blocks and marks the request as
   waiting for remote KV data.
4. The D worker resolves each selected worker endpoint. On a metadata cache
   miss it sends a control-plane metadata request to the P worker's
   `handshake_port`.
5. The D worker obtains the P worker's `te_rpc_port` and registered KV-cache
   layout, then builds Mooncake transfer tasks from the local/remote block ID
   pairs.
6. Mooncake pulls the selected blocks from the P worker's registered NPU
   memory into the D worker's local NPU KV cache.
7. After all required shards report completion, D notifies P so the P request's
   delayed KV blocks can be released, and decode proceeds.

## Concrete 16-Card P Example

For `glm53-1p1d-baseline-0-prefill-0-0`, the observed configuration is
`data_parallel_size=1`, `tensor_parallel_size=16`, and one P Pod requesting 16
Ascend 910 devices. Its P-side mapping is therefore:

| P rank | Parallel role | KV ownership |
| ---: | --- | --- |
| 0 | TP0 | TP0 model shard and local KV blocks |
| 1 | TP1 | TP1 model shard and local KV blocks |
| ... | ... | ... |
| 15 | TP15 | TP15 model shard and local KV blocks |

The scheduler and all 16 workers are processes managed by the executor. Shared
memory carries scheduler/worker control messages, HCCL carries rank-to-rank
model-parallel traffic, and Mooncake exposes each worker's registered KV memory
to the D side. The 16 workers do not merge their KV caches into one host
shared-memory pool.

## Key Takeaways

- A P Pod is an engine instance; its workers are the local parallel ranks.
- With TP=16 and one NPU per rank, one P Pod has 16 worker processes and 16
  NPU KV-cache shards.
- `engine_id` identifies the P engine. `handshake_port` and `te_rpc_port`
  identify a particular worker endpoint.
- D learns which P blocks to pull from request transfer parameters and the
  parallel-layout mapping; it learns where those blocks live from the P
  worker's handshake metadata.
- Shared memory is used for local process control. Mooncake transfers KV data
  through registered NPU memory, while HCCL handles intra-engine NPU
  communication.
