# Cross-Node Transport Protocols in LMCache

## Overview

When LMCache processes run on **different physical nodes** (machines), CUDA IPC cannot be used since it only works within a single machine. LMCache provides several alternative transport mechanisms for distributed KV cache sharing.

## Important: GPUDirect Technologies Clarification

NVIDIA's **GPUDirect** is an umbrella term for several distinct technologies:

1. **GPUDirect RDMA** (for networking)
   - **Purpose:** GPU ↔ GPU communication **over network**
   - **Mechanism:** Direct GPU memory access over InfiniBand/RoCE, bypassing CPU
   - **Used in LMCache:** Via NIXL library with UCX transport for P2P/PD backends
   - **Bandwidth:** 10-100 GB/s depending on network hardware

2. **GPUDirect Storage (GDS)**
   - **Purpose:** GPU ↔ **Storage/Filesystem** I/O
   - **Mechanism:** Direct GPU-to-disk transfers, bypassing CPU buffers
   - **Used in LMCache:** `gds_backend.py` for local disk caching, NIXL GDS backend for file I/O
   - **Bandwidth:** Depends on storage (NVMe: 5-15 GB/s, network storage: varies)

3. **GPUDirect P2P**
   - **Purpose:** GPU ↔ GPU within **same machine**
   - **Mechanism:** PCIe/NVLink peer-to-peer transfers
   - **Used in LMCache:** Indirectly via CUDA operations, not the primary cross-node mechanism
   - **Bandwidth:** 100+ GB/s (NVLink), 10-30 GB/s (PCIe)

**Key Takeaway:** When discussing cross-node GPU communication, we mean **GPUDirect RDMA**, not GDS. GDS is for storage I/O, not network transfers.

## Key Architectural Difference

### Same-Machine (Multiprocess Mode)
```
┌─────────────┐
│  Worker A   │──┐
└─────────────┘  │ CUDA IPC (zero-copy)
┌─────────────┐  │
│  Worker B   │──┼──► LMCache Server (GPU operations via CUDA)
└─────────────┘  │
┌─────────────┐  │
│  Worker C   │──┘
└─────────────┘
    Single Node
```

### Cross-Node (Distributed Mode)
```
┌──────────────────┐         Network          ┌──────────────────┐
│    Node A        │                          │    Node B        │
│  ┌────────────┐  │  ┌──────────────────┐   │  ┌────────────┐  │
│  │ vLLM       │  │  │ Control: ZMQ     │   │  │ vLLM       │  │
│  │ Worker     │──┼──┤ Data: NIXL/TCP   ├───┼──│ Worker     │  │
│  └────────────┘  │  └──────────────────┘   │  └────────────┘  │
│  ┌────────────┐  │                          │  ┌────────────┐  │
│  │ LMCache    │  │                          │  │ LMCache    │  │
│  │ Local CPU  │  │                          │  │ Local CPU  │  │
│  └────────────┘  │                          │  └────────────┘  │
└──────────────────┘                          └──────────────────┘
```

## Cross-Node Transport Options

LMCache supports three main approaches for cross-node KV cache sharing:

### 1. P2P Backend (Peer-to-Peer)

**File:** `lmcache/v1/storage_backend/p2p_backend.py`

**Architecture:**
- **Control plane:** ZeroMQ (ROUTER-DEALER pattern) with MessagePack
- **Data plane:** Transfer channels (NIXL or socket-based)
- **Coordination:** Optional cache controller for peer discovery

**How it works:**

1. **Peer Discovery:**
   - Workers register with a centralized cache controller
   - Controller maintains a registry of all peers and their URLs
   - Workers query controller to find which peer has specific KV caches

2. **Control Messaging (ZMQ):**
   ```python
   # Message types in P2PBackend
   - BatchedLookupAndGetMsg    # Request KV from peer
   - BatchedLookupAndGetRetMsg # Response with hit count
   - BatchedLookupAndPutMsg    # Send KV to peer
   - BatchedLookupAndPutRetMsg # Acknowledge receipt
   ```

3. **Data Transfer:**
   - Uses `TransferChannel` abstraction
   - Actual KV cache data transferred via NIXL or sockets
   - Workers maintain local CPU backend as cache

4. **Flow Example:**
   ```
   Worker A (needs KV)
       ↓
   1. Lookup in local cache → MISS
       ↓
   2. Query controller: "Who has key X?"
       ↓
   3. Controller: "Worker B has it at tcp://nodeB:5555"
       ↓
   4. ZMQ to Worker B: BatchedLookupAndGetMsg
       ↓
   5. Worker B → Worker A: Transfer KV via NIXL
       ↓
   6. Worker A caches locally
   ```

**Configuration:**
```python
config.enable_p2p = True
config.controller_pull_url = "tcp://controller:6000"
config.nixl_backends = ["POSIX", "GDS"]  # Transfer channel options
```

**Key Components:**
- `P2PBackend` - Main backend implementation
- `PeerInfo` - Tracks peer connection metadata
- `TransferChannel` - Handles actual data movement
- `LocalCPUBackend` - Local cache for received KV

### 2. PD Backend (Disaggregated Prefill/Decode)

**File:** `lmcache/v1/storage_backend/pd_backend.py`

**Architecture:**
- Specialized for disaggregated LLM serving
- **Sender (Prefill nodes):** Generate KV caches and push to receiver
- **Receiver (Decode nodes):** Receive and store KV caches for decoding

**How it works:**

1. **Roles:**
   - **Sender:** Prefill-only workers (generate KV, don't store)
   - **Receiver:** Decode workers (receive and cache KV for generation)

2. **Protocol (ZMQ-based):**
   ```python
   # Message types in PDBackend
   - AllocRequest     # Sender: "Allocate buffer for N chunks"
   - AllocResponse    # Receiver: "Buffers ready at indexes X, Y, Z"
   - ProxyNotif       # Notify proxy about completion
   ```

3. **Transfer Flow:**
   ```
   Prefill Worker (Sender)          Decode Worker (Receiver)
        │                                    │
        │  1. AllocRequest (keys, size)      │
        ├────────────────────────────────────►
        │                                    │
        │  2. Allocate buffers               │
        │     AllocResponse (mem indexes)    │
        ◄────────────────────────────────────┤
        │                                    │
        │  3. Transfer data via NIXL         │
        ├═══════════════════════════════════►│
        │                                    │
        │  4. ProxyNotif (completion)        │
        ├────────────────────────────────────►
        │                                    │
   ```

4. **Zero-Copy Design:**
   - Pre-allocates GPU/CPU buffers on receiver
   - Uses NIXL for RDMA-like transfers when available
   - Sender writes directly to receiver's buffer

**Configuration:**
```python
# Sender (Prefill) config
config.enable_pd = True
config.pd_role = "sender"
config.pd_proxy_host = "decode_node"
config.pd_proxy_port = 7000

# Receiver (Decode) config
config.enable_pd = True
config.pd_role = "receiver"
config.pd_peer_host = "0.0.0.0"
config.pd_peer_init_port = [7001, 7002]  # Per TP rank
config.pd_peer_alloc_port = [7003, 7004]
```

**Use Case:**
- Disaggregated serving: separate prefill and decode clusters
- Prefill nodes generate KV, push to decode nodes
- Decode nodes cache and reuse for generation

### 3. Remote Backend (External Storage)

**File:** `lmcache/v1/storage_backend/remote_backend.py`

**Architecture:**
- Connects to external storage services (S3, custom backends)
- **Control plane:** HTTP/HTTPS or custom protocol
- **Data plane:** Serialized KV cache via network

**How it works:**

1. **Backend Connectors:**
   - `S3Connector` - AWS S3 storage
   - `ExternalConnector` - Custom HTTP-based backends
   - `SageMakerHyperPodConnector` - AWS SageMaker integration
   - Plugin system for custom backends

2. **Serialization:**
   - KV caches serialized before network transfer
   - Multiple serialization options:
     - `torch_serde` - PyTorch native serialization
     - `safetensors` - Safe tensor format
     - `msgpack` - Compressed binary format

3. **Flow:**
   ```
   Worker → Serialize KV → HTTP PUT → Remote Storage
   Worker → HTTP GET → Deserialize → GPU KV
   ```

4. **Local Caching:**
   - Optional `LocalCPUBackend` as L1 cache
   - Remote storage as L2/persistent cache
   - Reduces network calls for frequently used KV

**Configuration:**
```python
config.remote_url = "s3://bucket/prefix"  # or http://storage-service
config.remote_serde = "safetensors"
config.local_cpu_cache_size_gb = 10  # Optional local cache
```

## Transfer Channel Implementations

### NIXL Channel

**File:** `lmcache/v1/transfer_channel/nixl_channel.py`

**NIXL** (Network I/O Acceleration Library) - High-performance data transfer library built on UCX

**Features:**
- **RDMA support** via UCX (InfiniBand/RoCE) for GPU↔GPU over network
- **GPUDirect RDMA** when RDMA hardware available
- **GPU Direct Storage (GDS)** for GPU↔Disk I/O
- Fallback to optimized socket/TCP when RDMA unavailable
- Zero-copy transfers when possible

**Supported Backends:**
```python
"GDS"      # GPU Direct Storage (for file I/O, not network)
"GDS_MT"   # Multi-threaded GDS (for file I/O, not network)
"POSIX"    # Standard POSIX file I/O
"HF3FS"    # Hadoop filesystem
"OBJ"      # Object storage
```

**Network Transfers:**
- Uses UCX transport layer underneath
- UCX environment variable controls transport: `UCX_TLS=cuda_ipc,cuda_copy,tcp`
- For RDMA: UCX automatically selects optimal transport (InfiniBand, RoCE, etc.)

**Initialization:**
1. Sender creates NIXL agent with buffer metadata
2. ZMQ handshake exchanges NIXL metadata
3. NIXL establishes optimized transfer path (RDMA if available)
4. Pre-allocates and registers memory buffers

**Transfer:**
```python
# Async send/receive via NIXL
await nixl_agent.send_async(buffer_ptr, size, remote_handle)
await nixl_agent.recv_async(buffer_ptr, size, remote_handle)
```

### Socket Channel

**File:** `lmcache/v1/transfer_channel/py_socket_channel.py`

**Features:**
- Pure Python/socket-based fallback
- No special hardware requirements
- Uses ZMQ for control plane
- TCP sockets for data plane (to be implemented)

**Current Status:**
- Base class with ZMQ handshake logic
- Native socket data plane: TODO
- Mock implementation available for testing

## Transport Protocol Summary

| Approach | Control Plane | Data Plane | Use Case | Requires Controller |
|----------|--------------|------------|----------|-------------------|
| **P2P Backend** | ZMQ (MessagePack) | NIXL/Socket | Multi-worker cache sharing | Optional (recommended) |
| **PD Backend** | ZMQ (MessagePack) | NIXL | Disaggregated prefill/decode | No |
| **Remote Backend** | HTTP/Custom | HTTP/S3/Custom | Persistent/shared storage | No |
| **Multiprocess** | ZMQ (MessagePack) | CUDA IPC | Same-machine only | No |

## Detailed Message Flow: P2P Example

### Scenario: Worker A needs KV from Worker B

```
┌────────────┐         ┌────────────┐         ┌────────────┐
│  Worker A  │         │ Controller │         │  Worker B  │
└──────┬─────┘         └─────┬──────┘         └──────┬─────┘
       │                     │                       │
       │ 1. Lookup key X     │                       │
       ├────────────────────►│                       │
       │                     │                       │
       │ 2. Key X at Worker B│                       │
       │    (tcp://B:5555)   │                       │
       ◄────────────────────┤                       │
       │                     │                       │
       │ 3. Connect to Worker B                      │
       │    (ZMQ DEALER socket)                      │
       ├─────────────────────────────────────────────►
       │                     │                       │
       │ 4. BatchedLookupAndGetMsg                   │
       │    - keys: [X]                              │
       │    - mem_indexes: [local_buf_idx]           │
       ├═════════════════════════════════════════════►
       │                     │                       │
       │                     │    5. Worker B reads  │
       │                     │       from local cache│
       │                     │                       ◄──┐
       │                     │                       │  │
       │ 6. Transfer via NIXL                        │  │
       │    (RDMA if available, else TCP)            │  │
       ◄═════════════════════════════════════════════┤  │
       │                     │                       │  │
       │ 7. BatchedLookupAndGetRetMsg                │  │
       │    - num_hit_chunks: 1                      │  │
       ◄─────────────────────────────────────────────┤  │
       │                     │                       ├──┘
       │ 8. Store in local cache                     │
       ├──┐                  │                       │
       │  │                  │                       │
       ├──┘                  │                       │
       │                     │                       │
```

## Performance Considerations

### Network Bandwidth
- **CUDA IPC (same machine):** 100+ GB/s (PCIe/NVLink)
- **NIXL with GPUDirect RDMA:** 10-100 GB/s (InfiniBand/RoCE)
- **NIXL with TCP:** 1-10 GB/s (10GbE/100GbE)
- **HTTP/S3:** 0.1-1 GB/s (varies widely)

**Note:** GPUDirect RDMA allows GPU-to-GPU transfers over RDMA networks, bypassing CPU. This is different from GPUDirect Storage (GDS), which is for GPU-to-disk I/O.

### Latency
- **CUDA IPC:** ~10 μs
- **NIXL RDMA:** ~1-10 ms
- **TCP Sockets:** ~1-50 ms
- **HTTP/S3:** ~10-100+ ms

### Tradeoffs

| Method | Throughput | Latency | Setup Complexity | Hardware Req |
|--------|-----------|---------|------------------|--------------|
| CUDA IPC | Highest | Lowest | Low | Single machine |
| NIXL+GPUDirect RDMA | High | Low | Medium | RDMA NICs (InfiniBand/RoCE) |
| NIXL+TCP | Medium | Medium | Low | Standard NICs |
| HTTP/S3 | Low | High | Low | None |

## Configuration Examples

### Multi-Node P2P with RDMA
```yaml
# Worker nodes
enable_p2p: true
controller_pull_url: tcp://controller:6000
nixl_backends: ["POSIX"]  # NIXL will use UCX for network transfers (RDMA if available)
p2p_socket_recv_timeout_ms: 5000
p2p_socket_send_timeout_ms: 5000

# Environment variable for UCX (enables RDMA)
# UCX_TLS=rc,cuda_copy  # For InfiniBand RDMA
# UCX_TLS=tcp,cuda_copy # Fallback to TCP

# Controller node
lmcache_controller --host 0.0.0.0 --port 6000
```

**Note:** The `nixl_backends` parameter here refers to NIXL's I/O backend (POSIX for files, GDS for GPU-direct file I/O). The actual network transport (RDMA vs TCP) is determined by UCX configuration.

### Disaggregated Prefill/Decode
```yaml
# Prefill cluster (senders)
enable_pd: true
pd_role: sender
pd_proxy_host: decode-node-1
pd_proxy_port: 7000
pd_buffer_size: 1073741824  # 1GB
pd_buffer_device: cuda
nixl_backends: ["GDS", "POSIX"]

# Decode cluster (receivers)
enable_pd: true
pd_role: receiver
pd_peer_host: 0.0.0.0
pd_peer_init_port: [7001, 7002, 7003, 7004]  # Per TP rank
pd_peer_alloc_port: [7101, 7102, 7103, 7104]
pd_buffer_size: 1073741824
pd_buffer_device: cuda
```

### Remote S3 Backend
```yaml
remote_url: s3://my-bucket/lmcache-kv/
remote_serde: safetensors
local_cpu_cache_size_gb: 20  # Local L1 cache
blocking_timeout_secs: 30
```

## Key Differences from Multiprocess Mode

| Aspect | Multiprocess (Same Machine) | Cross-Node (Distributed) |
|--------|---------------------------|-------------------------|
| **GPU Data Sharing** | CUDA IPC (zero-copy) | Serialization + Network |
| **Process Communication** | Shared GPU memory | ZMQ + Transfer channels |
| **Synchronization** | CUDA Events (IPC) | ZMQ messages |
| **Setup Complexity** | Low | Medium-High |
| **Hardware Requirements** | None (GPU local) | Network (RDMA optional) |
| **Scalability** | Single machine only | Multi-node clusters |
| **Latency** | Microseconds | Milliseconds |

## Recommendations

### For Maximum Performance (RDMA hardware available):
```
Use P2P Backend with NIXL + GPUDirect RDMA
- Lowest latency for cross-node transfers
- Near-local performance (10-100 GB/s)
- Requires InfiniBand/RoCE NICs
- Set UCX_TLS=rc,cuda_copy (InfiniBand) or UCX_TLS=tcp,cuda_copy (fallback)
```

### For Disaggregated Serving:
```
Use PD Backend with NIXL
- Separate prefill and decode clusters
- Efficient resource allocation
- Optimized for one-way transfers
```

### For Persistent Storage:
```
Use Remote Backend with S3
- Survives worker restarts
- Shared across sessions
- Good for long-lived KV caches
```

### For Testing/Development:
```
Use P2P Backend with TCP fallback
- Works on standard networks
- No special hardware needed
- Lower performance but functional
```
