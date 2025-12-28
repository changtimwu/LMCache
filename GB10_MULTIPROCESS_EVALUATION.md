# Multiprocess Mode Evaluation for GB10 Deployment

## Executive Summary

**Verdict: ❌ Multiprocess mode is NOT suitable for your GB10 deployment.**

**Reason**: Multiprocess mode is designed for **same-node** GPU sharing via CUDA IPC, while your use case requires **cross-node** KV cache sharing across multiple physically separate GB10 devices.

## What is Multiprocess Mode?

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Single GPU Node (Same Physical Machine)       │
│                                                 │
│  ┌──────────────┐         ┌──────────────┐     │
│  │ vLLM Worker  │         │ vLLM Worker  │     │
│  │ (GPU 0-3)    │         │ (GPU 4-7)    │     │
│  └──────┬───────┘         └──────┬───────┘     │
│         │                        │             │
│         │   CUDA IPC (via /dev/shm)            │
│         └────────────┬───────────┘             │
│                      ↓                          │
│         ┌────────────────────────┐              │
│         │ LMCache MP Server      │              │
│         │ (localhost:5555)       │              │
│         │ - CPU buffer (60GB)    │              │
│         │ - ZeroMQ messaging     │              │
│         │ - CUDA IPC transfers   │              │
│         └────────────────────────┘              │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Key Characteristics

1. **Same-Node Only**
   - Uses CUDA IPC (Inter-Process Communication)
   - Requires shared `/dev/shm` on the **same physical machine**
   - Cannot work across network boundaries

2. **Communication Mechanism**
   - **Control plane**: ZeroMQ over TCP (localhost)
   - **Data plane**: CUDA IPC shared memory (zero-copy GPU memory access)
   - GPU tensor pointers are shared via `CudaIPCWrapper` class

3. **GPU Requirement**
   - Server **must have GPU access** for CUDA IPC
   - Calls `torch.cuda.init()` on startup (server.py:537)
   - Unconditionally imports CUDA operations (server.py:36)

4. **Deployment Pattern**
   - **DaemonSet**: One LMCache server per Kubernetes node
   - Multiple vLLM pods on same node connect to local server
   - Uses `status.hostIP` to discover node-local server

### Technical Implementation

**CUDA IPC Wrapper** (`lmcache/v1/multiprocess/custom_types.py:18-83`):
```python
class CudaIPCWrapper:
    def __init__(self, tensor: torch.Tensor):
        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()  # ← Requires same-node GPU
        self.handle = handle
        self.device_uuid = ...

    def to_tensor(self):
        storage = torch.UntypedStorage._new_shared_cuda(...)  # ← IPC
        return t.view(self.shape)
```

**Server Architecture** (`lmcache/v1/multiprocess/server.py`):
- Uses `MPCacheEngine` with `MPStorageManager`
- CPU buffer with LRU eviction
- Handles: STORE, LOOKUP, RETRIEVE, REGISTER_KV_CACHE operations
- Message queue: ZeroMQ DEALER-ROUTER pattern

### Experimental Status

From `multiprocess_mode.rst:11-15`:
> This is an experimental feature and is under active development. Please expect breaking changes in the future.
>
> Currently, the multi-process mode only supports CPU offloading without eviction. It is not recommended for production use.

**Current Limitations**:
- ❌ No proper eviction policy (basic LRU only)
- ❌ CPU offloading only (no disk, remote, or distributed backends)
- ❌ Thread safety concerns (noted in TODOs)
- ❌ No storage backend plugins
- ❌ No distributed mode with sharding (future work)

## Your GB10 Deployment Requirements

### Your Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  GB10 Node #1    │     │  GB10 Node #2    │     │  GB10 Node #3    │
│  192.168.100.11  │     │  192.168.100.12  │     │  192.168.100.13  │
│  ┌────────────┐  │     │  ┌────────────┐  │     │  ┌────────────┐  │
│  │ vLLM Worker│  │     │  │ vLLM Worker│  │     │  │ vLLM Worker│  │
│  └────────────┘  │     │  └────────────┘  │     │  └────────────┘  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         │         ConnectX-7 Network (RDMA capable)       │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ↓
                   ┌──────────────────────────┐
                   │  Central Server          │
                   │  192.168.100.1           │
                   │  - CPU Cache (256GB)     │
                   │  - Disk Cache (2TB NVMe) │
                   └──────────────────────────┘
```

### Key Differences

| Aspect | Multiprocess Mode | Your GB10 Case |
|--------|------------------|----------------|
| **Topology** | Same-node (single machine) | Cross-node (multiple machines) |
| **Workers** | Multiple GPU workers on 1 node | Multiple GB10 devices (separate nodes) |
| **Network** | localhost IPC | 100 Gbps Ethernet with RDMA |
| **Data Transfer** | CUDA IPC (shared memory) | Network protocol (TCP/RDMA) |
| **Server Location** | Co-located (same node) | Centralized (separate node) |
| **Use Case** | Multi-GPU tensor parallelism | Distributed serving fleet |

## Why Multiprocess Mode Doesn't Work for You

### 1. CUDA IPC is Same-Node Only

**CUDA IPC Limitation**:
```python
# From custom_types.py:65
storage = tensor.untyped_storage()
handle = storage._share_cuda_()  # ← Creates IPC handle

# This handle only works for processes on the SAME machine
# sharing /dev/shm
```

**What happens across nodes**:
- CUDA IPC handles are **invalid** across network
- Requires shared `/dev/shm` filesystem (same kernel)
- GPU memory pointers are **not** network-addressable

### 2. Architecture Mismatch

**Multiprocess mode assumes**:
- Multiple vLLM instances on **one GPU node**
- Connecting to **localhost** cache server
- Example from K8s config:
  ```yaml
  # vllm-deployment.yaml:41-42
  "lmcache.mp.host": "tcp://${HOST_IP}"  # ← Same node IP
  "lmcache.mp.port": 6555
  ```

**Your requirement**:
- Multiple GB10 **nodes** (physically separate)
- Connecting to **remote** central server (192.168.100.1)
- Network-based KV cache sharing

### 3. No Cross-Node Support

From the documentation (`multiprocess_mode.rst:8-9`):
> In the future, each GPU node will have a single LMCache process running in multi-process mode. These LMCache processes will be interconnected to form a distributed KV cache service.

**Translation**: Cross-node distributed mode is **planned future work**, not currently implemented.

**Current state** (`multiprocess_mode.rst:356-361`):
> Future Work:
> - Distributed mode with sharding.

## Correct Approaches for Your GB10 Deployment

### Option 1: LM Server (Client-Server) ✅ Recommended

**Why it fits**:
- ✅ True client-server architecture (semantically correct)
- ✅ Works across network boundaries
- ✅ Each GB10 connects to central server via TCP
- ✅ Can use RDMA sockets for performance
- ✅ Simple configuration

**Configuration**:

**Central Server** (192.168.100.1):
```bash
# Start LM Server
lmcache_server 192.168.100.1 65432 cpu
```

**Each GB10 Worker**:
```yaml
# /etc/lmcache/worker_config.yaml
remote_url: "lm://192.168.100.1:65432"
local_cpu_cache_size_gb: 16
```

**Performance**: 5-8 GB/s with RDMA sockets

See: `GB10_CLIENT_SERVER_ADDENDUM.md` for complete details.

### Option 2: P2P Backend (Asymmetric) ⚠️ Higher Performance

**Why it could work**:
- ✅ Supports cross-node via NIXL/RDMA
- ✅ Best performance (10-12 GB/s)
- ✅ Leverages ConnectX-7 RDMA fully

**Why it's problematic**:
- ❌ Semantically wrong (designed for P2P, not client-server)
- ❌ Requires cache controller
- ❌ More complex setup

**Configuration**: See `GB10_DEPLOYMENT_GUIDE.md`

## Comparison Matrix

| Feature | Multiprocess Mode | LM Server | P2P Backend |
|---------|------------------|-----------|-------------|
| **Cross-Node** | ❌ No | ✅ Yes | ✅ Yes |
| **GB10 Compatible** | ❌ No | ✅ Yes | ✅ Yes |
| **RDMA Support** | N/A (same-node) | ✅ Via sockets | ✅ Native NIXL |
| **Semantic Fit** | N/A | ✅ Client-server | ⚠️ P2P (confusing) |
| **Setup Complexity** | Low | Low | Medium |
| **Production Ready** | ❌ Experimental | ✅ Stable | ✅ Stable |
| **Performance** | N/A | 5-8 GB/s | 10-12 GB/s |
| **Eviction Policy** | ❌ Basic | ✅ Full | ✅ Full |
| **Storage Backends** | ❌ CPU only | ✅ All backends | ✅ All backends |

## When to Use Multiprocess Mode

Multiprocess mode **is suitable** for these scenarios:

### ✅ Use Case 1: Multi-GPU Single Node

```
One DGX A100 node with 8 GPUs running multiple vLLM instances:
- GPU 0-3: vLLM instance #1 (TP=4)
- GPU 4-7: vLLM instance #2 (TP=4)
→ Both connect to localhost:5555 LMCache server
```

### ✅ Use Case 2: Kubernetes GPU Nodes

```
K8s cluster with GPU nodes:
- Each node: DaemonSet runs one LMCache server
- Multiple vLLM pods per node share the local cache
- Pods use ${HOST_IP} to find node-local server
```

### ✅ Use Case 3: Docker Compose Same-Host

```
docker-compose.yml on one machine:
  lmcache-server:
    ipc: host
    network_mode: host

  vllm-worker-1:
    ipc: host  # ← Shares /dev/shm
    network_mode: host
```

### ❌ NOT Suitable For:

- ❌ Cross-node distributed serving (your case)
- ❌ Separate physical machines
- ❌ Cloud VM instances without shared GPU
- ❌ Central cache server architecture

## Migration Path (If You Were Using Multiprocess Mode)

**Hypothetical scenario**: If you mistakenly deployed multiprocess mode thinking it works cross-node:

### What would happen:
```bash
# On central server (192.168.100.1)
python3 -m lmcache.v1.multiprocess.server --host 0.0.0.0 --port 6555

# On GB10 worker (192.168.100.11) - trying to connect
vllm serve ... \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", \
    "kv_connector_extra_config": {"lmcache.mp.host": "tcp://192.168.100.1", \
                                   "lmcache.mp.port": 6555}}'
```

### Expected failures:
1. **Connection succeeds** (ZeroMQ TCP works across network)
2. **KV cache registration succeeds** (metadata exchange works)
3. **Data transfer FAILS** (CUDA IPC handles are invalid across nodes)

**Error message**:
```
RuntimeError: Unable to open IPC handle from remote process
CUDA error: invalid device pointer
```

### Fix: Migrate to LM Server
```bash
# Replace multiprocess server with LM server
lmcache_server 192.168.100.1 65432 cpu

# Update worker config to use Remote Backend
# Change from: LMCacheMPConnector
# To: Remote Backend with lm:// URL
remote_url: "lm://192.168.100.1:65432"
```

## Recommendations for Your GB10 Deployment

### 1. **Immediate Action**: Use LM Server

Deploy the client-server architecture documented in `GB10_CLIENT_SERVER_ADDENDUM.md`:

```bash
# Central Server
bash deploy_lm_server.sh

# Each GB10 Worker
cat > /etc/lmcache/worker_config.yaml <<EOF
remote_url: "lm://192.168.100.1:65432"
local_cpu_cache_size_gb: 16
chunk_size: 256
EOF

vllm serve meta-llama/Llama-2-7b-hf --enable-lmcache ...
```

**Benefits**:
- ✅ Works across your GB10 nodes
- ✅ Semantically correct architecture
- ✅ Simple to deploy and debug
- ✅ Good performance (5-8 GB/s with RDMA sockets)

### 2. **Performance Optimization**: Add RDMA Sockets

Once basic setup works, optimize with RDMA-accelerated TCP:

```bash
# Install rsocket library on all nodes
apt-get install rdma-core librdmacm1

# Start server with RDMA preload
LD_PRELOAD=/usr/lib/rsocket/librspreload.so \
  lmcache_server 192.168.100.1 65432 cpu

# Start workers with RDMA preload
LD_PRELOAD=/usr/lib/rsocket/librspreload.so \
  vllm serve ...
```

**Expected improvement**: 1-5 GB/s → 5-8 GB/s

### 3. **Monitor and Watch Multiprocess Mode Development**

Keep an eye on multiprocess mode for future distributed features:

From roadmap:
> Future Work:
> - Distributed mode with sharding

When distributed mode is implemented (likely 6-12 months), re-evaluate whether it fits your use case.

## Conclusion

**For your GB10 deployment with multiple physically separate nodes**:

| Approach | Verdict | Reason |
|----------|---------|--------|
| **Multiprocess Mode** | ❌ Not Suitable | Same-node only, CUDA IPC limitation |
| **LM Server (Client-Server)** | ✅ **Recommended** | Cross-node, semantically correct, 5-8 GB/s |
| **P2P Backend** | ⚠️ Alternative | Cross-node, 10-12 GB/s, semantically confusing |

**Action Plan**:
1. Deploy LM Server architecture (see `GB10_CLIENT_SERVER_ADDENDUM.md`)
2. Test basic functionality with TCP
3. Optimize with RDMA sockets if needed
4. Monitor multiprocess mode roadmap for future distributed features

**Do NOT attempt to use multiprocess mode** for cross-node deployment - it will fail at the CUDA IPC layer even though ZeroMQ connections succeed.
