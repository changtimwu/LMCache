# GB10 Deployment: Client-Server vs. P2P Clarification

## Important Architectural Correction

The main deployment guide (`GB10_DEPLOYMENT_GUIDE.md`) uses **P2P Backend** configuration, which is **semantically incorrect** for a centralized client-server architecture. This addendum provides the **correct client-server approach** and explains the trade-offs.

## The Issue: P2P ≠ Client-Server

### What We Actually Need (Your Use Case):

```
┌─────────────────────┐
│  Central Server     │  ← Authority (server)
│  (Cache storage)    │
└──────────┬──────────┘
           │
     ┌─────┼─────┬──────────┐
     ↓     ↓     ↓          ↓
  [GB10] [GB10] [GB10]  [GB10]  ← Clients
```

- **Hierarchical**: Central server provides, workers consume
- **One-way**: Workers fetch from server only
- **Simple**: No peer discovery needed

### What P2P Backend Provides:

```
     Worker A ←──→ Worker B
        ↑              ↑
        └──────┬───────┘
               ↓
          Worker C + Server
```

- **Symmetric**: All nodes are equal peers
- **Multi-way**: Any worker can fetch from any other worker
- **Complex**: Requires cache controller for peer discovery

## Solution: True Client-Server Architecture

LMCache has a **proper client-server implementation** using the `lm://` protocol.

### How It Works

**Central Server:**
- Runs `lmcache_server` process
- Listens on TCP socket (e.g., port 65432)
- Stores KV caches in CPU memory / disk
- Responds to GET/PUT/EXISTS requests

**GB10 Workers:**
- Use `RemoteBackend` with `lm://` URL
- Connect to central server via TCP
- Fetch KV caches on demand
- Maintain local L1 CPU cache

### Architecture Comparison

| Aspect | P2P Backend (Original Guide) | LM Server (Correct Approach) |
|--------|-----------------------------|-----------------------------|
| **Semantics** | ❌ Peer-to-peer (wrong) | ✅ Client-server (correct) |
| **Protocol** | ZMQ + NIXL | TCP sockets |
| **RDMA Support** | ✅ Via NIXL | ❌ Plain TCP (but could use RDMA sockets) |
| **Controller Needed** | ✅ Yes | ❌ No |
| **Complexity** | High | Low |
| **Performance** | ~10-12 GB/s (RDMA) | ~1-10 GB/s (TCP) |

## Recommended Approach: Client-Server with RDMA Optimization

Since you have **ConnectX-7 NICs**, you can optimize TCP to use RDMA:

### Option 1: LM Server with RDMA Sockets (Recommended)

Use `lmcache_server` but bind to RDMA-capable network interface.

**Benefits:**
- ✅ Semantically correct (client-server)
- ✅ Leverages RDMA hardware
- ✅ Simple configuration
- ✅ No cache controller needed

**Performance:**
- With RDMA-accelerated TCP: 5-8 GB/s
- With standard TCP: 1-5 GB/s

### Option 2: Keep P2P but Document as Asymmetric

Use P2P backend but **clearly document** it's an asymmetric client-server topology.

**Benefits:**
- ✅ Best performance (10-12 GB/s via NIXL)
- ✅ Built-in RDMA support

**Drawbacks:**
- ❌ Semantically confusing
- ❌ Requires cache controller
- ❌ More complex

## Corrected Configuration Files

### Central Server: LM Server (Client-Server)

#### Configuration: `/etc/lmcache/server_config.yaml`

```yaml
# LMCache Central Server Configuration (CLIENT-SERVER)
# Using standalone lmcache_server for true client-server architecture

# Server binding
host: 192.168.100.1
port: 65432

# Storage Configuration
storage_backend: hybrid  # CPU + Disk

# CPU Cache
local_cpu_cache_size_gb: 256

# Disk Cache
enable_local_disk: true
local_disk_path: /mnt/nvme/lmcache
local_disk_size_gb: 2000

# Cache Configuration
chunk_size: 256

# Logging
log_level: INFO
```

#### Deployment Script: `deploy_lm_server.sh`

```bash
#!/bin/bash
# Deploy LMCache Server (True Client-Server)

set -euo pipefail

SERVER_IP="192.168.100.1"
SERVER_PORT="65432"
CPU_CACHE_GB="256"
DISK_PATH="/mnt/nvme/lmcache"
DISK_SIZE_GB="2000"

# Install dependencies
apt-get update
apt-get install -y python3-pip libibverbs-dev librdmacm-dev

pip3 install lmcache torch transformers

# Create directories
mkdir -p /etc/lmcache
mkdir -p /var/log/lmcache
mkdir -p ${DISK_PATH}

# Create systemd service
cat > /etc/systemd/system/lmcache-server.service <<EOF
[Unit]
Description=LMCache Server (Client-Server Architecture)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/lmcache

# Enable RDMA-accelerated TCP (optional but recommended)
Environment="LD_PRELOAD=/usr/lib/rsocket/librspreload.so"
Environment="RSOCKET_IOMAP_SIZE=64m"

ExecStart=/usr/local/bin/lmcache_server ${SERVER_IP} ${SERVER_PORT} cpu
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/lmcache/server.log
StandardError=append:/var/log/lmcache/server.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo "✅ LMCache Server installed"
echo ""
echo "Start server:"
echo "  systemctl start lmcache-server"
echo ""
echo "Enable auto-start:"
echo "  systemctl enable lmcache-server"
echo ""
echo "Test connection:"
echo "  nc -zv ${SERVER_IP} ${SERVER_PORT}"
```

### GB10 Workers: Remote Backend Clients

#### Configuration: `/etc/lmcache/worker_config.yaml`

```yaml
# LMCache Worker Configuration (CLIENT-SERVER)
# Uses Remote Backend to connect to central server

# Remote Backend Configuration (CLIENT-SERVER)
remote_url: "lm://192.168.100.1:65432"
remote_serde: safetensors  # Serialization format
blocking_timeout_secs: 30

# Local L1 Cache
local_cpu_cache_size_gb: 16

# Cache Configuration
chunk_size: 256

# Logging
log_level: INFO
```

#### vLLM Startup Script: `/opt/lmcache/start_vllm.sh`

```bash
#!/bin/bash
# Start vLLM with LMCache Remote Backend (Client-Server)

set -euo pipefail

MODEL_NAME="${1:-meta-llama/Llama-2-7b-hf}"
WORKER_ID="${2:-1}"

# Set configuration
export LMCACHE_CONFIG_FILE=/etc/lmcache/worker_config.yaml

# Start vLLM
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_NAME} \
  --enable-lmcache \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1

echo "✅ vLLM worker ${WORKER_ID} started"
echo "   Connected to: lm://192.168.100.1:65432"
echo "   Serving on: http://0.0.0.0:8000"
```

## Performance Optimization: RDMA-Accelerated TCP

To leverage your ConnectX-7 RDMA capability with TCP sockets:

### Install librspreload (RDMA Sockets Preload)

```bash
# On all nodes (server + workers)
apt-get install librdmacm-dev

# Build rsocket (RDMA sockets library)
git clone https://github.com/linux-rdma/rdma-core.git
cd rdma-core
mkdir build && cd build
cmake ..
make
make install

# Or use package manager
apt-get install rdma-core librdmacm1
```

### Enable RDMA Sockets

```bash
# For lmcache_server process
LD_PRELOAD=/usr/lib/rsocket/librspreload.so lmcache_server 192.168.100.1 65432 cpu

# For vLLM workers
LD_PRELOAD=/usr/lib/rsocket/librspreload.so python3 -m vllm.entrypoints.openai.api_server ...
```

**Expected Performance:**
- Without RDMA sockets: 1-5 GB/s (standard TCP)
- With RDMA sockets: 5-8 GB/s (RDMA-accelerated TCP)
- P2P with NIXL: 10-12 GB/s (optimal but semantically wrong)

## Deployment Decision Matrix

| Your Priority | Recommended Approach | Performance | Correctness |
|--------------|---------------------|-------------|-------------|
| **Semantic Correctness** | LM Server + Remote Backend | 1-5 GB/s | ✅ Perfect |
| **Semantic + Performance** | LM Server + RDMA Sockets | 5-8 GB/s | ✅ Perfect |
| **Maximum Performance** | P2P Backend (asymmetric) | 10-12 GB/s | ⚠️ Confusing |

## My Final Recommendation

**Use LM Server (Client-Server) with RDMA Socket Acceleration:**

### Why:
1. ✅ **Semantically correct**: True client-server architecture
2. ✅ **Good performance**: 5-8 GB/s with RDMA sockets
3. ✅ **Simpler**: No cache controller needed
4. ✅ **Clearer**: Workers explicitly connect to server URL
5. ✅ **Scalable**: Easy to add more workers

### Configuration Summary:

**Central Server:**
```bash
# Install and start
bash deploy_lm_server.sh
systemctl start lmcache-server
```

**Each GB10 Worker:**
```yaml
# /etc/lmcache/worker_config.yaml
remote_url: "lm://192.168.100.1:65432"
local_cpu_cache_size_gb: 16
```

```bash
# Start vLLM
bash /opt/lmcache/start_vllm.sh meta-llama/Llama-2-7b-hf 1
```

## Performance Expectations Revised

| Configuration | Bandwidth | Latency | Architecture |
|--------------|-----------|---------|--------------|
| **LM Server (TCP)** | 1-5 GB/s | ~5-20 ms | ✅ Client-Server |
| **LM Server (RDMA)** | 5-8 GB/s | ~1-5 ms | ✅ Client-Server |
| **P2P (NIXL)** | 10-12 GB/s | ~0.1-1 ms | ⚠️ Asymmetric P2P |

### Transfer Time Examples (Llama-2-7B, 1GB KV cache):

| Configuration | Transfer Time | Speedup vs Regeneration |
|--------------|---------------|------------------------|
| **LM Server (TCP)** | 200-1000 ms | 2-10x |
| **LM Server (RDMA)** | 125-200 ms | 5-16x |
| **P2P (NIXL)** | 80-100 ms | 10-25x |
| Regeneration | ~2000 ms | 1x baseline |

## Migration Guide

If you already deployed using P2P backend (from main guide):

### Step 1: Stop P2P Services

```bash
# On central server
systemctl stop lmcache-controller
systemctl stop lmcache-server

# On workers
systemctl stop vllm-worker
```

### Step 2: Deploy LM Server

```bash
# On central server
bash deploy_lm_server.sh
systemctl start lmcache-server
```

### Step 3: Update Worker Configs

```bash
# On each worker
cat > /etc/lmcache/worker_config.yaml <<EOF
remote_url: "lm://192.168.100.1:65432"
local_cpu_cache_size_gb: 16
chunk_size: 256
log_level: INFO
EOF
```

### Step 4: Restart Workers

```bash
# On each worker
systemctl restart vllm-worker
```

### Step 5: Verify

```bash
# Check server logs
tail -f /var/log/lmcache/server.log

# Should see:
# "Client connected from 192.168.100.11:xxxxx"
# "GET request for key=..."
# "Returned 1.2GB in 150ms"
```

## Conclusion

The **LM Server approach** is the **correct architectural choice** for your centralized KV cache use case:

- ✅ **True client-server**: Workers are clients, server is authority
- ✅ **Simpler deployment**: No controller, direct TCP connection
- ✅ **Good performance**: 5-8 GB/s with RDMA sockets (5-16x speedup)
- ✅ **Clear semantics**: No confusion about peer roles

The P2P approach in the main guide achieves **better raw performance** (10-12 GB/s) but is **semantically incorrect** and more complex. Use it only if you need absolute maximum performance and can tolerate the architectural confusion.

For **most use cases**, the **LM Server with RDMA sockets** provides the **best balance** of correctness, performance, and simplicity.
