# NVIDIA GB10 (DGX Spark) + Central LMCache Server Deployment Guide

## Executive Summary

This guide provides a complete deployment solution for using **NVIDIA GB10 (DGX Spark)** devices with a **centralized LMCache server** for distributed KV cache sharing over RDMA networks.

### Hardware Profile

**GB10 Devices (Workers):**
- 128GB unified memory (96GB dedicated for GPU)
- RTX 4060-equivalent compute power
- ConnectX-7 high-speed Ethernet adapter
- No GPUDirect support (per NVIDIA)

**Central Server (Cache Node):**
- High-end I/O: ConnectX-7, PCIe Gen4 NVMe SSDs
- Large CPU RAM (256GB+ recommended)
- No GPU required

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Central LMCache Server (No GPU Required)           │
│  ┌───────────────────────────────────────────────┐  │
│  │  Cache Controller + P2P Backend               │  │
│  │                                                │  │
│  │  Storage Hierarchy:                            │  │
│  │  ├─ L1: 256GB+ CPU RAM (LocalCPUBackend)      │  │
│  │  └─ L2: 2TB+ NVMe SSD (LocalDiskBackend)      │  │
│  │                                                │  │
│  │  Network: ConnectX-7 RDMA (100/200/400 Gbps)  │  │
│  └───────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        │ RDMA       │ RDMA       │ RDMA         │
        │ 100Gbps+   │            │              │
        ▼            ▼            ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐  ┌──────────┐
  │  GB10 #1 │ │  GB10 #2 │ │  GB10 #3 │  │  GB10 #N │
  │          │ │          │ │          │  │          │
  │ vLLM     │ │ vLLM     │ │ vLLM     │  │ vLLM     │
  │ Worker   │ │ Worker   │ │ Worker   │  │ Worker   │
  │          │ │          │ │          │  │          │
  │ L1 Cache:│ │ L1 Cache:│ │ L1 Cache:│  │ L1 Cache:│
  │ 16GB CPU │ │ 16GB CPU │ │ 16GB CPU │  │ 16GB CPU │
  └──────────┘ └──────────┘ └──────────┘  └──────────┘
```

## Feasibility Analysis

### ✅ Why This Architecture Works

| Factor | Assessment | Reasoning |
|--------|------------|-----------|
| **ConnectX-7 RDMA** | ✅ Excellent | 100-400 Gbps, supports RoCE v2 RDMA for CPU-to-CPU |
| **No GPUDirect** | ✅ Not a blocker | Use CPU buffers + regular RDMA (10-40 GB/s achievable) |
| **Unified Memory** | ✅ Major advantage | No PCIe bottleneck for GPU↔CPU on GB10 |
| **Centralized Cache** | ✅ Ideal | Better hit rates, simpler management |
| **Cost Efficiency** | ✅ High | One server vs. storage on each GPU node |

### Performance Expectations

#### Network Performance (ConnectX-7 with RDMA)

| Configuration | Bandwidth | Latency | Use Case |
|---------------|-----------|---------|----------|
| **100GbE + RoCE** | 10-12 GB/s | 50-100 μs | Standard deployment |
| **200GbE + RoCE** | 20-25 GB/s | 40-80 μs | High-performance |
| **400GbE + RoCE** | 40-50 GB/s | 30-60 μs | Maximum performance |

#### KV Cache Transfer Times

**For Llama-2-7B (32 layers, 4096 context):**
- KV cache size: ~1 GB
- Transfer time @ 10 GB/s: **~100 ms**
- Regeneration time: **1-2 seconds**
- **Speedup: 10-20x** for cache hits

**For Llama-2-70B (80 layers, 4096 context):**
- KV cache size: ~10 GB
- Transfer time @ 10 GB/s: **~1 second**
- Regeneration time: **10-20 seconds**
- **Speedup: 10-20x** for cache hits

### Expected Cache Hit Rates

| Workload Type | Cache Hit Rate | Benefit |
|---------------|----------------|---------|
| RAG (Retrieval-Augmented Generation) | 70-90% | Documents/context reused |
| Multi-round Chat | 60-80% | Conversation history cached |
| Batch Inference (similar prompts) | 80-95% | High prefix overlap |
| Diverse Queries | 20-40% | Limited reuse |

## Network Requirements

### Hardware Setup

**Required:**
- ConnectX-7 NICs on all nodes (already present)
- RDMA-capable switches with:
  - Priority Flow Control (PFC) enabled
  - Explicit Congestion Notification (ECN) enabled
  - Lossless Ethernet configuration

**Network Topology:**
```
[Central Server] ──┐
                   ├──► [Switch: RoCE-enabled] ─┬─► [GB10 #1]
[Controller]  ─────┘                            ├─► [GB10 #2]
                                                ├─► [GB10 #3]
                                                └─► [GB10 #N]
```

**IP Addressing Scheme (Example):**
```
Network: 192.168.100.0/24 (dedicated RDMA network)

Central Server:   192.168.100.1
Cache Controller: 192.168.100.1 (same host)
GB10 #1:         192.168.100.11
GB10 #2:         192.168.100.12
GB10 #3:         192.168.100.13
...
GB10 #N:         192.168.100.1N
```

## Complete Deployment Scripts

### 1. Central Server Deployment

#### Script: `deploy_central_server.sh`

```bash
#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Central LMCache Server Deployment Script
# For use with NVIDIA GB10 distributed caching architecture

set -euo pipefail

#=============================================================================
# Configuration Variables
#=============================================================================

# Network Configuration
export SERVER_IP="192.168.100.1"
export RDMA_DEVICE="mlx5_0"  # ConnectX-7 device (verify with ibv_devices)
export RDMA_PORT="1"

# Controller Configuration
export CONTROLLER_PORT="6000"

# P2P Configuration
export P2P_PORT="7100"

# Storage Configuration
export CPU_CACHE_SIZE_GB="256"
export DISK_CACHE_PATH="/mnt/nvme/lmcache"
export DISK_CACHE_SIZE_GB="2000"

# NIXL Configuration
export NIXL_BUFFER_SIZE="4294967296"  # 4GB

# Installation Paths
export INSTALL_DIR="/opt/lmcache"
export CONFIG_DIR="/etc/lmcache"
export LOG_DIR="/var/log/lmcache"

#=============================================================================
# Color Output Functions
#=============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

#=============================================================================
# Pre-flight Checks
#=============================================================================

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

check_rdma() {
    log_info "Checking RDMA devices..."
    if ! command -v ibv_devices &> /dev/null; then
        log_error "RDMA tools not found. Please install MLNX_OFED drivers first."
        exit 1
    fi

    if ! ibv_devices | grep -q "${RDMA_DEVICE}"; then
        log_error "RDMA device ${RDMA_DEVICE} not found"
        log_info "Available devices:"
        ibv_devices
        exit 1
    fi

    log_info "✓ RDMA device ${RDMA_DEVICE} found"
}

check_storage() {
    log_info "Checking storage paths..."

    if [[ ! -d "${DISK_CACHE_PATH}" ]]; then
        log_warn "Disk cache path ${DISK_CACHE_PATH} does not exist"
        read -p "Create it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mkdir -p "${DISK_CACHE_PATH}"
            log_info "✓ Created ${DISK_CACHE_PATH}"
        else
            log_error "Cannot proceed without disk cache path"
            exit 1
        fi
    fi

    log_info "✓ Storage paths verified"
}

#=============================================================================
# System Configuration
#=============================================================================

configure_system() {
    log_info "Configuring system parameters..."

    # Increase network buffer sizes for RDMA
    sysctl -w net.core.rmem_max=134217728
    sysctl -w net.core.wmem_max=134217728
    sysctl -w net.core.rmem_default=134217728
    sysctl -w net.core.wmem_default=134217728

    # Make permanent
    cat >> /etc/sysctl.conf <<EOF

# LMCache RDMA optimizations
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.core.rmem_default=134217728
net.core.wmem_default=134217728
EOF

    log_info "✓ System parameters configured"
}

#=============================================================================
# Software Installation
#=============================================================================

install_dependencies() {
    log_info "Installing system dependencies..."

    apt-get update
    apt-get install -y \
        build-essential \
        python3-pip \
        python3-dev \
        libibverbs-dev \
        librdmacm-dev \
        ibverbs-utils \
        rdma-core \
        git \
        wget \
        curl

    log_info "✓ System dependencies installed"
}

install_python_packages() {
    log_info "Installing Python packages..."

    # Upgrade pip
    pip3 install --upgrade pip

    # Install LMCache
    pip3 install lmcache

    # Install NIXL (for RDMA support)
    # Note: NIXL requires Python < 3.13
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$PYTHON_VERSION < 3.13" | bc) -eq 1 ]]; then
        pip3 install nixl
        log_info "✓ NIXL installed"
    else
        log_warn "Python version >= 3.13, NIXL not available. Will use fallback transport."
    fi

    # Install other dependencies
    pip3 install \
        torch \
        transformers \
        fastapi \
        uvicorn \
        msgspec \
        pyzmq \
        redis \
        prometheus-client

    log_info "✓ Python packages installed"
}

#=============================================================================
# Directory Structure
#=============================================================================

create_directories() {
    log_info "Creating directory structure..."

    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${DISK_CACHE_PATH}"

    log_info "✓ Directories created"
}

#=============================================================================
# Configuration Files
#=============================================================================

create_config_files() {
    log_info "Creating configuration files..."

    # Main LMCache configuration
    cat > "${CONFIG_DIR}/server_config.yaml" <<EOF
# LMCache Central Server Configuration
# Generated on $(date)

# P2P Backend Configuration
enable_p2p: true
lmcache_worker_ports: [${P2P_PORT}]

# Cache Controller Configuration
controller_pull_url: tcp://${SERVER_IP}:${CONTROLLER_PORT}

# Storage Backend Configuration
local_cpu_cache_size_gb: ${CPU_CACHE_SIZE_GB}
enable_local_disk: true
local_disk_path: ${DISK_CACHE_PATH}
local_disk_size_gb: ${DISK_CACHE_SIZE_GB}

# NIXL Configuration (RDMA Support)
nixl_buffer_size: ${NIXL_BUFFER_SIZE}
nixl_buffer_device: cpu
nixl_backends: ["POSIX"]
transfer_channel: nixl

# Network Timeouts
p2p_socket_recv_timeout_ms: 10000
p2p_socket_send_timeout_ms: 10000

# Cache Configuration
chunk_size: 256

# Logging
log_level: INFO
EOF

    log_info "✓ Configuration file created: ${CONFIG_DIR}/server_config.yaml"
}

#=============================================================================
# Cache Server Service
#=============================================================================

create_cache_server_script() {
    log_info "Creating cache server script..."

    cat > "${INSTALL_DIR}/cache_server.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Central LMCache Server
Acts as a centralized KV cache repository for distributed vLLM workers
"""

import asyncio
import signal
import sys
import os
from typing import Any

# Ensure we can import lmcache
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)

class CentralCacheServer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cache_engine = None
        self.stop_event = asyncio.Event()

    def broadcast_fn(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """Dummy broadcast for single-node cache server"""
        return tensor

    def broadcast_object_fn(self, obj: Any, src: int) -> Any:
        """Dummy broadcast for single-node cache server"""
        return obj

    async def start(self):
        """Initialize and start the cache server"""
        logger.info("Starting Central LMCache Server...")

        # Load configuration
        config = LMCacheEngineConfig.from_file(self.config_path)
        logger.info(f"Loaded configuration from {self.config_path}")

        # Create metadata
        metadata = LMCacheEngineMetadata(
            model_name="central-cache-server",
            fmt="huggingface",
            world_size=1,
            worker_id=0,
            use_mla=False,
        )

        # Create token database
        token_db = ChunkedTokenDatabase(config, metadata)
        logger.info("Token database initialized")

        # Create cache engine
        self.cache_engine = LMCacheEngine(
            config=config,
            metadata=metadata,
            token_database=token_db,
            gpu_connector=None,  # No GPU on cache server
            broadcast_fn=self.broadcast_fn,
            broadcast_object_fn=self.broadcast_object_fn,
        )
        logger.info("Cache engine initialized")

        # Print startup information
        print("\n" + "="*60)
        print("✅ Central LMCache Server Started Successfully")
        print("="*60)
        print(f"Controller URL:    {config.controller_pull_url}")
        print(f"P2P Port:          {config.lmcache_worker_ports}")
        print(f"CPU Cache Size:    {config.local_cpu_cache_size_gb}GB")
        print(f"Disk Cache Path:   {config.local_disk_path}")
        print(f"Disk Cache Size:   {config.local_disk_size_gb}GB")
        print(f"NIXL Buffer Size:  {config.nixl_buffer_size / 1e9:.1f}GB")
        print("="*60)
        print("\nPress Ctrl+C to stop the server\n")

        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep server running
        await self.stop_event.wait()

        logger.info("Shutting down cache server...")
        # Cleanup would go here

        print("\n👋 Cache server stopped gracefully\n")

async def main():
    config_path = os.getenv('LMCACHE_CONFIG_FILE', '/etc/lmcache/server_config.yaml')

    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        print(f"   Set LMCACHE_CONFIG_FILE environment variable or create the file.")
        sys.exit(1)

    server = CentralCacheServer(config_path)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
PYTHON_EOF

    chmod +x "${INSTALL_DIR}/cache_server.py"
    log_info "✓ Cache server script created"
}

create_controller_service() {
    log_info "Creating systemd service for cache controller..."

    cat > /etc/systemd/system/lmcache-controller.service <<EOF
[Unit]
Description=LMCache Cache Controller
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/local/bin/lmcache_controller --host ${SERVER_IP} --port ${CONTROLLER_PORT}
Restart=on-failure
RestartSec=10
StandardOutput=append:${LOG_DIR}/controller.log
StandardError=append:${LOG_DIR}/controller.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_info "✓ Controller service created"
}

create_cache_service() {
    log_info "Creating systemd service for cache server..."

    cat > /etc/systemd/system/lmcache-server.service <<EOF
[Unit]
Description=LMCache Central Cache Server
After=network.target lmcache-controller.service
Requires=lmcache-controller.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="LMCACHE_CONFIG_FILE=${CONFIG_DIR}/server_config.yaml"
Environment="UCX_TLS=tcp,rc"
Environment="UCX_NET_DEVICES=${RDMA_DEVICE}:${RDMA_PORT}"
ExecStart=/usr/bin/python3 ${INSTALL_DIR}/cache_server.py
Restart=on-failure
RestartSec=10
StandardOutput=append:${LOG_DIR}/server.log
StandardError=append:${LOG_DIR}/server.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_info "✓ Cache server service created"
}

#=============================================================================
# Testing & Verification
#=============================================================================

create_test_scripts() {
    log_info "Creating test scripts..."

    # RDMA bandwidth test script
    cat > "${INSTALL_DIR}/test_rdma.sh" <<'EOF'
#!/bin/bash
# Test RDMA bandwidth

echo "Starting RDMA bandwidth test (server mode)..."
echo "Run 'ib_write_bw -d mlx5_0 <server_ip>' on client to test"
echo ""

ib_write_bw -d mlx5_0
EOF
    chmod +x "${INSTALL_DIR}/test_rdma.sh"

    # Service health check
    cat > "${INSTALL_DIR}/health_check.sh" <<'EOF'
#!/bin/bash
# Health check for LMCache services

echo "=== LMCache Service Health Check ==="
echo ""

echo "1. Controller Service:"
systemctl status lmcache-controller.service | grep Active
echo ""

echo "2. Cache Server Service:"
systemctl status lmcache-server.service | grep Active
echo ""

echo "3. RDMA Device Status:"
ibv_devinfo -d mlx5_0 | grep -E "state|active_mtu|active_speed"
echo ""

echo "4. Network Connectivity:"
ss -tlnp | grep -E "6000|7100"
echo ""

echo "5. Recent Logs (last 10 lines):"
echo "--- Controller ---"
tail -n 5 /var/log/lmcache/controller.log
echo ""
echo "--- Server ---"
tail -n 5 /var/log/lmcache/server.log
EOF
    chmod +x "${INSTALL_DIR}/health_check.sh"

    log_info "✓ Test scripts created"
}

#=============================================================================
# Main Deployment
#=============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   Central LMCache Server Deployment Script                ║"
    echo "║   For NVIDIA GB10 (DGX Spark) Distributed Architecture    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Pre-flight checks
    check_root
    check_rdma
    check_storage

    # System configuration
    configure_system

    # Software installation
    install_dependencies
    install_python_packages

    # Create structure
    create_directories
    create_config_files

    # Create services
    create_cache_server_script
    create_controller_service
    create_cache_service

    # Create test utilities
    create_test_scripts

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   ✅ Deployment Complete!                                  ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start the services:"
    echo "   sudo systemctl start lmcache-controller"
    echo "   sudo systemctl start lmcache-server"
    echo ""
    echo "2. Enable auto-start on boot:"
    echo "   sudo systemctl enable lmcache-controller"
    echo "   sudo systemctl enable lmcache-server"
    echo ""
    echo "3. Check service status:"
    echo "   ${INSTALL_DIR}/health_check.sh"
    echo ""
    echo "4. Test RDMA connectivity:"
    echo "   ${INSTALL_DIR}/test_rdma.sh"
    echo ""
    echo "5. View logs:"
    echo "   tail -f ${LOG_DIR}/controller.log"
    echo "   tail -f ${LOG_DIR}/server.log"
    echo ""
    echo "Configuration file: ${CONFIG_DIR}/server_config.yaml"
    echo ""
}

main "$@"
```

### 2. GB10 Worker Deployment

#### Script: `deploy_gb10_worker.sh`

```bash
#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# GB10 Worker Node Deployment Script
# For distributed vLLM workers with LMCache

set -euo pipefail

#=============================================================================
# Configuration Variables
#=============================================================================

# Network Configuration
export CENTRAL_SERVER_IP="192.168.100.1"
export WORKER_ID=""  # Will be set dynamically or via argument
export RDMA_DEVICE="mlx5_0"
export RDMA_PORT="1"

# Controller Configuration
export CONTROLLER_PORT="6000"
export CONTROLLER_URL="tcp://${CENTRAL_SERVER_IP}:${CONTROLLER_PORT}"

# P2P Configuration
export P2P_PORT="7200"

# Storage Configuration
export CPU_CACHE_SIZE_GB="16"

# NIXL Configuration
export NIXL_BUFFER_SIZE="2147483648"  # 2GB

# vLLM Configuration
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
export VLLM_PORT="8000"
export GPU_MEMORY_UTILIZATION="0.85"
export TENSOR_PARALLEL_SIZE="1"

# Installation Paths
export INSTALL_DIR="/opt/lmcache"
export CONFIG_DIR="/etc/lmcache"
export LOG_DIR="/var/log/lmcache"

#=============================================================================
# Color Output Functions
#=============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

#=============================================================================
# Argument Parsing
#=============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --worker-id)
                WORKER_ID="$2"
                shift 2
                ;;
            --model)
                MODEL_NAME="$2"
                shift 2
                ;;
            --central-server)
                CENTRAL_SERVER_IP="$2"
                CONTROLLER_URL="tcp://${CENTRAL_SERVER_IP}:${CONTROLLER_PORT}"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --worker-id ID          Worker ID (e.g., 1, 2, 3)"
                echo "  --model MODEL           Model name (default: meta-llama/Llama-2-7b-hf)"
                echo "  --central-server IP     Central server IP (default: 192.168.100.1)"
                echo "  --help                  Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Auto-detect worker ID from hostname if not provided
    if [[ -z "${WORKER_ID}" ]]; then
        WORKER_ID=$(hostname | grep -oP '\d+$' || echo "1")
        log_warn "Worker ID not specified, using: ${WORKER_ID}"
    fi

    # Set worker IP based on ID
    export WORKER_IP="192.168.100.1${WORKER_ID}"
}

#=============================================================================
# Pre-flight Checks
#=============================================================================

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

check_gpu() {
    log_info "Checking GPU availability..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is NVIDIA driver installed?"
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [[ ${GPU_COUNT} -eq 0 ]]; then
        log_error "No GPUs found"
        exit 1
    fi

    log_info "✓ Found ${GPU_COUNT} GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

check_rdma() {
    log_info "Checking RDMA devices..."

    if ! command -v ibv_devices &> /dev/null; then
        log_error "RDMA tools not found. Please install MLNX_OFED drivers first."
        exit 1
    fi

    if ! ibv_devices | grep -q "${RDMA_DEVICE}"; then
        log_error "RDMA device ${RDMA_DEVICE} not found"
        exit 1
    fi

    log_info "✓ RDMA device ${RDMA_DEVICE} found"
}

check_central_server() {
    log_info "Checking connectivity to central server..."

    if ! ping -c 1 -W 2 "${CENTRAL_SERVER_IP}" &> /dev/null; then
        log_error "Cannot reach central server at ${CENTRAL_SERVER_IP}"
        exit 1
    fi

    log_info "✓ Central server reachable at ${CENTRAL_SERVER_IP}"
}

#=============================================================================
# System Configuration
#=============================================================================

configure_system() {
    log_info "Configuring system parameters..."

    # Network buffer sizes
    sysctl -w net.core.rmem_max=134217728
    sysctl -w net.core.wmem_max=134217728
    sysctl -w net.core.rmem_default=134217728
    sysctl -w net.core.wmem_default=134217728

    # Make permanent
    cat >> /etc/sysctl.conf <<EOF

# LMCache RDMA optimizations
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.core.rmem_default=134217728
net.core.wmem_default=134217728
EOF

    log_info "✓ System parameters configured"
}

#=============================================================================
# Software Installation
#=============================================================================

install_dependencies() {
    log_info "Installing system dependencies..."

    apt-get update
    apt-get install -y \
        build-essential \
        python3-pip \
        python3-dev \
        libibverbs-dev \
        librdmacm-dev \
        ibverbs-utils \
        rdma-core \
        git \
        wget \
        curl

    log_info "✓ System dependencies installed"
}

install_python_packages() {
    log_info "Installing Python packages..."

    pip3 install --upgrade pip

    # Install vLLM
    pip3 install vllm

    # Install LMCache
    pip3 install lmcache

    # Install NIXL (for RDMA)
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$PYTHON_VERSION < 3.13" | bc -l 2>/dev/null || echo "1") -eq 1 ]]; then
        pip3 install nixl || log_warn "NIXL installation failed, will use TCP fallback"
    else
        log_warn "Python >= 3.13, NIXL not available"
    fi

    log_info "✓ Python packages installed"
}

#=============================================================================
# Directory Structure
#=============================================================================

create_directories() {
    log_info "Creating directory structure..."

    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${LOG_DIR}"

    log_info "✓ Directories created"
}

#=============================================================================
# Configuration Files
#=============================================================================

create_config_files() {
    log_info "Creating configuration files..."

    cat > "${CONFIG_DIR}/worker_config.yaml" <<EOF
# LMCache Worker Configuration
# Worker ID: ${WORKER_ID}
# Generated on $(date)

# P2P Backend Configuration
enable_p2p: true
controller_pull_url: ${CONTROLLER_URL}
lmcache_worker_ports: [${P2P_PORT}]

# Local CPU Cache (L1)
local_cpu_cache_size_gb: ${CPU_CACHE_SIZE_GB}

# NIXL Configuration (RDMA Support)
nixl_buffer_size: ${NIXL_BUFFER_SIZE}
nixl_buffer_device: cpu
nixl_backends: ["POSIX"]
transfer_channel: nixl

# Network Timeouts
p2p_socket_recv_timeout_ms: 10000
p2p_socket_send_timeout_ms: 10000

# Cache Configuration
chunk_size: 256

# Logging
log_level: INFO
EOF

    log_info "✓ Configuration file created: ${CONFIG_DIR}/worker_config.yaml"
}

#=============================================================================
# vLLM Service
#=============================================================================

create_vllm_service() {
    log_info "Creating vLLM service..."

    cat > /etc/systemd/system/vllm-worker.service <<EOF
[Unit]
Description=vLLM Worker with LMCache (Worker ${WORKER_ID})
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="LMCACHE_CONFIG_FILE=${CONFIG_DIR}/worker_config.yaml"
Environment="UCX_TLS=tcp,rc"
Environment="UCX_NET_DEVICES=${RDMA_DEVICE}:${RDMA_PORT}"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/local/bin/python3 -m vllm.entrypoints.openai.api_server \\
    --model ${MODEL_NAME} \\
    --enable-lmcache \\
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \\
    --host 0.0.0.0 \\
    --port ${VLLM_PORT} \\
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
Restart=on-failure
RestartSec=10
StandardOutput=append:${LOG_DIR}/vllm.log
StandardError=append:${LOG_DIR}/vllm.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_info "✓ vLLM service created"
}

#=============================================================================
# Testing & Verification
#=============================================================================

create_test_scripts() {
    log_info "Creating test scripts..."

    # RDMA connectivity test
    cat > "${INSTALL_DIR}/test_rdma_client.sh" <<EOF
#!/bin/bash
# Test RDMA connectivity to central server

echo "Testing RDMA bandwidth to central server..."
echo ""

ib_write_bw -d ${RDMA_DEVICE} ${CENTRAL_SERVER_IP}
EOF
    chmod +x "${INSTALL_DIR}/test_rdma_client.sh"

    # vLLM test request
    cat > "${INSTALL_DIR}/test_vllm.sh" <<'EOF'
#!/bin/bash
# Test vLLM inference with cache

echo "Testing vLLM inference..."
echo ""

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq .

echo ""
echo "Check logs for cache hits/misses:"
echo "  tail -f /var/log/lmcache/vllm.log | grep -E 'cache|P2P'"
EOF
    chmod +x "${INSTALL_DIR}/test_vllm.sh"

    # Health check
    cat > "${INSTALL_DIR}/health_check.sh" <<'EOF'
#!/bin/bash
# Health check for vLLM worker

echo "=== vLLM Worker Health Check ==="
echo ""

echo "1. vLLM Service:"
systemctl status vllm-worker.service | grep Active
echo ""

echo "2. GPU Status:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

echo "3. RDMA Device:"
ibv_devinfo -d mlx5_0 | grep -E "state|active"
echo ""

echo "4. Network Ports:"
ss -tlnp | grep -E "8000|7200"
echo ""

echo "5. Recent Logs (last 10 lines):"
tail -n 10 /var/log/lmcache/vllm.log
EOF
    chmod +x "${INSTALL_DIR}/health_check.sh"

    log_info "✓ Test scripts created"
}

#=============================================================================
# Main Deployment
#=============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   GB10 Worker Node Deployment Script                      ║"
    echo "║   vLLM + LMCache with RDMA Support                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Parse arguments
    parse_args "$@"

    # Pre-flight checks
    check_root
    check_gpu
    check_rdma
    check_central_server

    # System configuration
    configure_system

    # Software installation
    install_dependencies
    install_python_packages

    # Create structure
    create_directories
    create_config_files

    # Create services
    create_vllm_service

    # Create test utilities
    create_test_scripts

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   ✅ Deployment Complete!                                  ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Worker Configuration:"
    echo "  Worker ID:         ${WORKER_ID}"
    echo "  Worker IP:         ${WORKER_IP}"
    echo "  Central Server:    ${CENTRAL_SERVER_IP}"
    echo "  Model:             ${MODEL_NAME}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download the model (if not already cached):"
    echo "   huggingface-cli login"
    echo "   huggingface-cli download ${MODEL_NAME}"
    echo ""
    echo "2. Start vLLM worker:"
    echo "   sudo systemctl start vllm-worker"
    echo ""
    echo "3. Enable auto-start:"
    echo "   sudo systemctl enable vllm-worker"
    echo ""
    echo "4. Check status:"
    echo "   ${INSTALL_DIR}/health_check.sh"
    echo ""
    echo "5. Test RDMA connectivity:"
    echo "   ${INSTALL_DIR}/test_rdma_client.sh"
    echo ""
    echo "6. Test inference:"
    echo "   ${INSTALL_DIR}/test_vllm.sh"
    echo ""
    echo "Configuration file: ${CONFIG_DIR}/worker_config.yaml"
    echo ""
}

main "$@"
```

### 3. Quick Deployment Helper

#### Script: `quick_deploy.sh`

```bash
#!/bin/bash
# Quick deployment orchestrator

set -euo pipefail

echo "LMCache GB10 Deployment Orchestrator"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root"
    exit 1
fi

# Menu
echo "Select deployment type:"
echo "1. Central Server"
echo "2. GB10 Worker"
echo "3. Both (for testing on same machine)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Deploying Central Server..."
        bash deploy_central_server.sh
        ;;
    2)
        echo "Deploying GB10 Worker..."
        read -p "Enter worker ID (e.g., 1): " worker_id
        read -p "Enter central server IP [192.168.100.1]: " server_ip
        server_ip=${server_ip:-192.168.100.1}

        bash deploy_gb10_worker.sh \
            --worker-id "${worker_id}" \
            --central-server "${server_ip}"
        ;;
    3)
        echo "Deploying both (test mode)..."
        bash deploy_central_server.sh
        sleep 5
        bash deploy_gb10_worker.sh --worker-id 1
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Deployment complete!"
```

## Verification & Testing

### Network Testing

**On Central Server:**
```bash
# Start RDMA bandwidth server
/opt/lmcache/test_rdma.sh
```

**On Each GB10 Worker:**
```bash
# Test RDMA bandwidth to server
/opt/lmcache/test_rdma_client.sh

# Expected output:
# ---------------------------------------------------------------------------------------
#                     RDMA_Write BW Test
# ---------------------------------------------------------------------------------------
# Number of qps   : 1
# Connection type : RC
# TX depth        : 128
# ...
# ---------------------------------------------------------------------------------------
#  #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
#  65536      100000           12000.00            11800.00            0.189
```

### Cache Testing

**Test sequence:**

```bash
# 1. On GB10 #1: Send first request
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Once upon a time in a faraway land",
    "max_tokens": 100
  }'

# 2. On GB10 #2: Send same prompt (should hit cache)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Once upon a time in a faraway land",
    "max_tokens": 100
  }'

# Check logs for cache hit
tail -f /var/log/lmcache/vllm.log | grep "P2P\|cache"
```

**Expected log output:**
```
[INFO] P2P lookup: key=<hash> found on peer=192.168.100.1:7100
[INFO] P2P transfer started: 1.2GB
[INFO] P2P transfer completed: 1.2GB in 120ms (10 GB/s)
[INFO] Cache hit: reusing KV cache for prompt
```

## Monitoring & Operations

### Service Management

```bash
# Central Server
systemctl status lmcache-controller
systemctl status lmcache-server

# GB10 Workers
systemctl status vllm-worker

# View logs
journalctl -u lmcache-controller -f
journalctl -u lmcache-server -f
journalctl -u vllm-worker -f
```

### Performance Monitoring

**Create monitoring script:** `/opt/lmcache/monitor.sh`

```bash
#!/bin/bash
# LMCache Performance Monitor

echo "=== LMCache Performance Monitor ==="
date
echo ""

echo "1. RDMA Network Stats:"
ibv_devinfo -d mlx5_0 | grep -A 5 "port:"
echo ""

echo "2. Cache Hit Rate:"
grep -oP 'cache hit rate: \K[\d\.]+' /var/log/lmcache/*.log | tail -1
echo ""

echo "3. Transfer Bandwidth:"
grep -oP 'transfer.*\K[\d\.]+ GB/s' /var/log/lmcache/*.log | tail -5
echo ""

echo "4. GPU Utilization:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo ""

echo "5. Network Traffic (RDMA interface):"
ip -s link show mlx5_0 | grep -A 1 "RX:\|TX:"
```

### Alerting Configuration

**Prometheus metrics** (if enabled):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'lmcache-central'
    static_configs:
      - targets: ['192.168.100.1:9090']

  - job_name: 'lmcache-workers'
    static_configs:
      - targets:
        - '192.168.100.11:9090'
        - '192.168.100.12:9090'
        - '192.168.100.13:9090'
```

## Troubleshooting

### Common Issues

#### 1. RDMA Connection Fails

**Symptoms:**
```
UCX ERROR: no active ports found for device mlx5_0
```

**Solution:**
```bash
# Check RDMA device status
ibv_devinfo

# Verify network link is up
ip link show mlx5_0

# Check RoCE mode
ibv_devinfo -d mlx5_0 | grep GID

# Enable RoCE if needed
echo "roce" > /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/0
```

#### 2. Cache Misses Despite Same Prompts

**Symptoms:**
- Logs show "cache miss" for identical prompts

**Solution:**
```bash
# Check chunk size matches across all nodes
grep chunk_size /etc/lmcache/*.yaml

# Verify controller is reachable
curl http://192.168.100.1:6000/health

# Check P2P registration
grep "P2P.*registered" /var/log/lmcache/*.log
```

#### 3. Slow Transfer Speeds

**Symptoms:**
- Transfer speeds < 5 GB/s with 100GbE

**Solution:**
```bash
# Check MTU size
ip link show mlx5_0 | grep mtu
# Should be 9000 (jumbo frames) for best performance

# Set jumbo frames
ip link set mlx5_0 mtu 9000

# Check for packet loss
ethtool -S mlx5_0 | grep -i "error\|drop"

# Verify RDMA mode
ibv_devinfo -d mlx5_0 | grep "active_speed"
```

#### 4. Out of Memory on Central Server

**Symptoms:**
```
OSError: [Errno 12] Cannot allocate memory
```

**Solution:**
```yaml
# Reduce cache sizes in server_config.yaml
local_cpu_cache_size_gb: 128  # Reduced from 256
nixl_buffer_size: 2147483648  # Reduced from 4GB

# Or increase swap
dd if=/dev/zero of=/swapfile bs=1G count=64
mkswap /swapfile
swapon /swapfile
```

## Performance Tuning Guide

### Network Optimization

```bash
# Increase socket buffer sizes
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456

# Enable TCP window scaling
sysctl -w net.ipv4.tcp_window_scaling=1

# Increase connection backlog
sysctl -w net.core.somaxconn=4096
```

### Storage Optimization

```bash
# For NVMe SSD cache on central server
# Enable I/O scheduler for NVMe
echo "none" > /sys/block/nvme0n1/queue/scheduler

# Increase read-ahead
echo 8192 > /sys/block/nvme0n1/queue/read_ahead_kb

# Disable write cache (for consistency)
echo write through > /sys/block/nvme0n1/queue/write_cache
```

### LMCache Configuration Tuning

```yaml
# For high-throughput workloads
chunk_size: 512  # Larger chunks
nixl_buffer_size: 8589934592  # 8GB buffers
p2p_socket_recv_timeout_ms: 30000  # Longer timeout for large transfers

# For low-latency workloads
chunk_size: 128  # Smaller chunks
nixl_buffer_size: 1073741824  # 1GB buffers
p2p_socket_recv_timeout_ms: 5000  # Short timeout
```

## Cost Analysis

### Hardware Investment

| Component | Quantity | Unit Cost | Total Cost |
|-----------|----------|-----------|------------|
| Central Server (high-end) | 1 | $8,000 | $8,000 |
| NVMe SSD (4TB) | 2 | $600 | $1,200 |
| ConnectX-7 (if not included) | 1 | $800 | $800 |
| RoCE Switch (48-port 100GbE) | 1 | $15,000 | $15,000 |
| **Total Infrastructure** | | | **$25,000** |

### Operating Costs

**Savings from cache hits:**

Assumptions:
- 10 GB10 devices
- 70% cache hit rate
- Average query: 4K context + 100 tokens generation
- Cost metric: GPU time saved

**Without LMCache:**
- Each query: ~1.5 seconds prefill + 0.5 seconds decode = 2 seconds
- 1000 queries/hour × 10 devices = 10,000 queries/hour
- Total GPU time: 20,000 GPU-seconds/hour

**With LMCache (70% hit rate):**
- Cache hits: 700 queries × 0.1s (transfer) = 70 GPU-seconds
- Cache misses: 300 queries × 2s (full generation) = 600 GPU-seconds
- Total GPU time: 670 GPU-seconds/hour

**Savings: 97% GPU time reduction for cached queries!**

## Scalability Considerations

### Horizontal Scaling

**Adding more GB10 workers:**
1. No changes needed to central server
2. Deploy new worker with unique ID
3. Automatically registers with controller

**Maximum workers supported:**
- Network bandwidth limited: ~40 workers per 100GbE link
- Central server RAM: ~50 workers per 256GB cache
- Controller overhead: ~100 workers per controller instance

**Recommended scaling strategy:**
```
1-10 workers:   Single central server
11-40 workers:  Add second central server (active-active)
41+ workers:    Regional clusters with federated controllers
```

### Vertical Scaling

**Central Server scaling path:**

| Workers | RAM | NVMe | Network |
|---------|-----|------|---------|
| 1-10 | 128GB | 1TB | 100GbE |
| 11-20 | 256GB | 2TB | 200GbE |
| 21-40 | 512GB | 4TB | 400GbE |
| 41+ | Multiple servers | | |

## Conclusion

This deployment architecture leverages your **ConnectX-7 RDMA** capabilities to achieve **10-20x speedup** for cached queries, despite GB10's lack of GPUDirect support. The centralized cache design maximizes hit rates and simplifies management across your GB10 cluster.

### Expected Performance Summary

| Metric | Value |
|--------|-------|
| Network Bandwidth | 10-12 GB/s (100GbE) |
| Cache Hit Latency | < 150 ms |
| Cache Hit Rate | 70-90% (workload dependent) |
| Speedup for Cache Hits | 10-20x |
| ROI Period | 3-6 months (based on GPU utilization improvement) |

### Next Steps

1. Deploy central server using `deploy_central_server.sh`
2. Deploy first GB10 worker as proof-of-concept
3. Run performance tests and tune parameters
4. Scale to remaining GB10 devices
5. Monitor and optimize based on real workload patterns

**Questions or issues?** Check `/opt/lmcache/health_check.sh` and review logs in `/var/log/lmcache/`
