# GPU Dependencies in LMCache

## Overview

This document clarifies LMCache's GPU/CUDA dependencies and when they are required.

## Question: Does LMCache Require a GPU?

**Short answer:** No, but it depends on which components you're using.

### CUDA Source Files

LMCache includes several CUDA kernel files in `csrc/`:
- `mem_kernels.cu` - Memory copy/manipulation kernels
- `ac_enc.cu` / `ac_dec.cu` - Arithmetic coding compression/decompression
- `cal_cdf.cu` - CDF calculation for compression
- `pos_kernels.cu` - Positional encoding operations
- `mem_alloc.cpp` - Memory allocation utilities

These are compiled into `lmcache.c_ops` during the build process.

### CPU Fallback Support

LMCache **can run without CUDA** through:

1. **Build-time option:**
   ```bash
   NO_CUDA_EXT=1 pip install -e . --no-build-isolation
   ```

2. **Runtime fallbacks:**
   The codebase conditionally imports CUDA operations:
   ```python
   if torch.cuda.is_available():
       import lmcache.c_ops as lmc_ops
   else:
       import lmcache.non_cuda_equivalents as lmc_ops
   ```

   See:
   - `lmcache/v1/memory_management.py:24-29`
   - `lmcache/v1/lazy_memory_allocator.py:31-36`

3. **Pure Python implementations:**
   `lmcache/non_cuda_equivalents.py` provides CPU-based implementations of memory operations (pinned memory allocation, etc.)

### Deployment Context

While LMCache can run without GPUs, it's typically deployed alongside LLM serving engines (vLLM, SGLang) which **do require GPUs** for inference. LMCache's role:
- Store/retrieve KV caches from CPU memory, disk, or remote storage
- Offload KV caches from GPU to save GPU memory
- Share KV caches across multiple serving instances

## Multiprocess Mode: GPU Dependencies

### Two Server Implementations

LMCache has **two different server implementations** with different GPU requirements:

#### 1. Multiprocess Server (`lmcache/v1/multiprocess/server.py`)

**Requires GPU/CUDA code:**
- Unconditional import: `import lmcache.c_ops as lmc_ops` (line 36)
- Initializes CUDA: `torch.cuda.init()` (line 537)
- Contains `GPUCacheContext` class that manages GPU KV cache tensors
- Performs GPU↔CPU transfers using CUDA streams and events
- Uses CUDA IPC (Inter-Process Communication) for GPU memory sharing

**Why GPU code is needed:**
This server acts as a shared cache coordinator for multiple GPU workers. It needs CUDA operations to:
- Receive KV caches from GPU memory via CUDA IPC
- Efficiently copy data between GPU and CPU using CUDA kernels (`lmc_ops.multi_layer_kv_transfer()`)
- Synchronize GPU operations across processes using CUDA events
- Manage CUDA streams for async operations

**Key GPU-related components:**
- `CudaIPCWrapper` class for inter-process GPU memory sharing
- `GPUCacheContext` manages GPU tensor pointers, streams, and buffers
- CUDA event handling for synchronization

#### 2. Standalone V1 Server (`lmcache/v1/server/__main__.py`)

**Can be GPU-free:**
- Takes a `device` parameter (defaults to "cpu")
- Can run with pure CPU storage backends
- Doesn't directly use CUDA operations
- Suitable for CPU-only cache server deployments

**Usage:**
```bash
lmcache_server <host> <port> <storage>
# storage can be "cpu" (default) or other backends
```

## What CUDA Kernels Actually Do

The CUDA kernels are not just for "GPU memory read/write" in general. They provide:

1. **Fast GPU→CPU transfers**
   - Uses pinned memory for faster PCIe transfers
   - CUDA streams for asynchronous operations
   - Optimized kernel-based memory copies

2. **Inter-process GPU memory sharing**
   - CUDA IPC handles for sharing GPU buffers between processes
   - Zero-copy access to GPU memory across process boundaries

3. **Memory operation optimizations**
   - Custom kernels for layout transformations
   - Batch operations for multiple layers
   - Stream-based parallelism

4. **Compression operations**
   - Arithmetic coding on GPU for faster compression
   - CDF calculations for statistical encoding

## Summary

| Component | GPU Required? | Why? |
|-----------|---------------|------|
| LMCache Core | No | Has CPU fallbacks in `non_cuda_equivalents.py` |
| Multiprocess Server | **Yes** | Needs CUDA IPC and efficient GPU↔CPU transfers |
| Standalone V1 Server | No | Can use CPU-only storage backends |
| vLLM/SGLang Integration | Yes* | The serving engines themselves require GPUs |
| Storage Backends | No | CPU, disk, remote storage are all CPU-based |

*While LMCache integration points with vLLM/SGLang use GPU code to interact with the serving engines, the caching infrastructure itself can be CPU-only.

## Typical Deployment Architecture

```
┌─────────────────────────────────────┐
│  vLLM/SGLang Worker (GPU required)  │
│  ┌──────────────────────────────┐   │
│  │ LLM Inference (GPU)          │   │
│  │ GPU Connector (CUDA code)    │   │
│  └──────────────┬───────────────┘   │
└─────────────────┼───────────────────┘
                  │ CUDA IPC
                  ▼
      ┌───────────────────────┐
      │ Multiprocess Server   │
      │ (Needs CUDA for IPC)  │
      │ GPU↔CPU transfers     │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │ Storage Backends      │
      │ (CPU-only)            │
      │ - CPU Memory          │
      │ - Disk                │
      │ - Remote (S3, etc)    │
      └───────────────────────┘
```

## Recommendations

- **For development/testing without GPU:** Build with `NO_CUDA_EXT=1` and use standalone server with CPU backend
- **For production with vLLM/SGLang:** Use multiprocess server (requires CUDA) for best performance
- **For ROCm/AMD GPUs:** Build with `BUILD_WITH_HIP=1` for AMD GPU support
