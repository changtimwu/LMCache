# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMCache is an LLM serving engine extension that reduces TTFT (Time To First Token) and increases throughput by caching and reusing KV (key-value) caches across serving instances. It integrates with serving engines like vLLM and SGLang.

## Common Commands

### Installation and Setup
```bash
# Install from source (recommended for development)
pip install -e . --no-build-isolation

# Install build dependencies first
pip install -r requirements/build.txt

# Install test dependencies
pip install -r requirements/test.txt

# Install linting tools
pip install -r requirements/lint.txt
```

### Build Commands
```bash
# Build with CUDA (default)
python setup.py build_ext --inplace

# Build with ROCm/HIP
BUILD_WITH_HIP=1 python setup.py build_ext --inplace

# Build without CUDA extensions (source distribution)
NO_CUDA_EXT=1 python setup.py sdist
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/v1/test_gpu_connector.py

# Run tests with coverage
pytest tests/ --cov=lmcache --cov-report=html

# Run tests with specific markers
pytest tests/ -m "not no_shared_allocator"

# Run tests verbosely with logs
pytest tests/ -v --log-cli-level=INFO
```

### Code Quality
```bash
# Install pre-commit hooks (required before committing)
pip install -r requirements/lint.txt
pre-commit install

# Run all code quality checks manually
pre-commit run --all-files

# Individual checks:
# - Formatting: ruff format
# - Linting: ruff check --fix
# - Type checking: mypy lmcache tests benchmarks
# - C++ formatting: clang-format -i csrc/*.cu csrc/*.cpp csrc/*.h
# - Spell checking: codespell --toml pyproject.toml
# - Import sorting: isort lmcache tests benchmarks
```

### Running LMCache Components
```bash
# Run v1 cache server (main cache instance)
lmcache_server

# Run v1 cache controller (centralized coordination)
lmcache_controller

# Run v0 server (legacy)
lmcache_v0_server
```

## Architecture

### Directory Structure
- `lmcache/` - Main Python package
  - `v1/` - Current stable API (primary development focus)
  - `integration/` - Integration with vLLM, SGLang
  - `storage_backend/` - Legacy storage backends (v0)
  - `server/` - Legacy server implementation (v0)
- `csrc/` - CUDA/C++ kernel implementations (memory operations, compression)
- `tests/` - Test suite (primarily pytest-based)
- `examples/` - Usage examples and integration patterns
- `benchmarks/` - Performance benchmarking scripts
- `docs/` - Documentation source (available at docs.lmcache.ai)

### Core Architecture (v1)

**CacheEngine** (`lmcache/v1/cache_engine.py`)
- Main orchestrator for caching operations
- Converts GPU KV caches to CPU MemoryObjs for storage
- Manages asynchronous storage and retrieval via StorageBackends
- Supports prefetching to optimize retrieval latency

**StorageBackends** (`lmcache/v1/storage_backend/`)
- `abstract_backend.py` - Base interface for all storage backends
- `local_cpu_backend.py` - CPU memory storage
- `local_disk_backend.py` - Local disk storage
- `gds_backend.py` - GPU Direct Storage backend
- `p2p_backend.py` - Peer-to-peer KV cache sharing
- `pd_backend.py` - Disaggregated prefill/decode backend
- `remote_backend.py` - Remote storage (S3, external services)
- `nixl_storage_backend.py` - NIXL integration

**GPUConnectors** (`lmcache/v1/gpu_connector.py`)
- Bridge between serving engines and LMCache
- `VLLMPagedMemGPUConnectorV2/V3` - vLLM paged attention integration
- `VLLMBufferLayerwiseGPUConnector` - Layer-wise buffer management
- `SGLangLayerwiseGPUConnector` - SGLang integration
- Handle engine-specific memory formats and data transfer

**TokenDatabase** (`lmcache/v1/token_database.py`)
- Maps token sequences to cached KV locations
- `ChunkedTokenDatabase` - Chunk-based lookup (default)
- `SegmentTokenDatabase` - Segment-based lookup

**CacheController** (`lmcache/v1/cache_controller/`)
- Centralized coordination for distributed cache instances
- Manages registration, heartbeat, eviction policies
- Provides P2P lookup for distributed KV sharing

**Memory Management** (`lmcache/v1/memory_management.py`)
- `MemoryAllocatorInterface` - Base for memory allocators
- `PagedTensorMemoryAllocator` - Paged memory allocation
- `CuFileMemoryAllocator` - GPU Direct Storage memory
- `MemoryObj` - Abstraction for cached KV data

### Integration Points

**vLLM Integration** (`lmcache/integration/vllm/`)
- `lmcache_connector_v1.py` - Main vLLM v1 adapter
- `vllm_v1_adapter.py` - vLLM v1 API integration
- Works with vLLM's attention mechanisms (PagedAttention, Flash Attention)

**SGLang Integration** (`lmcache/integration/sglang/`)
- `sglang_adapter.py` - SGLang integration adapter

### CUDA Kernels (`csrc/`)
- `mem_kernels.cu` - Memory copy/manipulation kernels
- `ac_enc.cu` / `ac_dec.cu` - Arithmetic coding compression/decompression
- `cal_cdf.cu` - CDF calculation for compression
- `pos_kernels.cu` - Positional encoding operations
- `mem_alloc.cpp` - Memory allocation utilities
- Built via PyTorch C++ extensions in setup.py

### Configuration System

Configuration is loaded from:
1. YAML files (if specified)
2. Environment variables (prefixed with `LMCACHE_`)
3. Command-line overrides

Key config files:
- `lmcache/v1/config.py` - Main v1 configuration
- `lmcache/config.py` - Legacy v0 configuration
- Config aliases map deprecated names to current names (see `_CONFIG_ALIASES`)

## Development Workflow

### Making Changes
1. Create a feature branch from `dev` (not `main`)
2. Install pre-commit hooks: `pre-commit install`
3. Make changes and ensure tests pass
4. Code quality checks run automatically on commit
5. Push and create PR against `dev` branch

### Testing Strategy
- Unit tests in `tests/v1/` for v1 components
- Integration tests in `.buildkite/` directory
- Use pytest markers to skip specific tests (e.g., `@pytest.mark.no_shared_allocator`)
- Tests use `pytest.ini` config for logging and markers

### Code Quality Standards
- Python: Ruff for linting/formatting, MyPy for type checking, isort for imports
- C++/CUDA: clang-format with project .clang-format config
- All Python files must include SPDX license header: `# SPDX-License-Identifier: Apache-2.0`
- Run `pre-commit run --all-files` before pushing

### CUDA/ROCm Development
- CUDA is the default build target
- ROCm/HIP support via hipify: set `BUILD_WITH_HIP=1`
- CUDA kernels in `csrc/` are automatically hipified to `csrc_hip/`
- Torch version pinned to 2.8.0 for builds (see pyproject.toml)

## Key Dependencies
- PyTorch 2.x (flexible runtime, pinned 2.8.0 for builds)
- transformers >= 4.51.1
- vLLM (no pinned version - user's choice)
- fastapi/uvicorn (for API servers)
- redis (for distributed coordination)
- cupy-cuda12x (CUDA utilities)
- nixl (optional, Python < 3.13 only)

## Important Notes

### v0 vs v1
- `lmcache/v1/` is the current active development path
- `lmcache/` (v0) is legacy but still maintained
- When adding features, target v1 unless explicitly working on v0 compatibility

### Storage Backend Selection
Storage backends are configured via `LMCacheEngineConfig`:
- Local: CPU, disk, GDS (GPU Direct Storage)
- Remote: S3, external backends via plugins
- Distributed: P2P sharing, disaggregated prefill/decode
- Plugins: Custom backends via `storage_plugins` config

### vLLM Version Compatibility
LMCache supports multiple vLLM versions through connector versioning:
- `lmcache_connector_v1.py` - Current vLLM v1
- `lmcache_connector_v1_085.py` - vLLM 0.8.5 compatibility
- Check vLLM version detection in integration code

### Multi-process Mode
LMCache supports multi-process serving:
- `lmcache/v1/multiprocess/` - Multi-process coordination
- Message queue-based communication between workers
- Separate cache server process for shared state

### Build System
- Uses setuptools with setuptools_scm for versioning
- Version written to `lmcache/_version.py`
- PyTorch C++ extensions for CUDA kernels
- Supports both wheel builds (cibuildwheel) and editable installs
- CUDA architectures: 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0 (see pyproject.toml)
