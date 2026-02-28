# CUDA GEMM - Hello World of HPC

This project implements General Matrix Multiply (GEMM): `C = alpha * A * B + beta * C`

GEMM is the fundamental operation in deep learning and scientific computing, making it the perfect "Hello World" for High Performance Computing (HPC).

## What is GEMM?

For matrices:
- A: M × K matrix
- B: K × N matrix
- C: M × N matrix

The operation computes: `C[i,j] = alpha * sum(A[i,k] * B[k,j]) + beta * C[i,j]`

## Implementations Included

### 1. CPU Reference (`gemm_cpu`)
- Triple nested loop implementation
- Used to verify GPU results

### 2. GPU Naive (`gemm_naive`)
- Each thread computes one element of C
- Direct global memory access pattern
- Simple but not optimized

### 3. GPU Tiled (`gemm_tiled`)
- Uses **shared memory tiling** to reduce global memory bandwidth
- Loads tiles of A and B into fast shared memory
- Reuses data within each tile before loading next
- Typically 5-10x faster than naive on modern GPUs

## Build & Run

```bash
# Build
make

# Run
make run
# or
./gemm

# Clean
make clean
```

## Key Concepts Demonstrated

| Concept | Description |
|---------|-------------|
| **Grid/Block Organization** | 2D thread blocks mapping to matrix elements |
| **Shared Memory** | Fast on-chip memory for data reuse |
| **Tiling** | Breaking computation into cache-friendly chunks |
| **Memory Coalescing** | Optimizing global memory access patterns |
| **Synchronization** | `__syncthreads()` for thread coordination |

## Performance Expectations

On an NVIDIA V100 GPU with 1024×1024 matrices:
- CPU (single core): ~1-2 seconds
- GPU Naive: ~50-100 ms
- GPU Tiled: ~5-20 ms

The tiled version should achieve significantly higher GFLOPS by reducing global memory traffic.

## Further Optimizations (Not Implemented)

For production use, consider:
- cuBLAS (NVIDIA's highly optimized library)
- Register tiling and warp-level primitives
- Tensor Cores (mixed precision)
- Double buffering / pipeline overlapping
