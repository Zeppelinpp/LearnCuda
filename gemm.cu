/**
 * CUDA GEMM (General Matrix Multiply) - Hello World of HPC
 * Computes C = alpha * A * B + beta * C
 *
 * This file contains multiple implementations:
 * 1. CPU reference implementation
 * 2. Naive GPU implementation
 * 3. Tiled GPU implementation with shared memory
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Matrix dimensions (can be changed)
#ifndef M
#define M 512 // Rows of A and C
#endif
#ifndef N
#define N 512 // Cols of B and C
#endif
#ifndef K
#define K 512 // Cols of A, Rows of B
#endif

// Tile size for shared memory optimization
#define TILE_SIZE 32

// Error checking macro
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------
void gemm_cpu(const float *A, const float *B, const float *C_in, float *C_out,
              int m, int n, int k, float alpha, float beta) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C_out[i * n + j] = alpha * sum + beta * C_in[i * n + j];
    }
  }
}

// ---------------------------------------------------------------------------
// GPU Kernel: Naive Implementation
// Each thread computes one element of C
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float *A, const float *B, const float *C_in,
                           float *C_out, int m, int n, int k, float alpha,
                           float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int l = 0; l < k; l++) {
      sum += A[row * k + l] * B[l * n + col];
    }
    C_out[row * n + col] = alpha * sum + beta * C_in[row * n + col];
  }
}

// ---------------------------------------------------------------------------
// GPU Kernel: Tiled Implementation with Shared Memory
// Uses tiling to reduce global memory accesses
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const float *A, const float *B, const float *C_in,
                           float *C_out, int m, int n, int k, float alpha,
                           float beta) {
  // Shared memory tiles
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  // Loop over tiles
  for (int tile_idx = 0; tile_idx < (k + TILE_SIZE - 1) / TILE_SIZE;
       tile_idx++) {
    // Load tile from A into shared memory
    if (row < m && tile_idx * TILE_SIZE + threadIdx.x < k) {
      tile_A[threadIdx.y][threadIdx.x] =
          A[row * k + tile_idx * TILE_SIZE + threadIdx.x];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load tile from B into shared memory
    if (col < n && tile_idx * TILE_SIZE + threadIdx.y < k) {
      tile_B[threadIdx.y][threadIdx.x] =
          B[(tile_idx * TILE_SIZE + threadIdx.y) * n + col];
    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product for this tile
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result
  if (row < m && col < n) {
    C_out[row * n + col] = alpha * sum + beta * C_in[row * n + col];
  }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------
void init_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

void print_matrix(const char *name, const float *mat, int rows, int cols,
                  int max_print = 8) {
  printf("%s (%dx%d):\n", name, rows, cols);
  int print_rows = (rows < max_print) ? rows : max_print;
  int print_cols = (cols < max_print) ? cols : max_print;

  for (int i = 0; i < print_rows; i++) {
    for (int j = 0; j < print_cols; j++) {
      printf("%8.4f ", mat[i * cols + j]);
    }
    if (cols > max_print)
      printf("...");
    printf("\n");
  }
  if (rows > max_print)
    printf("...\n");
  printf("\n");
}

bool verify_results(const float *ref, const float *result, int size,
                    float tolerance = 1e-3) {
  for (int i = 0; i < size; i++) {
    if (fabs(ref[i] - result[i]) > tolerance) {
      printf("Mismatch at index %d: ref=%.6f, result=%.6f\n", i, ref[i],
             result[i]);
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  // Print GPU info
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("========================================\n");
  printf("CUDA GEMM - Hello World of HPC\n");
  printf("========================================\n");
  printf("GPU: %s\n", prop.name);
  printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Tile size: %d\n", TILE_SIZE);
  printf("----------------------------------------\n\n");

  // Allocate host memory
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  float *h_A = (float *)malloc(size_A);
  float *h_B = (float *)malloc(size_B);
  float *h_C_in = (float *)malloc(size_C);
  float *h_C_cpu = (float *)malloc(size_C);
  float *h_C_naive = (float *)malloc(size_C);
  float *h_C_tiled = (float *)malloc(size_C);

  // Initialize matrices
  srand(42);
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);
  init_matrix(h_C_in, M, N);

  float alpha = 1.0f;
  float beta = 0.0f;

  // Print small sample of input matrices
  print_matrix("Matrix A", h_A, M, K, 4);
  print_matrix("Matrix B", h_B, K, N, 4);

  // Allocate device memory
  float *d_A, *d_B, *d_C_in, *d_C_out;
  CHECK_CUDA(cudaMalloc(&d_A, size_A));
  CHECK_CUDA(cudaMalloc(&d_B, size_B));
  CHECK_CUDA(cudaMalloc(&d_C_in, size_C));
  CHECK_CUDA(cudaMalloc(&d_C_out, size_C));

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C_in, h_C_in, size_C, cudaMemcpyHostToDevice));

  // ========================================================================
  // CPU Reference
  // ========================================================================
  printf("Running CPU reference...\n");
  auto cpu_start = std::chrono::high_resolution_clock::now();
  gemm_cpu(h_A, h_B, h_C_in, h_C_cpu, M, N, K, alpha, beta);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpu_time = cpu_end - cpu_start;
  printf("CPU time: %.4f seconds\n\n", cpu_time.count());

  // ========================================================================
  // GPU Naive Implementation
  // ========================================================================
  printf("Running GPU naive implementation...\n");

  dim3 block_size(16, 16);
  dim3 grid_size((N + block_size.x - 1) / block_size.x,
                 (M + block_size.y - 1) / block_size.y);

  // Warmup
  gemm_naive<<<grid_size, block_size>>>(d_A, d_B, d_C_in, d_C_out, M, N, K,
                                        alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark
  auto gpu_naive_start = std::chrono::high_resolution_clock::now();
  gemm_naive<<<grid_size, block_size>>>(d_A, d_B, d_C_in, d_C_out, M, N, K,
                                        alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto gpu_naive_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_naive_time =
      gpu_naive_end - gpu_naive_start;

  // Copy result back
  CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_out, size_C, cudaMemcpyDeviceToHost));

  bool naive_correct = verify_results(h_C_cpu, h_C_naive, M * N);
  printf("Naive kernel: %.4f seconds, Verification: %s\n",
         gpu_naive_time.count(), naive_correct ? "PASSED" : "FAILED");

  // Calculate GFLOPS
  double flops = 2.0 * M * N * K; // multiply-add = 2 FLOPs
  double naive_gflops = flops / gpu_naive_time.count() / 1e9;
  printf("Naive GFLOPS: %.2f\n\n", naive_gflops);

  // ========================================================================
  // GPU Tiled Implementation
  // ========================================================================
  printf("Running GPU tiled implementation...\n");

  dim3 tiled_block_size(TILE_SIZE, TILE_SIZE);
  dim3 tiled_grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);

  // Warmup
  gemm_tiled<<<tiled_grid_size, tiled_block_size>>>(d_A, d_B, d_C_in, d_C_out,
                                                    M, N, K, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark
  auto gpu_tiled_start = std::chrono::high_resolution_clock::now();
  gemm_tiled<<<tiled_grid_size, tiled_block_size>>>(d_A, d_B, d_C_in, d_C_out,
                                                    M, N, K, alpha, beta);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto gpu_tiled_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_tiled_time =
      gpu_tiled_end - gpu_tiled_start;

  // Copy result back
  CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C_out, size_C, cudaMemcpyDeviceToHost));

  bool tiled_correct = verify_results(h_C_cpu, h_C_tiled, M * N);
  printf("Tiled kernel: %.4f seconds, Verification: %s\n",
         gpu_tiled_time.count(), tiled_correct ? "PASSED" : "FAILED");

  double tiled_gflops = flops / gpu_tiled_time.count() / 1e9;
  printf("Tiled GFLOPS: %.2f\n\n", tiled_gflops);

  // ========================================================================
  // Summary
  // ========================================================================
  printf("========================================\n");
  printf("Performance Summary\n");
  printf("========================================\n");
  printf("CPU Reference:   %.4f s\n", cpu_time.count());
  printf("GPU Naive:       %.4f s (%.2fx speedup)\n", gpu_naive_time.count(),
         cpu_time.count() / gpu_naive_time.count());
  printf("GPU Tiled:       %.4f s (%.2fx speedup)\n", gpu_tiled_time.count(),
         cpu_time.count() / gpu_tiled_time.count());
  printf("\nGFLOPS:\n");
  printf("  Naive: %.2f\n", naive_gflops);
  printf("  Tiled: %.2f (%.2fx faster than naive)\n", tiled_gflops,
         tiled_gflops / naive_gflops);
  printf("========================================\n");

  // Print sample output
  print_matrix("Result C (CPU)", h_C_cpu, M, N, 4);

  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C_in);
  free(h_C_cpu);
  free(h_C_naive);
  free(h_C_tiled);
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C_in));
  CHECK_CUDA(cudaFree(d_C_out));

  return 0;
}
