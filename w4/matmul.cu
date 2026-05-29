#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 1024
#define TILE_SIZE 32
#define M 5000
#define N 4000
#define K 3000


// A: M x K, B: K x N, out: M x N
// A and B are stored in row-major order
__global__ void naive_matmul(float* A, float* B, float* out, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * N + col];
    }
    out[row * N + col] = sum;
  }
}

// tiled_matmul
__global__ void tiled_matmul(float* A, float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = threadIdx.y, tx = threadIdx.x;
  // bank = (ty * TILE_SIZE + i) % 32 (Warp size = 32)
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1]; // padding for smem bank conflict
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

  float sum = 0.0f;
  // load data into tile
  for (int tile_offset = 0; tile_offset < K; tile_offset += TILE_SIZE) {
    tile_A[ty][tx] = tile_offset + tx  < K && row < M ? A[row * K + tile_offset + tx] : 0.0f;
    tile_B[ty][tx] = tile_offset + ty < K && col < N ? B[(tile_offset + ty) * N + col] : 0.0f;
    __syncthreads();
    // compute partial dot product for this tile
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += tile_A[ty][i] * tile_B[i][tx];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

int main() {
  float *A, *B, *C;
  A = (float*)malloc(M * K * sizeof(float));
  B = (float*)malloc(K * N * sizeof(float));
  C = (float*)malloc(M * N * sizeof(float));
  float *d_A, *d_B, *d_C;
  for (int i = 0; i < M * K; i ++) {
    A[i] = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < K * N; i ++) {
    B[i] = (float)rand() / (float)RAND_MAX;
  }
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size((N + TILE_SIZE - 1) / TILE_SZIE, (M + TILE_SIZE - 1) / TILE_SIZE);

  // naive_matmul
  // warmup
  cudaMemset(d_C, 0, M * N * sizeof(float));
  naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  cudaEventRecord(stop);
  cudaDeviceSynchronize(); // CPU wait for kernel to finish
  float ms_naive = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // tiled_matmul
  cudaMemset(d_C, 0, M * N * sizeof(float));
  tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  cudaEventRecord(start);
  tiled_matmul<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  float ms_tiled = 0;
  cudaEventElapsedTime(&ms_tiled, start, stop);
  printf("naive: %.3f ms, tiled: %.3f ms\n", ms_naive, ms_tiled);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
