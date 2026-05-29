#include <cuda_runtime.h>

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
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  // bank = (ty * TILE_SIZE + i) % 32 (Warp size = 32)
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1]; // padding for smem bank conflict
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

  float sum = 0.0f;
  // load data into tile
  for (int tile_offset = 0; tile_offset < K; tile_offset += TILE_SIZE) {
    int a_check = tile_offset + tx  < K && row < M;
    tile_A[ty][tx] = a_check ? A[row * K + tile_offset + tx] : 0.0f;

    int b_check = tile_offset + ty < K && col < N;
    tile_B[ty][tx] = b_check ? B[(tile_offset + ty) * N + col] : 0.0f;
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
