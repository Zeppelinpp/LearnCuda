#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define BLOCK_SIZE 256

__global__ void rowSumKernel(
  const float* __restrict__ A, float* __restrict__ output, int M, int N
) {
  int row = blockIdx.x;
  
  if (row >= M) {
    return;
  }

  int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    sum += A[row * N + i];
  }

  __shared__ float sdata[BLOCK_SIZE + BLOCK_SIZE / 32];
  int idx = tid + (tid >> 5);
  sdata[idx] = sum;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      int idxCur = tid + (tid >> 5);
      int idxNext = (tid + stride) + ((tid + stride) >> 5);
      sdata[idxCur] += sdata[idxNext];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[row] = sdata[0];
  }
}

__global__ void rowSumKernelShuffle(
  const float* __restrict__ A, float* __restrict__ output, int M, int N
) {
  int row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int lane = tid % 32;
  int warpId = tid / 32;

  float sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    if (i < N) {
      sum += A[row * N + i];
    }
  }

  #pragma unroll
  for (int offset = 32 / 2 ; offset  > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, sum, offset);
    sum += other;
  }

  __shared__ float warpSums[BLOCK_SIZE / 32];
  if (lane == 0) {
    warpSums[warpId] = sum;
  }
  __syncthreads(); // 所有warp 的 lane 0 有规约结果，warp 外还是需要 __syncthreads

  if (tid < BLOCK_SIZE / 32) {
    sum = warpSums[tid];
    #pragma unroll
    for (int offset = BLOCK_SIZE / 32 / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid == 0) {
      output[row] = sum;
    }
  }
}


void testRowSum() {
  int M = 4;
  int N = 1024;

  std::vector<float> h_A(M * N, 1.0f);
  std::vector<float> h_result(M);

  float *d_A, *d_output;

  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_output, M * sizeof(float));

  cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 block(BLOCK_SIZE); // 一维
  dim3 grid(M);

  rowSumKernel<<<grid, block>>>(d_A, d_output, M, N);

  // copy back to host
  cudaMemcpy(h_result.data(), d_output, M * sizeof(float), cudaMemcpyDeviceToHost);

  // free device memory
  cudaFree(d_A);
  cudaFree(d_output);

  for (int i = 0; i < M; i++) {
    printf("Row %d: rowSum=%.1f, expected=%.1f\n", i, h_result[i], (float)N);
  }
}

