#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#defin BLOCK_SIZE 128

__global__ void layernorm_naive(
  const float* __restrict__ x, // __restrict__ 编译器优化提示，指针所在内存区域不会和别的 __restrict__ 指针重叠
  const float* __restrict__ gamma,
  const float* __restrict__ beta,
  float* __restrict__ y,
  int batch_size,
  int seq_len,
  int hidden_dim,
  float eps = 1e-5
) {
  // token idx
  int token_idx = blockIdx.x;
  if (token_idx >= batch_size * seq_len) return;

  const float* x_token = x + token_idx * hidden_dim;
  float* y_token = y + token_idx * hidden_dim; // token_idx 已经蕴含了seq 和 bz 的信息

  // mean
  float sum = 0.0f;
  int tid = threadIdx.x;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
      // sum += x[x_token * hidden_dim + i * hidden_dim / blockDim.x + j];
      // x_token 已经是指针，不需要再乘hidden_dim,直接offset就行
      sum += x_token[i];
  }

  // tree reduction -> mean
  __shared__ float sharedMean[BLOCK_SIZE];
  sharedMean[tid] = sum;
  for (int stride = blockDim.x /2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedMean[tid] += sharedMean[tid + stride];
    }
    __syncthreads();
  }

  __shared__ float mean;
  if (tid == 0) {
    mean =sharedMean[0] / hidden_dim;
  }
  __syncthreads();

  // varience
  __shared__ float sharedVar[BLOCK_SIZE];
  float var_sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float diff = x_token[i] - mean;
    var_sum += diff * diff;
  }
  sharedVar[tid] = var_sum;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedVar[tid] += sharedVar[tid + stride];
    }
    __syncthreads();
  }
  __shared__ float inv_std;
  if (tid == 0) {
      float var = sharedVar[0] / hidden_dim;
      inv_std = 1.0f / sqrt(var + eps);
  }
  __syncthreads();

  // normalize + scale/shift
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float norm = (x_token[i] - mean) * inv_std;
    y_token[i] = norm * gamma[i] + beta[i];
  }
}

