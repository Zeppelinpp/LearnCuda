#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 128

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

// 2026-04-04
// Warp shuffle version

__global__ void layernorm_warp_shuffle(
  const float* __restrict__ x, // __restrict__ 编译器优化提示，指针所在内存区域不会和别的 __restrict__ 指针重叠
  const float* __restrict__ gamma,
  const float* __restrict__ beta,
  float* __restrict__ y,
  int batch_size,
  int seq_len,
  int hidden_dim,
  float eps = 1e-5
) {
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;
  if (token_idx >= batch_size * seq_len) return;

  const float* x_token = x + token_idx * hidden_dim;
  float *y_token = y + token_idx * hidden_dim;

  // warp mean
  unsigned mask = 0xffffffff;
  float sum = 0.0f;
  __shared__ float mean;

  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    sum += x_token[i];
  }
  for (int offset = 32 / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(mask, sum, offset);
    sum += other;
  }

  __shared__ float warpSum[BLOCK_SIZE / 32];
  if (tid%32 == 0) {
    warpSum[tid/32] = sum;
  }

  __syncthreads();
  if (tid < 32) {
    float val = (tid < 4) ? warpSum[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(mask, val, offset);
    }
    if (tid == 0) {
      mean = val / hidden_dim;
    }
  }

  __syncthreads();
  // varience
  float var_sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float diff = x_token[i] - mean;
    var_sum += diff * diff;
  }

  for (int offset = 32 / 2; offset > 0; offset >>= 1) {
    float other_var_sum = __shfl_down_sync(mask, var_sum, offset);
    var_sum += other_var_sum;
  }
  __shared__ float warpVarSum[BLOCK_SIZE / 32];
  if (tid % 32 == 0) {
    warpVarSum[tid/32] = var_sum;
  }
  __syncthreads();
  // reduction
  __shared__ float inv_std;
  if (tid < 32) {
    float val_var_sum = (tid < 4) ? warpVarSum[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      val_var_sum += __shfl_down_sync(mask, val_var_sum, offset);
    }
    if (tid == 0) {
      float std = val_var_sum / hidden_dim;
      inv_std = 1.0f / sqrt(std + eps);
    }
  }
  __syncthreads();

  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float norm = (x_token[i] - mean) * inv_std;
    y_token[i] = norm * gamma[i] + beta[i];
  }
}

// ==================== CPU 参考实现 ====================
void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                   float* y, int batch_size, int seq_len, int hidden_dim, float eps) {
  int num_tokens = batch_size * seq_len;
  for (int token = 0; token < num_tokens; token++) {
    const float* x_token = x + token * hidden_dim;
    float* y_token = y + token * hidden_dim;

    // 计算 mean
    float sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) sum += x_token[i];
    float mean = sum / hidden_dim;

    // 计算 variance
    float var_sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
      float diff = x_token[i] - mean;
      var_sum += diff * diff;
    }
    float inv_std = 1.0f / sqrtf(var_sum / hidden_dim + eps);

    // normalize + scale/shift
    for (int i = 0; i < hidden_dim; i++) {
      float norm = (x_token[i] - mean) * inv_std;
      y_token[i] = norm * gamma[i] + beta[i];
    }
  }
}

// ==================== 检查结果 ====================
bool check_result(const float* gpu_out, const float* cpu_out, int n, float tol) {
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float diff = fabsf(gpu_out[i] - cpu_out[i]);
    if (diff > tol) {
      if (errors < 5) {
        printf("  位置 %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", i, gpu_out[i], cpu_out[i], diff);
      }
      errors++;
    }
  }
  if (errors > 0) {
    printf("  共 %d 个错误 (超过 tolerance %.6f)\n", errors, tol);
    return false;
  }
  return true;
}

// ==================== Main ====================
int main() {
  // 测试配置
  int batch_size = 8;
  int seq_len = 512;
  int hidden_dim = 768;
  int num_tokens = batch_size * seq_len;
  int n = num_tokens * hidden_dim;

  printf("测试配置: batch=%d, seq_len=%d, hidden_dim=%d\n", batch_size, seq_len, hidden_dim);
  printf("总元素数: %d (%.2f MB)\n\n", n, n * sizeof(float) / 1024.0f / 1024.0f);

  // 分配 host 内存
  float *h_x = (float*)malloc(n * sizeof(float));
  float *h_gamma = (float*)malloc(hidden_dim * sizeof(float));
  float *h_beta = (float*)malloc(hidden_dim * sizeof(float));
  float *h_y_naive = (float*)malloc(n * sizeof(float));
  float *h_y_warp = (float*)malloc(n * sizeof(float));
  float *h_y_cpu = (float*)malloc(n * sizeof(float));

  // 初始化数据
  srand(42);
  for (int i = 0; i < n; i++) h_x[i] = (float)rand() / RAND_MAX - 0.5f;
  for (int i = 0; i < hidden_dim; i++) h_gamma[i] = 1.0f;  // 简化：gamma=1
  for (int i = 0; i < hidden_dim; i++) h_beta[i] = 0.0f;   // 简化：beta=0

  // 分配 device 内存
  float *d_x, *d_gamma, *d_beta, *d_y;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_gamma, hidden_dim * sizeof(float));
  cudaMalloc(&d_beta, hidden_dim * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));

  // 拷贝数据到 GPU
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

  // 设置 kernel 配置
  int threads = BLOCK_SIZE;  // 128
  int blocks = num_tokens;   // 每个 block 处理一个 token

  // ==================== CPU 参考 ====================
  printf("[1] CPU 参考实现...\n");
  layernorm_cpu(h_x, h_gamma, h_beta, h_y_cpu, batch_size, seq_len, hidden_dim, 1e-5);
  printf("    完成\n\n");

  // ==================== Naive 版本 ====================
  printf("[2] Naive Shared Memory 版本...\n");
  cudaMemset(d_y, 0, n * sizeof(float));

  // warmup
  layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, batch_size, seq_len, hidden_dim);
  cudaDeviceSynchronize();

  // 计时（重复 100 次取平均）
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    layernorm_naive<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, batch_size, seq_len, hidden_dim);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float naive_ms;
  cudaEventElapsedTime(&naive_ms, start, stop);
  naive_ms /= 100.0f;

  // 拷贝结果回 host
  cudaMemcpy(h_y_naive, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  // 检查正确性
  bool naive_ok = check_result(h_y_naive, h_y_cpu, n, 1e-4);
  printf("    %s, 平均耗时: %.3f ms\n\n", naive_ok ? "结果正确" : "结果错误", naive_ms);

  // ==================== Warp Shuffle 版本 ====================
  printf("[3] Warp Shuffle 版本...\n");
  cudaMemset(d_y, 0, n * sizeof(float));

  // warmup
  layernorm_warp_shuffle<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, batch_size, seq_len, hidden_dim);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    layernorm_warp_shuffle<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, batch_size, seq_len, hidden_dim);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float warp_ms;
  cudaEventElapsedTime(&warp_ms, start, stop);
  warp_ms /= 100.0f;

  cudaMemcpy(h_y_warp, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  bool warp_ok = check_result(h_y_warp, h_y_cpu, n, 1e-4);
  printf("    %s, 平均耗时: %.3f ms\n", warp_ok ? "结果正确" : "结果错误", warp_ms);

  // 计算加速比
  printf("    相比 naive 版本: %.2fx 加速\n\n", naive_ms / warp_ms);

  // ==================== 清理 ====================
  free(h_x); free(h_gamma); free(h_beta);
  free(h_y_naive); free(h_y_warp); free(h_y_cpu);
  cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  return (naive_ok && warp_ok) ? 0 : 1;
}
