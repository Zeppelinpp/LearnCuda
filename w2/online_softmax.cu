#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__device__ void merge_online(
  float m_a, float s_a, float m_b, float s_b,
  float* m_out, float* s_out
) {
  if (m_a >= m_b) {
    *m_out = m_a;
    *s_out = s_a + s_b * expf(m_b - m_a);
  } else {
    *m_out = m_b;
    *s_out = s_a * expf(m_a - m_b) + s_b;
  }
}


__global__ void online_softmax(
  const float* __restrict__ input, 
  float* __restrict__ output,
  int rows, int cols
) {
  int row = blockIdx.x;
  if (row >= rows) return;

  float max = -INFINITY;
  float rowSum = 0.0f;

  for (int i = 0; i < cols; i++) {
    float curVal = output[row * cols + i];
    float preamx = max;
    max = fmaxf(max, curVal);
    rowSum = rowSum * expf(premax - max) + expf(curVal - max);
  }

  for (int i = 0; i < cols; i++) {
    output[row * cols + i] = expf(input[row * cols + i] - max) / rowSum;
  }
}

__global__ void online_softmax_parallel(
  const float* __restrict__ input,
  float* __restrict__ output,
  int rows, int cols
) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int blockSize = blockDim.x;

  if (row >= rows) return;

  // 每个线程处理一个 chunk
  int chunkSize = (cols + blockSize - 1) / blockSize;
  int start = tid * chunkSize;
  int end = min(start + chunkSize, cols);

  // Step 1: 每个线程用 online softmax 处理自己的 chunk
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (int i = start; i < end; i++) {
    float x = input[row * cols + i];
    float old_max = local_max;
    local_max = fmaxf(local_max, x);
    local_sum = local_sum * expf(old_max - local_max) + expf(x - local_max);
  }

  // Step 2: 存到 shared memory
  __shared__ float s_max[256];
  __shared__ float s_sum[256];

  s_max[tid] = local_max;
  s_sum[tid] = local_sum;
  __syncthreads();

  // Step 3: tree reduction
  for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      merge_online(s_max[tid], s_sum[tid], s_max[tid + stride], s_sum[tid + stride],
                   &s_max[tid], &s_sum[tid]);
    }
    __syncthreads();
  }

  // Step 4: 用全局 (max, sum) 计算输出
  float global_max = s_max[0];
  float global_sum = s_sum[0];

  for (int i = start; i < end; i++) {
    float x = input[row * cols + i];
    output[row * cols + i] = expf(x - global_max) / global_sum;
  }
}

void cpu_softmax(const float* input, float* output, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
      max_val = fmaxf(max_val, input[r * cols + i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
      sum += expf(input[r * cols + i] - max_val);
    }
    for (int i = 0; i < cols; i++) {
      output[r * cols + i] = expf(input[r * cols + i] - max_val) / sum;
    }
  }
}

int main() {
  int rows = 2;
  int cols = 1024;
  size_t size = rows * cols * sizeof(float);

  // 分配 host 内存
  float *h_input = (float*)malloc(size);
  float *h_output = (float*)malloc(size);
  float *h_ref = (float*)malloc(size);

  // 初始化数据
  srand(42);
  for (int i = 0; i < rows * cols; i++) {
    h_input[i] = (float)(rand() % 100) / 10.0f - 5.0f;  // [-5, 5]
  }

  // 分配 device 内存
  float *d_input, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);

  // 拷贝到 device
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // 启动 kernel（并行版本）
  online_softmax_parallel<<<rows, 256>>>(d_input, d_output, rows, cols);

  // 拷贝结果回 host
  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

  // CPU 参考计算
  cpu_softmax(h_input, h_ref, rows, cols);

  // 验证
  float max_error = 0.0f;
  for (int i = 0; i < rows * cols; i++) {
    float err = fabsf(h_output[i] - h_ref[i]);
    if (err > max_error) max_error = err;
  }
  printf("Max error: %e\n", max_error);
  printf("Test %s\n", max_error < 1e-5 ? "PASSED" : "FAILED");

  // 释放内存
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);
  free(h_ref);

  return 0;
}
