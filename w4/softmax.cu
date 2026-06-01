#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define PAD(i) ((i) + ((i) >> 5))

// Naive Softmax
// y = exp(x - max) / sum(exp(x - max))
// M rows, N cols, 1 block for each row, 1-dim block
__global__ void softmaxNaive(const float *input, float *output, int M, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // find rowMax
  __shared__ float sMax[BLOCK_SIZE + (BLOCK_SIZE >> 5)];
  float blockMax = -INFINITY;
  for (int i = tid; i < N; i += blockDim.x) {
    blockMax = fmaxf(blockMax, input[row * N + i]);
  }
  sMax[PAD(tid)] = blockMax;
  __syncthreads();

  // tree reduction for whole row max
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sMax[PAD(tid)] = fmaxf(sMax[PAD(tid)], sMax[PAD(tid+stride)]);
    }
    __syncthreads();
  }
  float rowMax = sMax[PAD(0)];
  __syncthreads();

  // Exp and local sum
  float localSum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    float expVal = expf(input[row * N + i] - rowMax);
    output[row * N + i] = expVal; // wirte back the expVal with - rowMax
    localSum += expVal;
  }

  // reduction for sum, set sSum[tid] = localSum, prepare for tree reduction
  __shared__ float sSum[BLOCK_SIZE + (BLOCK_SIZE >> 5)];
  sSum[PAD(tid)] = localSum;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sSum[PAD(tid)] = sSum[PAD(tid)] + sSum[PAD(tid + stride)];
    }
    __syncthreads();
  }
  float rowSum = sSum[PAD(0)];
  __syncthreads();

  // normalize
  for (int i = tid; i < N; i += blockDim.x) {
    output[row * N + i] /= rowSum;
  }
}

__global__ void onlineSoftmax(const float *input, float *output, int M, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  
  // surrogate & max_i, online softmax update -> block sum into shared memory
  float surrogate = 0.0f;
  float max_i = -INFINITY;
  __shared__ float sSum[BLOCK_SIZE + (BLOCK_SIZE >> 5)];
  __shared__ float sMax[BLOCK_SIZE + (BLOCK_SIZE >> 5)];
  for (int i = tid; i < N; i += blockDim.x) {
    float cur_val = input[row * N + i];
    float old_max = max_i;
    float new_max = fmaxf(max_i, cur_val);
    surrogate = surrogate * expf(old_max - new_max) + expf(cur_val - new_max);
    max_i = new_max;
  }
  sSum[PAD(tid)] = surrogate;
  sMax[PAD(tid)] = max_i;
  __syncthreads();

  // tree reduction for whole row sum, online merge
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float m1 = sMax[PAD(tid)], m2 = sMax[PAD(tid + stride)];
      float s1 = sSum[PAD(tid)], s2 = sSum[PAD(tid + stride)];
      float new_max = fmaxf(m1, m2);
      sSum[PAD(tid)] = s1 * expf(m1 - new_max) + s2 * expf(m2 - new_max);
      sMax[PAD(tid)] = new_max;
    }
    __syncthreads();
  }
  float rowSum = sSum[PAD(0)];
  float rowMax = sMax[PAD(0)];
  __syncthreads();

  for (int i = tid; i < N; i += blockDim.x) {
    output[row * N + i] = expf(input[row * N + i] - rowMax) / rowSum;
  }
}

__global__ void onlineSoftmax_vectorized(const float *input, float *output, int M, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  const int VEC = 4;
  
  // surrogate & max_i, online softmax update -> block sum into shared memory
  float surrogate = 0.0f;
  float max_i = -INFINITY;
  __shared__ float sSum[BLOCK_SIZE + (BLOCK_SIZE >> 5)];
  __shared__ float sMax[BLOCK_SIZE + (BLOCK_SIZE >> 5)];

  auto process_one = [&](float cur_val) {
    float old_max = max_i;
    float new_max = fmaxf(max_i, cur_val);
    surrogate = surrogate * expf(old_max - new_max) + expf(cur_val - new_max);
    max_i = new_max;
  }
  for (int i = tid * VEC; i < N; i += blockDim.x * VEC) {
    if (i + VEC <= N) {
      float cur_val = *reinterpret_cast<const float4*>(&input[row * N + i]);
      process_one(cur_val.x);
      process_one(cur_val.y);
      process_one(cur_val.z);
      process_one(cur_val.w);
    } else {
      for (int j = 0; j < N - i; ++j) {
        process_one(input[row * N + i + j]);
      }
    }
  }
  sSum[PAD(tid)] = surrogate;
  sMax[PAD(tid)] = max_i;
  __syncthreads();

  // tree reduction for whole row sum, online merge
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float m1 = sMax[PAD(tid)], m2 = sMax[PAD(tid + stride)];
      float s1 = sSum[PAD(tid)], s2 = sSum[PAD(tid + stride)];
      float new_max = fmaxf(m1, m2);
      sSum[PAD(tid)] = s1 * expf(m1 - new_max) + s2 * expf(m2 - new_max);
      sMax[PAD(tid)] = new_max;
    }
    __syncthreads();
  }
  float rowSum = sSum[PAD(0)];
  float rowMax = sMax[PAD(0)];
  __syncthreads();

  for (int i = tid + VEC; i < N; i += blockDim.x * VEC) {
    if (i + VEC <= N) {
      float4 v = *reinterpret_cast<const float4*>(&input[row * N + i]);
      float4 out;
      out.x = expf(v.x - rowMax) / rowSum;
      out.y = expf(v.y - rowMax) / rowSum;
      out.z = expf(v.z - rowMax) / rowSum;
      out.w = expf(v.w - rowMax) / rowSum;
      *reinterpret_cast<float4*>(&output[row * N + i]) = out;
    } else {
      for (int j = 0; j < N - i; ++j) output[row * N + i + j] = expf(input[row * N + i + j] - rowMax) / rowSum;
    }
  }
}

// CPU reference
void softmaxCPU(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        float maxVal = -INFINITY;
        for (int i = 0; i < N; i++)
            maxVal = fmaxf(maxVal, input[row * N + i]);

        float sumVal = 0.0f;
        for (int i = 0; i < N; i++) {
            output[row * N + i] = expf(input[row * N + i] - maxVal);
            sumVal += output[row * N + i];
        }
        for (int i = 0; i < N; i++)
            output[row * N + i] /= sumVal;
    }
}

int main() {
    int M = 1 << 15;  // rows
    int N = 1 << 15;  // cols (must be <= BLOCK_SIZE * stride coverage)

    size_t bytes = M * N * sizeof(float);

    float* h_input  = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref    = (float*)malloc(bytes);

    // Initialize with random values
    for (int i = 0; i < M * N; i++)
        h_input[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_input, *d_output;
    cudaMalloc(&d_input,  bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(M);

    // ==================== softmaxNaive ====================
    printf("\n========== softmaxNaive ==========\n");

    // Warmup
    int warmup_runs = 5;
    for (int i = 0; i < warmup_runs; i++) softmaxNaive<<<grid, block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    // Benchmark with cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int bench_runs = 10;
    cudaEventRecord(start);
    for (int i = 0; i < bench_runs; i++) {
        softmaxNaive<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    float ms_per = ms_total / bench_runs;
    float naive_ms = ms_per;

    // FLOPs: sub(1) + exp(1) + add(1) + div(1) ≈ 4 per element
    // (max reduction is negligible for large N, or can be counted as 1 extra)
    double ops = 5.0 * M * N;
    double gflops = (ops / (ms_per * 1e-3)) / 1e9;

    // HBM traffic: read input twice (max + exp), write output twice (exp + norm), read output once (norm)
    double traffic = 5.0 * M * N * sizeof(float);
    double bw_gb_s = (traffic / (ms_per * 1e-3)) / 1e9;

    printf("Grid: %d, Block: %d\n", grid.x, block.x);
    printf("Time: %.3f ms (avg of %d runs)\n", ms_per, bench_runs);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bw_gb_s);

    // Verify correctness once
    softmaxNaive<<<grid, block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    softmaxCPU(h_input, h_ref, M, N);

    float maxErr = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_output[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("Max error: %e\n", maxErr);
    printf("%s\n", maxErr < 1e-5f ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ==================== onlineSoftmax ====================
    printf("\n========== onlineSoftmax ==========\n");

    // Warmup
    for (int i = 0; i < warmup_runs; i++) onlineSoftmax<<<grid, block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_runs; i++) {
        onlineSoftmax<<<grid, block>>>(d_input, d_output, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms_total, start, stop);
    ms_per = ms_total / bench_runs;
    float online_ms = ms_per;

    // onlineSoftmax: two passes over input, one write to output
    // per-element ops ≈ fmaxf(1) + 2*expf + mul + add + expf + div = ~7
    ops = 7.0 * M * N;
    gflops = (ops / (ms_per * 1e-3)) / 1e9;

    // HBM traffic: read input twice (online pass + final normalize), write output once
    traffic = 3.0 * M * N * sizeof(float);
    bw_gb_s = (traffic / (ms_per * 1e-3)) / 1e9;

    printf("Grid: %d, Block: %d\n", grid.x, block.x);
    printf("Time: %.3f ms (avg of %d runs)\n", ms_per, bench_runs);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bw_gb_s);

    // Verify correctness once
    onlineSoftmax<<<grid, block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    maxErr = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_output[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("Max error: %e\n", maxErr);
    printf("%s\n", maxErr < 1e-5f ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n========== Speedup ==========\n");
    printf("naive:   %.3f ms\n", naive_ms);
    printf("online:  %.3f ms\n", online_ms);
    printf("Speedup: %.2fx\n", naive_ms / online_ms);

    free(h_input); free(h_output); free(h_ref);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
