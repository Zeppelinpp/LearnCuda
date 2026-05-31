#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256

// Naive Softmax
// y = exp(x - max) / sum(exp(x - max))
// M rows, N cols, 1 block for each row, 1-dim block
__global__ void softmaxNaive(const float *input, float *output, int M, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // find rowMax
  __shared__ float sMax[BLOCK_SIZE];
  float blockMax = -INFINITY;
  for (int i = tid; i < N; i += blockDim.x) {
    blockMax = fmaxf(blockMax, input[row * N + i]);
  }
  sMax[tid] = blockMax;
  __syncthreads();

  // tree reduction for whole row max
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
    }
    __syncthreads();
  }
  float rowMax = sMax[0];
  __syncthreads();

  // Exp and local sum
  float localSum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    float expVal = expf(input[row * N + i] - rowMax);
    output[row * N + i] = expVal; // wirte back the expVal with - rowMax
    localSum += expVal;
  }

  // reduction for sum, set sSum[tid] = localSum, prepare for tree reduction
  __shared__ float sSum[BLOCK_SIZE];
  sSum[tid] = localSum;
  __syncthreads();
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sSum[tid] = sSum[tid] + sSum[tid + stride];
    }
    __syncthreads();
  }
  float rowSum = sSum[0];
  __syncthreads();

  // normalize
  for (int i = tid; i < N; i += blockDim.x) {
    output[row * N + i] /= rowSum;
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

    // Warmup
    softmaxNaive<<<grid, block>>>(d_input, d_output, M, N);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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

    free(h_input); free(h_output); free(h_ref);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
