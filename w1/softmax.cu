#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void softmaxNaive(
    const float* input, float* output, int M, int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Phase 1: Find rowMax
    __shared__ float sharedMax[BLOCK_SIZE + BLOCK_SIZE / 32];  // bank padding
    float localMax = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        localMax = fmaxf(localMax, input[row * N + i]);
    }
    sharedMax[tid + (tid >> 5)] = localMax;
    __syncthreads();

    // Tree reduction for max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int cur = tid + (tid >> 5);
            int nxt = (tid + stride) + ((tid + stride) >> 5);
            sharedMax[cur] = fmaxf(sharedMax[cur], sharedMax[nxt]);
        }
        __syncthreads();
    }
    float rowMax = sharedMax[0];
    __syncthreads();

    // Phase 2: Compute exp and accumulate local sum
    float localSum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float expVal = expf(input[row * N + i] - rowMax);
        output[row * N + i] = expVal;
        localSum += expVal;
    }

    // Phase 3: Reduce for sum
    __shared__ float sharedSum[BLOCK_SIZE + BLOCK_SIZE / 32];  // bank padding
    sharedSum[tid + (tid >> 5)] = localSum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int cur = tid + (tid >> 5);
            int nxt = (tid + stride) + ((tid + stride) >> 5);
            sharedSum[cur] += sharedSum[nxt];
        }
        __syncthreads();
    }
    float rowSum = sharedSum[0];
    __syncthreads();

    // Phase 4: Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = row * N + i;
        output[idx] = output[idx] / rowSum;
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
    int M = 1024;  // rows
    int N = 1024;  // cols (must be <= BLOCK_SIZE * stride coverage)

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
    softmaxNaive<<<grid, block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // CPU reference
    softmaxCPU(h_input, h_ref, M, N);

    // Verify
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
