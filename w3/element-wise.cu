#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define N (1 << 26) // 64M elements
#define WARMUP 5
#define REPEATS 1000

template<typename Launcher>
void benchmark(const char* name, Launcher launch) {
    // malloc, cudaMalloc, cudaMemcpy, etc. can be done inside the launch function
    size_t bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < WARMUP; i++) {
        launch(d_a, d_b, d_c);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < REPEATS; i++) {
        launch(d_a, d_b, d_c);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= REPEATS;
    double bytes_total = 3.0 * bytes;
    double bw = bytes_total / 1e9 / (ms / 1e3);
    printf("Launch %s: %.3f ms, Bandwidth: %.2f GB/s\n", name, ms, bw);

    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}

__global__ void naive_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c
) {
    // Current thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // check bounds
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
} // Naive Add: 0.593 ms, Bandwidth: 1356.93 GB/s

// float4: CUDA内置的向量类型，包含4个float分量，可以同时处理4个float数据
// a = [a0, a1, a2, a3, a4, a5, a6, a7, ...]
// const float* a  -> [[a0, a1, a2, a3], [a4, a5, a6, a7], ...]
__global__ void add_vec4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int N4
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N4) {
        float4 av = a[tid];
        float4 bv = b[tid];
        float4 cv;
        cv.x = av.x + bv.x;
        cv.y = av.y + bv.y;
        cv.z = av.z + bv.z;
        cv.w = av.w + bv.w;
        c[tid] = cv;
    }
}

// Grid-stride loop: 每个线程处理多个元素，直到处理完所有元素
// Stride大小确定: Grid内总线程数 = gridDim.x * blockDim.x
__global__ void add_grid_stride(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;

    benchmark("Naive Add", [](float* a, float* b, float* c) {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        naive_add<<<grid, BLOCK_SIZE>>>(a, b, c);
    });

    benchmark("Vec4 Add", [](float* a, float* b, float* c) {
        int N4 = N / 4;
        int grid = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_vec4<<<grid, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c),
            N4
        );
    });

    benchmark("Grid-Stride Add", [num_sms](float* a, float* b, float* c) {
        // 获取设备SM数量
        int grid = num_sms * 512;
        add_grid_stride<<<grid, BLOCK_SIZE>>>(a, b, c);
    });

    return 0;
}