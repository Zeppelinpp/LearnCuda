#include <cuda_runtime.h>
#include <cstdio>

int main() {
    printf("Testing CUDA...\n");

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Found %d CUDA devices\n", deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);

        // Simple allocation test
        float* d_ptr;
        err = cudaMalloc(&d_ptr, 1024 * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("Allocation successful\n");

        cudaFree(d_ptr);
        printf("Test passed!\n");
    }

    return 0;
}
