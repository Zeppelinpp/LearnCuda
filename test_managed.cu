// Test 1: __managed__ alone
__managed__ int test1_var;

// Test 2: __device__ __managed__
__device__ __managed__ int test2_var;

__global__ void kernel() {
    test1_var = 1;
    test2_var = 2;
}

int main() {
    test1_var = 10;
    test2_var = 20;

    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
