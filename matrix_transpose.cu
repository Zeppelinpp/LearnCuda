#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32
// Navie matrix transpose - 一个线程处理一个元素
// out[col][row] = in[row][col]
__global__ void transposeNaive(float *out, const float *in, int width,
                               int height) {
  // 计算当前线程负责的全局坐标
  // 矩阵按照row-major在内存中存储
  // 按照笛卡尔坐标系习惯,x -> col, y -> row
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界检查，grid size 无法被block size 整除的情况
  if (row < height && col < width) {
    // 读输入矩阵的 i,j, 输出为输出矩阵的j,i
    // 读取时 row * width + col, 输出访存是 col * height + row (row
    // major存储特性)
    out[col * height + row] = in[row * width + col];
  }
}

__global__ void transposeTiled(float *out, const float *in, int width,
                               int height) {
  // 把一个tile中的数据从全局内存合并读取到share memory
  // 在Share memory 中做转置
  // 然后转置后的Tile合并写回内存

  // share memory 声明
  __shared__ float tile[TILE_SIZE][TILE_SIZE];

  // 当前线程在全局矩阵中的位置 read
  int col_in = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row_in = blockIdx.y * TILE_SIZE + threadIdx.y;

  // 写出位置，整个tile被转置
  int row_out = blockIdx.x * TILE_SIZE + threadIdx.y;
}

void initMatrix(float *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = (float)i;
  }
}

bool checkResult(float *out, float *in, int width, int height) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      float expected = in[row * width + col];
      float actual = out[col * height + row];
      if (expected != actual) {
        printf("Error at (%d, %d): expected %f, got %f\n", row, col, expected,
               actual);
        return false;
      }
    }
  }
  return true;
}

void showMatrix(float *matrix, int width, int height) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      printf("%.2f ", matrix[row * width + col]);
    }
    printf("\n");
  }
}

int main() {
  int width = 3;
  int height = 3;
  int size = width * height * sizeof(float); // 分配多少内存

  // CPU 内存
  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);
  initMatrix(h_in, width * height);

  // device memory
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // copy data to gpu
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // thread block and grid size
  dim3 blockSize(16, 16); // 256 threads in 256/32 = 8 warp
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  // launch kernel
  transposeNaive<<<gridSize, blockSize>>>(d_out, d_in, width, height);

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  printf("====== Orignal ======\n");
  showMatrix(h_in, width, height);
  printf("====== Transposed ======\n");
  showMatrix(h_out, width, height);
  if (checkResult(h_out, h_in, width, height)) {
    printf("PASSED\n");
  }

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out);

  return 0;
}
