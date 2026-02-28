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
  // 声明 shared memory，+1 padding 避免 bank conflict
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  // ========== Step 1: 从全局内存读入 shared memory ==========
  // 当前线程负责读取的原矩阵位置
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (row < height && col < width) {
    // 按行优先存入 tile：行=threadIdx.y, 列=threadIdx.x
    tile[threadIdx.y][threadIdx.x] = in[row * width + col];
  }

  __syncthreads();

  // ========== Step 2: 从 shared memory 写出到全局内存 ==========
  // 为了实现转置，原矩阵 (r, c) 的元素要写到转置矩阵的 (c, r)
  // 在线程协作方案中：
  // - 线程 (tx, ty) 读取原矩阵 (bx*T+tx, by*T+ty) 到 tile[ty][tx]
  // - 线程 (tx, ty) 将 tile[tx][ty] 写出到转置矩阵 (bx*T+ty, by*T+tx)
  // 这样相邻线程 (tx, ty) 和 (tx+1, ty) 写出的地址是连续的（合并写入）

  // 线程 (tx, ty) 负责写转置后矩阵的 (bx*T+ty, by*T+tx) 位置
  int out_row = blockIdx.x * TILE_SIZE + threadIdx.y; // 转置后的行 = bx*T + ty
  int out_col = blockIdx.y * TILE_SIZE + threadIdx.x; // 转置后的列 = by*T + tx

  // 转置后矩阵是 width × height（width行，height列）
  if (out_row < width && out_col < height) {
    // 从 tile[tx][ty] 读取，这个位置存储的是原矩阵 (bx*T+ty, by*T+tx)
    out[out_row * height + out_col] = tile[threadIdx.x][threadIdx.y];
  }
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

// 性能测试辅助函数
float benchmarkNaive(float *d_out, float *d_in, int width, int height,
                     dim3 gridSize, dim3 blockSize, int iterations) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  transposeNaive<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++) {
    transposeNaive<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iterations;
}

float benchmarkTiled(float *d_out, float *d_in, int width, int height,
                     dim3 gridSize, dim3 blockSize, int iterations) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  transposeTiled<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++) {
    transposeTiled<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iterations;
}

int main() {
  // 大矩阵测试
  int width = 4096;
  int height = 8192;
  int size = width * height * sizeof(float);
  int iterations = 100;

  printf("Matrix size: %d x %d = %.2f MB\n", width, height,
         size / (1024.0 * 1024.0));
  printf("Iterations: %d\n\n", iterations);

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

  // ==================== Naive 版本 ====================
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  float naiveTime = benchmarkNaive(d_out, d_in, width, height, gridSize,
                                   blockSize, iterations);

  // 验证结果
  transposeNaive<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  bool naivePassed = checkResult(h_out, h_in, width, height);

  printf("Naive Transpose:\n");
  printf("  Grid:  (%d, %d), Block: (%d, %d)\n", gridSize.x, gridSize.y,
         blockSize.x, blockSize.y);
  printf("  Time:  %.3f ms\n", naiveTime);
  printf("  Bandwidth: %.2f GB/s\n",
         (2.0 * size / (1024.0 * 1024.0 * 1024.0)) / (naiveTime / 1000.0));
  printf("  Result: %s\n\n", naivePassed ? "PASSED" : "FAILED");

  // ==================== Tiled 版本 ====================
  dim3 tileBlockSize(TILE_SIZE, TILE_SIZE); // 32x32 = 1024 threads
  dim3 tileGridSize((width + TILE_SIZE - 1) / TILE_SIZE,
                    (height + TILE_SIZE - 1) / TILE_SIZE);

  float tiledTime = benchmarkTiled(d_out, d_in, width, height, tileGridSize,
                                   tileBlockSize, iterations);

  // 验证结果
  transposeTiled<<<tileGridSize, tileBlockSize>>>(d_out, d_in, width, height);
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  bool tiledPassed = checkResult(h_out, h_in, width, height);

  printf("Tiled Transpose:\n");
  printf("  Grid:  (%d, %d), Block: (%d, %d)\n", tileGridSize.x, tileGridSize.y,
         tileBlockSize.x, tileBlockSize.y);
  printf("  Time:  %.3f ms\n", tiledTime);
  printf("  Bandwidth: %.2f GB/s\n",
         (2.0 * size / (1024.0 * 1024.0 * 1024.0)) / (tiledTime / 1000.0));
  printf("  Result: %s\n\n", tiledPassed ? "PASSED" : "FAILED");

  // ==================== 性能对比 ====================
  printf("Performance Comparison:\n");
  printf("  Speedup: %.2fx\n", naiveTime / tiledTime);

  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out);

  return 0;
}
