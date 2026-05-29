# Kernel / Block 形状设计的心智模型

## 核心原则：输出驱动

Grid/Block 的维度由**输出数据的维度**和**并行粒度**决定。

```
输出数据结构          →  Grid 维度          →  Block 维度
1D array/vector       →  1D grid            →  1D block
2D matrix/image       →  2D grid            →  2D block
3D volume             →  3D grid            →  3D block
```

## 决策树

```
每个输出元素之间是否独立？
├── 是（element-wise）
│   └── 每个线程负责一个输出元素
│       └── Grid shape = 输出 shape
│
└── 否（row-wise, reduction, scan）
    └── 每个线程负责一组输出元素
        └── Grid shape = 输出组的数量
```

## 1. Element-Wise：1D 足够

```c
// 输出：N 个元素的向量
__global__ void add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

dim3 block(256);
dim3 grid((N + 255) / 256);
```

**为什么是 1D：**
- 输出是 1D 数组，天然对应 1D 索引
- 一个 warp 的 32 线程访问 `c[idx]`~`c[idx+31]` → 连续地址 → 完美合并
- 2D/3D 会增加索引计算开销，没有任何收益

## 2. Row-Wise 处理：看并行度

场景：对矩阵每行做 softmax / sum / normalize。

### 方案 A：1D Block，每线程处理一行

```c
__global__ void row_sum(float* mat, float* out, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int col = 0; col < N; col++)
        sum += mat[row * N + col];
    out[row] = sum;
}

dim3 block(256);
dim3 grid((M + 255) / 256);   // M 行
```

**适用：** 行内操作串行，行与行之间完全独立。  
**优势：** 简单，无 smem 同步开销。  
**劣势：** 行很长时，一个线程串行遍历，内存延迟无法隐藏（除非同时跑很多 block）。

### 方案 B：2D Block，每线程处理一个元素 + Block 内 Reduce

```c
__global__ void row_sum_tiled(float* mat, float* out, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    // block 内所有线程并行读一行，再 tree-reduce
}

dim3 block(256);
dim3 grid(M);   // 每行一个 block
```

**适用：** 行内元素间需要协同（如求和、max、softmax）。  
**优势：** 行内并行，用 smem 做 reduce，快得多。  
**关键：** 此时 2D 不是必须的，1D block 也能做（row = blockIdx.x），但用 `threadIdx.x` 做列索引。

## 3. MatMul：2D 必须

输出是 M×N 矩阵，天然 2D。一维 block 强行映射会丢掉行列的直观语义。

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;  // M 方向
int col = blockIdx.x * blockDim.x + threadIdx.x;  // N 方向
```

## 4. 铁律：threadIdx.x → 连续内存

无论几维 block，**threadIdx.x 必须映射到内存连续的方向**。

| 场景 | 连续维度 | threadIdx.x 对应 |
|------|----------|------------------|
| 1D element-wise | 数组索引 | `idx` |
| Row-major matmul | 列 (N) | `col` |
| Col-major matmul | 行 (M) | `row` |
| Image (row-major) | 列 (width) | `x` 坐标 |

违反这条 = warp 内内存访问分散 = performance 崩盘。

## 5. Block Size 选择

```
总线程数 = blockDim.x * blockDim.y * blockDim.z
```

- 必须是 **32 的倍数**（1 个 warp）
- 推荐 **128~512**，常见：256
- 不能超过 GPU 的 max threads/block（通常 1024）
- 也要考虑 smem / register 限制，太高会拉低 occupancy

## 6. 总结速查

| 计算类型 | Grid | Block | 线程职责 |
|----------|------|-------|----------|
| Element-wise add | 1D | 1D | 1 线程 = 1 输出元素 |
| Vector dot product | 1D | 1D | 1 线程块 = partial reduce |
| Row softmax | 1D | 1D | 1 block = 1 行，线程并行 reduce |
| MatMul | 2D | 2D | 1 线程 = 1 个 C[i][j] |
| 2D Convolution | 2D/3D | 2D/3D | 1 线程 = 1 个输出 pixel |
| 3D Volume | 3D | 3D | 1 线程 = 1 个 voxel |
