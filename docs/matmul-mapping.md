# MatMul Thread/Block 映射

## 核心思想

输出矩阵 `C` 的形状是 **M × N**（M 行，N 列）。  
GPU 用一个 **2D Grid** 覆盖整个输出矩阵，**每个线程负责计算 C 的一个元素**。

## 坐标映射

```
          N (列)  →  x 方向
       ┌───────────────────┐
    M  │                   │
    (行)│    C = A × B      │
    ↓  │                   │
    y  └───────────────────┘
```

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;  // [0, M)
int col = blockIdx.x * blockDim.x + threadIdx.x;  // [0, N)
```

- **x 维度 ↔ 列 (N)**：因为内存是 row-major，`col` 增加意味着在内存中步进一个元素（stride=1），满足 memory coalescing。
- **y 维度 ↔ 行 (M)**：`row` 增加意味着跳过一整行（stride=N）。

## Grid / Block 结构

```
GridDim  = ( ceil(N / blockDim.x), ceil(M / blockDim.y) )
BlockDim = ( TX, TY )   例如 (32, 32) 或 (16, 16)
```

```
Block(0,0)          Block(1,0)          Block(2,0)
┌──────────┐       ┌──────────┐       ┌──────────┐
│(0,0) (0,1)│       │(0,32)(0,33)│       │ ...      │
│(1,0) (1,1)│       │(1,32)     │       │          │
│ ...      │       │ ...      │       │          │
└──────────┘       └──────────┘       └──────────┘

Block(0,1)
┌──────────┐
│(32,0)    │
│ ...      │
└──────────┘
```

- 每个 Block 是一个线程组，共享 Shared Memory（Tiling 优化时用）。
- 每个线程算一个 `C[row][col]`，循环遍历 `k = 0..K-1`：
  ```c
  float sum = 0;
  for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
  }
  out[row * N + col] = sum;
  ```

## Shared Memory Bank Conflict

### 为什么会冲突

Shared Memory 被分成 **32 个 bank**（cc ≥ 3.0）。  
一个 warp（32 线程）如果同时访问**同一个 bank 的不同地址**，访问会被串行化。

在 tiled matmul 中，compute 阶段访问 `tile_A[ty][i]`：
```
bank = (ty * TILE_SIZE + i) % 32
```

当 **TILE_SIZE 是 32 的倍数**（如 32、64）时：
- ty=0 的行起始 bank = i % 32
- ty=1 的行起始 bank = (32 + i) % 32 = i % 32
- 同一 warp 内 ty=0 和 ty=1 的线程同时访问**同一个 bank 的不同地址** → **2-way bank conflict**

TILE_SIZE=16 时，`16 % 32 ≠ 0`，相邻行天然错开 16 个 bank，**没有 conflict**。但加 padding 是防御性写法，改 TILE_SIZE 时不用回头修。

### Padding 原理

```c
__shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
```

每行多加 1 个 float，行 stride 从 `TILE_SIZE` 变成 `TILE_SIZE + 1`：
```
bank = (ty * (TILE_SIZE + 1) + i) % 32
```

因为 `TILE_SIZE + 1 ≡ 1 (mod 32)`，相邻行的起始 bank 永远相差 1，**彻底打散 conflict**。

### 需要调整的地方

**不需要任何调整。**

Load 阶段和 compute 阶段的索引逻辑完全不变：
```c
tile_A[ty][tx] = ...;      // 写还是 tile_A[ty][tx]
sum += tile_A[ty][i] * ...; // 读还是 tile_A[ty][i]
```

编译器会自动根据声明的 `[TILE_SIZE][TILE_SIZE+1]` 计算正确的内存偏移。你只需改声明，逻辑代码一模一样。

## 关键记忆点

| 概念 | 对应关系 |
|------|----------|
| `blockIdx.x` / `threadIdx.x` | 列方向 (N) |
| `blockIdx.y` / `threadIdx.y` | 行方向 (M) |
| 一个线程 | 一个输出元素 `C[row][col]` |
| 一个 Block | 一片连续的子矩阵 |
| smem padding | 声明 `[TILE][TILE+1]`，代码逻辑不变 |
