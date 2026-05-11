# Element Wise 算子

以 element-wise add (`c = a + b`) 为练习，理解 memory-bound kernel 的优化路径和核心概念。

## 1. 算子特征：Memory-Bound

| 维度 | 数值 |
|------|------|
| 每个元素的计算量 | 1 FLOP (一次加法) |
| 每个元素的内存访问 | 读 2 个 float + 写 1 个 float = **12 bytes** |
| 算术强度 (Arithmetic Intensity) | 1 / 12 ≈ **0.083 FLOP/byte** |

**对比 GPU 的能力比例**（以 A100 为例）：
- FP32 算力 ≈ 19.5 TFLOPS
- HBM 带宽 ≈ 1.55 TB/s
- 平衡点 ≈ 19.5 / 1.55 ≈ **12.6 FLOP/byte**

Element-wise add 的算术强度 0.08，**差了 150 倍**——瓶颈 100% 在内存带宽上。

**优化目标 = 怎么尽量打满 HBM 带宽。**

## 2. 有效带宽

```
Effective BW = (读字节数 + 写字节数) / kernel 时间
             = 3 * N * sizeof(float) / time
```

A100 HBM 峰值约 1555 GB/s，kernel 能跑到 **1300~1450 GB/s**（80~93%）就已经很好了。

## 3. V1：Naive 一线程一元素

每个线程处理一个 float：

```
Block 0           Block 1           Block 2
[T0 T1 ... T255]  [T0 T1 ... T255]  [T0 T1 ... T255]
 │                 │                 │
 ▼                 ▼                 ▼
a[0..255]         a[256..511]       a[512..767]
```

同一个 warp 内 32 个线程访问连续 32 个 float → 硬件**自动合并 (coalesce)** 成 1 个 128-byte memory transaction。

## 4. V2：向量化加载 (float4)

### 4.1 为什么做向量化

V1 每个线程只搬 4 byte，memory controller 的 in-flight 请求"含金量"低。V2 让每个线程一次搬 16 byte（4 个 float 打包成 float4）：

```
float 数组:
地址:   0x1000  0x1004  0x1008  0x100C  0x1010  0x1014  0x1018  0x101C
       ├───────┴───────┴───────┴───────┤├───────┴───────┴───────┴───────┤
       │           float4[0]            ││           float4[1]            │
       └────────────────────────────────┘└────────────────────────────────┘
         16 bytes                           16 bytes

float*  d_a:  d_a[0]  d_a[1]  d_a[2]  d_a[3]  d_a[4] ...
float4* p:    p[0]=(a0,a1,a2,a3)        p[1]=(a4,a5,a6,a7) ...
```

### 4.2 reinterpret_cast

把 `float*` 重新解释为 `float4*`，让编译器生成 `LD.E.128`（一次 load 128-bit）指令：

```
float4* p = reinterpret_cast<float4*>(d_a);
// "把这块内存按每 16 byte 一组来读"
```

`reinterpret_cast` 是 C++ 四种 cast 之一，语义是**重新解释位模式**（不管类型语义关联）。CUDA 向量化场景下必须使用它。

### 4.3 指针 `*` 的双重含义

```
声明时:  float *h_a;       // h_a 是一个指针变量，存的是 64-bit 内存地址
使用时:  *h_a = 3.14;      // 解引用：去 h_a 存的地址读写值
         h_a[i] = 3.14;    // 等价于 *(h_a + i)
```

`cudaMalloc` 返回的地址赋值给 `float*`，表示"从该地址开始，每 4 byte 是一个 float"。`reinterpret_cast` 把它改成"每 16 byte 是一个 float4"。

### 4.4 V2 比 V1 快的三个层面

| 对比 | V1 (float) | V2 (float4) |
|------|-----------|-------------|
| 每 warp 数据量 | 128 B | 512 B |
| 处理 N 元素的指令数 | 3N 条 | 3N/4 条 |
| In-flight 含金量 | 256 req × 4 B = 1 KB | 256 req × 16 B = 4 KB |

核心：指令发射减少 + Memory-Level Parallelism (MLP) ×4 → 更容易把 HBM bus 打满。

## 5. Benchmark 方法学

```
1. cudaMalloc / 初始化数据
2. Warmup (5~10 次)      ← GPU 时钟频率爬升、cache 预热
3. cudaEventRecord(start)
   for (i = 0; i < repeats; i++) kernel<<<...>>>()
   cudaEventRecord(stop)
   cudaEventSynchronize(stop)
4. 计算 BW = 3 * N * 4 * repeats / (ms * 1e6) GB/s
```

关键点：
- `cudaEventRecord` 是**异步**的，必须 `Synchronize` 后才能读时间
- 不要把 `cudaMemcpy` 算进 kernel 时间
- repeats 100~1000 次取平均，消除单次计时噪声

## 6. 横测结果与分析

实际跑在 A100-PCIE-40GB（理论峰值 ~1555 GB/s）：

| 版本 | 时间 | 带宽 | 占峰值 |
|------|------|------|--------|
| V1 Naive | 0.593 ms | 1358 GB/s | **87%** |
| V2 Vec4 | 0.591 ms | 1364 GB/s | **88%** |

**Vec4 只快了 0.4%**——因为 V1 已经快到天花板了。A100 的 memory controller 对简单 coalesced pattern 已经足够高效，NVCC 甚至可能已经做了部分自动向量化。

**结论**：在这个 workload 上，naive → vec4 的边际收益递减。如果 naive 只跑到 60%，vec4 通常能拉到 85%+。当 naive 本身已经 87% 时，再想提升只能换方向（如 grid-stride 减少调度开销）。
