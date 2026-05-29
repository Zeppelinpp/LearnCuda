# CUDA Event 计时

测 GPU kernel 真实运行时间的标准做法。所有 kernel benchmark 都会用到。

## 1. 为什么需要它：CUDA 的异步执行模型

CUDA kernel launch 是**异步**的：

```
CPU 调用 kernel<<<...>>>()
    ↓
   把 kernel 塞进 GPU 的 stream 队列（μs 级开销）
    ↓
   CPU 立即返回，继续执行下一行
    ↓
   GPU 后台按顺序执行队列里的 kernel
```

后果：**CPU 的 wall-clock 时间 ≠ GPU kernel 的真实运行时间**。
用 `clock_gettime` 测出来的是 "把 kernel 塞进队列的时间"，根本不是 kernel 本身的执行时间。

## 2. 核心思想：在队列里"打卡"

CUDA Event 的本质：**插入到 stream 队列里的特殊标记**，GPU 执行到它时盖一个时间戳（精度 0.5μs）。

```
CPU 时间线                       GPU Stream 队列
─────────                       ────────────────
cudaEventCreate(&start)     ─→  申请打卡器对象
cudaEventRecord(start)      ─→  [start ⏱]      ← 插入
                                 
kernel<<<>>>()              ─→  [kernel 1 ]
kernel<<<>>>()              ─→  [kernel 2 ]
   ...                          [   ...   ]
cudaEventRecord(stop)       ─→  [stop  ⏱]      ← 插入
                                      ↓
                            GPU 按顺序执行:
                            执行 start → 盖时间戳 t₀
                            执行 kernel × N
                            执行 stop  → 盖时间戳 t₁

cudaEventSynchronize(stop)  ─★ CPU 阻塞，等 GPU 真的执行到 stop
cudaEventElapsedTime        ─→ 返回 t₁ - t₀ (ms)
```

## 3. 三个 API 的角色

| API | 在干什么 | 类比 |
|-----|---------|------|
| `cudaEventCreate(&e)` | 申请 event 对象（类似 malloc） | 买一个打卡器 |
| `cudaEventRecord(e)` | 把 event 推进 GPU 队列，异步 | 把打卡器丢进流水线 |
| `cudaEventSynchronize(e)` | CPU 阻塞等待 event 被 GPU 执行完 | 站在终点等打卡器出来 |
| `cudaEventElapsedTime(&ms, s, t)` | 读两个 event 的时间戳差 | 查后台记录 |
| `cudaEventDestroy(e)` | 释放 event 对象 | 扔掉打卡器 |

## 4. 标准模板（5 步）

```
cudaEvent_t start, stop;
cudaEventCreate(&start);           // ① 申请
cudaEventCreate(&stop);

cudaEventRecord(start);            // ② 插入起点
for (int i = 0; i < repeats; i++) {
    kernel<<<grid, block>>>(...);  //   推 N 个 kernel
}
cudaEventRecord(stop);             // ③ 插入终点

cudaEventSynchronize(stop);        // ④ 等 stop 完成
float ms;
cudaEventElapsedTime(&ms, start, stop);  // ⑤ 读时间差

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

## 5. `cudaDeviceSynchronize` vs `cudaEventSynchronize`

| | `cudaEventSynchronize(stop)` | `cudaDeviceSynchronize()` |
|--|------------------------------|---------------------------|
| 等待范围 | 只等这个 event 之前的 stream 操作 | 等 device 上**所有 stream**全部做完 |
| 用途 | Event 计时后读取结果 | Host 读 GPU 数据前确保完成 |
| 性能影响 | 小（只等必要工作） | 大（全局栅栏，破坏流水线） |

**计时场景**：
- 简单 benchmark：可以用 `cudaDeviceSynchronize()` 替代 event sync，但会多等其它 stream 的工作
- 严谨 benchmark：必须用 `cudaEventSynchronize(stop)`，只等目标 stream

**Host 读数据场景**：
```c
kernel<<<...>>>(d_C, ...);
cudaDeviceSynchronize();              // 确保 kernel 写完了
cudaMemcpy(C, d_C, size, D2H);        // 再拷回 CPU
// 或者直接用同步 memcpy：
cudaMemcpy(C, d_C, size, D2H);        // pageable memory 下会隐式 sync
```

## 6. 常见坑

| 坑 | 后果 | 解决 |
|---|------|------|
| 漏掉 `cudaEventSynchronize` / `cudaDeviceSynchronize` | 读到错误时间戳（GPU 还没跑完） | 必须 sync 后才能读时间 |
| 不做 warmup | 前几次 kernel 慢（GPU clock 爬升 + cache 冷） | 先跑 5~10 次不计时 |
| repeats 太少 | 单次 < 1ms 时计时噪声大 | repeats 100~1000，结果取平均 |
| 把 `cudaMemcpy` 算进计时 | PCIe 传输时间污染 kernel 时间 | Record 只包 kernel，不包 memcpy |
| 跨 stream 用同一组 event | 时间戳不对应 | 确保 event 和 kernel 在同一 stream |

## 6. 跟 `std::chrono` 的对比

| 维度 | `cudaEvent` | `std::chrono` |
|------|------------|---------------|
| 测的对象 | GPU 真实执行时间 | CPU wall-clock |
| 异步友好 | ✓ 在 GPU 队列里打卡 | ✗ kernel 异步返回，测不准 |
| 精度 | 0.5 μs | ns 级，但对 GPU 没意义 |
| 用法 | benchmark kernel | benchmark host 代码 |

**结论**：测 GPU 必须用 `cudaEvent`，测 CPU 用 `chrono`。
