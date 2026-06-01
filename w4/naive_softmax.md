# Navie Softmax分析

## 访存模式

1. 从 `HBM读` 加载数据进行block内求max -> 写入到shared memory
2. 对shared memory中block级别的max进行tree reduction得到整行的max
3. 重新从 `HBM读` 行内各元素与行max进行 expVal = exp(x - max)
4. `写` output[row * N + i] = expVal -> `HBM写`, 同时维护block内的localSum
5. 对localSum进行tree reduction求和 -> sum(exp(x-max))
6. `HBM读` output内容，做 /= rowSum 得到整行的softmax结果
7. `写` output 到HBM得到最终结果

## 重复访存:
1,3 重复对input的元素进行 `HBM读`
4,6 对output进行了`写`然后又重新`读`回并更新，最后再 `写` 回HBM

## 思路
能否同时更新max和sum？

---
## 实验结果

```bash
========= softmaxNaive ==========
Grid: 32768, Block: 256
Time: 18.245 ms (avg of 10 runs)
GFLOPS: 294.25
Bandwidth: 1177.00 GB/s
Max error: 5.784386e-10
PASS

========== onlineSoftmax ==========
Grid: 32768, Block: 256
Time: 13.598 ms (avg of 10 runs)
GFLOPS: 552.74
Bandwidth: 947.55 GB/s
Max error: 5.711627e-10
PASS

========== onlineSoftmax_vectorized ==========
Grid: 32768, Block: 256
Time: 9.337 ms (avg of 10 runs)
GFLOPS: 804.95
Bandwidth: 1379.92 GB/s
Max error: 5.711627e-10
PASS

========== Speedup ==========
naive:   18.245 ms
online:  13.598 ms
Speedup(online) vs naive: 1.34x
vector:  9.337 ms
Speedup(vectorized & online) vs naive:  1.95x
Speedup(vectorized & online) vs online: 1.46x
```
