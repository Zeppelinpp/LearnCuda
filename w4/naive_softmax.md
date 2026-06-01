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
## Baseline 结果
```bash
Grid: 32768, Block: 256
Time: 18.259 ms (avg of 10 runs)
GFLOPS: 294.03
Bandwidth: 1176.11 GB/s
```

寄存器/shared memory 暂存expVal,统一写出HBM
```bash

```
