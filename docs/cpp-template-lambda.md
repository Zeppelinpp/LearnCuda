# C++ Template & Lambda

## 1. Template：编译期生成专用代码

```
template<typename T>
void f(T x) { ... }

f(3)     → 编译器生成 f(int)
f(3.14)  → 编译器生成 f(float)
f(λ)     → 编译器生成 f(lambda_type)  ← CUDA benchmark 用的就是这个
```

**核心优势：零运行时开销**。编译器在编译期就确定了具体类型，直接内联展开，没有虚函数调用。

对比 `std::function`：

| 方式 | 时机 | 开销 |
|------|------|------|
| `template<typename F>` | 编译期推断 | 0，直接内联 |
| `std::function<void(...)>` | 运行期擦除 | 虚表查找，有间接跳转 |

CUDA 性能代码永远优先 template。

## 2. Lambda 语法

```
[捕获列表](参数列表) { 函数体 }
    ↑          ↑           ↑
  能访问      接收什么      做什么
  哪些外部    参数
  变量
```

**捕获方式**：

| 写法 | 含义 |
|------|------|
| `[]` | 不捕获任何外部变量 |
| `[x]` | 按值捕获 x（拷贝一份） |
| `[&x]` | 按引用捕获 x（修改影响外部） |
| `[=]` | 按值捕获所有用到的变量 |
| `[&]` | 按引用捕获所有用到的变量 |

## 3. 在 CUDA Benchmark 中的典型模式

```
template<typename Launcher>
void benchmark(const char* name, Launcher launch) {
    // 共用流程：malloc / cudaMalloc / warmup / timing / free
    // ...
    launch(d_a, d_b, d_c);   // ★ 可变部分：编译器自动内联
    // ...
}

// 调用：编译器自动推断 Launcher 类型
benchmark("Naive", [](float* a, float* b, float* c) {
    naive_add<<<grid, block>>>(a, b, c);
});

benchmark("Vec4", [](float* a, float* b, float* c) {
    add_vec4<<<grid, block>>>(
        reinterpret_cast<const float4*>(a), ...);
});
```

每加一个 kernel 版本，只需写 5~10 行 lambda。

## 4. 为什么空捕获 `[]` 就够了？

```
#define N (1 << 26)       // 宏：预处理阶段直接文本替换

[](float* a, ...) {
    int grid = (N + ...) / BLOCK_SIZE;
    // 编译器看到的：((1 << 26) + ...) / 256
    // 没有"外部变量"需要捕获
}
```

如果改用普通变量 `int total_N = 1 << 26;`，则必须捕获：`[total_N]`。
