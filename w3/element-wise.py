import torch

N = 1 << 26
a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)

_ = a + b

def benchmark_add(func, *args, name="Add", n_warmup=5, n_repeat=20):
    for _ in range(n_warmup):
        func(*args)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        func(*args)
    end.record()
    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end) / n_repeat
    return elapsed_time

ms = benchmark_add(lambda x, y: torch.add(x, y), a, b, name="Element-wise Addition")
bytes_total = 3 * N * 4
bw = (bytes_total / 1e9 / (ms / 1000.0))

print(f"Pytorch add: {ms:.2f} ms, Bandwidth: {bw:.2f} GB/s")
