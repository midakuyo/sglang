# 2차: CUDA graph 캡처로 런치 오버헤드 제거 → 그래프 안 실비용. int8_scaled_mm(CUTLASS) vs torch._int_mm(cuBLAS)+스케일 에필로그 비교
import torch
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
dev = "cuda"
H, I = 5376, 21504
QKV, O = 32 * 256 + 2 * 16 * 256, 32 * 256
layers = {"qkv": (QKV, H), "o": (H, O), "gate_up": (2 * I, H), "down": (H, I)}
L = 60
params = sum(N * K for N, K in layers.values()) * L
Ws = {n: (torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev), torch.rand(N, dtype=torch.float32, device=dev) * 0.01) for n, (N, K) in layers.items()}
Wt = {n: W.t() for n, (W, _) in Ws.items()}
def graph_time(fn, reps=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps
print(f"{'M':>3} | {'cutlass':>8} {'cublas':>8} {'quant':>6} | cutlass GB/s  cublas GB/s   (ms, 60층 전체, 그래프 리플레이)")
for M in (1, 3, 8, 12, 24, 48, 96):
    xs = {"qkv": torch.randn(M, H, dtype=torch.bfloat16, device=dev), "o": torch.randn(M, O, dtype=torch.bfloat16, device=dev),
          "down": torch.randn(M, I, dtype=torch.bfloat16, device=dev)}
    xs["gate_up"] = xs["qkv"]
    xq = {n: per_token_quant_int8(x) for n, x in xs.items()}
    def cutlass():
        for n, (N, K) in layers.items():
            q, sc = xq[n]
            for _ in range(L): int8_scaled_mm(q, Wt[n], sc, Ws[n][1], out_dtype=torch.bfloat16, bias=None)
    def cublas():
        for n, (N, K) in layers.items():
            q, sc = xq[n]
            for _ in range(L):
                y = torch._int_mm(q, Wt[n])            # [M,N] int32
                (y.to(torch.float32) * sc * Ws[n][1]).to(torch.bfloat16)
    def quant():
        for n, x in xs.items():
            for _ in range(L): per_token_quant_int8(x)
    tc = graph_time(cutlass)
    try:
        tb = graph_time(cublas)
    except Exception as ex:
        tb = float("nan"); print("cublas fail:", str(ex)[:80])
    tq = graph_time(quant)
    print(f"{M:3d} | {tc:8.2f} {tb:8.2f} {tq:6.2f} | {params/tc/1e6:10.0f} {params/tb/1e6:12.0f}", flush=True)
