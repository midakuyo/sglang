# verify 비어텐션 스케일링 진단: Gemma4-31B 실형상 int8 GEMM(sgl_kernel int8_scaled_mm + per_token_quant_int8)과
# lm_head bf16 GEMM을 M(토큰 수)별로 계측 → "M=48(bs16×3)에서 컴퓨트 바운드" 가설 검증
import torch
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
dev = "cuda"
H, I, V = 5376, 21504, 262144
QKV, O = 32 * 256 + 2 * 16 * 256, 32 * 256
layers = {"qkv": (QKV, H), "o": (H, O), "gate_up": (2 * I, H), "down": (H, I)}
L = 60
def timeit(fn, iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters
Ws = {}
for name, (N, K) in layers.items():
    W = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
    Ws[name] = (W.t(), torch.rand(N, dtype=torch.float32, device=dev) * 0.01)
lm = torch.randn(V, H, dtype=torch.bfloat16, device=dev)
params = sum(N * K for N, K in layers.values()) * L
print(f"linear params {params/1e9:.1f}B (+lm_head {V*H/1e9:.2f}B); weight bytes/layer-set {params/1e9:.1f}GB int8")
print(f"{'M':>3} | {'gemm ms':>8} {'quant ms':>8} {'lm_head':>7} | {'total':>6} | {'eff TOPS':>8} {'GB/s':>6}")
for M in (1, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    xs = {"qkv": x, "gate_up": x, "o": torch.randn(M, O, dtype=torch.bfloat16, device=dev), "down": torch.randn(M, I, dtype=torch.bfloat16, device=dev)}
    gemm = quant = 0.0
    for name, (N, K) in layers.items():
        Wt, ws = Ws[name]; xin = xs[name]
        xq, xsc = per_token_quant_int8(xin)
        try:
            f = lambda: int8_scaled_mm(xq, Wt, xsc, ws, out_dtype=torch.bfloat16, bias=None)
            f()
        except Exception as ex:
            ws2 = ws.view(N, 1)
            f = lambda: int8_scaled_mm(xq, Wt, xsc, ws2, out_dtype=torch.bfloat16, bias=None)
            f()
        gemm += timeit(f) * L
        quant += timeit(lambda: per_token_quant_int8(xin)) * L
    lmh = timeit(lambda: torch.matmul(x, lm.t()))
    tot = gemm + quant + lmh
    flops = 2 * params * M
    print(f"{M:3d} | {gemm:8.2f} {quant:8.2f} {lmh:7.2f} | {tot:6.1f} | {flops/gemm/1e9:8.1f} {params/gemm/1e6:6.0f}", flush=True)
