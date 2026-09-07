# W4A8 Triton GEMM: 정확성(디퀀트 fp32 기준) + Gemma4 형상 M 스캔 벤치 vs int8_scaled_mm / bf16 matmul
import sys, torch, triton
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.w4a8_int_gemm import pack_w4_int8, w4a8_int_gemm, GROUP
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sgl_kernel import int8_scaled_mm
dev = "cuda"; torch.manual_seed(0)
H, I = 5376, 21504
shapes = {"qkv": (16384, H), "o": (H, 8192), "gate_up": (2 * I, H), "down": (H, I)}
def make(N, K):
    q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s = (torch.rand(N, K // GROUP, device=dev) * 0.02 + 0.005).to(torch.bfloat16)
    return q, s
def ref(a_q, a_s, q, s):
    w = q.float() * s.float().repeat_interleave(GROUP, dim=1)  # [N, K]
    return (a_q.float() * a_s.float()) @ w.t()
def timeit(fn, iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters
# ---- correctness
print("== correctness (max rel err vs fp32 dequant reference)")
for name, (N, K) in shapes.items():
    q, s = make(N, K); wp, ws = pack_w4_int8(q, s)
    for M in (1, 3, 24, 300):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        a_q, a_s = per_token_quant_int8(x)
        y = w4a8_int_gemm(a_q, a_s, wp, ws)
        r = ref(a_q, a_s, q, s)
        err = ((y.float() - r).abs().max() / r.abs().max()).item()
        print(f"  {name:8s} M={M:4d}: max|diff|/max|ref| = {err:.2e}  {'OK' if err < 1e-2 else 'FAIL'}")
# ---- bench
print("== bench: ms per Gemma4 layer-set x60 (qkv+o+gate_up+down), GB/s of int4 weights")
params = sum(N * K for N, K in shapes.values())
print(f"{'M':>5} | {'w4a8 ms':>8} {'GB/s':>6} | {'int8 ms':>8} {'GB/s':>6} | {'bf16 ms':>8}")
Ws = {}
for name, (N, K) in shapes.items():
    q, s = make(N, K); wp, ws = pack_w4_int8(q, s)
    w8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
    Ws[name] = (wp, ws, w8.t(), torch.rand(N, device=dev) * 0.01, torch.randn(N, K, dtype=torch.bfloat16, device=dev))
for M in (1, 3, 8, 16, 24, 48, 128, 2048):
    t4 = t8 = tb = 0.0
    for name, (N, K) in shapes.items():
        wp, ws, w8t, s8, wb = Ws[name]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        a_q, a_s = per_token_quant_int8(x)
        t4 += timeit(lambda: w4a8_int_gemm(a_q, a_s, wp, ws)) * 60
        t8 += timeit(lambda: int8_scaled_mm(a_q, w8t, a_s, s8, out_dtype=torch.bfloat16, bias=None)) * 60
        tb += timeit(lambda: x @ wb.t()) * 60 if M <= 512 else 0
    print(f"{M:5d} | {t4:8.2f} {params*60/2/t4/1e6:6.0f} | {t8:8.2f} {params*60/t8/1e6:6.0f} | {tb:8.2f}", flush=True)
