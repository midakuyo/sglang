# Marlin-QQQ 포팅 테스트: 정확성(커널이 쓰는 int8 가중치 기준 fp32 참조) + Gemma4 형상 M 스캔 vs int8_scaled_mm
import sys, torch
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.marlin_qqq import marlin_qqq_gemm, marlin_qqq_workspace, qqq_pack_from_int4
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sgl_kernel import int8_scaled_mm
dev = "cuda"; torch.manual_seed(0)
H, I = 5376, 21504
shapes = {"qkv": (16384, H), "o": (H, 8192), "gate_up": (2 * I, H), "down": (H, I)}
params = sum(N * K for N, K in shapes.values()) * 60
def timeit(fn, iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters
print("== correctness (kernel vs fp32 reference on the kernel's own int8 weights)")
for name, (N, K) in shapes.items():
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s_g = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    B, s_ch_p, s_grp_p, q8, s_ch = qqq_pack_from_int4(q4, s_g)
    ws = marlin_qqq_workspace(N, dev)
    for M in (1, 3, 24, 300):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        a_q, a_s = per_token_quant_int8(x)
        y = marlin_qqq_gemm(a_q, B, a_s, s_ch_p, s_grp_p, ws, M, N, K)
        r = (a_q.float() @ q8.float()) * s_ch * a_s.float()
        err = ((y.float() - r).abs().max() / r.abs().max()).item()
        print(f"  {name:8s} M={M:4d}: rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
print("== bench: ms per 60-layer set, GB/s of weight bytes (int4 vs int8)")
print(f"{'M':>5} | {'qqq ms':>8} {'GB/s':>6} | {'int8 ms':>8} {'GB/s':>6}")
Ws = {}
for name, (N, K) in shapes.items():
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s_g = (torch.rand(N, K // 128, device=dev) * 0.02).to(torch.bfloat16)
    B, s_ch_p, s_grp_p, _, _ = qqq_pack_from_int4(q4, s_g)
    w8 = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev)
    Ws[name] = (B, s_ch_p, s_grp_p, marlin_qqq_workspace(N, dev), w8.t(), torch.rand(N, device=dev) * 0.01)
for M in (1, 3, 8, 16, 24, 48, 128, 512, 2048):
    tq = t8 = 0.0
    for name, (N, K) in shapes.items():
        B, s_ch_p, s_grp_p, ws, w8t, s8 = Ws[name]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev); a_q, a_s = per_token_quant_int8(x)
        tq += timeit(lambda: marlin_qqq_gemm(a_q, B, a_s, s_ch_p, s_grp_p, ws, M, N, K)) * 60
        t8 += timeit(lambda: int8_scaled_mm(a_q, w8t, a_s, s8, out_dtype=torch.bfloat16, bias=None)) * 60
    print(f"{M:5d} | {tq:8.2f} {params/2/tq/1e6:6.0f} | {t8:8.2f} {params/t8/1e6:6.0f}", flush=True)
