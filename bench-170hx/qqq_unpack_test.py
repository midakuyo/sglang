import sys, torch
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.marlin_qqq import qqq_pack_from_int4, qqq_unpack_to_int4, qqq_unpack_to_int8
from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
dev = "cuda"; torch.manual_seed(0)
def timeit(fn, iters=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters
tot = 0.0
for name, (N, K) in {"qkv": (16384, 5376), "o": (5376, 8192), "gate_up": (43008, 5376), "down": (5376, 21504)}.items():
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s_g = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    B, s_ch_p, s_grp_p, q8_ref, s_ch, s_grp = qqq_pack_from_int4(q4, s_g)
    q4_back = qqq_unpack_to_int4(B, K, N)
    ok4 = torch.equal(q4_back, q4.t())
    q8_back = qqq_unpack_to_int8(B, s_grp, K, N)
    ok8 = torch.equal(q8_back, q8_ref.t().contiguous())
    t = timeit(lambda: qqq_unpack_to_int8(B, s_grp, K, N)); tot += t * 60
    # int8 path end-to-end check vs kernel's int8 reference
    x = torch.randn(300, K, dtype=torch.bfloat16, device=dev); a_q, a_s = per_token_quant_int8(x)
    y8 = int8_scaled_mm(a_q, q8_back.t(), a_s, s_ch.reshape(-1), out_dtype=torch.bfloat16, bias=None)
    r = (a_q.float() @ q8_ref.float()) * s_ch * a_s.float()
    err = ((y8.float() - r).abs().max() / r.abs().max()).item()
    print(f"{name:8s} unpack int4 exact={ok4} int8 exact={ok8}  unpack {t:.2f} ms/layer  int8_scaled_mm rel err {err:.2e}", flush=True)
print(f"unpack total per forward (60 layers): {tot:.0f} ms")
