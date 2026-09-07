# W4A8 커널 절제: MODE 0 full / 1 no-unpack / 2 no-dot(sum) / 3 load-only  + 타일·stage 스윕 (M=3, Gemma4 4형상 ×60층)
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.w4a8_int_gemm import pack_w4_int8, GROUP
dev = "cuda"; torch.manual_seed(0)
H, I = 5376, 21504
shapes = {"qkv": (16384, H), "o": (H, 8192), "gate_up": (2 * I, H), "down": (H, I)}
params = sum(N * K for N, K in shapes.values()) * 60

@triton.jit
def k(a_ptr, w_ptr, s_ptr, part_ptr, M, N, G, gps, stride_am, stride_wk, stride_sk, stride_ps, stride_pm,
      MODE: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    HALF: tl.constexpr = GROUP // 2
    pid_m = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_j = tl.arange(0, HALF)
    mask_m = offs_m < M; mask_n = offs_n < N
    g0 = pid_k * gps; g1 = tl.minimum(g0 + gps, G)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    a_lo_ptrs = a_ptr + offs_m[:, None] * stride_am + g0 * GROUP + offs_j[None, :]; a_hi_ptrs = a_lo_ptrs + HALF
    w_ptrs = w_ptr + (g0 * HALF + offs_j[:, None]) * stride_wk + offs_n[None, :]; s_ptrs = s_ptr + g0 * stride_sk + offs_n
    for g in range(g0, g1):
        a_lo = tl.load(a_lo_ptrs, mask=mask_m[:, None], other=0); a_hi = tl.load(a_hi_ptrs, mask=mask_m[:, None], other=0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
        s = tl.load(s_ptrs, mask=mask_n, other=0.0)
        if MODE == 0:
            lo = (w << 4).to(tl.int8, bitcast=True) >> 4; hi = w.to(tl.int8, bitcast=True) >> 4
            part = tl.dot(a_lo, lo) + tl.dot(a_hi, hi); acc += part.to(tl.float32) * s[None, :]
        elif MODE == 1:
            lo = w.to(tl.int8, bitcast=True); part = tl.dot(a_lo, lo) + tl.dot(a_hi, lo); acc += part.to(tl.float32) * s[None, :]
        elif MODE == 2:
            lo = (w << 4).to(tl.int8, bitcast=True) >> 4; hi = w.to(tl.int8, bitcast=True) >> 4
            acc += (tl.sum(lo.to(tl.float32), 0)[None, :] + tl.sum(hi.to(tl.float32), 0)[None, :] + tl.sum(a_lo.to(tl.float32), 1)[:, None] * 0) * s[None, :]
        else:
            acc += (tl.sum(w.to(tl.float32), 0)[None, :] + tl.sum(a_lo.to(tl.float32), 1)[:, None] * 0) * s[None, :]
        a_lo_ptrs += GROUP; a_hi_ptrs += GROUP; w_ptrs += HALF * stride_wk; s_ptrs += stride_sk
    tl.store(part_ptr + pid_k * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])

def run(a_q, wp, ws, mode, bm, bn, nw, ns, split_target=432):
    M, K = a_q.shape; K2, N = wp.shape; G = K // GROUP
    programs = triton.cdiv(M, bm) * triton.cdiv(N, bn); split = max(1, min(G, -(-split_target // programs)))
    gps = -(-G // split); split = -(-G // gps)
    part = torch.empty((split, M, N), dtype=torch.float32, device=dev)
    k[(triton.cdiv(M, bm), triton.cdiv(N, bn), split)](a_q, wp, ws, part, M, N, G, gps, a_q.stride(0), wp.stride(0), ws.stride(0), part.stride(0), part.stride(1),
        MODE=mode, GROUP=GROUP, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns)
    return part
def timeit(fn, iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters
M = 3
Ws = {}
for name, (N, K) in shapes.items():
    q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s = (torch.rand(N, K // GROUP, device=dev) * 0.02).to(torch.bfloat16)
    wp, ws = pack_w4_int8(q, s); a_q = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev); Ws[name] = (a_q, wp, ws)
def total(mode, bm, bn, nw, ns, st=432):
    t = 0.0
    for name in shapes: a_q, wp, ws = Ws[name]; t += timeit(lambda: run(a_q, wp, ws, mode, bm, bn, nw, ns, st)) * 60
    return t
print(f"{'mode':>14} {'bm':>3} {'bn':>4} {'nw':>3} {'ns':>3} {'split>=':>8} | {'ms':>7} {'GB/s':>6}")
for label, mode in (("full", 0), ("no-unpack", 1), ("no-dot", 2), ("load-only", 3)):
    t = total(mode, 16, 128, 4, 4); print(f"{label:>14} {16:3d} {128:4d} {4:3d} {4:3d} {432:8d} | {t:7.2f} {params/2/t/1e6:6.0f}", flush=True)
print("-- full, tile/stage sweep")
for bm, bn, nw, ns, st in ((16, 64, 4, 4, 432), (16, 128, 4, 2, 432), (16, 128, 4, 3, 432), (16, 128, 8, 4, 432), (16, 256, 8, 4, 432), (16, 256, 8, 3, 216), (16, 128, 4, 4, 864), (16, 128, 4, 4, 216), (32, 128, 4, 4, 432)):
    try:
        t = total(0, bm, bn, nw, ns, st); print(f"{'full':>14} {bm:3d} {bn:4d} {nw:3d} {ns:3d} {st:8d} | {t:7.2f} {params/2/t/1e6:6.0f}", flush=True)
    except Exception as ex: print(f"{'full':>14} {bm:3d} {bn:4d} {nw:3d} {ns:3d} {st:8d} | FAIL {str(ex)[:60]}")
