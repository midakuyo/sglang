# W4A8 디코드용 CUDA-core GEMV 변형 (M<=4): w 레이아웃 [N, K/2](K 연속), fp32 FMA, 그룹 스케일. M=3, Gemma4 4형상 ×60
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/src/python")
dev = "cuda"; torch.manual_seed(0); GROUP = 128
H, I = 5376, 21504
shapes = {"qkv": (16384, H), "o": (H, 8192), "gate_up": (2 * I, H), "down": (H, I)}
params = sum(N * K for N, K in shapes.values()) * 60

def pack_nk(w_q, w_scale):  # -> uint8 [N, K/2] (group-local lo/hi planes), fp32 [N, K/GROUP]
    N, K = w_q.shape; half = GROUP // 2
    q = w_q.view(N, K // GROUP, GROUP)
    packed = ((q[:, :, :half].to(torch.uint8) & 0xF) | ((q[:, :, half:].to(torch.uint8) & 0xF) << 4)).reshape(N, K // 2).contiguous()
    return packed, w_scale.to(torch.float32).contiguous()

@triton.jit
def gemv(a_ptr, w_ptr, s_ptr, part_ptr, M, N, G, gps, stride_am, stride_wn, stride_sn, stride_ps, stride_pm,
         GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    HALF: tl.constexpr = GROUP // 2
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M); offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_j = tl.arange(0, HALF)
    mask_m = offs_m < M; mask_n = offs_n < N
    g0 = pid_k * gps; g1 = tl.minimum(g0 + gps, G)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    a_lo_ptrs = a_ptr + offs_m[:, None] * stride_am + g0 * GROUP + offs_j[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + g0 * HALF + offs_j[None, :]
    s_ptrs = s_ptr + offs_n * stride_sn + g0
    for g in range(g0, g1):
        a_lo = tl.load(a_lo_ptrs, mask=mask_m[:, None], other=0).to(tl.float32)           # [BM, HALF]
        a_hi = tl.load(a_lo_ptrs + HALF, mask=mask_m[:, None], other=0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask_n[:, None], other=0)                                # [BN, HALF] uint8
        lo = ((w << 4).to(tl.int8, bitcast=True) >> 4).to(tl.float32)
        hi = (w.to(tl.int8, bitcast=True) >> 4).to(tl.float32)
        part = tl.sum(a_lo[:, None, :] * lo[None, :, :], 2) + tl.sum(a_hi[:, None, :] * hi[None, :, :], 2)  # [BM, BN]
        s = tl.load(s_ptrs, mask=mask_n, other=0.0)
        acc += part * s[None, :]
        a_lo_ptrs += GROUP; w_ptrs += HALF; s_ptrs += 1
    tl.store(part_ptr + pid_k * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])

def run(a_q, wp, ws, bm, bn, nw, ns, split_target):
    M, K = a_q.shape; N = wp.shape[0]; G = K // GROUP
    programs = triton.cdiv(N, bn); split = max(1, min(G, -(-split_target // programs)))
    gps = -(-G // split); split = -(-G // gps)
    part = torch.empty((split, M, N), dtype=torch.float32, device=dev)
    gemv[(programs, split)](a_q, wp, ws, part, M, N, G, gps, a_q.stride(0), wp.stride(0), ws.stride(0), part.stride(0), part.stride(1),
                            GROUP=GROUP, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns)
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
    wp, ws = pack_nk(q, s); a_q = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    Ws[name] = (a_q, wp, ws, q, s)
# correctness on one shape
a_q, wp, ws, q, s = Ws["o"]
part = run(a_q, wp, ws, 4, 32, 4, 3, 432).sum(0)
ref = a_q.float() @ (q.float() * s.float().repeat_interleave(GROUP, dim=1)).t()
print("gemv rel err:", ((part - ref).abs().max() / ref.abs().max()).item())
print(f"{'bm':>3} {'bn':>4} {'nw':>3} {'ns':>3} {'split>=':>8} | {'ms':>7} {'GB/s':>6}")
for bm, bn, nw, ns, st in ((4, 32, 4, 3, 432), (4, 32, 2, 3, 432), (4, 64, 4, 3, 432), (4, 64, 8, 3, 432), (4, 32, 4, 4, 864), (4, 16, 2, 3, 864), (4, 128, 8, 2, 432), (4, 64, 4, 2, 864)):
    try:
        t = 0.0
        for name in shapes: a_q, wp, ws, _, _ = Ws[name]; t += timeit(lambda: run(a_q, wp, ws, bm, bn, nw, ns, st)) * 60
        print(f"{bm:3d} {bn:4d} {nw:3d} {ns:3d} {st:8d} | {t:7.2f} {params/2/t/1e6:6.0f}", flush=True)
    except Exception as ex: print(f"{bm:3d} {bn:4d} {nw:3d} {ns:3d} {st:8d} | FAIL {str(ex)[:80]}")
