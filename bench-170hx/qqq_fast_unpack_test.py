# 구조적 순열 기반 고속 언팩: 프로그램당 (8 k-타일 = 그룹 128) × (1024-블록 = 64 n), 코얼레싱 로드/스토어 + permute
import sys, torch, triton, triton.language as tl
from triton.language.extra import libdevice
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.marlin_qqq import qqq_pack_from_int4, qqq_unpack_to_int8_triton

@triton.jit
def _qqq_unpack_fast_kernel(b_ptr, s_ptr, out_ptr, N, K, stride_bk, stride_sk, GROUP: tl.constexpr):
    g = tl.program_id(0)      # group index along K (8 k-tiles)
    blk = tl.program_id(1)    # 1024-element block along N (64 columns)
    offs_r = tl.arange(0, 8)          # k-tile within group
    offs_j = tl.arange(0, 128)        # int32 within block
    ptrs = b_ptr + (g * 8 + offs_r)[:, None] * stride_bk + blk * 128 + offs_j[None, :]
    w = tl.load(ptrs)                 # int32 [8, 128], coalesced
    # nibbles p = 0..7 -> last axis via joins: order x0..x7
    n0 = (w >> 0) & 0xF; n1 = (w >> 4) & 0xF; n2 = (w >> 8) & 0xF; n3 = (w >> 12) & 0xF
    n4 = (w >> 16) & 0xF; n5 = (w >> 20) & 0xF; n6 = (w >> 24) & 0xF; n7 = (w >> 28) & 0xF
    v = tl.join(tl.join(tl.join(n0, n1), tl.join(n2, n3)), tl.join(tl.join(n4, n5), tl.join(n6, n7)))  # [8,128,2,2,2]
    # measured layout: j = nt*32 + i_hh*16 + i_hl*4 + i_lo, p = p2*4 + p1*2 + p0
    #   k = r*16 + i_hl*4 + p0*2 + p2,   n = i_lo*16 + p1*8 + nt*2 + i_hh
    # nested tl.join appends the new axis last, so the joined nibble axes come
    # out as (bit0, bit1, bit2) of the nibble index p
    v = tl.reshape(v, (8, 4, 2, 4, 4, 2, 2, 2))  # r, nt, i_hh, i_hl, i_lo, p0, p1, p2
    v = tl.permute(v, (4, 6, 1, 2, 0, 3, 5, 7))  # i_lo, p1, nt, i_hh | r, i_hl, p0, p2
    v = tl.reshape(v, (64, 128))
    offs_n = blk * 64 + tl.arange(0, 64)
    offs_k = g * GROUP + tl.arange(0, 128)
    s = tl.load(s_ptr + g * stride_sk + offs_n)  # fp16 [64]
    q8 = libdevice.rint(((v - 8).to(tl.float16) * s[:, None]).to(tl.float32))
    q8 = tl.minimum(tl.maximum(q8, -128.0), 127.0)
    tl.store(out_ptr + offs_n[:, None] * K + offs_k[None, :], q8.to(tl.int8))

def unpack_fast(B, s_group, K, N, out=None):
    if out is None: out = torch.empty((N, K), dtype=torch.int8, device=B.device)
    _qqq_unpack_fast_kernel[(K // 128, N // 64)](B, s_group, out, N, K, B.stride(0), s_group.stride(0), GROUP=128, num_warps=4)
    return out

dev = "cuda"; torch.manual_seed(0)
def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters
tot = 0.0
for name, (N, K) in {"qkv": (16384, 5376), "o": (5376, 8192), "gate_up": (43008, 5376), "down": (5376, 21504)}.items():
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s_g = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    B, s_ch_p, s_grp_p, q8_ref, s_ch, s_grp = qqq_pack_from_int4(q4, s_g)
    ref = qqq_unpack_to_int8_triton(B, s_grp, K, N)
    try:
        got = unpack_fast(B, s_grp, K, N)
        ok = torch.equal(got, ref); mism = (got != ref).sum().item()
        t = timeit(lambda: unpack_fast(B, s_grp, K, N)); tot += t * 60
        print(f"{name:8s} exact={ok} mismatch={mism}  {t:.2f} ms/layer ({N*K*1.5/t/1e6:.0f} GB/s r+w)", flush=True)
    except Exception as ex:
        print(f"{name:8s} FAIL {type(ex).__name__}: {str(ex)[:300]}"); break
print(f"fast unpack total per forward: {tot:.0f} ms")
