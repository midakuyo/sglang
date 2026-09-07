# verify 커널 마이크로벤치: extend_attention_fwd(현행 verify 경로) vs verify_splitkv_fwd
# Gemma4 형상: h_q 32 / h_kv 16 / d 256 / l_ext 3, page64 산포 kv_indices, bf16, cuda 이벤트 계측
import sys, torch
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
dev, dt = "cuda", torch.bfloat16
HQ, HKV, D, L = 32, 16, 256, 3
POOL = 262144
torch.manual_seed(0)
kb = torch.randn(POOL, HKV, D, dtype=dt, device=dev); vb = torch.randn_like(kb)
def build(bs, P):
    kvp = torch.arange(0, bs * P + 1, P, dtype=torch.int32, device=dev)
    pages = torch.randperm(POOL // 64, device=dev)[: bs * (P // 64 + 1)]
    kvi = (pages[:, None] * 64 + torch.arange(64, device=dev)[None, :]).reshape(-1)[: bs * P].to(torch.int64)
    q = torch.randn(bs * L, HQ, D, dtype=dt, device=dev)
    k = torch.randn(bs * L, HKV, D, dtype=dt, device=dev); v = torch.randn_like(k)
    qo = torch.arange(0, bs * L + 1, L, dtype=torch.int32, device=dev)
    return q, k, v, qo, kvp, kvi
def timeit(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000
sm = 1 / D ** 0.5
shapes = [(1024, 1024), (1024, -1), (2100, -1), (4096, -1)]
print(" bs     P   win | extend us splitkv us | ran   maxdiff")
for P, win in shapes:
    for bs in [1, 4, 8, 16]:
        q, k, v, qo, kvp, kvi = build(bs, P)
        o1 = torch.empty(bs * L, HQ, D, dtype=dt, device=dev); o2 = torch.empty_like(o1)
        f1 = lambda: extend_attention_fwd(q, k, v, o1, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win)
        f2 = lambda: verify_splitkv_fwd(q, k, v, o2, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win, max_bs=64)
        t1 = timeit(f1); ran = f2()
        t2 = timeit(f2) if ran else float("nan")
        md = (o1.float() - o2.float()).abs().max().item() if ran else float("nan")
        print(f"{bs:3d} {P:5d} {win:5d} | {t1:9.0f} {t2:10.0f} | {str(ran):5} {md:.2e}", flush=True)
