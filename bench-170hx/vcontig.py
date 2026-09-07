# 페이지 배치 가설 검증: 산포(randperm 페이지) vs 연속 페이지 kv_indices에서 num_stages 1 vs 3 비교
import torch
import sglang.kernels.ops.attention.verify_splitkv as vs
dev, dt = "cuda", torch.bfloat16
HQ, HKV, D, L = 32, 16, 256, 3
POOL = 262144
torch.manual_seed(0)
kb = torch.randn(POOL, HKV, D, dtype=dt, device=dev); vb = torch.randn_like(kb)
def build(bs, P, contig):
    kvp = torch.arange(0, bs * P + 1, P, dtype=torch.int32, device=dev)
    if contig:
        kvi = torch.arange(bs * P, dtype=torch.int64, device=dev) + 4096
    else:
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
print("layout  stages | P1024b8 P1024b16 P2100b8 P2100b16")
for contig in (False, True):
    for ns in (1, 3):
        vs.STAGE1_NUM_STAGES = ns; vs._VK_CACHE.clear()
        row = []
        for P, win in ((1024, 1024), (2100, -1)):
            for bs in (8, 16):
                q, k, v, qo, kvp, kvi = build(bs, P, contig)
                o = torch.empty(bs * L, HQ, D, dtype=dt, device=dev)
                row.append(timeit(lambda: vs.verify_splitkv_fwd(q, k, v, o, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win, max_bs=64)))
        print(f"{'contig' if contig else 'random':7s} {ns:6d} | " + " ".join(f"{t:7.0f}" for t in row), flush=True)
