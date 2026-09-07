# split-KV verify 튜닝 스윕 (Ampere): n_splits × (BLOCK_N, num_warps) — choose_n_splits/block_config 몽키패치
import torch, itertools
import sglang.kernels.ops.attention.verify_splitkv as vs
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
shapes = [(1024, 1024), (2100, -1)]
BS = [1, 4, 8, 16]
inputs = {(P, bs): build(bs, P) for P, _ in shapes for bs in BS}
ref = {}
for P, win in shapes:
    for bs in BS:
        q, k, v, qo, kvp, kvi = inputs[(P, bs)]
        o = torch.empty(bs * L, HQ, D, dtype=dt, device=dev)
        vs.verify_splitkv_fwd(q, k, v, o, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win, max_bs=64)
        ref[(P, bs)] = o.clone()
vs._VK_CACHE.clear()
configs = [(n, bn, nw) for n in (2, 4, 8, 16, 32) for (bn, nw) in ((32, 4), (64, 4), (32, 8), (64, 8), (128, 4), (128, 8))]
hdr = "nsplit blockN warps | " + " ".join(f"P{P}b{bs:<2}" for P, _ in shapes for bs in BS)
print(hdr)
best = {}
for n, bn, nw in configs:
    vs.choose_n_splits = lambda s, n=n: n
    vs.block_config = lambda d, bn=bn, nw=nw: (bn, nw)
    vs._VK_CACHE.clear()
    row = []
    for P, win in shapes:
        for bs in BS:
            q, k, v, qo, kvp, kvi = inputs[(P, bs)]
            o = torch.empty(bs * L, HQ, D, dtype=dt, device=dev)
            f = lambda: vs.verify_splitkv_fwd(q, k, v, o, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win, max_bs=64)
            try:
                t = timeit(f)
                md = (o.float() - ref[(P, bs)].float()).abs().max().item()
                if md > 2e-2: t = float("nan")
            except Exception as ex:
                t = float("nan")
            row.append(t)
            if t == t and (P, bs) not in best or (t == t and t < best[(P, bs)][0]): best[(P, bs)] = (t, (n, bn, nw))
    print(f"{n:6d} {bn:6d} {nw:5d} | " + " ".join(f"{t:7.0f}" for t in row), flush=True)
print("BEST:", {k: (round(v[0]), v[1]) for k, v in sorted(best.items())})
