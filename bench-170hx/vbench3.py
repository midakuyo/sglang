# 마이크로벤치 v3: CUDA graph 캡처(런치 오버헤드 제거) + 층별 별도 KV 버퍼 순환(L2-cold) — 서버 조건 재현
import torch
import sglang.kernels.ops.attention.verify_splitkv as vs
dev, dt = "cuda", torch.bfloat16
HQ, HKV, D, L = 32, 16, 256, 3
NL = 24  # 순환 층 수 (bs16 P2100: 24×550MB=13GB는 과하므로 bs별로 조정)
torch.manual_seed(0)
sm = 1 / D ** 0.5
def timeit_graph(fn, reps=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000
print("stages | " + " ".join(f"{'P'+str(P)+'b'+str(bs):>10}" for P, bs in ((1024, 1), (1024, 8), (1024, 16), (2100, 8), (2100, 16))) + "   (us/층, graph+L2-cold)")
for ns in (1, 3):
    vs.STAGE1_NUM_STAGES = ns; vs._VK_CACHE.clear()
    row = []
    for P, bs in ((1024, 1), (1024, 8), (1024, 16), (2100, 8), (2100, 16)):
        win = 1024 if P == 1024 else -1
        nl = min(NL, int(6e9 // (bs * P * HKV * D * 2 * 2)))  # ≤6GB
        kvp = torch.arange(0, bs * P + 1, P, dtype=torch.int32, device=dev)
        kvi = torch.arange(bs * P, dtype=torch.int64, device=dev)
        q = torch.randn(bs * L, HQ, D, dtype=dt, device=dev); k = torch.randn(bs * L, HKV, D, dtype=dt, device=dev); v = torch.randn_like(k)
        qo = torch.arange(0, bs * L + 1, L, dtype=torch.int32, device=dev)
        o = torch.empty(bs * L, HQ, D, dtype=dt, device=dev)
        pools = [(torch.randn(bs * P, HKV, D, dtype=dt, device=dev), torch.randn(bs * P, HKV, D, dtype=dt, device=dev)) for _ in range(nl)]
        def fn():
            for kb, vb in pools:
                vs.verify_splitkv_fwd(q, k, v, o, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win, max_bs=64)
        t = timeit_graph(fn) / nl
        row.append(t)
        del pools; torch.cuda.empty_cache()
    print(f"{ns:6d} | " + " ".join(f"{t:10.0f}" for t in row), flush=True)
