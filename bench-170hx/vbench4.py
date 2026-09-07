# v4: v3(그래프+L2-cold) + 지속 부하 — 전력 캡이 걸린 상태에서 계측 (10초 예열 후 측정, 클럭 샘플 포함)
import torch, time, subprocess
import sglang.kernels.ops.attention.verify_splitkv as vs
dev, dt = "cuda", torch.bfloat16
HQ, HKV, D, L = 32, 16, 256, 3
torch.manual_seed(0); sm = 1 / D ** 0.5
def build_graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
def clocks():
    try: return subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    except Exception: return "?"
print("stages sustain |  P1024b8 P1024b16  P2100b8 P2100b16   (us/층)")
for ns in (3, 1):
    vs.STAGE1_NUM_STAGES = ns; vs._VK_CACHE.clear()
    for sustain in (0.0, 10.0):
        row = []
        for P, bs in ((1024, 8), (1024, 16), (2100, 8), (2100, 16)):
            win = 1024 if P == 1024 else -1
            nl = min(24, int(6e9 // (bs * P * HKV * D * 2 * 2)))
            kvp = torch.arange(0, bs * P + 1, P, dtype=torch.int32, device=dev); kvi = torch.arange(bs * P, dtype=torch.int64, device=dev)
            q = torch.randn(bs * L, HQ, D, dtype=dt, device=dev); k = torch.randn(bs * L, HKV, D, dtype=dt, device=dev); v = torch.randn_like(k)
            qo = torch.arange(0, bs * L + 1, L, dtype=torch.int32, device=dev); o = torch.empty(bs * L, HQ, D, dtype=dt, device=dev)
            pools = [(torch.randn(bs * P, HKV, D, dtype=dt, device=dev), torch.randn(bs * P, HKV, D, dtype=dt, device=dev)) for _ in range(nl)]
            def fn():
                for kb, vb in pools:
                    vs.verify_splitkv_fwd(q, k, v, o, kb, vb, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, sm_scale=sm, sliding_window_size=win, max_bs=64)
            g = build_graph(fn); g.replay(); torch.cuda.synchronize()
            t0 = time.time()
            while time.time() - t0 < sustain: g.replay()
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(20): g.replay()
            b.record(); torch.cuda.synchronize()
            row.append(a.elapsed_time(b) / 20 * 1000 / nl)
            cl = clocks() if sustain else ""
            del pools, g; torch.cuda.empty_cache()
        print(f"{ns:6d} {sustain:7.0f} | " + " ".join(f"{t:8.0f}" for t in row) + (f"   [{cl}]" if sustain else ""), flush=True)
