# 프리필 어텐션 마이크로벤치: extend_attention_fwd, 프리픽스 0 + q=N 토큰(콜드 프리필), SWA(W=1023) vs full, us/층
import sys, torch, importlib.util
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd as patched
spec = importlib.util.spec_from_file_location("ea_stock", "/ea_stock.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); stock = m.extend_attention_fwd
dev, dt = "cuda", torch.bfloat16
H, KV, D = 32, 16, 256
def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / iters * 1000
print(" bs  qlen   win |  stock us  patched us | maxdiff")
for bs, qlen in ((1, 2107), (1, 4096), (2, 2107), (1, 512)):
    for win in (1023, -1):
        torch.manual_seed(0)
        nq = bs * qlen
        q = torch.randn(nq, H, D, dtype=dt, device=dev); ke = torch.randn(nq, KV, D, dtype=dt, device=dev); ve = torch.randn_like(ke)
        kb = torch.randn(64, KV, D, dtype=dt, device=dev); vb = torch.randn_like(kb)
        o1 = torch.zeros(nq, H, D, dtype=dt, device=dev); o2 = torch.zeros_like(o1)
        qo = torch.arange(0, bs + 1, device=dev, dtype=torch.int32) * qlen
        kvp = torch.zeros(bs + 1, device=dev, dtype=torch.int32); kvi = torch.zeros(0, device=dev, dtype=torch.int64)
        f1 = lambda: stock(q, ke, ve, o1, kb, vb, qo, kvp, kvi, None, True, None, qlen, 1.0, 1.0, sm_scale=1 / D ** 0.5, sliding_window_size=win, page_size=64)
        f2 = lambda: patched(q, ke, ve, o2, kb, vb, qo, kvp, kvi, None, True, None, qlen, 1.0, 1.0, sm_scale=1 / D ** 0.5, sliding_window_size=win, page_size=64)
        t1 = timeit(f1); t2 = timeit(f2)
        md = (o1.float() - o2.float()).abs().max().item(); nan = torch.isnan(o2).any().item()
        print(f"{bs:3d} {qlen:5d} {win:5d} | {t1:9.0f} {t2:11.0f} | {md:.2e} nan={nan}", flush=True)
