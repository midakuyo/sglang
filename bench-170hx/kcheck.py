#!/usr/bin/env python3
"""Bit-equality check: stock extend_attention_fwd (loaded from /ea_stock.py) vs patched
(package path). Same inputs, sliding-window + custom-mask + page64 argument set."""
import importlib.util, torch
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd as patched
spec = importlib.util.spec_from_file_location("ea_stock", "/ea_stock.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); stock = m.extend_attention_fwd
dev, dt = "cuda", torch.bfloat16
PAGE, POOL = 64, 64 * 1024

def paged_indices(bs, plen, g):
    npages = POOL // PAGE; out = []
    for _ in range(bs):
        pages = torch.randperm(npages, generator=g)[: (plen + PAGE - 1) // PAGE]
        out.append((pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)[:plen])
    return torch.cat(out).to(dev, torch.int64)

def run(fn, bs, H, KV, D, plen, full_len, qlen, win, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(dev, dt)
    kb, vb = r(POOL, KV, D), r(POOL, KV, D)
    kv_indices = paged_indices(bs, plen, g)
    kv_indptr = torch.arange(0, bs + 1, device=dev, dtype=torch.int32) * plen
    nq = bs * qlen
    q, ke, ve = r(nq, H, D), r(nq, KV, D), r(nq, KV, D)
    o = torch.zeros(nq, H, D, device=dev, dtype=dt)
    qo_indptr = torch.arange(0, bs + 1, device=dev, dtype=torch.int32) * qlen
    per = qlen * (full_len + qlen)
    cm = torch.ones(bs * per, device=dev, dtype=torch.bool)
    mi = torch.arange(0, bs + 1, device=dev, dtype=torch.int64) * per
    offs = torch.full((bs,), max(full_len - plen, 0), device=dev, dtype=torch.int32) if win > 0 else None
    fn(q, ke, ve, o, kb, vb, qo_indptr, kv_indptr, kv_indices, cm, True, mi, qlen, 1.0, 1.0,
       sm_scale=1.0 / D ** 0.5, sliding_window_size=win, window_kv_offsets=offs, page_size=PAGE)
    torch.cuda.synchronize(); return o

cases = [
    ("SWA verify   bs8 q3  pre1023/2100 W1023", 8, 32, 16, 256, 1023, 2100, 3, 1023),
    ("SWA prefill  bs2 q64 pre2100 W1023 (tiles skipped)", 2, 32, 16, 256, 2100, 2100, 64, 1023),
    ("SWA prefill  bs2 q200 pre300 W1023 (partial)", 2, 32, 16, 256, 300, 300, 200, 1023),
    ("SWA short    bs4 q3  pre150/150 W1023", 4, 32, 16, 256, 150, 150, 3, 1023),
    ("FULL verify  bs4 q3  pre2100 no-window", 4, 32, 16, 512, 2100, 2100, 3, -1),
]
ok_all = True
for name, *a in cases:
    o1 = run(stock, *a); o2 = run(patched, *a)
    eq = torch.equal(o1, o2); mx = (o1.float() - o2.float()).abs().max().item()
    nan = torch.isnan(o2).any().item()
    ok_all &= eq and not nan
    print(f"{name:<52} bit-equal={eq}  max|diff|={mx:.3g}  nan={nan}")
print("ALL BIT-EQUAL" if ok_all else "MISMATCH")
