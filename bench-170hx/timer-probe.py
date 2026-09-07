#!/usr/bin/env python3
# /metrics 스냅숏 diff — 워크로드 전후 forward_execution_seconds_total{category} 등 카운터 변화량
import sys, subprocess, time, urllib.request, re
def snap():
    txt = urllib.request.urlopen("http://localhost:8001/metrics", timeout=10).read().decode()
    d = {}
    for ln in txt.splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        k, _, v = ln.rpartition(" ")
        try: d[k] = float(v)
        except: pass
    return d
a = snap(); t0 = time.time()
out = subprocess.run(sys.argv[1:], capture_output=True, text=True)
wall = time.time() - t0; b = snap()
print(out.stdout.strip()[-1500:]); print(out.stderr.strip()[-400:])
print("== wall %.1fs" % wall)
keys = sorted(set(a) | set(b))
for k in keys:
    if "_bucket" in k or k.endswith("_created"): continue
    dv = b.get(k, 0) - a.get(k, 0)
    if abs(dv) < 1e-9: continue
    if re.search(r"forward_execution|spec_accept|generation_tokens|prompt_tokens|num_requests_total|decode_|verify|draft|step", k):
        print("%-110s %+.3f" % (k[:110], dv))
