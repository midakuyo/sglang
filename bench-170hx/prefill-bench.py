#!/usr/bin/env python3
# 서버 프리필 벤치: 프리픽스 캐시 미스(매번 고유 내용) 프롬프트 L토큰 → max_tokens=1, TTFT(벽시계) + /metrics extend GPU초 델타
# 사용: prefill-bench.py <base_url> [salt]   (solo 512/2k/4k/8k ×2, cold burst 8×2k)
import json, sys, time, threading, urllib.request, random, re
BASE = sys.argv[1].rstrip("/"); SALT = sys.argv[2] if len(sys.argv) > 2 else str(int(time.time()))
URL = BASE + "/chat/completions"; MET = BASE.replace("/v1", "") + "/metrics"
WORDS = "apple river mountain silver quiet ocean lantern forest window candle marble thunder velvet garden copper meadow harbor".split()
def metrics():
    ext = 0.0
    for line in urllib.request.urlopen(MET, timeout=30).read().decode().splitlines():
        if line.startswith("sglang:forward_execution_seconds_total") and 'category="extend"' in line:
            ext += float(line.rsplit(" ", 1)[1])
    return ext
def prompt(n_words, tag):
    rnd = random.Random(hash((SALT, tag)) & 0xffffffff)
    return f"[{SALT}-{tag}] " + " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(n_words)) + "\n\nSummarize the list above in one word."
def one(text, out):
    b = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
         "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; pt = 0
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:]); u = d.get("usage")
        if u and u.get("prompt_tokens"): pt = u["prompt_tokens"]
        ch = d.get("choices")
        if ch and first is None and (ch[0].get("delta", {}).get("content") or ch[0].get("finish_reason")): first = time.time()
    out.append((first - t0 if first else time.time() - t0, pt))
print(f"{'case':14s} {'ptok':>6s} {'TTFT s':>7s} {'extend GPU s':>12s} {'tok/s GPU':>10s} {'tok/s wall':>10s}")
for L in (512, 2048, 4096, 8192):
    for r in range(2):
        text = prompt(int(L / 2.3), f"solo{L}-{r}")
        e0 = metrics(); out = []; one(text, out); e1 = metrics()
        ttft, pt = out[0]; ext = e1 - e0
        print(f"solo L={L:<5d}   {pt:6d} {ttft:7.3f} {ext:12.3f} {pt/ext if ext else 0:10.0f} {pt/ttft:10.0f}", flush=True)
for r in range(2):
    texts = [prompt(int(2048 / 2.3), f"burst{r}-{k}") for k in range(8)]
    e0 = metrics(); outs = [[] for _ in texts]
    ths = [threading.Thread(target=one, args=(t, o)) for t, o in zip(texts, outs)]
    for t in ths: t.start()
    for t in ths: t.join()
    e1 = metrics(); tt = sorted(o[0][0] for o in outs); pt = sum(o[0][1] for o in outs); ext = e1 - e0
    print(f"burst 8x2k r{r}   {pt:6d} p50 {tt[4]:5.3f} max {tt[-1]:5.3f} {ext:8.3f} {pt/ext if ext else 0:10.0f} {pt/tt[-1]:10.0f}", flush=True)
