#!/usr/bin/env python3
# steady-bench 변형: 요청별 (k, 토큰, tg, 생성 텍스트 머리/꼬리) 출력. 사용: steady-detail.py <base_url> <K> [max_tokens]
import json, sys, time, threading, urllib.request, os
URL = sys.argv[1].rstrip("/") + "/chat/completions"
K = int(sys.argv[2]); MT = int(sys.argv[3]) if len(sys.argv) > 3 else 300
base = json.load(open("/home/midakuyo/miru.json"))
res = [None] * K
def one(k):
    msgs = base["messages"][:-1] + [{"role": "user", "content": base["messages"][-1]["content"] + f" ({k})"}]
    b = {"model": "x", "messages": msgs, "max_tokens": MT, "temperature": 0, "ignore_eos": True,
         "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(b, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; ct = 0; txt = []
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:]); u = d.get("usage")
        if u and u.get("completion_tokens"): ct = u["completion_tokens"]
        ch = d.get("choices")
        if ch and ch[0].get("delta", {}).get("content"):
            if first is None: first = time.time()
            last = time.time(); txt.append(ch[0]["delta"]["content"])
    res[k] = (first - t0, last - t0, ct, ct / (last - first), "".join(txt))
ths = [threading.Thread(target=one, args=(k,)) for k in range(K)]
for t in ths: t.start()
for t in ths: t.join()
for k, r in enumerate(res):
    s = r[4].replace("\n", " ")
    print(f"k={k} ct={r[2]} tg={r[3]:6.1f} e2e={r[1]:.2f}s | {s[:70]} ... {s[-60:]}")
