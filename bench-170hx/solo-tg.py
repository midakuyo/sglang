#!/usr/bin/env python3
# 솔로 tg 프로브: miru.json 대화로 단일 스트리밍 요청 N회(기본 3), 220tok, 그리디. TTFT/tg/응답 앞부분 출력
import json, sys, time, urllib.request
URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1").rstrip("/") + "/chat/completions"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
base = json.load(open("/home/midakuyo/miru.json"))
for i in range(N):
    b = {"model": "x", "messages": base["messages"], "max_tokens": 220, "temperature": 0,
         "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(b, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; ct = pt = None; text = []
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:]); u = d.get("usage")
        if u: ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
        ch = d.get("choices")
        if ch and ch[0].get("delta", {}).get("content"):
            if first is None: first = time.time()
            last = time.time(); text.append(ch[0]["delta"]["content"])
    tg = ct / (last - first) if ct and last and last > first else float("nan")
    txt = "".join(text)[:100]
    print(f"run{i}: ttft {first - t0:.2f}s  tg {tg:.1f} tok/s  ct {ct} pt {pt}  | {txt!r}", flush=True)
