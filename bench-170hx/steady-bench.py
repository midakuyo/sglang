#!/usr/bin/env python3
# 정상상태 배치 벤치: 같은 접두사(miru.json) K발 동시, ignore_eos로 max_tokens까지 전부 생성 → bs=K 고정.
# 출력: 개별 TTFT/tg, 디코드 구간 벽시계, 반복당 벽시계(= 디코드 벽시계 / 검증 패스 수는 timer-probe 메트릭과 결합)
# 사용: steady-bench.py <base_url> <K> [max_tokens=160]
import json, sys, time, threading, urllib.request
URL = sys.argv[1].rstrip("/") + "/chat/completions"
K = int(sys.argv[2]); MT = int(sys.argv[3]) if len(sys.argv) > 3 else 160; SALT = sys.argv[4] if len(sys.argv) > 4 else ""
base = json.load(open("/home/midakuyo/miru.json"))
res = [None] * K
def one(k):
    msgs = base["messages"][:-1] + [{"role": "user", "content": base["messages"][-1]["content"] + f" ({k}{SALT})"}]
    b = {"model": "x", "messages": msgs, "max_tokens": MT, "temperature": 0, "ignore_eos": True,
         "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, json.dumps(b, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; ct = 0
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:]); u = d.get("usage")
        if u and u.get("completion_tokens"): ct = u["completion_tokens"]
        ch = d.get("choices")
        if ch and ch[0].get("delta", {}).get("content"):
            if first is None: first = time.time()
            last = time.time()
    res[k] = (first - t0, last - t0, ct, ct / (last - first))
ths = [threading.Thread(target=one, args=(k,)) for k in range(K)]
T0 = time.time()
for t in ths: t.start()
for t in ths: t.join()
T1 = time.time()
ttft = sorted(r[0] for r in res); e2e = max(r[1] for r in res); tot = sum(r[2] for r in res)
tg = sorted(r[3] for r in res)
dec = e2e - ttft[-1]
print(f"K={K} mt={MT}: TTFT p50 {ttft[len(ttft)//2]:.2f} max {ttft[-1]:.2f}s | e2e {e2e:.2f}s | decode wall {dec:.2f}s | "
      f"per-req tg p50 {tg[len(tg)//2]:.1f} min {tg[0]:.1f} | aggregate {tot/dec:.0f} tok/s | tokens {tot}")
