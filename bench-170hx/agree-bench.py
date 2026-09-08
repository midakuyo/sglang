#!/usr/bin/env python3
# 두 서버의 출력 일치도 측정 (컨테이너 안에서 실행, --network host).
#   gen  : 프롬프트 묶음을 greedy 96토큰 생성 → 파일 저장 (prompt ids + output ids + logprobs)
#   score: 저장된 (prompt+continuation)을 teacher-forced 채점 → 토큰별 logprob 차·top-1 일치율
# 사용: agree-bench.py gen  <base_url> <model_dir> <out.json>
#       agree-bench.py score <base_url> <model_dir> <in.json>
import json, sys, urllib.request, math
mode, base, mdir, path = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], sys.argv[4]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(mdir)
PROMPTS = [
    "한국의 사계절을 각각 한 문장으로 설명해줘.", "파이썬으로 피보나치 수열을 구하는 함수를 짧게 작성해줘.",
    "친구가 시험에 떨어져서 우울해해. 위로하는 말을 세 줄로 써줘.", "커피와 녹차의 카페인 차이를 간단히 알려줘.",
    "디스코드 봇이 음성 채널에서 할 수 있는 재미있는 기능 다섯 가지를 제안해줘.", "물의 끓는점이 고도에 따라 달라지는 이유는?",
    "Explain the difference between TCP and UDP in three sentences.", "Write a haiku about a rainy night in Seoul.",
    "GPU 텐서코어가 int8 연산에서 빠른 이유를 초등학생에게 설명하듯 말해줘.", "오늘 저녁 메뉴로 떡볶이와 파스타 중 하나를 골라주고 이유를 말해줘.",
    "List three common mistakes when learning Korean and how to fix them.", "양자화(quantization)가 LLM 품질에 미치는 영향을 두 문단으로 설명해줘.",
    "고양이가 상자를 좋아하는 이유를 과학적으로 설명해줘.", "Summarize the plot of Romeo and Juliet in five sentences.",
    "1부터 100까지 3의 배수의 합은? 풀이 과정을 보여줘.", "가상 아바타 캐릭터 '미루'가 방에서 혼자 노는 장면을 짧게 묘사해줘.",
]
try:
    base_msgs = json.load(open("/home/midakuyo/miru.json"))["messages"]
    PROMPTS_MSGS = [[{"role": "user", "content": p}] for p in PROMPTS] + [base_msgs]
except Exception:
    PROMPTS_MSGS = [[{"role": "user", "content": p}] for p in PROMPTS]

def post(body):
    req = urllib.request.Request(base + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

if mode == "gen":
    items = []
    for msgs in PROMPTS_MSGS:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list): ids = ids["input_ids"]
        r = post({"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 96}, "return_logprob": True})
        out = [(lp, tid) for lp, tid, *_ in r["meta_info"]["output_token_logprobs"]]
        items.append({"prompt_ids": ids, "out_ids": [t for _, t in out], "out_lp": [lp for lp, _ in out], "text": r["text"]})
        print(f"gen {len(ids):5d}+{len(out):3d} | {r['text'][:60]!r}", flush=True)
    json.dump(items, open(path, "w")); print("saved", path)
else:
    items = json.load(open(path)); tot = 0; agree = 0; dsum = 0.0; dabs = 0.0; first_agree = 0; n_items = 0
    for it in items:
        P = it["prompt_ids"]; C = it["out_ids"]; ref_lp = it["out_lp"]
        r = post({"input_ids": P + C, "sampling_params": {"temperature": 0, "max_new_tokens": 0}, "return_logprob": True,
                  "logprob_start_len": len(P), "top_logprobs_num": 1})
        mi = r["meta_info"]; inp = mi["input_token_logprobs"]; top = mi["input_top_logprobs"]
        # input_token_logprobs[i] scores token at position logprob_start_len+i given prefix; first entry may be None
        got = [(e[0], e[1]) for e in inp if e is not None and e[0] is not None]
        tops = [t for t in top if t]
        n = min(len(got), len(C), len(tops))
        # align from the end (the first continuation token's score is at index 0 or 1 depending on server convention)
        got = got[-n:]; tops = tops[-n:]; C2 = C[-n:]; ref2 = ref_lp[-n:]
        for i in range(n):
            lp, tid = got[i]; assert tid == C2[i], (tid, C2[i])
            am = tops[i][0][1]
            tot += 1; agree += int(am == C2[i]); d = ref2[i] - lp; dsum += d; dabs += abs(d)
            if i == 0: first_agree += int(am == C2[i]); n_items += 1
    print(f"tokens {tot} | top-1 agreement {agree/tot:.3%} | first-token agreement {first_agree}/{n_items} | mean(ref_lp - lp) {dsum/tot:+.4f} nats | mean|dlp| {dabs/tot:.4f}")
