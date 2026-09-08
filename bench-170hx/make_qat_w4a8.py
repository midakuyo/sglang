# QAT w4a16-ct 복사본에 토큰별 동적 int8 활성 설정을 추가한 디렉터리 생성(가중치 파일은 심볼릭 링크)
import json, os, shutil
SRC = "/home/midakuyo/data/models/gemma4-qat-w4a16"; DST = "/home/midakuyo/data/models/gemma4-qat-w4a8"
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f.startswith(".") or f == "config.json": continue
    d = os.path.join(DST, f)
    if not os.path.lexists(d): os.symlink(os.path.join("..", "gemma4-qat-w4a16", f), d)
c = json.load(open(os.path.join(SRC, "config.json")))
q = c.get("quantization_config") or c["text_config"]["quantization_config"]
for gname, g in q["config_groups"].items():
    g["input_activations"] = {"actorder": None, "block_structure": None, "dynamic": True, "group_size": None, "num_bits": 8,
                              "observer": None, "observer_kwargs": {}, "scale_dtype": None, "strategy": "token",
                              "symmetric": True, "type": "int", "zp_dtype": None}
json.dump(c, open(os.path.join(DST, "config.json"), "w"), indent=2, ensure_ascii=False)
print("written", DST); print(json.dumps({k: v for k, v in q["config_groups"].items()}, ensure_ascii=False)[:600])
print(os.listdir(DST))
