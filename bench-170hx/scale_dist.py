# W4A8 체크포인트 그룹 스케일 분포: 층 최대값 기준 12bit 정수화(s16=round(s/smax*4096))에서 뭉개지는 그룹 비율
import glob, json, torch, collections
from safetensors import safe_open
files = sorted(glob.glob("/models/gemma4-lokesh-w4a8/*.safetensors"))
stats = []
for f in files:
    with safe_open(f, "pt", device="cpu") as sf:
        for k in sf.keys():
            if not k.endswith("weight_scale"): continue
            s = sf.get_tensor(k).float().abs()
            if s.dim() != 2: continue
            smax = s.max(); s16 = (s / smax * 4096).round()
            colmax = s.max(dim=1, keepdim=True).values  # per-N max over groups
            r16 = (s / colmax * 4096).round()
            stats.append((k, s.shape, (smax / s.median()).item(),
                          (s16 < 64).float().mean().item(), (s16 < 16).float().mean().item(), (s16 == 0).float().mean().item(),
                          (r16 < 64).float().mean().item(), (colmax / s.min(dim=1).values.unsqueeze(1)).median().item()))
stats.sort(key=lambda x: -x[3])
print(f"{'tensor':70s} {'shape':>16s} smax/med  s16<64  s16<16  s16==0 | r16<64(col-norm) colmax/colmin(med)")
for k, shp, r, a, b, c, d, e in stats[:14]:
    print(f"{k[:70]:70s} {str(tuple(shp)):>16s} {r:8.1f} {a:7.3f} {b:7.3f} {c:7.4f} | {d:7.3f} {e:8.2f}")
import statistics
print("== all linears:", len(stats), "median frac s16<64 =", round(statistics.median(x[3] for x in stats), 4),
      "median frac s16<16 =", round(statistics.median(x[4] for x in stats), 4),
      "median smax/med =", round(statistics.median(x[2] for x in stats), 1),
      "| col-norm median r16<64 =", round(statistics.median(x[6] for x in stats), 4))
