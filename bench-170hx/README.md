# bench-170hx — CMP 170HX(GA100, 64GB 언락) 실험 하네스

perf-dev 브랜치의 verify 커널 작업(2026-09-07)에 쓴 스크립트 모음. 경로는 negroni(`/home/midakuyo`) 기준 하드코딩.
컨테이너 = `sglang-overlay:50c1bf0`(v0.5.19-cu129 컴파일 스택) + `python/sglang` 디렉터리 ro 마운트.

| 파일 | 용도 |
|---|---|
| `run-sgl-dev.sh` | 포크 소스 마운트 서버 기동 (GPU1, lokesh INT8, MTP 2/3, page64, device_timer, :8001) |
| `vtest.sh` | 컨테이너 안 `test_verify_splitkv.py` + `vbench.py` |
| `vbench.py` | extend_attention_fwd vs verify_splitkv_fwd, Gemma4 형상, bs×문맥×윈도 |
| `vsweep.py` / `vsweep2.py` | n_splits×BLOCK_N×warps / num_stages 스윕 (몽키패치) |
| `vcontig.py` | 페이지 연속 vs 산포 kv_indices 대조 |
| `gemm-scan.py` / `gemm-scan2.py` | int8_scaled_mm M-스캔 (eager / CUDA graph 캡처, cuBLAS 대조) |
| `steady-bench.py` | ignore_eos로 bs 고정한 정상상태 tg/반복 벽시계 |
| `solo-tg.py` | 단일 스트리밍 220tok ×N |
| `ab-run.sh` | 솔로 + `miru-load-bench.py` multi 8 3 을 `timer-probe.py`로 감싼 한 세트 |
| `timer-probe.py` | `/metrics` 전후 diff (`forward_execution_seconds_total{category}` ÷ `cuda_graph_passes_total` = 패스당 GPU ms) |

## 09-07 결과 요약 (lokesh INT8, MTP 2/3, 175W)
verify ms/패스 bs1/4/8/16: 스톡 52/—/97(bs4)/— → 커밋 1~5 29.3/33.7/38.7/47.9.
per-req tg bs1/8/16: 36/15/— → 72/54/40. multi S=8 R=3 tg 15.3 → 48 (vLLM quark W8A8+MTP3: 48). 250W: verify −5~9%.
교훈: 마이크로벤치는 후보 선별용(num_stages −20% → 서버 −1.5%), 판정은 서버 A/B. GQA(바이트 절감)는 그대로 전이.
