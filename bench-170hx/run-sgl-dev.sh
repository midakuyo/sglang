#!/bin/bash
# SGLang 포크 개발 루프: ~/sglang(perf-dev) 파이썬 패키지를 통째로 오버레이 이미지에 마운트.
# 순수 Python/Triton 변경은 이미지 재빌드 없이 이 스크립트 재실행만으로 반영.
# GPU1(블로워) 고정. device_timer + /metrics 로 verify/draft GPU 시간 분리 계측.
IMG=sglang-overlay:50c1bf0
MODEL=${MODEL:-/models/gemma4-lokesh-int8}
docker rm -f sgl-dev sgl-dev-prev 2>/dev/null
docker run -d --device nvidia.com/gpu=1 -p 8001:8000 --name sgl-dev \
  -v /home/midakuyo/data/models:/models \
  -v /home/midakuyo/sglang/python/sglang:/src/python/sglang:ro \
  -v /home/midakuyo/triton-cache:/root/.cache/sglang \
  -e SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1 \
  -e SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 -e SGLANG_DEVICE_TIMER_LAYER_GROUPS=${LG:-0} -e SGLANG_DEVICE_TIMER_LAYER_GROUPS_LOG=${LGLOG:-0} -e SGLANG_PROBE_ATTN_ABLATE=${ABL:-} \
  "$IMG" \
  python3 -m sglang.launch_server --model-path "$MODEL" \
  --host 0.0.0.0 --port 8000 --mem-fraction-static 0.8 --trust-remote-code \
  --page-size 64 --enable-metrics \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /models/gemma4-31b-assistant \
  --speculative-num-steps 2 --speculative-num-draft-tokens 3 --speculative-eagle-topk 1 "$@"
echo "sgl-dev (fork ~/sglang $(git -C ~/sglang rev-parse --short HEAD), GPU1) :8001"
