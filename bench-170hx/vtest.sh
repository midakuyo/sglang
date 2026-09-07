#!/bin/bash
# 포크 소스를 마운트한 컨테이너에서 split-KV verify 단위 테스트 + 마이크로벤치 (GPU1)
docker run --rm --device nvidia.com/gpu=${GPU:-1} \
  -v ~/sglang/python/sglang:/src/python/sglang:ro -v ~/sglang/test:/src/test:ro \
  -v ~/vbench.py:/vbench.py:ro sglang-overlay:50c1bf0 \
  bash -c "cd /src/test/registered/attention && echo == unit test && python3 test_verify_splitkv.py 2>&1 | tail -${TESTTAIL:-8}; echo == vbench; python3 /vbench.py 2>&1 | tail -25"
