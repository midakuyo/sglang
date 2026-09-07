#!/bin/bash
# 서버 A/B 한 세트: 솔로 tg ×3 (timer-probe로 verify/draft GPU ms 분리) → multi S=8 R=3 load-bench (동일)
cd ~
echo "##### SOLO"; python3 ~/timer-probe.py python3 ~/solo-tg.py http://localhost:8001/v1 3
echo "##### MULTI S=8 R=3"; python3 ~/timer-probe.py python3 ~/miru-load-bench.py http://localhost:8001/v1 x multi 8 3 ${OFF:-0}
