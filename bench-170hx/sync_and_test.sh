#!/bin/bash
# Usage: sync_and_test.sh "<commit message or ->" "<command to run inside the sglang container>" 
# 1) rsync the edited fork subtrees from /tmp/sglang-fork (local) to negroni ~/sglang (branch w4a8-marlin)
# 2) run the command inside sglang-overlay:50c1bf0 on GPU0 with the fork mounted at /src (JIT cache persisted)
# 3) if a commit message is given (not "-"), git commit the synced paths on negroni (does not push)
set -u
MSG="${1:--}"; CMD="${2:-echo no-cmd}"
PATHS="python/sglang/kernels python/sglang/srt/layers/quantization python/sglang/test test/registered/kernels/ops/quantization test/registered/quant bench-170hx"
cd /tmp/sglang-fork || exit 1
EXIST=""; for p in $PATHS; do [ -e "$p" ] && EXIST="$EXIST $p"; done
tar cf - $EXIST | ssh -o BatchMode=yes midakuyo@192.168.1.15 'cd ~/sglang && tar xf - && git status --short | wc -l'
ssh -o BatchMode=yes midakuyo@192.168.1.15 "set -o pipefail; cd ~/sglang && docker run --rm --device nvidia.com/gpu=${GPU:-1} -v ~/sglang/python/sglang:/src/python/sglang:ro -v ~/sglang/test:/src/test:ro -v ~/sglang/bench-170hx:/bench:ro -v ~/triton-cache:/root/.cache/sglang -e SGLANG_JIT_VERBOSE=1 sglang-overlay:50c1bf0 bash -c 'set -o pipefail; cd /src && $CMD' 2>&1 | grep -viE 'nvidia|license|container image|CUDA Version|^====|^\$' | tail -80"
RC=${PIPESTATUS[0]}
if [ "$MSG" != "-" ] && [ "$RC" = "0" ]; then
  ssh -o BatchMode=yes midakuyo@192.168.1.15 "cd ~/sglang && git add -A $EXIST && git -c user.name=midakuyo -c user.email=midakuyo@gmail.com commit -q -m \"$MSG\" && git log --oneline -1"
fi
exit $RC
