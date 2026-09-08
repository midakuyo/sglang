#!/bin/bash
# Compile-only check of the shared Marlin headers (see marlin_mma_check.cu).
# Run inside the sglang container from /src. Uses exactly the include paths and
# nvcc flags the JIT loader (ninja.py) uses, so the check matches JIT builds.
set -e
INC=$(python3 -c "from sglang.kernels.jit.utils.compile.toolchain import base_include_paths; from sglang.kernels.jit.utils.compile.paths import DEFAULT_INCLUDE; print(' '.join('-I'+p for p in list(base_include_paths())+list(DEFAULT_INCLUDE)))")
FLAGS=$(python3 -c "from sglang.kernels.jit.utils.compile import toolchain as t; from sglang.kernels.jit.utils.arch import get_default_target_flags; print(' '.join(t.base_cuda_flags()+t.target_flags()+get_default_target_flags()))")
NVCC=$(python3 -c "from sglang.kernels.jit.utils.compile import toolchain as t; print(t.device_compiler_path())")
SRC=$(dirname "$(readlink -f "$0")")/marlin_mma_check.cu
CSRC=$(python3 -c "from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH; print(KERNEL_PATH/'csrc')")
echo "$NVCC $FLAGS"
"$NVCC" $FLAGS $INC -I"$CSRC" -c "$SRC" -o /tmp/marlin_mma_check.o
cuobjdump -sass /tmp/marlin_mma_check.o | grep -oE "IMMA\.[0-9x]+(\.[A-Z0-9]+)*|HMMA\.[0-9x]+(\.[A-Z0-9]+)*|LDGSTS(\.[A-Z0-9]+)*" | sort | uniq -c
echo "marlin_mma_check: OK"
