// Standalone compile check for the shared Marlin headers ported from vLLM main
// (S2 of the W4A8 Marlin port). Nothing else includes marlin_mma.h until the
// dense template is replaced (S3), so this instantiates every path we rely on.
// Build: bash bench-170hx/marlin_mma_check.sh (inside the sglang container).
#include <sgl_kernel/scalar_type.hpp>
#include <sgl_kernel/utils.cuh>

#include "gemm/marlin/dequant.h"
#include "gemm/marlin/marlin.cuh"
#include "gemm/marlin/marlin_dtypes.cuh"
#include "gemm/marlin/marlin_mma.h"

namespace sglang {
namespace device::marlin {

using S8 = MarlinScalarType<host::kS8.id()>;
using F16 = MarlinScalarType<host::kFloat16.id()>;
using BF16 = MarlinScalarType<host::kBFloat16.id()>;
using F8 = MarlinScalarType<host::kFE4M3fn.id()>;

static_assert(std::is_same<S8::scalar_t, int8_t>::value);
static_assert(std::is_same<S8::FragA, Vec<int32_t, 4>>::value);
static_assert(std::is_same<S8::FragB, Vec<int32_t, 2>>::value);
static_assert(std::is_same<S8::FragC, Vec<float, 4>>::value);
static_assert(std::is_same<MarlinScalarType2<int8_t>::scalar_t, int8_t>::value);
// legacy alias used by marlin_moe/* and the old dense template
static_assert(std::is_same<ScalarType<fp16_t>::FragS, Vec<fp16x2_t, 1>>::value);
static_assert(std::is_same<ScalarType<bf16_t>::scalar_t2, bf16x2_t>::value);

__global__ void s2_check_kernel(int q, int zp, float* out, int8_t* sm) {
  // int8 activation path: A/B are int32 fragments, C is int32 (in float slots)
  S8::FragA a8;
  S8::FragB b8, b8b;
  S8::FragC c8{};
  dequant<int32_t, host::kU4B8.id(), true>(q, reinterpret_cast<int32_t*>(&b8));
  sub_zp_and_dequant<int32_t, host::kU4.id(), true>(q, reinterpret_cast<int32_t*>(&b8b), zp);
  a8[0] = q; a8[1] = q + 1; a8[2] = q + 2; a8[3] = q + 3;
  mma<host::kS8.id(), false, 32>(a8, b8, c8);
  mma<host::kS8.id(), false, 16>(a8, b8, c8, 1);
  mma_trans<host::kS8.id(), false, 32>(a8, b8, b8b, c8);
  mma_trans<host::kS8.id(), false, 16>(a8, b8, b8b, c8);

  // fp16 / bf16 paths (existing users)
  F16::FragA a16;
  F16::FragB b16;
  F16::FragC c16{};
  dequant<fp16x2_t, host::kU4B8.id(), false>(q, reinterpret_cast<fp16x2_t*>(&b16));
  dequant<fp16x2_t, host::kU8B128.id(), false>(q, reinterpret_cast<fp16x2_t*>(&b16));
  mma<host::kFloat16.id(), false>(a16, b16, c16);
  mma_trans<host::kFloat16.id(), false>(a16, b16, b16, c16);
  BF16::FragA abf;
  BF16::FragB bbf;
  BF16::FragC cbf{};
  dequant<bf16x2_t, host::kU4B8.id(), false>(q, reinterpret_cast<bf16x2_t*>(&bbf));
  dequant<bf16x2_t, host::kFE2M1f.id(), true>(q, reinterpret_cast<bf16x2_t*>(&bbf));
  mma<host::kBFloat16.id(), false>(abf, bbf, cbf);
  mma_trans<host::kBFloat16.id(), false>(abf, bbf, bbf, cbf);

  // fp8 path
  F8::FragA af8;
  F8::FragB bf8;
  F8::FragC cf8{};
  dequant<fp8x4_e4m3_t, host::kU4B8.id(), true>(q, reinterpret_cast<fp8x4_e4m3_t*>(&bf8));
  mma<host::kFE4M3fn.id(), false, 32>(af8, bf8, cf8);

  // scale dequant: legacy 1-param (old dense template) and 2-param (MoE / new)
  fp16x2_t s16[4];
  bf16x2_t sbf[4];
  dequant_fp8_scales<fp16x2_t>(q, s16);
  dequant_fp8_scales<bf16x2_t>(q, sbf);
  dequant_fp8_scales<fp16x2_t, host::kFE4M3fn.id()>(q, s16);
  dequant_fp8_scales<bf16x2_t, host::kFE4M3fn.id()>(q, sbf);
  dequant_fp8_scales<bf16x2_t, host::kFE8M0fnu.id()>(q, sbf);

  // cp.async helpers
  __shared__ int4 smem[4];
  cp_async1_ca_pred(&smem[0], sm, q > 0);
  cp_async2_ca_pred(&smem[1], sm, q > 0);
  cp_async4_ca_pred(&smem[2], sm, q > 0);
  cp_async4_pred(&smem[3], sm, q > 0);
  cp_async_fence();
  cp_async_wait<0>();

  out[threadIdx.x] = c8[0] + c16[0] + cbf[0] + cf8[0] + ScalarType<fp16_t>::num2float(__low2half(s16[0])) +
                     __bfloat162float(__low2bfloat16(sbf[0]));
}

}  // namespace device::marlin
}  // namespace sglang

void s2_check_launch(int q, int zp, float* out, int8_t* sm) {
  sglang::device::marlin::s2_check_kernel<<<1, 32>>>(q, zp, out, sm);
}
