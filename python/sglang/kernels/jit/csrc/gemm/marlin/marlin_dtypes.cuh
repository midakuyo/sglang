#ifndef _data_types_cuh
#define _data_types_cuh
#include <sgl_kernel/scalar_type.hpp>
#include <sgl_kernel/utils.cuh>

#include "marlin.cuh"

namespace sglang {

namespace device::marlin {

template <long scalar_type_id>
class MarlinScalarType {};

template <>
class MarlinScalarType<host::kFloat16.id()> {
 public:
  using scalar_t = fp16_t;
  using scalar_t2 = fp16x2_t;
  using scalar_t4 = fp16x2_t;
  using scalar_32bit_t = fp16x2_t;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<fp16x2_t, 4>;
  using FragB = Vec<fp16x2_t, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<fp16x2_t, 1>;
  using FragS0 = Vec<fp8x2_e4m3_t, 1>;
  using FragZP = Vec<fp16x2_t, 4>;

  static __device__ float inline num2float(const fp16_t x) {
    return __half2float(x);
  }

  static __device__ fp16x2_t inline num2num2(const fp16_t x) {
    return __half2half2(x);
  }

  static __device__ fp16x2_t inline nums2num2(const fp16_t x1, const fp16_t x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ fp16_t inline float2num(const float x) {
    return __float2half(x);
  }

  static __host__ __device__ float2 inline num22float2(const fp16x2_t x) {
    return __half22float2(x);
  }
};

template <>
class MarlinScalarType<host::kBFloat16.id()> {
 public:
  using scalar_t = bf16_t;
  using scalar_t2 = bf16x2_t;
  using scalar_t4 = bf16x2_t;
  using scalar_32bit_t = bf16x2_t;

  using FragA = Vec<bf16x2_t, 4>;
  using FragB = Vec<bf16x2_t, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<bf16x2_t, 1>;
  using FragS0 = Vec<fp8x2_e4m3_t, 1>;
  using FragZP = Vec<bf16x2_t, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const bf16_t x) {
    return __bfloat162float(x);
  }

  static __device__ bf16x2_t inline num2num2(const bf16_t x) {
    return __bfloat162bfloat162(x);
  }

  static __device__ bf16x2_t inline nums2num2(const bf16_t x1,
                                                  const bf16_t x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ bf16_t inline float2num(const float x) {
    return __float2bfloat16(x);
  }

  static __host__ __device__ float2 inline num22float2(const bf16x2_t x) {
    return __bfloat1622float2(x);
  }
#endif
};

template <>
class MarlinScalarType<host::kFE4M3fn.id()> {
 public:
  using scalar_t = fp8_e4m3_t;
  using scalar_t2 = fp8x2_e4m3_t;
  using scalar_t4 = fp8x4_e4m3_t;
  using scalar_32bit_t = fp8x4_e4m3_t;

  using FragA = Vec<fp8x4_e4m3_t, 4>;
  using FragB = Vec<fp8x4_e4m3_t, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<fp8x2_e4m3_t, 4>;

  static __host__ __device__
      float2 inline num22float2(const fp8x2_e4m3_t x) {
    return (float2)x;
  }
};

template <>
class MarlinScalarType<host::kS8.id()> {
 public:
  using scalar_t = int8_t;
  using scalar_t2 = int16_t;
  using scalar_t4 = int32_t;
  using scalar_32bit_t = int32_t;

  using FragA = Vec<int32_t, 4>;
  using FragB = Vec<int32_t, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<int16_t, 4>;
};

template <typename scalar_t>
class MarlinScalarType2 {};

template <>
class MarlinScalarType2<fp16_t> : public MarlinScalarType<host::kFloat16.id()> {};

template <>
class MarlinScalarType2<bf16_t>
    : public MarlinScalarType<host::kBFloat16.id()> {};

template <>
class MarlinScalarType2<fp8_e4m3_t>
    : public MarlinScalarType<host::kFE4M3fn.id()> {};

template <>
class MarlinScalarType2<int8_t> : public MarlinScalarType<host::kS8.id()> {};

// Compatibility alias: the MoE template (marlin_moe/marlin_template.h) and the
// pre-vLLM-main dense template still address the type-keyed table as
// ScalarType<scalar_t>.
template <typename T>
using ScalarType = MarlinScalarType2<T>;

}  // namespace device::marlin

}  // namespace sglang

#endif
