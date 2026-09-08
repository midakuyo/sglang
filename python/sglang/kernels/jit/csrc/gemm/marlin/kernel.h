#pragma once

#include <sgl_kernel/scalar_type.hpp>

#include "marlin.cuh"
#include "marlin_dtypes.cuh"

// Kernel parameter list shared by the declaration below, the explicit
// instantiations in gptq_marlin.cuh and the sm<80 stub in marlin_template.h.
// Ported from vLLM main csrc/libtorch_stable/quantization/marlin/kernel.h
// (int8/fp8 activations: a_scales_ptr; fp32 global scale; fused bias).
#define MARLIN_KERNEL_PARAMS                                                                                  \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,     \
      const int4 *__restrict__ b_bias_ptr, const float *__restrict__ a_scales_ptr,                            \
      const int4 *__restrict__ scales_ptr, const float *__restrict__ global_scale_ptr,                        \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n, \
      int prob_k, int lda, int *locks, bool has_bias, bool use_atomic_add, bool use_fp32_reduce,              \
      int max_shared_mem

namespace sglang {

namespace device::marlin {
template <
    const host::ScalarTypeId a_type_id,  // A (activation) ScalarType id
    const host::ScalarTypeId b_type_id,  // B (weight) ScalarType id
    const host::ScalarTypeId c_type_id,  // C (output) ScalarType id
    const host::ScalarTypeId s_type_id,  // B scale ScalarType id
    const int threads,                   // number of threads in a threadblock
    const int thread_m_blocks,           // number of 16x16 blocks in the m
                                         // dimension (batchsize) of the
                                         // threadblock
    const int thread_n_blocks,           // same for n dimension (output)
    const int thread_k_blocks,           // same for k dimension (reduction)
    const bool m_block_size_8,           // whether m_block_size == 8
                                         // only works when thread_m_blocks == 1
    const int stages,                    // number of stages for the async global->shared
                                         // fetch pipeline
    const int group_blocks,              // number of consecutive 16x16 blocks
                                         // with a separate quantization scale
    const bool is_zp_float               // is zero point of float16 type?
    >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}  // namespace device::marlin

}  // namespace sglang
