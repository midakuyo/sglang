/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#pragma once
#include <cstdio>
#include <cstdlib>

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "kernel.h"
#include "marlin_template.h"

namespace sglang {

namespace device::marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    int size_m,
    int size_k,
    int lda,
    int block_rows) {}

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    int size_m,
    int size_k,
    int lda,
    int block_rows) {
  auto start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int input_row_stride = lda * sizeof(half) / 16;
  int output_row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int input_offset = row * input_row_stride;
    int output_offset = row * output_row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + input_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + output_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(
    thread_config_t const& th_config,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * stages * 2;  // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);        // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;
    return tb_scales * stages;
  }
}

int get_kernel_cache_size(
    thread_config_t const& th_config,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    bool is_zp_float,
    bool is_a_8bit,
    int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }
  // int8 activations also stage per-token activation scales (16 * m_blocks floats)
  int sh_a_s_size = is_a_8bit ? 16 * thread_m_blocks * 4 : 0;

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_a_s_size;

  return total_size;
}

bool is_valid_config(
    thread_config_t const& th_config,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    bool is_zp_float,
    bool is_a_8bit,
    int stages,
    int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config,
      thread_m_blocks,
      prob_m,
      prob_n,
      prob_k,
      num_bits,
      group_size,
      has_act_order,
      is_k_full,
      has_zp,
      is_zp_float,
      is_a_8bit,
      stages);
  return cache_size <= max_shared_mem;
}

// Kernel selection. The JIT builds one translation unit per (a_dtype, c_dtype)
// module (see gptq_marlin.py), so the lists below are the explicit
// instantiation set of that module (vLLM generates the same matrix with
// generate_kernels.py). A_T / C_T are the compile-time activation / output
// ScalarTypes of the module; S_TYPE is the scale type.
#define _GET_IF(                                                                                                    \
    A_TYPE,                                                                                                         \
    W_TYPE,                                                                                                         \
    C_TYPE,                                                                                                         \
    S_TYPE,                                                                                                         \
    THREAD_M_BLOCKS,                                                                                                \
    THREAD_N_BLOCKS,                                                                                                \
    THREAD_K_BLOCKS,                                                                                                \
    M_BLOCK_SIZE_8,                                                                                                 \
    GROUP_BLOCKS,                                                                                                   \
    NUM_THREADS,                                                                                                    \
    IS_ZP_FLOAT)                                                                                                    \
  else if (                                                                                                         \
      a_type == A_TYPE && b_type == W_TYPE && c_type == C_TYPE && s_type == S_TYPE &&                              \
      thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                                  \
      thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS &&    \
      num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) {                                                   \
    kernel = Marlin<                                                                                                \
        A_TYPE.id(),                                                                                                \
        W_TYPE.id(),                                                                                                \
        C_TYPE.id(),                                                                                                \
        S_TYPE.id(),                                                                                                \
        NUM_THREADS,                                                                                                \
        THREAD_M_BLOCKS,                                                                                            \
        THREAD_N_BLOCKS,                                                                                            \
        THREAD_K_BLOCKS,                                                                                            \
        M_BLOCK_SIZE_8,                                                                                             \
        pipe_stages,                                                                                                \
        GROUP_BLOCKS,                                                                                               \
        IS_ZP_FLOAT>;                                                                                               \
  }

// 16-bit activation families (a_type == c_type == s_type unless noted)
// COMMON: group_blocks in [-1, 2, 4, 8], is_zp_float == false
// BIGGROUP: group_blocks in [-1, 8] (fp8 weights)
// FP4: nvfp4 (e2m1) weights, e4m3 scales, group_blocks == 1
// FZP: float zero points (fp16 compute only)
// ACT: act_order (group_blocks == 0)
#define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false)   \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false)   \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF(W_TYPE)            \
  COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  COMMON_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
  COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)

#define BIGGROUP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)              \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF(W_TYPE)            \
  BIGGROUP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  BIGGROUP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  BIGGROUP_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  BIGGROUP_GET_IF_M234(W_TYPE, 4, 8, 128)

#define FP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                                  \
  _GET_IF(C_T, W_TYPE, C_T, host::kFE4M3fn, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, host::kFE4M3fn, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define FP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                                 \
  _GET_IF(C_T, W_TYPE, C_T, host::kFE4M3fn, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, host::kFE4M3fn, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, host::kFE4M3fn, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define FP4_GET_IF(W_TYPE)            \
  FP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  FP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  FP4_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  FP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  FP4_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  FP4_GET_IF_M234(W_TYPE, 4, 8, 128)

// We currently have 4-bit models only with group_blocks == 4
#define FZP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                    \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF(W_TYPE)            \
  FZP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  FZP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  FZP_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  FZP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  FZP_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  FZP_GET_IF_M234(W_TYPE, 4, 8, 128)

#define ACT_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                     \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS, false)  \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 1, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)                   \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 2, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 3, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
  _GET_IF(C_T, W_TYPE, C_T, C_T, 4, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF(W_TYPE)            \
  ACT_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  ACT_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  ACT_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  ACT_GET_IF_M234(W_TYPE, 16, 4, 256) \
  ACT_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  ACT_GET_IF_M234(W_TYPE, 4, 8, 128)

// int8 activation family (vLLM PR #24722): a_type kS8, output/scales C_T,
// symmetric uint4b8 weights, group 128 (group_blocks 8), no m_block_size_8.
// Thread configs follow vLLM generate_kernels.py: for thread_m_blocks == 1
// only (128,128,256) among the 256-thread configs, for > 1 only (64,256,256).
#define A8_GET_IF_M1(W_TYPE, GROUP_BLOCKS)                                                            \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 1, 8, 8, false, GROUP_BLOCKS, 256, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 1, 8, 4, false, GROUP_BLOCKS, 128, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 1, 4, 8, false, GROUP_BLOCKS, 128, false)

#define A8_GET_IF_M234(W_TYPE, GROUP_BLOCKS)                                                          \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 2, 16, 4, false, GROUP_BLOCKS, 256, false)                     \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 2, 8, 4, false, GROUP_BLOCKS, 128, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 2, 4, 8, false, GROUP_BLOCKS, 128, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 3, 16, 4, false, GROUP_BLOCKS, 256, false)                     \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 3, 8, 4, false, GROUP_BLOCKS, 128, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 3, 4, 8, false, GROUP_BLOCKS, 128, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 4, 16, 4, false, GROUP_BLOCKS, 256, false)                     \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 4, 8, 4, false, GROUP_BLOCKS, 128, false)                      \
  _GET_IF(host::kS8, W_TYPE, C_T, C_T, 4, 4, 8, false, GROUP_BLOCKS, 128, false)

#define A8_GET_IF(W_TYPE, GROUP_BLOCKS) A8_GET_IF_M1(W_TYPE, GROUP_BLOCKS) A8_GET_IF_M234(W_TYPE, GROUP_BLOCKS)

template <typename T>
constexpr host::ScalarType marlin_scalar_type_of() {
  if constexpr (std::is_same_v<T, int8_t>) {
    return host::kS8;
  } else if constexpr (std::is_same_v<T, fp16_t>) {
    return host::kFloat16;
  } else {
    static_assert(std::is_same_v<T, bf16_t>, "unsupported Marlin dtype");
    return host::kBFloat16;
  }
}

template <typename a_scalar_t, typename c_scalar_t>
MarlinFuncPtr get_marlin_kernel(
    const host::ScalarType a_type,
    const host::ScalarType b_type,
    const host::ScalarType c_type,
    const host::ScalarType s_type,
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    bool has_act_order,
    bool has_zp,
    int group_blocks,
    int num_threads,
    bool is_zp_float) {
  constexpr host::ScalarType C_T = marlin_scalar_type_of<c_scalar_t>();
  auto kernel = MarlinDefault;

  if constexpr (std::is_same_v<a_scalar_t, int8_t>) {
    if (false) {
    }
    A8_GET_IF(host::kU4B8, 8)
  } else {
    static_assert(std::is_same_v<a_scalar_t, c_scalar_t>, "16-bit activations must match the output dtype");
    if (false) {
    }
    COMMON_GET_IF(host::kU4)
    COMMON_GET_IF(host::kU4B8)
    COMMON_GET_IF(host::kU8B128)

    FP4_GET_IF(host::kFE2M1f)

    BIGGROUP_GET_IF(host::kFE4M3fn)

    ACT_GET_IF(host::kU4B8)
    ACT_GET_IF(host::kU8B128)

    if constexpr (std::is_same_v<c_scalar_t, fp16_t>) {
      if (false) {
      }
      FZP_GET_IF(host::kU4)
    }
  }

  return kernel;
}

template <typename a_scalar_t, typename c_scalar_t>
exec_config_t determine_exec_config(
    const host::ScalarType& a_type,
    const host::ScalarType& b_type,
    const host::ScalarType& c_type,
    const host::ScalarType& s_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    bool is_a_8bit,
    int stages,
    int max_shared_mem,
    int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config,
            thread_m_blocks,
            prob_m,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            has_act_order,
            is_k_full,
            has_zp,
            is_zp_float,
            is_a_8bit,
            stages,
            max_shared_mem - 512)) {
      continue;
    }

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel = get_marlin_kernel<a_scalar_t, c_scalar_t>(
        a_type,
        b_type,
        c_type,
        s_type,
        thread_m_blocks,
        th_config.thread_n / 16,
        th_config.thread_k / 16,
        m_block_size_8,
        has_act_order,
        has_zp,
        group_blocks,
        th_config.num_threads,
        is_zp_float);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

template <typename a_scalar_t, typename c_scalar_t>
void marlin_mm(
    const void* A,
    const void* B,
    void* C,
    void* C_tmp,
    void* b_bias,
    void* a_s,
    void* b_s,
    void* g_s,
    void* zp,
    void* g_idx,
    void* perm,
    void* a_tmp,
    int prob_m,
    int prob_n,
    int prob_k,
    int lda,
    void* workspace,
    int workspace_size,
    host::ScalarType const& a_type,
    host::ScalarType const& b_type,
    host::ScalarType const& c_type,
    host::ScalarType const& s_type,
    bool has_bias,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k_init,
    int thread_n_init,
    int threads_init,
    int bps_init,
    bool marlin_debug,
    bool occ2,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  host::RuntimeCheck(
      prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      host::RuntimeCheck(group_size != -1);
      group_blocks = group_size / 16;
      host::RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    } else {
      host::RuntimeCheck(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      host::RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, sms);
    host::LaunchKernel(sms, default_threads, stream)(
        permute_cols_kernel, A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
    A_ptr = a_tmp_ptr;
    lda = prob_k;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  host::RuntimeCheck(max_shared_mem > 0);

  // This JIT only targets sm_80+ (4-stage pipeline); Turing configs are not built.
  constexpr int stages = pipe_stages;

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, threads_init > 0 ? threads_init : default_threads};
      exec_cfg = exec_config_t{bps_init > 0 ? bps_init : 1, thread_tfg};
      host::RuntimeCheck(
          workspace_size >= sms * exec_cfg.blocks_per_sm,
          "SGLANG_MARLIN_CFG blocks_per_sm needs workspace >= sms * blocks_per_sm");
      host::RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);
      host::RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config<a_scalar_t, c_scalar_t>(
          a_type,
          b_type,
          c_type,
          s_type,
          prob_m_split,
          prob_n,
          prob_k,
          thread_m_blocks,
          m_block_size_8,
          num_bits,
          group_size,
          has_act_order,
          is_k_full,
          has_zp,
          is_zp_float,
          is_a_8bit,
          stages,
          max_shared_mem,
          sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        // Low occupancy: prefer the narrower tile so more blocks are in flight.
        if (prob_n / thread_tfg.thread_n * div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <= sms) {
          if (is_valid_config(
                  {128, 64, 128},
                  thread_m_blocks,
                  prob_m_split,
                  prob_n,
                  prob_k,
                  num_bits,
                  group_size,
                  has_act_order,
                  is_k_full,
                  has_zp,
                  is_zp_float,
                  is_a_8bit,
                  stages,
                  max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      // Occupancy 2: on parts with few SMs (CMP 170HX: 70) two 128-thread
      // {thread_k 64, thread_n 128} blocks per SM beat one 256-thread block at
      // every M measured (bench-170hx/marlin_a8_cfg_sweep.py: -17% at M<=16,
      // -8% at M=64, ~-1% at M=2048, int8 and 16-bit activations alike). The
      // lock workspace must hold sms * 2 entries (locks are indexed by block).
      // Above M=512 the gain vanishes (16-bit activations: 2-5% slower at
      // M=2048 than {64,256,256}), so large prefills keep the vLLM choice.
      if (occ2 && prob_m <= 512 && thread_tfg.thread_k != -1 && workspace_size >= 2 * sms) {
        thread_config_t occ_cfg{64, 128, 128};
        int group_blocks = 0;
        if (!has_act_order) group_blocks = group_size == -1 ? -1 : group_size / 16;
        if (is_valid_config(
                occ_cfg,
                thread_m_blocks,
                prob_m_split,
                prob_n,
                prob_k,
                num_bits,
                group_size,
                has_act_order,
                is_k_full,
                has_zp,
                is_zp_float,
                is_a_8bit,
                stages,
                max_shared_mem / 2 - 1024) &&
            get_marlin_kernel<a_scalar_t, c_scalar_t>(
                a_type,
                b_type,
                c_type,
                s_type,
                thread_m_blocks,
                occ_cfg.thread_n / 16,
                occ_cfg.thread_k / 16,
                m_block_size_8,
                has_act_order,
                has_zp,
                group_blocks,
                occ_cfg.num_threads,
                is_zp_float) != MarlinDefault) {
          thread_tfg = occ_cfg;
          exec_cfg = {2, thread_tfg};
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (marlin_debug) {
      std::printf(
          "[marlin] M=%d (split %d, m_blocks %d, m8 %d) N=%d K=%d a8=%d -> thread_k %d thread_n %d threads %d bps %d blocks %d\n",
          prob_m, prob_m_split, thread_m_blocks, m_block_size_8, prob_n, prob_k, (int)is_a_8bit,
          thread_k, thread_n, num_threads, exec_cfg.blocks_per_sm, blocks);
    }
    if (exec_cfg.blocks_per_sm > 1) max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    host::RuntimeCheck(
        is_valid_config(
            thread_tfg,
            thread_m_blocks,
            prob_m_split,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            has_act_order,
            is_k_full,
            has_zp,
            is_zp_float,
            is_a_8bit,
            stages,
            max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ",
        thread_m_blocks,
        ", thread_k = ",
        thread_tfg.thread_k,
        ", thread_n = ",
        thread_tfg.thread_n,
        ", num_threads = ",
        thread_tfg.num_threads,
        " for MKN = [",
        prob_m,
        ", ",
        prob_k,
        ", ",
        prob_n,
        "] and num_bits = ",
        num_bits,
        ", prob_m_split = ",
        prob_m_split,
        ", group_size = ",
        group_size,
        ", has_act_order = ",
        has_act_order,
        ", is_k_full = ",
        is_k_full,
        ", has_zp = ",
        has_zp,
        ", is_zp_float = ",
        is_zp_float,
        ", is_a_8bit = ",
        is_a_8bit,
        ", max_shared_mem_new = ",
        max_shared_mem_new);

    auto kernel = get_marlin_kernel<a_scalar_t, c_scalar_t>(
        a_type,
        b_type,
        c_type,
        s_type,
        thread_m_blocks,
        thread_n_blocks,
        thread_k_blocks,
        m_block_size_8,
        has_act_order,
        has_zp,
        group_blocks,
        num_threads,
        is_zp_float);

    if (kernel == MarlinDefault) {
      host::Panic(
          "Unsupported shapes: MNK = [",
          prob_m,
          ", ",
          prob_n,
          ", ",
          prob_k,
          "]",
          ", a_type = ",
          a_type.str(),
          ", b_type = ",
          b_type.str(),
          ", c_type = ",
          c_type.str(),
          ", s_type = ",
          s_type.str(),
          ", has_act_order = ",
          has_act_order,
          ", num_groups = ",
          num_groups,
          ", group_size = ",
          group_size,
          ", prob_m_split = ",
          prob_m_split,
          ", thread_m_blocks = ",
          thread_m_blocks,
          ", thread_n_blocks = ",
          thread_n_blocks,
          ", thread_k_blocks = ",
          thread_k_blocks,
          ", num_threads = ",
          num_threads,
          ", num_bits = ",
          num_bits);
    }

    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new));

    bool part_use_atomic_add = use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    host::LaunchKernel(blocks, num_threads, stream, max_shared_mem_new)(
        kernel,
        A_ptr,
        B_ptr,
        C_ptr,
        C_tmp_ptr,
        bias_ptr,
        a_s_ptr,
        b_s_ptr,
        g_s_ptr,
        zp_ptr,
        g_idx_ptr,
        num_groups,
        prob_m_split,
        prob_n,
        prob_k,
        lda,
        locks,
        has_bias,
        part_use_atomic_add,
        use_fp32_reduce,
        max_shared_mem_new);

    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

#endif

}  // namespace device::marlin

template <typename a_scalar_t, typename c_scalar_t>
void gptq_marlin_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView a_scales,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView g_idx,
    tvm::ffi::TensorView perm,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView c_tmp,
    tvm::ffi::TensorView a_tmp,
    tvm::ffi::TensorView workspace,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  using namespace host;

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();
  constexpr ScalarType a_type = device::marlin::marlin_scalar_type_of<a_scalar_t>();
  constexpr ScalarType c_type = device::marlin::marlin_scalar_type_of<c_scalar_t>();
  const bool is_a_8bit = a_type.size_bits() == 8;
  // nvfp4 weights carry e4m3 group scales; everything else scales in the output dtype
  const ScalarType s_type = (b_q_type == kFE2M1f) ? ScalarType(kFE4M3fn) : c_type;

  // Bind symbolic sizes
  auto M = SymbolicSize{"M"};
  auto K = SymbolicSize{"K"};
  auto N = SymbolicSize{"N"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  // Verify a: [M, K]
  auto lda = SymbolicSize{"lda"};
  TensorMatcher({M, K}).with_strides({lda, 1}).with_dtype<a_scalar_t>().with_device(device).verify(a);

  int64_t size_m = M.unwrap();
  int64_t size_k = K.unwrap();

  // Verify b_q_weight: [K/tile_size, packed_N]
  RuntimeCheck(
      size_k % device::marlin::tile_size == 0,
      "size_k = ",
      size_k,
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  int64_t expected_bqw_dim0 = size_k / device::marlin::tile_size;
  auto bqw_dim0 = SymbolicSize{"bqw_dim0"};
  auto bqw_dim1 = SymbolicSize{"bqw_dim1"};
  bqw_dim0.set_value(expected_bqw_dim0);
  TensorMatcher({bqw_dim0, bqw_dim1}).with_dtype<int32_t>().with_device(device).verify(b_q_weight);

  RuntimeCheck(
      b_q_weight.size(1) % device::marlin::tile_size == 0,
      "b_q_weight.size(1) = ",
      b_q_weight.size(1),
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  int64_t actual_size_n = (b_q_weight.size(1) / device::marlin::tile_size) * pack_factor;
  N.set_value(actual_size_n);
  int64_t size_n = N.unwrap();

  // Verify stride alignment
  int64_t a_stride0 = a.stride(0);
  RuntimeCheck(a_stride0 % (is_a_8bit ? 16 : 8) == 0, "a.stride(0) must be divisible by 8 (16 for int8 activations)");

  // Verify b_scales: [num_groups, N]
  auto num_groups_sym = SymbolicSize{"num_groups"};
  TensorMatcher({num_groups_sym, N}).with_device(device).verify(b_scales);
  int num_groups = static_cast<int>(num_groups_sym.unwrap());

  // Verify c: [M, N]
  TensorMatcher({M, N}).with_dtype<c_scalar_t>().with_device(device).verify(c);

  // Optional fused bias [N] (output dtype) and per-token activation scales [M] (fp32, int8 activations only)
  const bool has_bias = b_bias.size(0) > 0;
  if (has_bias) {
    TensorMatcher({N}).with_dtype<c_scalar_t>().with_device(device).verify(b_bias);
  }
  const bool has_a_scales = a_scales.size(0) > 0;
  RuntimeCheck(has_a_scales == is_a_8bit, "a_scales must be given exactly when the activations are int8");
  if (has_a_scales) {
    TensorMatcher({M}).with_dtype<float>().with_device(device).verify(a_scales);
  }

  // Early return for zero-size M
  if (size_m == 0) return;

  // Determine has_act_order from g_idx/perm sizes
  int64_t g_idx_size = g_idx.size(0);
  int64_t perm_size = perm.size(0);
  bool has_act_order = g_idx_size > 0 && perm_size > 0;

  if (has_act_order) {
    RuntimeCheck(
        (g_idx_size == size_k && perm_size == size_k),
        "Unexpected g_idx.size(0) = ",
        g_idx_size,
        " and perm.size(0) = ",
        perm_size,
        ", where size_k = ",
        size_k);
  }

  // Determine has_zp from b_zeros size
  int64_t b_zeros_size = b_zeros.size(0);
  bool has_zp = b_zeros_size > 0;

  if (has_zp) {
    RuntimeCheck(
        b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
  } else {
    RuntimeCheck(
        b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,
        "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
        "has_zp = False. Got = ",
        b_q_type.str());
  }

  if (has_zp && is_zp_float) {
    RuntimeCheck(
        std::is_same<c_scalar_t, fp16_t>::value, "Computation type must be float16 (half) when using float zero points.");
  }

  // Verify b_zeros shape
  if (has_zp) {
    RuntimeCheck(b_zeros.dim() == 2, "b_zeros rank = ", b_zeros.dim(), " is not 2");
    if (is_zp_float) {
      RuntimeCheck(b_zeros.size(1) == size_n, "b_zeros dim 1 = ", b_zeros.size(1), " is not size_n = ", size_n);
      RuntimeCheck(
          num_groups == b_zeros.size(0), "b_zeros dim 0 = ", b_zeros.size(0), " is not num_groups = ", num_groups);
      RuntimeCheck(num_groups != -1, "num_groups must be != -1");
    } else {
      RuntimeCheck(
          b_zeros.size(0) == num_groups, "b_zeros dim 0 = ", b_zeros.size(0), " is not num_groups = ", num_groups);
      RuntimeCheck(
          b_zeros.size(1) == size_n / pack_factor,
          "b_zeros dim 1 = ",
          b_zeros.size(1),
          " is not size_n / pack_factor = ",
          size_n / pack_factor);
    }
  }

  // Verify global_scale
  int64_t global_scale_size = global_scale.size(0);
  if (global_scale_size > 0) {
    RuntimeCheck(b_q_type == kFE2M1f, "global_scale can only be used for float4_e2m1f.");
    auto GS = SymbolicSize{"gs"};
    TensorMatcher({GS}).with_dtype<float>().with_device(device).verify(global_scale);
  } else {
    RuntimeCheck(!(b_q_type == kFE2M1f), "the global_scale parameter must be passed for float4_e2m1f.");
  }

  // Derive group_size
  int group_size = -1;
  if (has_act_order) {
    if (is_k_full) {
      RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
      group_size = static_cast<int>(size_k / num_groups);
    } else {
      group_size = 0;
    }
  } else {
    if (num_groups > 1) {
      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
      group_size = static_cast<int>(size_k / num_groups);
    } else {
      group_size = -1;
    }
  }

  // Verify workspace and get device info
  RuntimeCheck(
      size_n % device::marlin::min_thread_n == 0,
      "size_n = ",
      size_n,
      ", is not divisible by min_thread_n = ",
      device::marlin::min_thread_n);

  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);

  int sms = -1;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

  RuntimeCheck(
      workspace.size(0) >= sms, "workspace.size(0) = ", workspace.size(0), " is below min_workspace_size = ", sms);

  // Hardcoded defaults (auto config)
  // Debug/tuning override: SGLANG_MARLIN_CFG="thread_k,thread_n,num_threads"
  // (e.g. 64,128,128) forces a thread config for every call; unset = auto.
  int thread_k_init = -1;
  int thread_n_init = -1;
  int threads_init = -1;
  int bps_init = -1;  // blocks per SM for the forced config (optional 4th field)
  if (const char* cfg = std::getenv("SGLANG_MARLIN_CFG")) {
    int k = -1, n = -1, t = -1, b = -1;
    int got = std::sscanf(cfg, "%d,%d,%d,%d", &k, &n, &t, &b);
    if (got >= 3) {
      thread_k_init = k;
      thread_n_init = n;
      threads_init = t;
      if (got == 4) bps_init = b;
    }
  }
  const bool marlin_debug = std::getenv("SGLANG_MARLIN_DEBUG") != nullptr;
  // Occupancy-2 heuristic (default on): SGLANG_MARLIN_OCC2=0 restores the vLLM choice.
  const char* occ2_env = std::getenv("SGLANG_MARLIN_OCC2");
  const bool occ2 = !(occ2_env && occ2_env[0] == '0');

  // Compute c_tmp and a_tmp pointers
  // c_tmp and a_tmp are pre-allocated by caller

  device::marlin::marlin_mm<a_scalar_t, c_scalar_t>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      a_scales.data_ptr(),
      b_scales.data_ptr(),
      global_scale.data_ptr(),
      b_zeros.data_ptr(),
      g_idx.data_ptr(),
      perm.data_ptr(),
      a_tmp.data_ptr(),
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      static_cast<int>(a_stride0),
      workspace.data_ptr(),
      (int)workspace.size(0),
      a_type,
      b_q_type,
      c_type,
      s_type,
      has_bias,
      has_act_order,
      is_k_full,
      has_zp,
      num_groups,
      group_size,
      dev,
      stream,
      thread_k_init,
      thread_n_init,
      threads_init,
      bps_init,
      marlin_debug,
      occ2,
      sms,
      use_atomic_add,
      use_fp32_reduce,
      is_zp_float);
}

}  // namespace sglang
