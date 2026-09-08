# vLLM Marlin W4A8‑INT8 → SGLang 포크 JIT 이식 계획 (Gemma4 31B / sm_80)

## 0. 결론 요약

- **베이스 = vLLM main `6b5a12c` (`/tmp/vllm/csrc/libtorch_stable/quantization/marlin/`)** 의 디바이스 파일을 **통째로** 가져오고 sglang 변환(namespace, `host::`, `sgl_kernel/*` include, tvm_ffi)을 재적용한다. PR 헝크 재적용은 불가(포크 템플릿이 v0.10.0 기반이라 51개 중 37개 충돌).
- 호스트 층(`gptq_marlin.cuh`의 TensorView 래퍼)은 **포크 것을 유지하고 패치**한다 (main `marlin.cu`의 로직만 옮김).
- 포크에는 `generate_kernels.py`가 없으므로 int8 인스턴스는 `_GET_IF` 계열 매크로에 **명시 목록**으로 추가하고, **별도 JIT 모듈 키 `gptq_marlin<int8_t, bf16_t>`** 로 분리해 fp16/bf16 모듈이 비대해지지 않게 한다.
- Gemma4(group 128, symmetric, uint4b8, bf16)용 최소 인스턴스는 **bf16 출력 × group_blocks=8 × 12 (thread cfg×m_blocks)** = 12개.
- 이식 후 `CompressedTensorsW4A8Int8` 스킴에서 `M <= QQQ_MAX_M` 분기를 Marlin‑A8로 교체하고, 대 M(unpack+CUTLASS) 교체 여부는 **벤치 결과로 결정**(현재로선 미지수).
- 반드시 얹을 두 가지 미병합 수정: **PR #48926**(스케일 `uint16_t*`→`int16_t*`), **PR #49862**(repack `size_k % 32` 검사).

---

## 1. 베이스 버전 선택과 근거

| 후보 | 판정 | 이유 |
|---|---|---|
| PR 헝크(`/tmp/pr24722.diff`)를 포크에 재적용 | **기각** | 포크 dense 템플릿은 vLLM v0.10.0(#22428 이전: `s_type_id`/`b_bias_ptr`/`write_result(bool)` 없음). dry‑run 결과 `marlin_template.h` 37/51, `gptq_marlin.cu` 17/18 헝크 실패. |
| 머지 커밋 `1656ad37` (`/tmp/vllm-pr`) | 참고용 | int8 디바이스 경로는 main과 동일. 그러나 호스트 `get_kernel_cache_size`가 int8 A를 2배 과대계상(`gptq_marlin.cu:189`), `pipe_stages` 상수 사용. |
| **vLLM main `6b5a12c`** | **채택** | int8 경로 바이트 동일(analysis 4 검증). 추가 이점: `marlin_mma.h` 분리, `is_8bit_scale`(MXFP8), `float global_scale_ptr`, `sh_a_size = stages*(tb_m*tb_k)*(is_a_8bit?1:2)` (`marlin.cu:201`), `{128,64,128}` 그리드 저충전 오버라이드(`marlin.cu:458-468`). |

main 전용으로 **버릴 것**: sm75 분기(`marlin_template.h:288-304`, `1849-1861`; `marlin_mma.h:22,47,104,154,179,236`의 `__CUDA_ARCH__ == 750` 블록; `marlin.cuh:52-95` fallback; `marlin.cu:407-412`의 `stages=2`). 타깃은 sm_80 단일이므로 `stages`는 포크의 `pipe_stages`(=4) 리터럴로 고정.

---

## 2. 파일별 이식 방식 (`python/sglang/kernels/jit/csrc/gemm/marlin/`)

공통 sglang 변환(모든 파일): `namespace MARLIN_NAMESPACE_NAME` → `namespace sglang { namespace device::marlin {`; `vllm::` → `host::`; `#include "core/scalar_type.hpp"` → `<sgl_kernel/scalar_type.hpp>`; torch/ATen include 제거 → `<sgl_kernel/utils.cuh>`; `half/nv_bfloat16/half2/nv_bfloat162` → `fp16_t/bf16_t/fp16x2_t/bf16x2_t` (`sgl_kernel/utils.cuh:59-66`); `__nv_fp8x4_e4m3` → `fp8x4_e4m3_t`.

### 2.1 `marlin.cuh` (87줄) — **패치**
- main `marlin.cuh:98-135`의 `cp_async1_ca_pred`(4B) / `cp_async2_ca_pred`(8B) / `cp_async4_ca_pred`(16B)를 포크 `#else` 블록(`marlin.cuh:46` 이후, `cp_async4_pred` 앞)에 삽입. PR 헝크가 그대로 적용됨(analysis 2 dry‑run 통과).
- main의 `constexpr int div_ceil`(`marlin.cuh:50`)은 **넣지 않음** — `sgl_kernel/utils.h:155`의 템플릿 `div_ceil`과 충돌.
- `max_par=16`, `max_thread_n=256`, `tile_k_size/tile_n_size` 등 상수는 그대로.

### 2.2 `marlin_dtypes.cuh` (81줄) — **main으로 교체 + 호환 alias**
- main `marlin_dtypes.cuh:17-149` 전체 채택: `template <long scalar_type_id> class MarlinScalarType`, 특수화 `kFloat16`(20), `kBFloat16`(59), `kFE4M3fn`(98), `kS8`(117-128: `scalar_t=int8_t, scalar_t2=int16_t, scalar_t4/scalar_32bit_t=int32_t, FragA=Vec<int32_t,4>, FragB=Vec<int32_t,2>, FragC=Vec<float,4>, FragZP=Vec<int16_t,4>`), `MarlinScalarType2<T>`(131-145).
- **MoE 호환 필수**: `marlin_moe/kernel.h:5`, `marlin_moe/marlin_template.h:26`이 이 파일을 include하며 `ScalarType<scalar_t>::{FragA,FragB,FragC,FragS,FragZP,scalar_t2,num2num2,num2float}`을 사용. 파일 끝에 다음을 추가:
  ```cpp
  template <typename T> using ScalarType = MarlinScalarType2<T>;
  ```
  `MarlinScalarType2<fp16_t>`, `<bf16_t>`, `<fp8_e4m3_t>`, `<int8_t>` 특수화가 필요 멤버(`FragS`, `num2num2`, `num2float`, `nums2num2`, `float2num`, `num22float2`)를 모두 상속하므로 MoE 모듈 컴파일 유지.
- bf16 특수화의 `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800` 가드는 포크 스타일대로 유지.

### 2.3 `dequant.h` (508줄) — **main으로 교체**
- main `dequant.h`(609줄) 전체. int8용 핵심: `dequant<int32_t, host::kU4B8.id(), true>`(main:494, `((q & 0x0F0F0F0F | 0x80808080) - 0x08080808) ^ 0x80808080`), `sub_zp_and_dequant<int32_t, host::kU4.id(), true>`(570).
- 포크가 유지하던 **1‑파라미터** `dequant_fp8_scales<scalar_t2>`(포크 425-457)는 삭제 — 새 dense 템플릿은 2‑파라미터 `dequant_fp8_scales<scalar_t2, s_type_id>`(main:518-560)를 쓰고, `marlin_moe/marlin_template.h`도 이미 2‑파라미터 버전만 사용(grep 확인: `dequant_fp8_scales<scalar_t2, s_type_id>` 2회, `dequant<scalar_t2, w_type_id, dequant_skip_flop>` 1회 — 둘 다 main 시그니처와 일치).
- 가드 `__CUDA_ARCH__ >= 750`(main:70)은 `>= 800`으로 되돌림.

### 2.4 `marlin_mma.h` — **신규 (main 268줄)**
- `template <host::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16> mma(FragA&, FragB&, FragC&, int idx=0)` 및 `mma_trans`. int8 k32 분기(main:126-131, 258-263)의 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite`.
- sm75 `m8n8k16` 분기 삭제. `use_fp16_accum` 파라미터는 유지(호출부 `mma<a_type_id, false, 32>` 일관성)하되 항상 false.
- 포크 `marlin_template.h:38-72`에 인라인돼 있던 구 `mma<typename scalar_t>`는 사라짐 (MoE 템플릿은 자체 사본을 가짐 — 확인: `marlin_moe/marlin_template.h`는 `../marlin/dequant.h, marlin.cuh, marlin_dtypes.cuh`만 include, mma는 자체 정의).

### 2.5 `kernel.h` (37줄) — **수동 재작성**
```cpp
#define MARLIN_KERNEL_PARAMS                                                   \
  const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C, \
  int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,               \
  const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr, \
  const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr, \
  const int* __restrict__ g_idx, int num_groups, int prob_m, int prob_n,       \
  int prob_k, int lda, int* locks, bool has_bias, bool use_atomic_add,         \
  bool use_fp32_reduce, int max_shared_mem

template <const host::ScalarTypeId a_type_id, const host::ScalarTypeId b_type_id,
          const host::ScalarTypeId c_type_id, const host::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void Marlin(MARLIN_KERNEL_PARAMS);
```
(main `kernel.h:8-43`과 동일; `scale2_ptr: const uint16_t*` → `global_scale_ptr: const float*`.)

### 2.6 `marlin_template.h` (1624줄) — **main(2081줄)으로 교체**
- 재적용 항목: 위 공통 변환 + `#include "marlin_mma.h"`(main:29) + `#include <sgl_kernel/scalar_type.hpp>` + `__CUDA_ARCH__ < 750` → `< 800`(main:39)과 그 안의 stub을 **12‑파라미터 시그니처**로 재작성(main:41-59 stub은 구 10‑파라미터라 kernel.h 선언과 불일치 — JIT는 sm_80만 컴파일하므로 실동작엔 무관하나 일관성 위해 수정).
- 삭제: sm75 블록(288-304, 1849-1861); `use_fp16_accum`은 `constexpr bool use_fp16_accum = false;` 한 줄로 고정.
- **적용할 수정 #48926**: main:1336-1340의 `reinterpret_cast<uint16_t*>` 4곳 → `reinterpret_cast<int16_t*>`. (int16 부호 스케일 대비.)
- 포크의 세 가지 semantic 편집(`is_8bit_scale`, `FragB frag_zp_0/1`, 닫는 중괄호)은 main 템플릿에 흡수됨(`is_8bit_scale = s_type.size_bits() == 8`, main:340).
- int8 경로 핵심 위치(검증용): `is_a_8bit` 339, `sh_a_s/sh_new` 365-366, a_scales cp.async 487/503, `matmul_a8` 1295-1362, `static_assert(group_blocks != 0 && group_blocks != 1)` 1813, 에필로그 `__int2float_rn * frag_a_s` 1864-1893.

### 2.7 `gptq_marlin.cuh` (1005줄) — **포크 유지 + 호스트 로직 패치** (main `marlin.cu` 기준)
1. `get_scales_cache_size`(포크 138-170): `pipe_stages` 그대로(스테이지 고정).
2. `get_kernel_cache_size`(171-211): main:190-224 본문으로 교체 — 인자 `bool has_bias, bool is_a_8bit` 추가, `sh_a_size = pipe_stages*(tb_m*tb_k)*(is_a_8bit ? 1 : 2)`, `sh_bias_size`, `tmp_size` 포함. **추가로** `+ (is_a_8bit ? 16*thread_m_blocks*4 : 0)` (sh_a_s 프리픽스, main도 안 세지만 명시).
3. `is_valid_config`(213-259): main:229-260 시그니처(`is_a_8bit` 추가). 반환 `cache_size + 512 <= max_shared_mem`(vLLM 슬랙 복원).
4. `_GET_IF`(261-278) → §3의 새 매크로.
5. `get_marlin_kernel`(395-430): `template <typename a_scalar_t, typename c_scalar_t>`; 인자에 `host::ScalarType a_type, b_type, c_type, s_type` 추가.
6. `determine_exec_config`(432-512): `is_a_8bit` 관통; `if (kernel == MarlinDefault) continue;`(505) 유지 — int8에서 `m_block_size_8` 커널이 없을 때 자연 폴백.
7. `marlin_mm`(514-784): 시그니처에 `void* b_bias, void* a_s, host::ScalarType a_type, c_type, s_type, bool has_bias` 추가; `bool is_a_8bit = a_type.size_bits() == 8;`(main:336); `m_block_size_8 = prob_m_split <= 8 && !is_a_8bit`(포크 626 ← main:438); `{128,64,128}` 오버라이드(main:458-468) 추가; 런치 인자 순서를 새 `MARLIN_KERNEL_PARAMS`에 맞춤(`A_ptr,B_ptr,C_ptr,C_tmp_ptr,bias_ptr,a_s_ptr,s_ptr,g_s_ptr,zp_ptr,g_idx_ptr,num_groups,prob_m_split,prob_n,prob_k,lda,locks,has_bias,part_use_atomic_add,use_fp32_reduce,max_shared_mem_new`); 분할 전진 `A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8)); a_s_ptr += prob_m_split;`(포크 780 ← main:535-537).
8. tvm_ffi 래퍼(790-1003): §4.

### 2.8 `gptq_marlin_repack.cuh` (366줄) — **커널 본문 main 교체 + 포크 래퍼 유지**
- 커널 템플릿(포크 35-46) → `template <int num_threads, int num_bits, bool has_perm, bool is_a_8bit>`; 본문(45-273)을 main `gptq_marlin_repack.cu:14-269`로 교체(`target_tile_n_size = tile_n_size/(is_a_8bit?2:1)`, `target_tile_k_size = tile_k_size*(is_a_8bit?2:1)`, `tc_row = (th_id%4)*(is_a_8bit?4:2)`(130), `static_assert(!is_a_8bit)` under has_perm(147), `pack_idx {0,4,1,5,2,6,3,7}`(211-220)). `<800` 빈 stub(34-44)도 4‑파라미터로.
- `CALL_IF_REPACK(NUM_BITS, HAS_PERM, IS_A_8BIT)`(276-289) 3인자화; 호출 목록 `(4,false,false) (4,true,false) (8,false,false) (8,true,false) (4,false,true) (8,false,true)`.
- 호스트 `gptq_marlin_repack(TensorView b_q_weight, TensorView perm, TensorView out, int64_t size_k, int64_t size_n, int64_t num_bits, bool is_a_8bit)`; **#49862 적용**: `RuntimeCheck(size_k % (tile_k_size * (is_a_8bit ? 2 : 1)) == 0)`; `RuntimeCheck(!(is_a_8bit && has_perm), "act_order unsupported for 8-bit activation")`. 출력 shape `{size_k/16, size_n*16/pack_factor}` 불변.

### 2.9 `awq_marlin_repack.cuh` (255줄) — 동일 패턴(선택). Gemma4엔 불필요하지만 uint4(zp) 대칭성을 위해 main `awq_marlin_repack.cu` 본문 교체 + `is_a_8bit` 인자. 우선순위 낮음(Phase 6).

### 2.10 `marlin_qqq.cuh` — **불변**. `sgl_kernel/tensor.h, utils.cuh`만 include하므로 영향 없음. 벤치 비교 대상으로 남긴다.

### 2.11 `marlin_moe/*` — **불변**, 단 2.2의 alias와 2.3의 2‑파라미터 `dequant_fp8_scales`로 컴파일 유지. 회귀 테스트 `test/registered/quant/test_marlin_moe.py`.

---

## 3. int8 커널 인스턴스화 (generate_kernels.py 없이)

포크는 `gptq_marlin.cuh:29`가 `marlin_template.h`를 include하고 `_GET_IF` 안의 `kernel = Marlin<...>`가 **암묵 인스턴스화**를 일으킨다. 그대로 활용한다.

### 3.1 새 매크로
```cpp
#define _GET_IF(A_TYPE, W_TYPE, C_TYPE, S_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS,   \
                THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
  else if (a_type == A_TYPE && b_type == W_TYPE && c_type == C_TYPE && s_type == S_TYPE && \
           thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&   \
           thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 &&     \
           group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS &&                  \
           is_zp_float == IS_ZP_FLOAT) {                                                  \
    kernel = Marlin<A_TYPE.id(), W_TYPE.id(), C_TYPE.id(), S_TYPE.id(), NUM_THREADS,     \
                    THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8,    \
                    pipe_stages, GROUP_BLOCKS, IS_ZP_FLOAT>;                              \
  }
```
기존 `COMMON/BIGGROUP/FP4/FZP/ACT_GET_IF` 계열은 `A_TYPE = C_TYPE = MarlinScalarType2<c_scalar_t>` 의 id(즉 `host::kFloat16`/`kBFloat16`), `S_TYPE = C_TYPE`(FP4 계열만 `S_TYPE = host::kFE4M3fn`)로 래핑한다.

### 3.2 int8 계열 (`A8_GET_IF`)
```cpp
// m_blocks == 1 : small_batch configs  (thread_k, thread_n, threads) → (k_blocks, n_blocks)
#define A8_GET_IF_M1(W, C, G)                          \
  _GET_IF(host::kS8, W, C, C, 1, 8, 8, false, G, 256, false)   /* {128,128,256} */ \
  _GET_IF(host::kS8, W, C, C, 1, 8, 4, false, G, 128, false)   /* { 64,128,128} */ \
  _GET_IF(host::kS8, W, C, C, 1, 4, 8, false, G, 128, false)   /* {128, 64,128} */
// m_blocks in {2,3,4} : large_batch configs
#define A8_GET_IF_M(W, C, M, G)                        \
  _GET_IF(host::kS8, W, C, C, M, 16, 4, false, G, 256, false)  /* { 64,256,256} */ \
  _GET_IF(host::kS8, W, C, C, M,  8, 4, false, G, 128, false)  /* { 64,128,128} */ \
  _GET_IF(host::kS8, W, C, C, M,  4, 8, false, G, 128, false)  /* {128, 64,128} */
#define A8_GET_IF(W, C, G) A8_GET_IF_M1(W,C,G) A8_GET_IF_M(W,C,2,G) A8_GET_IF_M(W,C,3,G) A8_GET_IF_M(W,C,4,G)
```
`get_marlin_kernel` 안에서:
```cpp
if constexpr (std::is_same_v<a_scalar_t, int8_t>) {
  constexpr auto C = MarlinScalarType2<c_scalar_t>::... // host::kBFloat16 or kFloat16
  A8_GET_IF(host::kU4B8, C, 8)          // Gemma4: group 128
#ifdef SGLANG_MARLIN_A8_FULL            // 테스트/범용: vLLM 전체 집합
  A8_GET_IF(host::kU4B8, C, -1) A8_GET_IF(host::kU4B8, C, 2) A8_GET_IF(host::kU4B8, C, 4)
  A8_GET_IF(host::kU4,   C, -1) ... (2,4,8)
#endif
} else { /* 기존 fp16/bf16 계열 */ }
```

### 3.3 Gemma4에 필요한 조합 (검증)
- `m_block_size_8 = false` 고정(int8은 호스트가 강제, main:438). `is_zp_float=false`, `group_blocks=8`(128/16), `stages=4`.
- 호스트 분할: `max_thread_m_blocks=4` → `prob_m_split` ≤ 64행 청크, `max_par=16`(N>4096) / 128(N≤4096). M=1..16 → m_blocks=1, 17..32 → 2, 33..48 → 3, 49..2048 → 4 청크 반복. 따라서 **m_blocks 1,2,3,4 모두 필요**.
- shape 허용성: K∈{5376,8192,21504} 모두 128의 배수 → thread_k 64/128 OK. N∈{16384,5376,43008} 모두 256의 배수 → thread_n 64/128/256 OK. TP=2 시 N=2688은 256 배수 아님 → `{64,256,256}` 탈락, `{64,128,128}` 폴백(정상 동작, 성능만 다름).
- smem(가장 큰 케이스 `{128,128,256}`, m_blocks=4, int8): sh_a 4·64·128 = 32 KB, sh_b 4·(128·128/8)·4 = 32 KB, sh_s 4·1·128·2 = 1 KB, sh_red 64·136·… < 32 KB, sh_a_s 256 B → ≈ 66 KB ≪ 163 KB(GA100 opt‑in). 머지 커밋의 2배 과대계상도 통과하지만 main 식을 채택.
- **최소 집합 = 12 커널(bf16 출력)**. fp16 출력까지 24. 전체 vLLM 집합은 192(별도 플래그).

### 3.4 모듈 키
`gptq_marlin.py:18-26`:
```python
@cache_once
def _jit_gptq_marlin_module(a_dtype, c_dtype):
    args = make_cpp_args(a_dtype, c_dtype)          # ("int8_t","bf16_t") / ("bf16_t","bf16_t")
    return load_jit("gptq_marlin", *args,
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{args}>")])
```
`make_cpp_args`는 `torch.int8 → "int8_t"`(`cpp_args.py:28`) 이미 지원. 캐시 키는 spec(args+파일 내용)에 대한 `compute_build_key`(`loader.py:109-114`)이므로 int8 모듈과 bf16 모듈이 별개 디렉터리에 캐시된다.

---

## 4. tvm_ffi 래퍼 / Python 래퍼 시그니처

### 4.1 C++ (`gptq_marlin.cuh:790-1003`)
```cpp
template <typename a_scalar_t, typename c_scalar_t>
void gptq_marlin_gemm(
    tvm::ffi::TensorView a,            // [M,K] a_scalar_t (int8 시 stride(0) % 16 == 0)
    tvm::ffi::TensorView b_q_weight,   // int32 [K/16, 2N]
    tvm::ffi::TensorView b_bias,       // c_scalar_t [N] (permute됨) 또는 numel 0   ← 신규
    tvm::ffi::TensorView b_scales,     // [num_groups, N] c_scalar_t (int8 시 int16 비트를 c dtype으로 view)
    tvm::ffi::TensorView a_scales,     // fp32 [M] contiguous 또는 numel 0         ← 신규
    tvm::ffi::TensorView global_scale, // fp32 [1] 또는 numel 0 (nvfp4)           ← dtype 변경(fp16→fp32)
    tvm::ffi::TensorView b_zeros, g_idx, perm, c, c_tmp, a_tmp, workspace,
    int64_t b_q_type_id, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);
```
내부 규칙:
- `a_type`: `std::is_same_v<a_scalar_t,int8_t> ? host::kS8 : (fp16_t ? kFloat16 : kBFloat16)`; `c_type`: `c_scalar_t`에서; `s_type = c_type`, 단 `b_type == kFE2M1f`면 `kFE4M3fn`(포크 nvfp4 경로는 e4m3 스케일 사용), `kFE4M3fn` 가중치+e8m0 스케일이면 `kFE8M0fnu`(선택).
- `TensorMatcher({M,K}).with_strides({lda,1}).with_dtype<a_scalar_t>()`(822), `c`는 `with_dtype<c_scalar_t>()`(860). `RuntimeCheck(a_stride0 % (is_a_8bit ? 16 : 8) == 0)`(852).
- `bool has_bias = b_bias.size(0) > 0;` `bool has_a_scales = a_scales.size(0) > 0; RuntimeCheck(has_a_scales == is_a_8bit)`; `TensorMatcher({M}).with_dtype<float>()`로 a_scales 검증(커널은 `a_scales_ptr[par_id*16*thread_m_blocks + threadIdx.x]`를 `threadIdx.x < prob_m` 프레디케이트로 읽으므로 M개면 충분).
- `global_scale` numel>0 시 `with_dtype<float>()`.
- 스텁: `spec.py:_wrapper_source`가 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(gptq_marlin_gemm, (gptq_marlin_gemm<int8_t, bf16_t>))`를 생성.

### 4.2 Python (`kernels/ops/quantization/gptq_marlin.py:36-117`)
```python
def gptq_marlin_gemm(a, c, b_q_weight, b_scales, global_scale, b_zeros, g_idx, perm,
                     workspace, b_q_type, size_m, size_n, size_k, is_k_full=True,
                     use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False,
                     *, a_scales: Optional[Tensor] = None, b_bias: Optional[Tensor] = None,
                     c_tmp: Optional[Tensor] = None) -> Tensor:
    c_dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else b_scales.dtype
    if c is None: c = torch.empty((size_m, size_n), dtype=c_dtype, device=device)   # 58-59 수정
    if a.dtype == torch.int8:
        assert a_scales is not None and a_scales.dtype == torch.float32 and a.stride(0) % 16 == 0
        a_scales = a_scales.reshape(-1)          # [M,1] → [M]
    a_tmp dtype = c_dtype                                                            # 86-89 수정
    global_scale_t = _or_empty(global_scale.float() if global_scale is not None else None, device, torch.float32)  # 92 수정
    module = _jit_gptq_marlin_module(a.dtype, c_dtype)
    module.gptq_marlin_gemm(a, b_q_weight, _or_empty(b_bias, device, c_dtype), b_scales,
                            _or_empty(a_scales, device, torch.float32), global_scale_t,
                            b_zeros_t, g_idx_t, perm_t, c, c_tmp, a_tmp, workspace, b_q_type.id, ...)
```
- `c_tmp`를 외부 주입 가능하게(스킴에서 `_scratch`로 재사용 → 그래프 캡처 시 할당 제거). 크기 공식 `sms * min(round_up(M,16),64) * 256` fp32(73-83)는 main:711-722와 동일.
- 기존 호출자(`marlin_utils.py:466-530`, `marlin_utils_fp4.py`, `marlin_utils_fp8.py`)는 kwargs 기본값으로 무변경 호환. 단 nvfp4 호출자가 넘기던 fp16 `global_scale`은 래퍼가 `.float()`로 변환(재검증 필요).

### 4.3 `gptq_marlin_repack.py:27-45`
`def gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit: bool = False)` → `module.gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits, is_a_8bit)`. 출력 shape 불변.

---

## 5. Repack / 스케일 경로와 기존 compressed‑tensors 텐서 매핑

스킴 `compressed_tensors_w4a8_int8.py`의 `create_weights`(150-179)는 그대로: `layer.weight` int8 `[N, K]`(값 −8..7), `layer.weight_scale` params_dtype(bf16) `[N, K/128]`. `process_weights_after_loading`(181-206)을 다음으로 교체:

```python
q4 = layer.weight.data                      # int8 [N,K]
s_g = layer.weight_scale.data               # bf16 [N,K/g]
N, K = q4.shape
# (1) signed int4 → uint4b8, GPTQ 행 패킹 [K/8, N] (vLLM quant_utils.pack_rows 동치)
q_u = ((q4.to(torch.int32) + 8) & 0xF).t().contiguous()          # [K,N]
packed = torch.zeros(K // 8, N, dtype=torch.int32, device=q4.device)
for i in range(8): packed |= q_u[i::8, :] << (4 * i)
# (2) int8 활성 전용 타일 레이아웃으로 repack
empty = torch.empty(0, dtype=torch.int32, device=q4.device)
B = gptq_marlin_repack(packed, empty, K, N, num_bits=4, is_a_8bit=True)   # int32 [K/16, 2N]
# (3) 스케일: [N,K/g] → [K/g,N] → scale_perm_single → int16 양자화
s = marlin_permute_scales(s_g.t().contiguous(), K, N, 128, is_a_8bit=True)  # bf16 [K/g,N]
s_q, input_global_scale = marlin_act_int8_process_scales(s)                  # int16 bits viewed as bf16, fp32 0-dim
layer.weight        = Parameter(B, requires_grad=False)
layer.weight_scale  = Parameter(s_q, requires_grad=False)
layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
layer.marlin_workspace = marlin_make_workspace(q4.device)   # marlin_utils.py:268-276, int32 zeros [sms]
```
- `marlin_utils.py:313-324 marlin_permute_scales`에 `is_a_8bit=False` 인자 추가: `if group_size < size_k and group_size != -1 and not is_a_8bit: scale_perm else: scale_perm_single` (main:470-488 동치). `get_scale_perms`(303-311)는 이미 동일.
- 신규 `marlin_act_int8_process_scales(s, q=4096)` (main:490-494 이식, q를 인자로 노출 — §8 오버플로 완화용):
  ```python
  smax = s.max(); factor = (smax.float() / q)
  s = (s / smax * q).round().to(torch.int16).view(s.dtype)
  return s, factor
  ```
  `view(s.dtype)`로 bf16 뷰를 유지해야 C++가 `c_type`을 `b_scales` dtype에서 얻는다(§4.1).
- 부호 확인: 로드 시 `assert (s_g > 0).all()` 대신 로그로 `negative_ratio`를 남기고, #48926 적용으로 음수도 정확히 처리(−4096..4096 ⊂ int16).
- 재현 검증용 Python 레퍼런스: vLLM `marlin_utils_test.py:33-52 marlin_permute_weights(is_a_8bit)` / `73-92 get_weight_perm(4, True)` / `54 marlin_weights`를 `python/sglang/test/test_marlin_utils.py`(42/57/76)에 `is_a_8bit` 인자로 이식.
- QQQ용 `qqq_pack_from_int4`(`marlin_qqq.py:100-125`)는 벤치 비교를 위해 남긴다(스킴에선 미사용).

---

## 6. 활성 양자화와 글로벌 스케일 폴딩

- 기존 `sglang.kernels.ops.quantization.int8_kernel.per_token_quant_int8(x, scale_dtype=fp32)`(59-81): absmax/127 대칭, 반환 `x_q int8 [M,K]`(contiguous, K%16==0 → stride 요건 충족), `scales fp32 [M,1]` — vLLM `per_token_quant_int8`와 수학 동일. 그대로 사용.
- `apply_weights`(208-230) 교체:
  ```python
  x_q, x_s = per_token_quant_int8(x_2d)                         # 커널 1
  a_scales = x_s.view(-1) * layer.input_global_scale            # 커널 2 (fp32 [M])
  out = gptq_marlin_gemm(x_q, None, layer.weight, layer.weight_scale, None, None, None, None,
                         layer.marlin_workspace, scalar_types.uint4b8, M, N, K,
                         is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
                         a_scales=a_scales, b_bias=None, c_tmp=_scratch("marlin_c_tmp", ...))
  if bias is not None: out = out + bias        # 1차. 2차에 marlin_permute_bias 캐시 후 b_bias로 융합
  ```
  출력 dtype = `weight_scale.dtype` = params_dtype = x.dtype → QQQ 경로의 `y.to(x.dtype)` 변환이 사라진다.
- 폴딩(2차): `per_token_quant_int8`의 Triton 커널에 `scale_mul: tl.constexpr`/런타임 스칼라 인자를 추가해 `scales = absmax/127 * scale_mul`로 커널 2를 제거. `input_global_scale`은 0‑dim fp32 텐서라 `.item()` 없이 포인터로 전달(그래프 안전).

---

## 7. 정확성 테스트 / 마이크로벤치 계획

### 7.1 단위 테스트 (`test/registered/kernels/ops/quantization/`)
1. **`test_gptq_marlin_repack.py`에 `is_a_8bit` 파라미터 추가**: `marlin_weights(q_w, K, N, 4, get_weight_perm(4, is_a_8bit), is_a_8bit)` 대비 `torch.testing.assert_close` **비트 정확**. K∈{32,128,5376}, N∈{64,256,5376}. (K=48 같은 16 mod 32는 #49862 적용으로 에러 기대.)
2. **신규 `test_gptq_marlin_w4a8_int8.py`** — 두 종류 레퍼런스:
   - **Ref‑A (커널 수학 정확 에뮬레이션, int64)**:
     `P_g[m,n] = Σ_{k∈g} a_q[m,k]·(w_u4[k,n]−8)` (int64) → `acc[m,n] = Σ_g P_g·s16[g,n]` (int64, s16 = round(s/s.max·4096)) → `out = acc.float() · (a_s[m]·s.max/4096)` → bf16. 허용치: `max|out−ker| ≤ 2·ulp_bf16(|ref|) + 1e-3·mean|ref|`(fp32 합산 순서 차이만 허용). 동시에 `acc.abs().max()`를 로그해 **2^31 대비 여유(headroom)** 를 기록.
   - **Ref‑B (참 수학)**: `(a_q·a_s) @ ((w_u4−8)·s_bf16)` → vLLM 허용치 `mean|out−ref|/mean|ref| < 0.04`. 이 값이 12‑bit 스케일 양자화 오차의 지표.
   - 매트릭스: Gemma4 4 shape(qkv 16384×5376, o 5376×8192, gate_up 43008×5376, down 5376×21504) × M∈{1,4,8,12,16,17,32,33,48,64,100,257,2048} × `use_fp32_reduce∈{T,F}` × c_dtype∈{bf16,fp16}; 소형 shape(64×128 등)로 세 thread cfg 모두 강제 노출(`prob_n` 선택으로 유도). 음수 스케일 케이스(50% 부호 반전) 1건.
3. **스킴 레벨**: `bench-170hx/qqq_scheme_test.py` 형식으로 `CompressedTensorsW4A8Int8` 신·구 경로를 같은 int4 가중치로 fp32 dequant 레퍼런스 대비 비교(각 경로의 오차, 경로 간 차이).
4. **회귀**: `test_gptq_marlin.py`(fp16/bf16 dense, act_order 포함), `test_nvfp4_marlin.py`(global_scale fp32 변경), `test_awq_marlin_repack.py`, `test_marlin_moe.py`(dtypes alias).
5. **E2E**: Gemma4 31B wikitext ppl + gsm8k 소표본, QQQ 경로 vs Marlin‑A8 경로 vs bf16(가능하면) — 스케일 양자화·오버플로의 실제 영향 판정.

### 7.2 마이크로벤치 (`bench-170hx/marlin_a8_bench.py`, `qqq_test.py` 골격 재사용)
- 대상: (a) Marlin‑A8 (b) `marlin_qqq_gemm` (c) `int8_scaled_mm`(상주 int8, unpack 비용 별도 열) (d) 참고용 W4A16 Marlin bf16.
- M ∈ {1,4,8,12,32,64,128,256,512,1024,2048}, 4 shape × 60층 합산 ms, weight GB/s, TOPS. 디코드 구간(M≤32)은 **CUDA graph replay**로도 측정(런치 오버헤드 포함).
- 양자화 커널 포함/미포함 두 열(per_token_quant + a_scales mul).
- 결정 규칙: M≤32에서 Marlin‑A8 ≤ QQQ 시간이면 QQQ 제거. M≥256에서 Marlin‑A8 ≥ 0.9×(int8_scaled_mm + 상각 unpack)이면 unpack+CUTLASS 경로 제거, 아니면 `QQQ_MAX_M`을 `MARLIN_A8_MAX_M`으로 이름만 바꿔 교차점 유지. **현재는 미지수** — GA100(170HX)에서 int8 mma+레지스터 dequant의 대 M 효율은 실측 전엔 판단 불가.

---

## 8. 리스크와 완화

| 리스크 | 정량/근거 | 완화 |
|---|---|---|
| **int32 누적 오버플로** (`matmul_a8`, main:1345-1359, 포화 없음) | 그룹당 ≤127·8·128=130,048; ×4096 → 5.3e8; K=21504는 168그룹 → 최악 9e10 ≈ 42×INT32_MAX. 실제 누적 범위는 **스레드블록 K‑슬라이스**(그리드에 따라 다름, 다중 슬라이스는 fp32 global reduce) | Ref‑A로 실제 Gemma4 활성에서 `max|acc|` 계측(down_proj 우선). 여유 < 8× 이면 `marlin_act_int8_process_scales(s, q)`의 q를 층별로 낮춤(4096→1024: 정밀도 2bit 손실, 오버플로 4× 완화; 커널 무변경). 최후 수단: down_proj만 CUTLASS 경로 유지. |
| **12‑bit 스케일 양자화** (층 전체 max 기준; `s < s.max/8192` 그룹은 0) | N=43008 gate_up에서 outlier 컬럼 하나가 전체 해상도를 깎음 | 로드 시 층별 `s.max/s.min`, 0으로 떨어진 그룹 수 로깅; Ref‑B 오차와 E2E ppl로 판정. 해결 불가 시 해당 층만 QQQ/CUTLASS. |
| **음수 스케일** (issue #48905) | 커널이 `uint16_t*`로 읽음 | #48926(`int16_t*`) 적용 + 음수 케이스 단위 테스트. |
| **repack K%32 미검사** (#49862) | Gemma4는 모두 32 배수 | 검사 추가로 fail‑closed. |
| **기존 fp16/bf16/nvfp4/MoE 경로 회귀** | 템플릿 통째 교체(새 persistent‑slice 스케줄러, bias 파라미터, fp32 global_scale) | 7.1‑4 회귀 테스트 필수. nvfp4 Python(`marlin_utils_fp4.py`)의 global_scale dtype 확인. |
| **JIT 컴파일 시간·캐시** | 단일 TU 암묵 인스턴스화. 헤더 변경으로 fp16/bf16 모듈도 1회 재빌드 | int8은 별도 모듈 키(`int8_t,bf16_t`)로 분리; 기본 12커널만, 전체 집합은 `SGLANG_MARLIN_A8_FULL` 매크로. 빌드 시간은 **실측 미지수**(예상 수 분). |
| **CUDA graph 안전성** | 핫패스 할당: `c`, `c_tmp`(4.6 MB fp32), `a_scales` | `c_tmp`는 `_scratch` 주입, `c`는 캡처 풀 할당(QQQ 경로와 동일하게 이미 허용). workspace lock은 커널이 스스로 리셋하므로 재사용 가능; 층별 70 int32. |
| **`k == 1` 그룹 접기 가정** (`b_sh_wr_iters == 2`) | 4개 thread cfg + 4bit 가중치에서만 성립 | 새 thread cfg 추가 금지(주석+`static_assert(b_sh_wr_iters == 2)`를 int8 분기에 추가). |
| **TP=2 N=2688** | `{64,256,256}` 불가 | `{64,128,128}` 자동 폴백, 성능 차이만 벤치로 확인. |
| **act_order / group 16 / m_block_size_8 / W8A8 미지원** | `static_assert`(main:1813), repack `static_assert(!is_a_8bit)` | 스킴 진입 조건: `group_size==128 && symmetric && no g_idx` (현 `_is_dynamic_token_w4a8_int`, `compressed_tensors.py:491`) 유지. |

---

## 9. 실행 순서 (각 단계가 컴파일·테스트 가능)

| 단계 | 작업 | 검증 |
|---|---|---|
| **S1** | `gptq_marlin_repack.cuh/.py`에 `is_a_8bit`(§2.8, §4.3) + `test_marlin_utils.py`에 `get_weight_perm(4, is_a_8bit)` 등 이식 | `test_gptq_marlin_repack.py[is_a_8bit=True]` 비트 정확. 독립 모듈이라 다른 파일에 무영향 |
| **S2** | `marlin.cuh`(cp_async helpers), `marlin_dtypes.cuh`(main + `ScalarType<T>` alias), `dequant.h`(main), `marlin_mma.h`(신규) | `test_marlin_moe.py` 통과(MoE만 이 헤더들을 쓰므로 dense 템플릿 교체 전에 호환성 확정) |
| **S3** | `kernel.h` 12‑파라미터, `marlin_template.h` main 교체(+#48926, sm75 제거), `gptq_marlin.cuh` 호스트 패치(§2.7) — **int8 계열은 아직 넣지 않고** 기존 계열만 새 `_GET_IF`로 래핑, 래퍼는 `<a_scalar_t, c_scalar_t>`로 확장하되 Python은 `(dtype, dtype)` 호출 | `test_gptq_marlin.py`, `test_nvfp4_marlin.py` 회귀 통과 = 템플릿 교체가 fp16/bf16 사용자에 무해 |
| **S4** | `A8_GET_IF` 12커널 추가, Python `gptq_marlin.py` 신 파라미터(§4.2), `marlin_utils.py`에 `marlin_permute_scales(is_a_8bit)`, `marlin_act_int8_process_scales` | 신규 `test_gptq_marlin_w4a8_int8.py` Ref‑A/Ref‑B, JIT 빌드 시간 기록 |
| **S5** | `compressed_tensors_w4a8_int8.py` 교체(§5, §6): 소 M 분기를 Marlin‑A8로, 대 M은 기존 unpack+CUTLASS 유지, `QQQ_MAX_M` → `MARLIN_A8_MAX_M` | 스킴 레벨 테스트, Gemma4 로드 + 짧은 생성 비교(QQQ vs A8) |
| **S6** | 벤치(§7.2) → 대 M 경로 존폐 결정; bias 융합(`marlin_permute_bias` 캐시 + `b_bias`), `per_token_quant_int8`에 global scale 폴딩 | 벤치 표, CUDA graph replay 측정 |
| **S7** | E2E ppl/gsm8k; 오버플로 headroom·스케일 분포 로그 검토 → q 조정 또는 층별 폴백 결정; `awq_marlin_repack` is_a_8bit(선택) | E2E 수치가 QQQ 대비 동등 이상이면 QQQ 커널·`qqq_pack_from_int4`를 스킴에서 제거(파일은 벤치용 보존) |

### 명시적 미지수
1. Gemma4 체크포인트 `weight_scale`의 부호·동적범위(층별 max/min 비) — S5에서 계측.
2. K=21504에서의 실제 int32 headroom — Ref‑A 계측 전까지 q=4096 유지 여부 미정.
3. GA100(CMP 170HX)에서 Marlin‑A8의 대 M 처리량 vs CUTLASS int8 — S6 실측.
4. 단일 TU JIT 빌드 시간(12 vs 192 커널) — S4 실측.
5. main 템플릿의 새 슬라이스 스케줄러가 기존 fp16/bf16/nvfp4 성능에 미치는 영향 — S3 회귀는 정확성만 보장, 성능은 별도 확인 필요.

### 참조 파일 (절대 경로)
- 포크: `/tmp/sglang-fork/python/sglang/kernels/jit/csrc/gemm/marlin/{kernel.h, marlin.cuh, marlin_dtypes.cuh, dequant.h, marlin_template.h, gptq_marlin.cuh, gptq_marlin_repack.cuh, awq_marlin_repack.cuh, marlin_qqq.cuh}`, `/tmp/sglang-fork/python/sglang/kernels/jit/csrc/gemm/marlin_moe/{kernel.h, marlin_template.h}`, `/tmp/sglang-fork/python/sglang/kernels/ops/quantization/{gptq_marlin.py, gptq_marlin_repack.py, marlin_qqq.py, int8_kernel.py}`, `/tmp/sglang-fork/python/sglang/srt/layers/quantization/marlin_utils.py`, `/tmp/sglang-fork/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8.py`, `/tmp/sglang-fork/python/sglang/test/test_marlin_utils.py`, `/tmp/sglang-fork/test/registered/kernels/ops/quantization/{test_gptq_marlin.py, test_gptq_marlin_repack.py}`, `/tmp/sglang-fork/bench-170hx/qqq_test.py`
- vLLM main: `/tmp/vllm/csrc/libtorch_stable/quantization/marlin/{marlin_template.h, marlin_mma.h, marlin_dtypes.cuh, dequant.h, marlin.cuh, kernel.h, gptq_marlin_repack.cu, marlin.cu, generate_kernels.py}`, `/tmp/vllm/vllm/model_executor/layers/quantization/utils/{marlin_utils.py, marlin_utils_test.py}`, `/tmp/vllm/tests/kernels/quantization/test_marlin_gemm.py`