"""W4A8-INT8 GPTQ-Marlin (vLLM PR #24722 port): int4 uint4b8 weights, group
scales quantised to int16, dynamic per-token int8 activations.

Ref-A emulates the kernel's integer arithmetic exactly (int64), Ref-B is the
true bf16 math; the kernel must match Ref-A to fp32-reduction noise and
Ref-B within the scale-quantisation error budget (vLLM uses 0.04).
"""

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.marlin_utils import (
    get_scale_perms,
    marlin_act_int8_process_scales,
    marlin_make_workspace,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import marlin_quantize

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

GROUP_SIZE = 128
# small shapes exercise every thread config; the last four are Gemma4 31B.
# K == group_size (single group) is not instantiated for int8 (group_blocks -1), so K >= 256.
SHAPES = [(256, 256), (512, 1024), (5376, 16384), (8192, 5376), (5376, 43008), (21504, 5376)]
MS = [1, 4, 8, 12, 16, 17, 32, 33, 48, 64, 100, 257, 2048]


def _pack_rows_uint4b8(q_w):
    """[K, N] int values in [0, 15] -> GPTQ int32 packed rows [K/8, N]."""
    K, N = q_w.shape
    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=q_w.device)
    for i in range(8):
        packed |= q_w[i::8, :].to(torch.int32) << (4 * i)
    return packed


def _run(size_k, size_n, size_m, dtype, use_fp32_reduce, negative_scales=False, group_size=GROUP_SIZE):
    device = "cuda"
    torch.manual_seed(size_k * 7 + size_n * 3 + size_m)
    b_weight = torch.randn((size_k, size_n), dtype=dtype, device=device) * 0.02
    w_ref, marlin_q_w, marlin_s, _, _, _ = marlin_quantize(
        b_weight, scalar_types.uint4b8, group_size, False, input_dtype=torch.int8
    )
    if negative_scales:
        # flip the sign of every other 32-column block; the int8 scale permutation
        # only reorders columns inside a 32-block, so the pattern is invariant
        sign = torch.where((torch.arange(size_n, device=device) // 32) % 2 == 0, 1.0, -1.0).to(dtype)
        marlin_s = marlin_s * sign
        w_ref = w_ref * sign
    x = torch.randn((size_m, size_k), dtype=dtype, device=device)
    a_q, a_s = per_token_quant_int8(x)  # int8 [M,K], fp32 [M,1]
    s_q, factor = marlin_act_int8_process_scales(marlin_s)
    a_scales = (a_s.reshape(-1) * factor).float()
    workspace = marlin_make_workspace(device)
    out = gptq_marlin_gemm(
        a_q, None, marlin_q_w, s_q, None, None, None, None, workspace, scalar_types.uint4b8,
        size_m, size_n, size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=use_fp32_reduce,
        a_scales=a_scales,
    )
    # Ref-B: true math with the dequantised activations
    a_ref = (a_q.float() * a_s.float()).to(dtype)
    ref_b = torch.matmul(a_ref.float(), w_ref.float())
    rel_b = (out.float() - ref_b).abs().mean() / ref_b.abs().mean()
    # Ref-A: exact kernel arithmetic. w_ref = (q - 8) * s where s = marlin_s (permuted),
    # kernel uses s16 = round(s / smax * 4096) and out = sum_g P_g * s16[g] * (a_s * smax / 4096)
    # marlin_s / s_q are in the kernel's permuted column order; undo it for the reference
    _, sps = get_scale_perms()
    inv = torch.argsort(torch.tensor(sps, device=device))
    unperm = lambda v: v.reshape(-1, len(sps))[:, inv].reshape(-1, size_n)
    s_orig = unperm(marlin_s).float()
    s16 = unperm(s_q.view(torch.int16)).float()  # exactly what the kernel reads
    q_signed = (w_ref.float() / s_orig.repeat_interleave(group_size, dim=0)).round()  # (q - 8), exact for our data
    assert q_signed.abs().max() <= 8
    acc = torch.zeros((size_m, size_n), dtype=torch.float64, device=device)
    for g in range(size_k // group_size):
        sl = slice(g * group_size, (g + 1) * group_size)
        p_g = a_q[:, sl].to(torch.float64) @ q_signed[sl, :].to(torch.float64)  # exact int
        acc += p_g * s16[g].to(torch.float64)[None, :]
    headroom = acc.abs().max().item() / 2**31
    ref_a = (acc * a_scales.to(torch.float64)[:, None]).to(torch.float32)
    diff_a = (out.float() - ref_a).abs()
    tol_a = 2 * torch.finfo(dtype).eps * ref_a.abs() + 2e-3 * ref_a.abs().mean()
    ok_a = bool((diff_a <= tol_a).float().mean() > 0.999)
    return rel_b.item(), ok_a, diff_a.max().item(), headroom


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("size_m", MS)
@pytest.mark.parametrize("use_fp32_reduce", [True, False])
def test_w4a8_int8_gemm(shape, size_m, use_fp32_reduce):
    size_k, size_n = shape
    if size_m > 64 and shape[0] * shape[1] > 20_000_000 and not use_fp32_reduce:
        pytest.skip("large shape: fp32-reduce variant only")
    rel_b, ok_a, max_a, headroom = _run(size_k, size_n, size_m, torch.bfloat16, use_fp32_reduce)
    print(f"K={size_k} N={size_n} M={size_m} fp32r={use_fp32_reduce}: relB={rel_b:.4f} maxA={max_a:.3e} acc/2^31={headroom:.3f}")
    assert headroom < 1.0, "int32 accumulation overflow"
    assert rel_b < 0.04, f"true-math error {rel_b}"
    # the bf16 k-split reduction (use_fp32_reduce=False) rounds partial sums, so
    # bit-exact agreement with the integer emulation only holds for fp32 reduce
    if use_fp32_reduce:
        assert ok_a, f"kernel deviates from exact integer emulation, max {max_a}"


def test_w4a8_int8_negative_scales():
    rel_b, ok_a, max_a, headroom = _run(512, 1024, 33, torch.bfloat16, True, negative_scales=True)
    assert rel_b < 0.04 and ok_a


def test_w4a8_int8_fp16_output():
    rel_b, ok_a, max_a, headroom = _run(512, 1024, 17, torch.float16, True)
    assert rel_b < 0.04 and ok_a


@pytest.mark.parametrize("shape", [(512, 1024), (5376, 16384), (21504, 5376)])
@pytest.mark.parametrize("size_m", [1, 12, 33, 64, 2048])
def test_w4a8_int8_gemm_group32(shape, size_m):
    """Google QAT w4a16-ct checkpoints use group_size 32 (group_blocks 2)."""
    size_k, size_n = shape
    rel_b, ok_a, max_a, headroom = _run(size_k, size_n, size_m, torch.bfloat16, True, group_size=32)
    print(f"g32 K={size_k} N={size_n} M={size_m}: relB={rel_b:.4f} maxA={max_a:.3e} acc/2^31={headroom:.3f}")
    assert headroom < 1.0, "int32 accumulation overflow"
    assert rel_b < 0.04, f"true-math error {rel_b}"
    assert ok_a, f"kernel deviates from exact integer emulation, max {max_a}"
