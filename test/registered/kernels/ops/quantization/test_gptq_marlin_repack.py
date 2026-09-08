import sys

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.utils import (
    gptq_quantize_weights,
    pack_rows,
    sort_weights,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import get_weight_perm, marlin_weights

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
    (257, 13, 11),
    (658, 13, 11),
]

# Shapes for the 8-bit-activation (32x32 tile) layout: cover the minimum
# K (one 32-row tile), the group size, and the Gemma4 hidden size 5376
# (= 21 * 256, i.e. 168 k-tiles / 168 n-tiles).
A8_SIZE_K = [32, 128, 5376]
A8_SIZE_N = [64, 256, 5376]


def _quantize_and_pack(size_k, size_n, quant_type, group_size, act_order):
    """Random GPTQ weight -> (q_w [K,N] unpacked, q_w_gptq [K/pack,N], sort_indices)."""
    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        b_weight, quant_type, group_size, act_order
    )

    q_w_gptq = pack_rows(q_w, quant_type.size_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    return q_w, q_w_gptq, sort_indices


@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type", [scalar_types.uint4b8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_repack(
    k_chunk, n_chunk, quant_type, group_size, act_order, mnk_factors
):
    m_factor, n_factor, k_factor = mnk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if size_k % group_size != 0:
        pytest.skip("size_k must be divisible by group_size")

    q_w, q_w_gptq, sort_indices = _quantize_and_pack(
        size_k, size_n, quant_type, group_size, act_order
    )

    marlin_layout_perm = get_weight_perm(quant_type.size_bits)
    q_w_marlin_ref = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, marlin_layout_perm
    )

    # Run JIT repack kernel
    jit_output = gptq_marlin_repack(
        q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits
    )

    torch.cuda.synchronize()

    # JIT should match the reference (computed from CPU marlin_weights)
    torch.testing.assert_close(jit_output, q_w_marlin_ref)


@pytest.mark.parametrize("size_k", A8_SIZE_K)
@pytest.mark.parametrize("size_n", A8_SIZE_N)
@pytest.mark.parametrize("quant_type", [scalar_types.uint4b8, scalar_types.uint8b128])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("is_a_8bit", [False, True])
def test_gptq_marlin_repack_a8(size_k, size_n, quant_type, group_size, is_a_8bit):
    """Bit-exact check of the repack kernel against the Python reference for
    both the fp16-activation (16x64) and 8-bit-activation (32x32) tile layouts.
    act_order is not supported with is_a_8bit, so only the non-perm path is
    exercised here (the perm path is covered by test_gptq_marlin_repack)."""
    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    if size_k % group_size != 0:
        pytest.skip("size_k must be divisible by group_size")

    q_w, q_w_gptq, sort_indices = _quantize_and_pack(
        size_k, size_n, quant_type, group_size, act_order=False
    )

    marlin_layout_perm = get_weight_perm(quant_type.size_bits, is_a_8bit)
    q_w_marlin_ref = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, marlin_layout_perm, is_a_8bit
    )

    jit_output = gptq_marlin_repack(
        q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits, is_a_8bit
    )
    torch.cuda.synchronize()

    # Output shape is identical in both layouts
    assert jit_output.shape == q_w_marlin_ref.shape
    torch.testing.assert_close(jit_output, q_w_marlin_ref, rtol=0, atol=0)


def test_gptq_marlin_repack_a8_rejects_k_not_multiple_of_32():
    """is_a_8bit uses 32-row k tiles: K = 48 (16 mod 32) must be rejected
    (vLLM #49862) instead of silently repacking a truncated tensor."""
    quant_type = scalar_types.uint4b8
    size_k, size_n = 48, 64
    q_w, q_w_gptq, sort_indices = _quantize_and_pack(
        size_k, size_n, quant_type, size_k, act_order=False
    )
    # Sanity: the fp16-activation layout accepts K = 48
    gptq_marlin_repack(q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits)
    with pytest.raises(Exception, match="not divisible"):
        gptq_marlin_repack(
            q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits, True
        )


def test_gptq_marlin_repack_a8_rejects_act_order():
    """act_order (non-empty perm) is not supported with is_a_8bit."""
    quant_type = scalar_types.uint4b8
    size_k, size_n = 128, 64
    q_w, q_w_gptq, sort_indices = _quantize_and_pack(
        size_k, size_n, quant_type, 32, act_order=True
    )
    assert sort_indices.numel() == size_k
    with pytest.raises(Exception, match="act_order"):
        gptq_marlin_repack(
            q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits, True
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
