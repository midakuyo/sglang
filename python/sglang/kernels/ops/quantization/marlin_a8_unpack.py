"""Unpack Marlin W4A8-INT8 (is_a_8bit) repacked weights back to per-channel int8
[N, K] for the CUTLASS int8 GEMM (large-M / prefill path of the hybrid W4A8
scheme).

Layout of the repacked tensor B (int32 [K/16, 2N], see
test_marlin_utils.marlin_permute_weights / get_weight_perm with is_a_8bit):
the matrix is cut into chunks of 32 k-rows x 32 n-cols (1024 nibbles = 128
int32 words, contiguous), chunk index c = (k // 32) * (N // 32) + n // 32.
Inside a chunk, source element (row = k % 32, nn = n % 32) lives in word

    q = 4 * (4 * col + (row % 16) // 4) + 2 * (nn // 16) + (nn % 16) // 8
    (col = nn % 8)

at nibble e = 2 * (row % 4) + row // 16 (bits 4e..4e+3), stored as uint4b8
(value + 8). Verified against the Python reference packer
(bench-170hx/a8_layout_check.py).

The int8 weights follow the QQQ convention: q8 = round(q4 * s_group[g, n]) with
s_group = s_g / s_channel and s_channel[n] = max_k |w[k, n]| / 127, so
w ~= s_channel[n] * q8 and the GEMM epilogue applies s_channel per output
channel and the per-token activation scale per row.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _a8_unpack_kernel(b_ptr, s_ptr, out_ptr, N, K, stride_sk, GROUP: tl.constexpr, NB: tl.constexpr):
    kb = tl.program_id(0)  # 32-row k block
    nb = tl.program_id(1)  # NB chunks of 32 columns
    row = tl.arange(0, 32)  # k within block
    nn = tl.arange(0, 32 * NB)  # n within this program's span
    n = nb * (32 * NB) + nn
    k = kb * 32 + row
    m = n // 32
    c16 = (nn % 32) % 16
    col = c16 % 8
    block = c16 // 8
    j = (nn % 32) // 16
    i = 4 * col[None, :] + (row[:, None] % 16) // 4  # [32, 32NB]
    q = 4 * i + 2 * j[None, :] + block[None, :]
    e = 2 * (row[:, None] % 4) + row[:, None] // 16
    chunk = kb * (N // 32) + m[None, :]
    word = 128 * chunk + q
    w = tl.load(b_ptr + word)  # int32 gather, all inside NB*512 B of contiguous chunks (L1/L2)
    v = (w >> (4 * e)) & 0xF
    g = k // GROUP  # [32] group index per k row
    s = tl.load(s_ptr + g[:, None] * stride_sk + n[None, :])  # fp16 [32, 32NB]
    q8 = libdevice.rint(((v - 8).to(tl.float16) * s).to(tl.float32))
    q8 = tl.minimum(tl.maximum(q8, -128.0), 127.0)
    tl.store(out_ptr + n[None, :] * K + k[:, None], q8.to(tl.int8))


def marlin_a8_unpack_to_int8(
    b_q_weight: torch.Tensor,
    s_group: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """b_q_weight: Marlin-A8 repacked int32 [K/16, 2N]; s_group: fp16 [K/g, N]
    (plain order, = s_g / s_channel). Returns int8 [N, K]."""
    assert size_k % 32 == 0 and size_n % 64 == 0 and size_k % group_size == 0
    assert b_q_weight.is_contiguous() and s_group.is_contiguous() and s_group.dtype == torch.float16
    if out is None:
        out = torch.empty((size_n, size_k), dtype=torch.int8, device=b_q_weight.device)
    NB = 2  # 64 columns per program
    _a8_unpack_kernel[(size_k // 32, size_n // (32 * NB))](
        b_q_weight, s_group, out, size_n, size_k, s_group.stride(0), GROUP=group_size, NB=NB, num_warps=4
    )
    return out


def a8_channel_scales(q4_kn: torch.Tensor, s_g_kn: torch.Tensor, group_size: int):
    """q4_kn: int8/int32 [K, N] signed int4 values; s_g_kn: [K/g, N] group scales.
    Returns (s_channel fp32 [N], s_group fp16 [K/g, N]) in the QQQ convention."""
    K, N = q4_kn.shape
    w_absmax = torch.zeros(N, dtype=torch.float32, device=q4_kn.device)
    step = 4096  # bound temporaries (q4 * s can be 231M elements)
    s_f = s_g_kn.float()
    for k0 in range(0, K, step):
        k1 = min(K, k0 + step)
        blk = q4_kn[k0:k1].float() * s_f[k0 // group_size:(k1 + group_size - 1) // group_size].repeat_interleave(group_size, dim=0)[: k1 - k0]
        w_absmax = torch.maximum(w_absmax, blk.abs().amax(dim=0))
    s_channel = (w_absmax / 127.0).clamp_min(1e-8)
    s_group = (s_f / s_channel[None, :]).to(torch.float16).contiguous()
    return s_channel.contiguous(), s_group
