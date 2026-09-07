"""W4A8 (int4 weights, int8 per-token activations) GEMM in Triton.

Target: Ampere-class GPUs without a CUTLASS/Machete W4A8 path. Weights are
symmetric int4 with a per-group (along K) scale, activations are int8 with a
per-token scale (as produced by per_token_quant_int8).

    y[m, n] = a_s[m] * sum_g  w_s[g, n] * sum_{k in g} a_q[m, k] * w_q[k, n]

Each K group is one iteration of the main loop: an int8 tensor-core dot gives
the exact int32 group partial sum, which is then scaled in fp32 and
accumulated. To avoid nibble interleaving in the kernel the packer stores,
for every group, the low nibble of byte j as k = j and the high nibble as
k = j + GROUP/2, so a group is two [BLOCK_M, GROUP/2] x [GROUP/2, BLOCK_N]
dots on the two nibble planes.

Small M (decode / MTP verify) is memory bound and needs far more programs in
flight than (M/BLOCK_M) x (N/BLOCK_N) offers, so the K groups are split across
SPLIT_K programs that write fp32 partials; a second tiny kernel sums them and
applies the per-token scale and bias.

Packed layout (both K-major so a [k, n] tile is contiguous along n):
    w_packed: uint8 [K // 2, N]
    w_scale:  fp32  [K // GROUP, N]
"""

import torch
import triton
import triton.language as tl

GROUP = 128


def pack_w4_int8(w_q: torch.Tensor, w_scale: torch.Tensor, group: int = GROUP):
    """w_q: int8 [N, K] holding int4 values in [-8, 7]; w_scale: [N, K // group].
    Returns (w_packed uint8 [K // 2, N], w_scale fp32 [K // group, N])."""
    N, K = w_q.shape
    assert K % group == 0 and group % 2 == 0
    half = group // 2
    q = w_q.view(N, K // group, group)
    lo = q[:, :, :half]  # k = g*group + j
    hi = q[:, :, half:]  # k = g*group + half + j
    packed = (lo.to(torch.uint8) & 0xF) | ((hi.to(torch.uint8) & 0xF) << 4)
    packed = packed.reshape(N, K // 2).t().contiguous()  # [K/2, N]
    scale = w_scale.to(torch.float32).t().contiguous()  # [K/group, N]
    return packed, scale


@triton.jit
def _w4a8_gemm_kernel(
    a_ptr,  # int8 [M, K]
    w_ptr,  # uint8 [K // 2, N]
    w_scale_ptr,  # fp32 [K // GROUP, N]
    part_ptr,  # fp32 [SPLIT_K, M, N]  (unscaled partial sums)
    M,
    N,
    G,  # number of K groups
    groups_per_split,
    stride_am,
    stride_wk,
    stride_sk,
    stride_ps,
    stride_pm,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    HALF: tl.constexpr = GROUP // 2
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, HALF)
    mask_m = offs_m < M
    mask_n = offs_n < N

    g0 = pid_k * groups_per_split
    g1 = tl.minimum(g0 + groups_per_split, G)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    a_lo_ptrs = a_ptr + offs_m[:, None] * stride_am + g0 * GROUP + offs_j[None, :]
    a_hi_ptrs = a_lo_ptrs + HALF
    w_ptrs = w_ptr + (g0 * HALF + offs_j[:, None]) * stride_wk + offs_n[None, :]
    s_ptrs = w_scale_ptr + g0 * stride_sk + offs_n

    for g in range(g0, g1):
        a_lo = tl.load(a_lo_ptrs, mask=mask_m[:, None], other=0)
        a_hi = tl.load(a_hi_ptrs, mask=mask_m[:, None], other=0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
        # low nibble: shift up then arithmetic shift down sign-extends; high
        # nibble: arithmetic shift of the byte reinterpreted as int8.
        lo = (w << 4).to(tl.int8, bitcast=True) >> 4
        hi = w.to(tl.int8, bitcast=True) >> 4
        part = tl.dot(a_lo, lo) + tl.dot(a_hi, hi)  # int32 [BLOCK_M, BLOCK_N]
        s = tl.load(s_ptrs, mask=mask_n, other=0.0)
        acc += part.to(tl.float32) * s[None, :]
        a_lo_ptrs += GROUP
        a_hi_ptrs += GROUP
        w_ptrs += HALF * stride_wk
        s_ptrs += stride_sk

    p_ptrs = part_ptr + pid_k * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :]
    tl.store(p_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _w4a8_reduce_kernel(
    part_ptr,  # fp32 [SPLIT_K, M, N]
    a_scale_ptr,  # fp32 [M]
    bias_ptr,
    c_ptr,  # [M, N]
    N,
    stride_ps,
    stride_pm,
    stride_cm,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + s * stride_ps + m * stride_pm + offs_n, mask=mask_n, other=0.0)
    out = acc * tl.load(a_scale_ptr + m)
    if HAS_BIAS:
        out += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(c_ptr + m * stride_cm + offs_n, out.to(c_ptr.dtype.element_ty), mask=mask_n)


def _config(M: int, N: int, G: int):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages, SPLIT_K)."""
    if M <= 16:
        bm, bn, nw, ns = 16, 128, 4, 4
    elif M <= 64:
        bm, bn, nw, ns = 64, 128, 4, 3
    else:
        bm, bn, nw, ns = 128, 128, 8, 3
    programs = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    split = max(1, min(G, -(-432 // programs)))  # aim for >= ~4 programs per SM
    return bm, bn, nw, ns, split


def w4a8_int_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    group: int = GROUP,
    config=None,
) -> torch.Tensor:
    M, K = a_q.shape
    K2, N = w_packed.shape
    assert K2 * 2 == K and K % group == 0
    assert a_q.dtype == torch.int8 and w_packed.dtype == torch.uint8
    G = K // group
    a_scale = a_scale.reshape(M).to(torch.float32)
    bm, bn, nw, ns, split = config or _config(M, N, G)
    gps = -(-G // split)
    split = -(-G // gps)  # drop empty splits
    part = torch.empty((split, M, N), dtype=torch.float32, device=a_q.device)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn), split)
    _w4a8_gemm_kernel[grid](
        a_q, w_packed, w_scale, part,
        M, N, G, gps,
        a_q.stride(0), w_packed.stride(0), w_scale.stride(0), part.stride(0), part.stride(1),
        GROUP=group, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns,
    )
    c = torch.empty((M, N), dtype=out_dtype, device=a_q.device)
    RB = 1024
    _w4a8_reduce_kernel[(M, triton.cdiv(N, RB))](
        part, a_scale, bias if bias is not None else c, c, N,
        part.stride(0), part.stride(1), c.stride(0),
        SPLIT_K=split, HAS_BIAS=bias is not None, BLOCK_N=RB, num_warps=4,
    )
    return c
