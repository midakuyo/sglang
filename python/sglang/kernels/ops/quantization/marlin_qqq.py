"""Marlin-QQQ W4A8 GEMM (int4 group/channel-scaled weights x int8 per-token
activations, int8 tensor cores; sm80+). JIT-built from
kernels/jit/csrc/gemm/marlin/marlin_qqq.cuh (port of vLLM v0.9.2 / HandH1998 QQQ).

Weight format (see qqq_pack_from_int4): weights are stored as unsigned int4
(q + 8) in Marlin tile order; the kernel re-quantises each int4 to int8 with
the fp16 per-group scale ratio s_group = s_g / s_channel and applies
s_channel (fp32, per output channel) and s_tok (fp32, per token) in the
epilogue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GROUP_SIZE = 128
TILE = 16
MIN_THREAD_N = 64
MAX_PAR = 16


@cache_once
def _jit_marlin_qqq_module() -> Module:
    return load_jit(
        "marlin_qqq",
        cuda_files=["gemm/marlin/marlin_qqq.cuh"],
        cuda_wrappers=[("marlin_qqq_gemm", "marlin_qqq_gemm")],
    )


def marlin_qqq_workspace(size_n: int, device) -> torch.Tensor:
    return torch.zeros((size_n // MIN_THREAD_N) * MAX_PAR, dtype=torch.int32, device=device)


def marlin_qqq_gemm(
    a_q: torch.Tensor,  # int8 [M, K]
    b_q_weight: torch.Tensor,  # int32 [K/16, 2N]
    s_tok: torch.Tensor,  # fp32 [M] (or [M, 1])
    s_ch: torch.Tensor,  # fp32 [1, N]
    s_group: torch.Tensor,  # fp16 [K/128, N] (numel 0 => per-channel)
    workspace: torch.Tensor,  # int32 zeros [>= N/64*16]
    size_m: int,
    size_n: int,
    size_k: int,
    c_tmp: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = a_q.device
    if c_tmp is None:
        c_tmp = torch.empty((MAX_PAR * 64, size_n), dtype=torch.int32, device=device)
    if out is None:
        out = torch.empty((size_m, size_n), dtype=torch.float16, device=device)
    _jit_marlin_qqq_module().marlin_qqq_gemm(
        a_q, b_q_weight, s_tok.reshape(-1).contiguous(), s_ch, s_group, workspace, c_tmp, out,
        size_m, size_n, size_k,
    )
    return out


# ---- packing (torch port of vLLM marlin_utils_test_qqq.py) ----------------

def _qqq_weight_perm(num_bits: int = 4, per_group: bool = True) -> torch.Tensor:
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (4 * (i % 4), 4 * (i % 4) + 1, 4 * (i % 4) + 2, 4 * (i % 4) + 3):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = torch.tensor(perm_list, dtype=torch.int64)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7] if per_group else [4, 0, 5, 1, 6, 2, 7, 3])
    return perm.reshape(-1, 8)[:, interleave].reshape(-1)


def _qqq_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in (0, 1, 8, 9, 16, 17, 24, 25)])
    return torch.tensor(scale_perm), torch.tensor(scale_perm_single)


def _marlin_permute_weights(q_w: torch.Tensor, size_k: int, size_n: int, perm: torch.Tensor) -> torch.Tensor:
    q_w = q_w.reshape(size_k // TILE, TILE, size_n // TILE, TILE).permute(0, 2, 1, 3)
    q_w = q_w.reshape(size_k // TILE, size_n * TILE)
    return q_w.reshape(-1, perm.numel())[:, perm.to(q_w.device)].reshape(q_w.shape)


def qqq_pack_from_int4(q4: torch.Tensor, s_g: torch.Tensor, group_size: int = GROUP_SIZE):
    """q4: int8 [N, K] signed int4 values in [-8, 7]; s_g: [N, K // group_size]
    per-group scales (w = q4 * s_g). Returns QQQ tensors
    (b_q_weight int32 [K/16, 2N], s_channel fp32 [1, N], s_group fp16 [K/g, N])
    and the effective int8 weights the kernel will use ([K, N] int8, for tests)."""
    N, K = q4.shape
    assert K % group_size == 0 and K % TILE == 0 and N % MIN_THREAD_N == 0
    q4t = q4.t().contiguous()  # [K, N]
    s_gt = s_g.t().float().contiguous()  # [K/g, N]
    w_ref = q4t.float() * s_gt.repeat_interleave(group_size, dim=0)
    s_channel = (w_ref.abs().amax(dim=0, keepdim=True) / 127.0).clamp_min(1e-8)  # [1, N] fp32
    s_group = (s_gt / s_channel).to(torch.float16)  # [K/g, N]
    # int8 the kernel materialises: round(q4 * s_group) in fp16
    q8 = torch.round((q4t.half() * s_group.repeat_interleave(group_size, dim=0)).float()).clamp(-128, 127).to(torch.int8)
    q_w = (q4t.to(torch.int32) + 8)  # unsigned int4 [K, N]
    q_w = _marlin_permute_weights(q_w, K, N, _qqq_weight_perm(4, True))
    packed = torch.zeros((q_w.shape[0], q_w.shape[1] // 8), dtype=torch.int32, device=q_w.device)
    for i in range(8):
        packed |= q_w[:, i::8] << (4 * i)
    scale_perm, scale_perm_single = _qqq_scale_perms()
    s_group_p = s_group.reshape(-1, scale_perm.numel())[:, scale_perm.to(s_group.device)].reshape(-1, N).contiguous()
    s_channel_p = s_channel.reshape(-1, scale_perm_single.numel())[:, scale_perm_single.to(s_channel.device)].reshape(1, N).contiguous()
    return packed, s_channel_p, s_group_p, q8, s_channel, s_group


# ---- unpack back to per-channel int8 (for the M-large / prefill path) --------

@cache_once
def _qqq_inverse_index(device_str: str) -> torch.Tensor:
    """For each of the 1024 logical positions of a permuted block, the (int32
    column, nibble) it was packed into: idx = col * 8 + nibble."""
    perm = _qqq_weight_perm(4, True)  # logical position p -> source index perm[p]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())  # source index s -> position inv[s] in the permuted block
    # packed int32 column j holds positions j*8 + i (i = nibble)
    return inv.to(device_str)


def qqq_unpack_to_int4(b_q_weight: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """Inverse of the packing in qqq_pack_from_int4: returns signed int4 values
    as int8 [K, N] in natural order."""
    rows = size_k // TILE
    # nibbles in packed order: position p = j*8 + i  -> [rows, size_n*16]
    nib = torch.stack([(b_q_weight >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(rows, size_n * TILE)
    inv = _qqq_inverse_index(str(b_q_weight.device))
    # nib[:, inv[s]] recovers source order s for every 1024-block
    q_w = nib.reshape(-1, inv.numel())[:, inv].reshape(rows, size_n * TILE)
    # undo the 16x16 tile layout
    q_w = q_w.reshape(rows, size_n // TILE, TILE, TILE).permute(0, 2, 1, 3).reshape(size_k, size_n)
    return (q_w - 8).to(torch.int8)


def qqq_unpack_to_int8(b_q_weight: torch.Tensor, s_group: torch.Tensor, size_k: int, size_n: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """The exact int8 [N, K] weights the QQQ kernel materialises (usable with
    int8_scaled_mm together with the per-channel scale). s_group: fp16
    [K/g, N] in natural (unpermuted) order."""
    q4 = qqq_unpack_to_int4(b_q_weight, size_k, size_n)  # [K, N]
    q8 = torch.round((q4.half() * s_group.repeat_interleave(group_size, dim=0)).float()).clamp(-128, 127)
    return q8.to(torch.int8).t().contiguous()  # [N, K]


# ---- Triton unpack (int4 QQQ tiles -> per-channel int8 [N, K]) ---------------

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _qqq_unpack_kernel(
    b_ptr,  # int32 [K/16, 2N]
    inv_ptr,  # int32 [1024]  logical position -> packed position within a block
    s_ptr,  # fp16 [K/G, N]
    out_ptr,  # int8 [N, K]
    N,
    K,
    stride_bk,  # = 2N
    stride_sk,  # = N
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    n = offs_n[:, None]
    k = offs_k[None, :]
    r = k // 16
    kk = k % 16
    ntile = n // 16
    nn = n % 16
    blk = ntile // 4
    s = (ntile % 4) * 256 + kk * 16 + nn  # logical position inside the 1024 block
    t = tl.load(inv_ptr + s)  # packed position
    col = blk * 128 + t // 8
    nib = t % 8
    packed = tl.load(b_ptr + r * stride_bk + col, mask=mask, other=0)
    q4 = ((packed >> (nib * 4)) & 0xF) - 8
    sc = tl.load(s_ptr + (k // GROUP) * stride_sk + n, mask=mask, other=0.0)
    q8 = libdevice.rint((q4.to(tl.float16) * sc).to(tl.float32))
    q8 = tl.minimum(tl.maximum(q8, -128.0), 127.0)
    tl.store(out_ptr + n * K + k, q8.to(tl.int8), mask=mask)


def qqq_unpack_to_int8_gather(
    b_q_weight: torch.Tensor,
    s_group: torch.Tensor,
    size_k: int,
    size_n: int,
    out: Optional[torch.Tensor] = None,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Same result as qqq_unpack_to_int8 (int8 [N, K]), in one gather kernel."""
    if out is None:
        out = torch.empty((size_n, size_k), dtype=torch.int8, device=b_q_weight.device)
    inv = _qqq_inverse_index(str(b_q_weight.device)).to(torch.int32)
    grid = (triton.cdiv(size_n, 64), triton.cdiv(size_k, 128))
    _qqq_unpack_kernel[grid](
        b_q_weight, inv, s_group, out, size_n, size_k, b_q_weight.stride(0), s_group.stride(0),
        GROUP=group_size, BLOCK_N=64, BLOCK_K=128, num_warps=4,
    )
    return out


# ---- fast unpack: the Marlin tile permutation is structured, so one program
# can load a whole (group of 8 k-tiles) x (1024-element block) coalesced,
# expand nibbles, permute in registers/smem and store a [64 n, 128 k] tile.
# Measured layout (see bench-170hx/perm_probe.py): with j = nt*32 + i_hh*16 +
# i_hl*4 + i_lo the int32 column and p the nibble index,
#   k = r*16 + i_hl*4 + p0*2 + p2,   n = i_lo*16 + p1*8 + nt*2 + i_hh.
# 4x faster than the element gather above (CMP 170HX: 74 vs 296 ms per
# 60-layer Gemma4 forward), bit-identical.


@triton.jit
def _qqq_unpack_fast_kernel(b_ptr, s_ptr, out_ptr, N, K, stride_bk, stride_sk, GROUP: tl.constexpr):
    g = tl.program_id(0)      # group index along K (8 k-tiles)
    blk = tl.program_id(1)    # 1024-element block along N (64 columns)
    offs_r = tl.arange(0, 8)          # k-tile within group
    offs_j = tl.arange(0, 128)        # int32 within block
    ptrs = b_ptr + (g * 8 + offs_r)[:, None] * stride_bk + blk * 128 + offs_j[None, :]
    w = tl.load(ptrs)                 # int32 [8, 128], coalesced
    # nibbles p = 0..7 -> last axis via joins: order x0..x7
    n0 = (w >> 0) & 0xF; n1 = (w >> 4) & 0xF; n2 = (w >> 8) & 0xF; n3 = (w >> 12) & 0xF
    n4 = (w >> 16) & 0xF; n5 = (w >> 20) & 0xF; n6 = (w >> 24) & 0xF; n7 = (w >> 28) & 0xF
    v = tl.join(tl.join(tl.join(n0, n1), tl.join(n2, n3)), tl.join(tl.join(n4, n5), tl.join(n6, n7)))  # [8,128,2,2,2]
    # measured layout: j = nt*32 + i_hh*16 + i_hl*4 + i_lo, p = p2*4 + p1*2 + p0
    #   k = r*16 + i_hl*4 + p0*2 + p2,   n = i_lo*16 + p1*8 + nt*2 + i_hh
    # nested tl.join appends the new axis last, so the joined nibble axes come
    # out as (bit0, bit1, bit2) of the nibble index p
    v = tl.reshape(v, (8, 4, 2, 4, 4, 2, 2, 2))  # r, nt, i_hh, i_hl, i_lo, p0, p1, p2
    v = tl.permute(v, (4, 6, 1, 2, 0, 3, 5, 7))  # i_lo, p1, nt, i_hh | r, i_hl, p0, p2
    v = tl.reshape(v, (64, 128))
    offs_n = blk * 64 + tl.arange(0, 64)
    offs_k = g * GROUP + tl.arange(0, 128)
    s = tl.load(s_ptr + g * stride_sk + offs_n)  # fp16 [64]
    q8 = libdevice.rint(((v - 8).to(tl.float16) * s[:, None]).to(tl.float32))
    q8 = tl.minimum(tl.maximum(q8, -128.0), 127.0)
    tl.store(out_ptr + offs_n[:, None] * K + offs_k[None, :], q8.to(tl.int8))

def _qqq_unpack_fast(B, s_group, K, N, out=None):
    if out is None: out = torch.empty((N, K), dtype=torch.int8, device=B.device)
    _qqq_unpack_fast_kernel[(K // 128, N // 64)](B, s_group, out, N, K, B.stride(0), s_group.stride(0), GROUP=128, num_warps=4)
    return out



def qqq_unpack_to_int8_triton(
    b_q_weight: torch.Tensor,
    s_group: torch.Tensor,
    size_k: int,
    size_n: int,
    out: Optional[torch.Tensor] = None,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """int8 [N, K] weights the QQQ kernel materialises (structured fast path;
    falls back to the gather kernel for shapes the tile path cannot cover)."""
    if group_size == GROUP_SIZE and size_k % 128 == 0 and size_n % 64 == 0:
        return _qqq_unpack_fast(b_q_weight, s_group, size_k, size_n, out=out)
    return qqq_unpack_to_int8_gather(b_q_weight, s_group, size_k, size_n, out=out, group_size=group_size)
