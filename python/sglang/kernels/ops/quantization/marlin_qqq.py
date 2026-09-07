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
