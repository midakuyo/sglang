"""compressed-tensors W4A8-INT8: int4 group-quantised weights (symmetric,
per-group scale along K) with dynamic per-token int8 activations, served on
Ampere with a Marlin kernel. Two kernel families, selected with
``SGLANG_W4A8_KERNEL``:

* ``marlin`` (default): the vLLM Marlin W4A8-INT8 kernel (PR #24722). The
  int4 weights stay exact (int4 -> int8 by shifting), group scales are
  quantised to a 12-bit integer grid per layer and folded per group into the
  int32 accumulator; the layer-wide factor goes into the per-token activation
  scale. One kernel for every M (decode, MTP verify and prefill).
* ``qqq``: Marlin-QQQ two-level format (per-channel fp32 scale + fp16 group
  ratio, q8 = round(q4 * ratio) materialised on the fly). Small M runs the QQQ
  kernel; large M (prefill) unpacks the same q8 into a shared scratch and runs
  the CUTLASS int8_scaled_mm.
"""

import os
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Parameter

from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.kernels.ops.quantization.marlin_qqq import (
    MAX_PAR,
    marlin_qqq_gemm,
    marlin_qqq_workspace,
    qqq_pack_from_int4,
    qqq_unpack_to_int8_triton,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.utils import is_cuda

__all__ = ["CompressedTensorsW4A8Int8"]

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import int8_scaled_mm
    from sgl_kernel.scalar_type import scalar_types

# "marlin": vLLM Marlin W4A8-INT8 for all M. "qqq": Marlin-QQQ + unpack/CUTLASS.
W4A8_KERNEL = os.environ.get("SGLANG_W4A8_KERNEL", "marlin")
assert W4A8_KERNEL in ("marlin", "qqq"), W4A8_KERNEL
# fp32 reduce scratch of the Marlin kernel: sms * max_m_block(64) * max_thread_n(256)
_MARLIN_C_TMP_PER_SM = 64 * 256
# Keep a resident per-channel int8 copy of every weight (+28.7 GB for a 31B
# model) so the large-M path skips the unpack kernel entirely.
KEEP_INT8 = os.environ.get("SGLANG_W4A8_KEEP_INT8", "0") == "1"
# M above which the CUTLASS int8 path is used. With a resident int8 copy the
# crossover is the kernel crossover (CMP 170HX: QQQ wins up to M~24, loses
# from M~48). Without it the unpack costs ~300 ms per 60-layer forward, which
# the QQQ kernel's large-M penalty (~0.2 ms/token) only exceeds past ~1.5k
# tokens, so only genuinely large prefills should pay for the unpack.
QQQ_MAX_M = int(os.environ.get("SGLANG_W4A8_QQQ_MAX_M", "32" if KEEP_INT8 else "1024"))

_SCRATCH: Dict[Tuple[str, str], torch.Tensor] = {}


def pack_rows_uint4(w_u: torch.Tensor) -> torch.Tensor:
    """GPTQ row packing on the GPU: [K, N] uint4 values (int32) -> int32 [K/8, N],
    8 consecutive K rows per word, row i in bits 4i..4i+3."""
    K, N = w_u.shape
    assert K % 8 == 0
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=w_u.device)
    for i in range(8):
        packed |= w_u[i::8, :] << (4 * i)
    return packed


def _scratch(key: str, shape, dtype, device) -> torch.Tensor:
    k = (key, str(device))
    t = _SCRATCH.get(k)
    numel = 1
    for s in shape:
        numel *= s
    if t is None or t.numel() < numel:
        t = torch.empty(numel, dtype=dtype, device=device)
        _SCRATCH[k] = t
    return t[:numel].view(*shape)


class _UnpackPrefetcher:
    """Overlaps the int4->int8 unpack of the *next* linear with the CUTLASS
    GEMM of the current one on a side stream (prefill is compute bound, the
    unpack is memory bound). The call order of W4A8 layers is recorded during
    the first large-M forward and reused afterwards; two scratch buffers
    alternate so the side stream never writes the buffer the main stream is
    reading. Only used outside CUDA graphs (prefill)."""

    def __init__(self):
        self.stream: Optional[torch.cuda.Stream] = None
        self.order: list = []  # layer objects in call order (first forward)
        self.recording = True
        self.pos = -1
        self.bufs = [None, None]
        self.cur = 0
        self.ready: Dict[int, Tuple[torch.Tensor, torch.cuda.Event]] = {}
        self.main_done: Optional[torch.cuda.Event] = None

    def _buf(self, i, N, K, device):
        need = N * K
        if self.bufs[i] is None or self.bufs[i].numel() < need:
            self.bufs[i] = torch.empty(need, dtype=torch.int8, device=device)
        return self.bufs[i][:need].view(N, K)

    def get(self, layer, device) -> torch.Tensor:
        N, K = layer.qqq_size_n, layer.qqq_size_k
        main = torch.cuda.current_stream(device)
        if self.recording:
            if self.order and layer is self.order[0]:
                self.recording = False  # a second forward started: order is complete
                self.pos = -1
            else:
                self.order.append(layer)
        if not self.recording:
            if self.pos + 1 < len(self.order) and self.order[self.pos + 1] is layer:
                self.pos += 1
            else:
                try:
                    self.pos = self.order.index(layer)
                except ValueError:
                    self.pos = -1
        entry = self.ready.pop(id(layer), None)
        if entry is not None:
            q8, ev = entry
            main.wait_event(ev)
        else:
            q8 = self._buf(self.cur, N, K, device)
            qqq_unpack_to_int8_triton(layer.weight, layer.qqq_s_grp_plain, K, N, out=q8)
        # Fence: the side stream may only overwrite the other buffer once the
        # GEMM that read it (previous layer, main stream) has been issued.
        self.main_done = torch.cuda.Event()
        # Prefetch the next layer in recorded order.
        if not self.recording and 0 <= self.pos < len(self.order) - 1:
            nxt = self.order[self.pos + 1]
            if id(nxt) not in self.ready:
                if self.stream is None:
                    self.stream = torch.cuda.Stream(device)
                self.cur ^= 1
                nb = self._buf(self.cur, nxt.qqq_size_n, nxt.qqq_size_k, device)
                # wait for everything issued so far on main (incl. the previous
                # GEMM that read this buffer) before overwriting it
                self.main_done.record(main)
                self.stream.wait_event(self.main_done)
                with torch.cuda.stream(self.stream):
                    qqq_unpack_to_int8_triton(nxt.weight, nxt.qqq_s_grp_plain, nxt.qqq_size_k, nxt.qqq_size_n, out=nb)
                    ev = torch.cuda.Event()
                    ev.record(self.stream)
                self.ready[id(nxt)] = (nb, ev)
        return q8


_PREFETCH = _UnpackPrefetcher()
# Off by default: the unpack kernel fills the GPU, so it does not overlap with
# the CUTLASS GEMM in practice (no gain measured on CMP 170HX).
PREFETCH_UNPACK = os.environ.get("SGLANG_W4A8_PREFETCH_UNPACK", "0") == "1"


class CompressedTensorsW4A8Int8(CompressedTensorsLinearScheme):
    def __init__(self, group_size: int, symmetric: bool = True):
        assert group_size == 128, "Marlin-QQQ supports group_size 128 only"
        assert symmetric, "asymmetric int4 weights are not supported"
        self.group_size = group_size

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        assert input_size_per_partition % self.group_size == 0
        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def _process_marlin(self, layer: torch.nn.Module, q4: torch.Tensor, s_g: torch.Tensor) -> None:
        N, K = q4.shape
        dev = q4.device
        # GPTQ layout: uint4b8 (= q4 + 8) packed 8-per-int32 along K -> repack to
        # the int8-activation tile layout
        w_u = (q4.to(torch.int32) + 8).t().contiguous()  # [K, N]
        packed = pack_rows_uint4(w_u)  # int32 [K/8, N]
        del w_u
        B = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), K, N, 4, is_a_8bit=True)
        del packed
        s_perm = marlin_permute_scales(s_g.t().contiguous(), K, N, self.group_size, is_a_8bit=True)  # [K/g, N]
        s16, factor = marlin_act_int8_process_scales(s_perm)
        layer.weight = Parameter(B, requires_grad=False)  # int32 [K/16, N*2]
        layer.weight_scale = Parameter(s16, requires_grad=False)  # int16 bits viewed as params dtype [K/g, N]
        layer.marlin_input_global_scale = factor.to(device=dev, dtype=torch.float32)
        layer.marlin_workspace = marlin_make_workspace(dev)
        layer.qqq_size_n = N
        layer.qqq_size_k = K
        torch.cuda.empty_cache()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        q4 = layer.weight.data  # int8 [N, K], values in [-8, 7]
        s_g = layer.weight_scale.data  # [N, K/g]
        if W4A8_KERNEL == "marlin":
            self._process_marlin(layer, q4, s_g)
            return
        N, K = q4.shape
        B, s_ch_p, s_grp_p, q8, s_ch, s_grp = qqq_pack_from_int4(q4, s_g, self.group_size)
        if KEEP_INT8:
            layer.qqq_q8 = Parameter(q8.t().contiguous(), requires_grad=False)  # int8 [N, K]
        else:
            layer.qqq_q8 = None
        del q8
        layer.qqq_size_n = N
        layer.qqq_size_k = K
        layer.weight = Parameter(B, requires_grad=False)  # int32 [K/16, 2N]
        layer.weight_scale = Parameter(s_grp_p, requires_grad=False)  # fp16 [K/g, N] (permuted)
        layer.qqq_s_ch = Parameter(s_ch_p, requires_grad=False)  # fp32 [1, N] (permuted)
        layer.qqq_s_ch_plain = Parameter(s_ch.reshape(-1).contiguous(), requires_grad=False)  # fp32 [N]
        layer.qqq_s_grp_plain = Parameter(s_grp, requires_grad=False)  # fp16 [K/g, N]
        layer.qqq_workspace = marlin_qqq_workspace(N, q4.device)
        # Linear layers are post-processed in module order, which is also the
        # forward call order for a decoder stack; seed the prefetch order so
        # the very first prefill already overlaps its unpacks.
        _PREFETCH.order.append(layer)
        _PREFETCH.recording = False
        torch.cuda.empty_cache()

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        N, K = layer.qqq_size_n, layer.qqq_size_k
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        x_q, x_s = per_token_quant_int8(x_2d)
        if W4A8_KERNEL == "marlin":
            a_scales = x_s.reshape(-1) * layer.marlin_input_global_scale
            sms = torch.cuda.get_device_properties(x.device).multi_processor_count
            c_tmp = _scratch("marlin_c_tmp", (sms * _MARLIN_C_TMP_PER_SM,), torch.float32, x.device)
            out = gptq_marlin_gemm(
                x_q, None, layer.weight, layer.weight_scale, None, None, None, None,
                layer.marlin_workspace, scalar_types.uint4b8, M, N, K,
                is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
                a_scales=a_scales, c_tmp=c_tmp,
            )
            if bias is not None:
                out = out + bias
        elif M <= QQQ_MAX_M:
            c_tmp = _scratch("qqq_c_tmp", (MAX_PAR * 64, N), torch.int32, x.device)
            y = marlin_qqq_gemm(
                x_q, layer.weight, x_s, layer.qqq_s_ch, layer.weight_scale,
                layer.qqq_workspace, M, N, K, c_tmp=c_tmp,
            )
            out = y.to(x.dtype)
            if bias is not None:
                out = out + bias
        else:
            q8 = layer.qqq_q8
            if q8 is None:
                if PREFETCH_UNPACK and not torch.cuda.is_current_stream_capturing():
                    q8 = _PREFETCH.get(layer, x.device)
                else:
                    q8 = _scratch("qqq_q8", (N, K), torch.int8, x.device)
                    qqq_unpack_to_int8_triton(layer.weight, layer.qqq_s_grp_plain, K, N, out=q8)
            out = int8_scaled_mm(x_q, q8.t(), x_s, layer.qqq_s_ch_plain, out_dtype=x.dtype, bias=bias)
        return out.reshape(*x.shape[:-1], N)
