# SPDX-License-Identifier: Apache-2.0
# Quark W8A8-INT8 linear scheme for SGLang.
#
# Serves AMD Quark exports whose global/layer quant config is
#   weight: int8, static, symmetric, per_channel (ch_axis 0) or per_tensor
#   input : int8, dynamic, symmetric, per_channel (== per-token) or per_tensor
# e.g. nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8. The checkpoint stores
# `weight` as int8 [N, K] and `weight_scale` as a 1-D [N] tensor (bf16 or fp32);
# no zero points and no input scales are serialized for the dynamic case.
#
# Kernel path is the same CUTLASS one used by compressed-tensors W8A8-INT8:
# per-token activation quantization + `int8_scaled_mm`.

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import is_cuda, set_weight_attrs

__all__ = ["QuarkW8A8Int8"]

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import int8_scaled_mm


class QuarkW8A8Int8(QuarkLinearScheme):
    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.weight_qscheme = cast(str, weight_config.get("qscheme"))
        if self.weight_qscheme not in ("per_channel", "per_tensor"):
            raise NotImplementedError(
                f"Quark W8A8-INT8: unsupported weight qscheme {self.weight_qscheme!r}"
            )
        if weight_config.get("symmetric") is not True:
            raise NotImplementedError(
                "Quark W8A8-INT8: only symmetric weight quantization is supported"
            )

        self.is_static_input_scheme = False
        self.input_symmetric = True
        if input_config is not None:
            self.is_static_input_scheme = not cast(bool, input_config.get("is_dynamic"))
            self.input_symmetric = input_config.get("symmetric") is not False
        if self.is_static_input_scheme:
            raise NotImplementedError(
                "Quark W8A8-INT8: static activation scales are not supported yet "
                "(only dynamic per-token/per-tensor activations)"
            )
        if not self.input_symmetric:
            raise NotImplementedError(
                "Quark W8A8-INT8: asymmetric activation quantization is not supported"
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # int8_scaled_mm (CUTLASS) needs Ampere and up.
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

        # WEIGHT: int8 [N, K]
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE. Quark serializes the per-channel scale as 1-D [N]
        # (possibly bf16); keep the parameter 1-D fp32 so the sharded copy is a
        # plain dtype cast, and view it as [N, 1] after loading (kernel layout).
        if self.weight_qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(output_size_per_partition, dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            set_weight_attrs(weight_scale, {"needs_scalar_to_array": True})
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.weight_qscheme == "per_tensor":
            # Fused modules (qkv / gate_up) carry one scale per shard; requantize
            # to a single max scale so the kernel always sees a uniform scale.
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        else:
            weight_scale = layer.weight_scale.data.to(torch.float32).view(-1, 1)
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
            # torch.compile requires a Parameter here.
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_q, x_scale = per_token_quant_int8(x)
        return int8_scaled_mm(
            x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias
        )
