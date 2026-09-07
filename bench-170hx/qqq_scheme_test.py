# 스킴 단위 테스트: create_weights → 가짜 로드 → process → apply (M=3 QQQ 경로, M=300 언팩+int8 경로) vs fp32 참조
import sys, torch
sys.path.insert(0, "/src/python")
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import CompressedTensorsW4A8Int8
from sglang.kernels.ops.quantization.marlin_qqq import qqq_pack_from_int4
dev = "cuda"; torch.manual_seed(0)
for name, (N, K) in {"o": (5376, 8192), "qkv": (16384, 5376)}.items():
    layer = torch.nn.Module()
    sch = CompressedTensorsW4A8Int8(128)
    sch.create_weights(layer, [N], K, torch.bfloat16, lambda *a, **k: None)
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s_g = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    layer.weight.data = q4.clone(); layer.weight_scale.data = s_g.clone()
    _, _, _, q8_ref, s_ch, _ = qqq_pack_from_int4(q4, s_g)
    sch.process_weights_after_loading(layer)
    for M in (3, 24, 300, 2107):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        y = sch.apply_weights(layer, x, None)
        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
        a_q, a_s = per_token_quant_int8(x)
        r = (a_q.float() @ q8_ref.float()) * s_ch * a_s.float()
        err = ((y.float() - r).abs().max() / r.abs().max()).item()
        print(f"{name:4s} M={M:5d} path={'qqq' if M <= 32 else 'unpack+int8'} dtype={y.dtype} rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
print("scheme test done")
