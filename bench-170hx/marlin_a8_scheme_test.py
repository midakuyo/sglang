# 스킴 단위 테스트(Marlin-A8 모드): create_weights → 가짜 로드 → process → apply vs 정확 int 참조
# SGLANG_W4A8_KERNEL=marlin python3 bench-170hx/marlin_a8_scheme_test.py
import os, sys, torch
sys.path.insert(0, "/src/python")
os.environ.setdefault("SGLANG_W4A8_KERNEL", "marlin")
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import CompressedTensorsW4A8Int8, W4A8_KERNEL
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
dev = "cuda"; torch.manual_seed(0)
print("kernel mode:", W4A8_KERNEL)
for name, (N, K) in {"o": (5376, 8192), "qkv": (16384, 5376), "down": (5376, 21504)}.items():
    layer = torch.nn.Module()
    sch = CompressedTensorsW4A8Int8(128)
    sch.create_weights(layer, [N], K, torch.bfloat16, lambda *a, **k: None)
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s_g = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    layer.weight.data = q4.clone(); layer.weight_scale.data = s_g.clone()
    w_true = q4.float() * s_g.float().repeat_interleave(128, dim=1)  # [N, K]
    sch.process_weights_after_loading(layer)
    for M in (1, 3, 12, 24, 64, 300, 2107):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        y = sch.apply_weights(layer, x, None)
        a_q, a_s = per_token_quant_int8(x)
        r = (a_q.float() * a_s.reshape(-1, 1).float()) @ w_true.t()
        err = ((y.float() - r).abs().max() / r.abs().max()).item()
        print(f"{name:4s} M={M:5d} dtype={y.dtype} rel err vs true W4 {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
    del layer; torch.cuda.empty_cache()
print("scheme test done")
