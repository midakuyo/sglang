# Marlin-A8 언팩 검증·시간: repack → unpack == QQQ 규약 q8 ; 층 4형상 언팩 ms ; 스킴 하이브리드(M>임계) vs A8 경로 오차
import os, sys, time, torch
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.kernels.ops.quantization.marlin_a8_unpack import a8_channel_scales, marlin_a8_unpack_to_int8
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import pack_rows_uint4
dev = "cuda"; torch.manual_seed(0)
for name, N, K, g in (("qkv", 16384, 5376, 128), ("o", 5376, 8192, 128), ("gate_up", 43008, 5376, 32), ("down", 5376, 21504, 32)):
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s_g = (torch.rand(N, K // g, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    packed = pack_rows_uint4((q4.to(torch.int32) + 8).t().contiguous())
    B = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), K, N, 4, is_a_8bit=True)
    s_ch, s_grp = a8_channel_scales(q4.t().contiguous(), s_g.t().contiguous(), g)
    # reference q8 (QQQ convention)
    ref = torch.round((q4.t().half() * s_grp.repeat_interleave(g, dim=0)).float()).clamp(-128, 127).to(torch.int8).t().contiguous()  # [N, K]
    out = marlin_a8_unpack_to_int8(B, s_grp, K, N, g)
    ok = torch.equal(out, ref)
    for _ in range(3): marlin_a8_unpack_to_int8(B, s_grp, K, N, g, out=out)
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(20): marlin_a8_unpack_to_int8(B, s_grp, K, N, g, out=out)
    e1.record(); torch.cuda.synchronize(); ms = e0.elapsed_time(e1) / 20
    gbps = (N * K // 2 + N * K) / (ms * 1e-3) / 1e9
    print(f"{name:8s} g={g:3d} unpack {'OK ' if ok else 'MISMATCH'} {ms:6.3f} ms  ({gbps:5.0f} GB/s in+out)  mismatches={int((out != ref).sum())}", flush=True)
    del B, packed, ref, out; torch.cuda.empty_cache()
# scheme-level: hybrid vs A8 path
os.environ["SGLANG_W4A8_KERNEL"] = "marlin"
from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w4a8_int8 as S
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
for g in (128, 32):
    N, K = 5376, 8192
    layer = torch.nn.Module(); sch = S.CompressedTensorsW4A8Int8(g)
    sch.create_weights(layer, [N], K, torch.bfloat16, lambda *a, **k: None)
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s_g = (torch.rand(N, K // g, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    layer.weight.data = q4.clone(); layer.weight_scale.data = s_g.clone()
    w_true = q4.float() * s_g.float().repeat_interleave(g, dim=1)
    sch.process_weights_after_loading(layer)
    for M in (700, 800, 2048):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        y = sch.apply_weights(layer, x, None)
        a_q, a_s = per_token_quant_int8(x); r = (a_q.float() * a_s.reshape(-1, 1).float()) @ w_true.t()
        err = ((y.float() - r).abs().max() / r.abs().max()).item()
        path = "cutlass" if M > S.MARLIN_A8_MAX_M else "marlin-a8"
        print(f"scheme g={g:3d} M={M:5d} path={path:9s} rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
print("done")
