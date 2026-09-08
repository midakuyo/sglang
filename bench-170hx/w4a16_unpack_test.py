# W4A16 하이브리드 검증: fp16 레이아웃 언팩 == 참조 q8 ; 언팩 ms ; wNa16 스킴 M<=768(Marlin bf16) vs M>768(int8) vs 참 수학
import os, sys, torch
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.kernels.ops.quantization.marlin_a8_unpack import a8_channel_scales, marlin_w4a16_unpack_to_int8, unpack_uint4b8_rows
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import pack_rows_uint4
dev = "cuda"; torch.manual_seed(0)
for name, N, K, g in (("qkv", 16384, 5376, 32), ("o", 5376, 8192, 32), ("gate_up", 43008, 5376, 32), ("down", 5376, 21504, 128)):
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s_g = (torch.rand(N, K // g, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    packed = pack_rows_uint4((q4.to(torch.int32) + 8).t().contiguous())  # [K/8, N]
    B = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), K, N, 4, is_a_8bit=False)
    s_ch, s_grp = a8_channel_scales(q4.t().contiguous(), s_g.t().contiguous(), g)
    ref = torch.round((q4.t().half() * s_grp.repeat_interleave(g, dim=0)).float()).clamp(-128, 127).to(torch.int8).t().contiguous()
    out = marlin_w4a16_unpack_to_int8(B, s_grp, K, N, g)
    ok = torch.equal(out, ref)
    for _ in range(3): marlin_w4a16_unpack_to_int8(B, s_grp, K, N, g, out=out)
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(20): marlin_w4a16_unpack_to_int8(B, s_grp, K, N, g, out=out)
    e1.record(); torch.cuda.synchronize(); ms = e0.elapsed_time(e1) / 20
    print(f"{name:8s} g={g:3d} unpack {'OK ' if ok else 'MISMATCH'} {ms:6.3f} ms  mismatches={int((out != ref).sum())}", flush=True)
    # row-unpack helper check
    q4_back = unpack_uint4b8_rows(packed.t().contiguous())
    print(f"{name:8s} unpack_uint4b8_rows {'OK' if torch.equal(q4_back, q4.t().contiguous()) else 'MISMATCH'}", flush=True)
    del B, packed, ref, out; torch.cuda.empty_cache()
# scheme level
from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_wNa16 as S
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
for g in (32, 128):
    N, K = 5376, 8192
    layer = torch.nn.Module(); sch = S.CompressedTensorsWNA16(strategy="group", num_bits=4, group_size=g, symmetric=True, actorder=None)
    sch.create_weights(layer, output_size=N, input_size=K, output_partition_sizes=[N], input_size_per_partition=K, params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev); s_g = (torch.rand(N, K // g, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    packed_nk = pack_rows_uint4((q4.to(torch.int32) + 8).t().contiguous()).t().contiguous()  # [N, K/8]
    layer.weight_packed.data = packed_nk.clone(); layer.weight_scale.data = s_g.clone()
    w_true = q4.float() * s_g.float().repeat_interleave(g, dim=1)
    sch.process_weights_after_loading(layer)
    print(f"scheme g={g}: hybrid={sch.hybrid}")
    for M in (16, 700, 800, 2048):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        y = sch.apply_weights(layer, x, None)
        if M > S.W4A16_INT8_PREFILL_MAX_M:
            a_q, a_s = per_token_quant_int8(x); r = (a_q.float() * a_s.reshape(-1, 1).float()) @ w_true.t(); path = "int8"
        else:
            r = x.float() @ w_true.t(); path = "marlin-bf16"
        err = ((y.float() - r).abs().max() / r.abs().max()).item()
        print(f"scheme g={g:3d} M={M:5d} path={path:11s} rel err {err:.2e} {'OK' if err < 2e-2 else 'FAIL'}", flush=True)
print("done")
