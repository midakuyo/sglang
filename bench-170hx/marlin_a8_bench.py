# W4A8 커널 비교 마이크로벤치 (CMP 170HX): Marlin-A8(fp32 reduce) / Marlin-QQQ / CUTLASS int8_scaled_mm(상주 q8) / W4A16 Marlin(bf16 act)
# python3 bench-170hx/marlin_a8_bench.py [graph]   — 'graph'면 CUDA 그래프에 30회 캡처해 재생(런치 오버헤드 제거)
import sys, torch
sys.path.insert(0, "/src/python")
from sgl_kernel import int8_scaled_mm
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.kernels.ops.quantization.marlin_qqq import MAX_PAR, marlin_qqq_gemm, marlin_qqq_workspace, qqq_pack_from_int4
from sglang.srt.layers.quantization.marlin_utils import marlin_act_int8_process_scales, marlin_make_workspace, marlin_permute_scales
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import pack_rows_uint4

USE_GRAPH = "graph" in sys.argv[1:]
dev = "cuda"; torch.manual_seed(0)
SHAPES = [("qkv", 16384, 5376), ("o", 5376, 8192), ("gate_up", 43008, 5376), ("down", 5376, 21504)]
MS = [1, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048]
sms = torch.cuda.get_device_properties(0).multi_processor_count

def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    if USE_GRAPH:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for _ in range(30):
                fn()
        torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(5):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / (5 * 30)
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

print(f"graph={USE_GRAPH}  columns: ms per call  (weight bytes int4 = N*K/2)")
print(f"{'layer':8s} {'M':>5s} | {'marlinA8':>9s} {'qqq':>9s} {'cutlass8':>9s} {'w4a16':>9s} | TB/s(A8)")
for name, N, K in SHAPES:
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s_g = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    # Marlin-A8
    w_u = (q4.to(torch.int32) + 8).t().contiguous()
    packed = pack_rows_uint4(w_u)
    B_a8 = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), K, N, 4, is_a_8bit=True)
    s16, factor = marlin_act_int8_process_scales(marlin_permute_scales(s_g.t().contiguous(), K, N, 128, is_a_8bit=True))
    # W4A16 Marlin (same weights, bf16 activations)
    B_16 = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), K, N, 4, is_a_8bit=False)
    s_16 = marlin_permute_scales(s_g.t().contiguous(), K, N, 128, is_a_8bit=False)
    del w_u, packed
    ws = marlin_make_workspace(dev)
    c_tmp = torch.empty(sms * 64 * 256, dtype=torch.float32, device=dev)
    # QQQ + resident q8 for CUTLASS
    B_q, s_ch_p, s_grp_p, q8, s_ch, s_grp = qqq_pack_from_int4(q4, s_g, 128)
    q8_nk = q8.t().contiguous()  # int8 [N, K]
    s_ch_plain = s_ch.reshape(-1).contiguous()
    ws_q = marlin_qqq_workspace(N, dev)
    c_tmp_q = torch.empty(MAX_PAR * 64, N, dtype=torch.int32, device=dev)
    del q8
    for M in MS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        x_q = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
        x_s = torch.rand(M, device=dev, dtype=torch.float32) * 0.01
        a_scales = x_s * factor.to(dev)
        f_a8 = lambda: gptq_marlin_gemm(x_q, None, B_a8, s16, None, None, None, None, ws, scalar_types.uint4b8, M, N, K, True, False, True, a_scales=a_scales, c_tmp=c_tmp)
        f_qq = lambda: marlin_qqq_gemm(x_q, B_q, x_s, s_ch_p, s_grp_p, ws_q, M, N, K, c_tmp=c_tmp_q)
        f_c8 = lambda: int8_scaled_mm(x_q, q8_nk.t(), x_s.reshape(-1, 1), s_ch_plain, out_dtype=torch.bfloat16)
        f_16 = lambda: gptq_marlin_gemm(x, None, B_16, s_16, None, None, None, None, ws, scalar_types.uint4b8, M, N, K, True, False, True, c_tmp=c_tmp)
        t_a8 = timeit(f_a8); t_qq = timeit(f_qq); t_c8 = timeit(f_c8); t_16 = timeit(f_16)
        tbps = (N * K / 2) / (t_a8 * 1e-3) / 1e12
        print(f"{name:8s} {M:5d} | {t_a8:9.4f} {t_qq:9.4f} {t_c8:9.4f} {t_16:9.4f} | {tbps:5.2f}", flush=True)
    del B_a8, B_16, B_q, q8_nk; torch.cuda.empty_cache()
