# Marlin-A8 스레드 구성 스윕(그래프 재생, 소량 M): SGLANG_MARLIN_CFG 강제 vs auto, fp32 reduce on/off
import os, sys, torch
sys.path.insert(0, "/src/python")
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_act_int8_process_scales, marlin_make_workspace, marlin_permute_scales
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import pack_rows_uint4
dev = "cuda"; torch.manual_seed(0)
SHAPES = [("qkv", 16384, 5376), ("o", 5376, 8192), ("gate_up", 43008, 5376), ("down", 5376, 21504)]
MS = [int(v) for v in os.environ.get("MS", "1,4,8,12,16,24,32,48,64").split(",")]
REPS = int(os.environ.get("REPS", "5"))  # min of REPS graph measurements (clock noise)
A16 = os.environ.get("DTYPE", "int8") == "bf16"  # bf16 activations (W4A16 path) instead of int8
GROUP = int(os.environ.get("GROUP", "128"))  # weight group size (32 for Google QAT checkpoints)
CFGS = os.environ.get("CFGS", "auto 128,128,256,1 128,128,256,2 64,256,256,1 64,256,256,2 64,128,128,1 64,128,128,2").split()
sms = torch.cuda.get_device_properties(0).multi_processor_count

def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(30):
            fn()
    torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(REPS):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(5):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 150)
    return best

print(f"dtype={'bf16 (W4A16)' if A16 else 'int8 (W4A8)'} group={GROUP} ms per call, graph replay, min of REPS; columns = cfg (thread_k,thread_n,threads,blocks_per_sm); nr = use_fp32_reduce=False")
hdr = f"{'layer':8s} {'M':>4s} | " + " ".join(f"{c:>12s}" for c in CFGS) + " | auto_nr"
print(hdr)
for name, N, K in SHAPES:
    q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
    s_g = (torch.rand(N, K // GROUP, device=dev) * 0.02 + 0.002).to(torch.bfloat16)
    packed = pack_rows_uint4((q4.to(torch.int32) + 8).t().contiguous())
    B = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), K, N, 4, is_a_8bit=not A16)
    if A16:
        s16 = marlin_permute_scales(s_g.t().contiguous(), K, N, GROUP, is_a_8bit=False); factor = torch.ones((), device=dev)
    else:
        s16, factor = marlin_act_int8_process_scales(marlin_permute_scales(s_g.t().contiguous(), K, N, GROUP, is_a_8bit=True))
    del packed
    ws = marlin_make_workspace(dev, max_blocks_per_sm=4)  # forced blocks_per_sm>1 needs sms*bps lock slots
    c_tmp = torch.empty(sms * 64 * 256, dtype=torch.float32, device=dev)
    for M in MS:
        x_q = torch.randn(M, K, dtype=torch.bfloat16, device=dev) if A16 else torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
        a_scales = None if A16 else torch.rand(M, device=dev) * 0.01 * factor.to(dev)
        row = []
        os.environ["SGLANG_MARLIN_DEBUG"] = "1"; os.environ.pop("SGLANG_MARLIN_CFG", None)
        gptq_marlin_gemm(x_q, None, B, s16, None, None, None, None, ws, scalar_types.uint4b8, M, N, K, True, False, True, a_scales=a_scales, c_tmp=c_tmp)
        torch.cuda.synchronize(); os.environ.pop("SGLANG_MARLIN_DEBUG", None)
        for cfg in CFGS:
            if cfg == "auto":
                os.environ.pop("SGLANG_MARLIN_CFG", None)
            else:
                os.environ["SGLANG_MARLIN_CFG"] = cfg
            try:
                t = timeit(lambda: gptq_marlin_gemm(x_q, None, B, s16, None, None, None, None, ws, scalar_types.uint4b8, M, N, K, True, False, True, a_scales=a_scales, c_tmp=c_tmp))
                row.append(f"{t:12.4f}")
            except Exception as e:
                row.append(f"{'n/a':>12s}")
                torch.cuda.synchronize()
        os.environ.pop("SGLANG_MARLIN_CFG", None)
        t_nr = timeit(lambda: gptq_marlin_gemm(x_q, None, B, s16, None, None, None, None, ws, scalar_types.uint4b8, M, N, K, True, False, False, a_scales=a_scales))
        print(f"{name:8s} {M:4d} | " + " ".join(row) + f" | {t_nr:.4f}", flush=True)
    del B; torch.cuda.empty_cache()
