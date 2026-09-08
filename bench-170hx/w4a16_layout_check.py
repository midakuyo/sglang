# Marlin fp16(is_a_8bit=False) 4bit 타일 레이아웃 닫힌식 검증 (16k x 64n 청크 = 128 워드)
import sys, torch
sys.path.insert(0, "/src/python")
from sglang.test.test_marlin_utils import marlin_weights, get_weight_perm
torch.manual_seed(0)
for K, N in ((256, 128), (512, 256), (1024, 512), (5376, 2048)):
    q = torch.randint(0, 16, (K, N), dtype=torch.int32)
    B = marlin_weights(q, K, N, 4, get_weight_perm(4, False), is_a_8bit=False)  # int32 [K/16, 2N]
    Bf = B.reshape(-1)
    k = torch.arange(K).view(K, 1).expand(K, N); n = torch.arange(N).view(1, N).expand(K, N)
    kb, row = k // 16, k % 16; m, nn = n // 64, n % 64
    j, c16 = nn // 16, nn % 16; block, col = c16 // 8, c16 % 8
    i = 4 * col + (row % 8) // 2
    r_idx = (row % 2) + 2 * (row // 8)
    t = block * 4 + r_idx
    e = torch.where(t % 2 == 0, t // 2, 4 + (t - 1) // 2)
    qw = i * 4 + j
    chunk = kb * (N // 64) + m
    word = 128 * chunk + qw
    rec = (Bf[word] >> (4 * e)) & 0xF
    ok = torch.equal(rec, q)
    print(f"K={K} N={N}: fp16-layout closed-form inverse {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        bad = (rec != q).nonzero()[:5]; print("first mismatches (k,n):", bad.tolist())
