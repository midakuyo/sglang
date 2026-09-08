# Marlin-A8(is_a_8bit) 타일 레이아웃 닫힌식 검증: 파이썬 참조 marlin_weights vs 공식 역변환
import sys, torch
sys.path.insert(0, "/src/python")
from sglang.test.test_marlin_utils import marlin_weights, get_weight_perm
torch.manual_seed(0)
for K, N in ((256, 128), (512, 256), (21504 // 21, 5376 // 21 * 2)):
    K = K // 32 * 32; N = N // 32 * 32
    q = torch.randint(0, 16, (K, N), dtype=torch.int32)
    B = marlin_weights(q, K, N, 4, get_weight_perm(4, True), is_a_8bit=True)  # int32 [K/16, 2N]
    Bf = B.reshape(-1)
    k = torch.arange(K).view(K, 1).expand(K, N); n = torch.arange(N).view(1, N).expand(K, N)
    kb, row = k // 32, k % 32; m, nn = n // 32, n % 32
    j, c16 = nn // 16, nn % 16; block, col = c16 // 8, c16 % 8
    i = 4 * col + (row % 16) // 4
    r_idx = (row % 4) + 4 * (row // 16)
    qw = i * 4 + j * 2 + block
    e = 2 * (row % 4) + row // 16
    chunk = kb * (N // 32) + m
    word = 128 * chunk + qw
    rec = (Bf[word] >> (4 * e)) & 0xF
    ok = torch.equal(rec, q)
    print(f"K={K} N={N}: closed-form inverse {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        bad = (rec != q).nonzero()[:5]; print("first mismatches (k,n):", bad.tolist())
