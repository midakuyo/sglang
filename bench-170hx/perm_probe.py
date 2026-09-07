import sys, torch
sys.path.insert(0, "/src/python")
from sglang.kernels.ops.quantization.marlin_qqq import _marlin_permute_weights, _qqq_weight_perm
K, N = 16, 64  # one k-tile row, one 1024 block
ids = torch.arange(K * N).reshape(K, N)  # id = k*64 + n
perm = _qqq_weight_perm(4, True)
pw = _marlin_permute_weights(ids, K, N, perm)  # [1, 1024]: permuted column c holds id
row = pw[0]
# packed column j nibble i <-> c = j*8 + i  (q_packed[:, j] |= q_w[:, i::8] << 4i)
print("c -> (k, n) for c in 0..15:", [(int(row[c]) // 64, int(row[c]) % 64) for c in range(16)])
print("c -> (k, n) for c in 256..263:", [(int(row[c]) // 64, int(row[c]) % 64) for c in range(256, 264)])
# test hypothesis: c = nt*256 + i*8 + p ; p -> (rr_lo, block, rr_hi) ; k = 4*(i%4) + 2*rr_hi + rr_lo ; n = nt*16 + block*8 + i//4
bad = 0
for c in range(1024):
    nt, rem = divmod(c, 256); i, p = divmod(rem, 8)
    rr_lo, r2 = divmod(p, 4); block, rr_hi = divmod(r2, 2)
    k = 4 * (i % 4) + 2 * rr_hi + rr_lo; n = nt * 16 + block * 8 + i // 4
    if int(row[c]) != k * 64 + n:
        if bad < 6: print("  mismatch c", c, "hyp", (k, n), "actual", (int(row[c]) // 64, int(row[c]) % 64))
        bad += 1
print("hypothesis mismatches:", bad, "/ 1024")
# brute-force: fit k,n as functions of (nt, i_hi, i_lo, p bits)
import itertools
tab = {}
for c in range(1024):
    nt, rem = divmod(c, 256); i, p = divmod(rem, 8)
    tab[(nt, i // 4, i % 4, p)] = (int(row[c]) // 64, int(row[c]) % 64)
print("p -> (k,n) for nt=0,i=0:", [tab[(0, 0, 0, p)] for p in range(8)])
print("i_lo -> (k,n) for nt=0,i_hi=0,p=0:", [tab[(0, 0, l, 0)] for l in range(4)])
print("i_hi -> (k,n) for nt=0,i_lo=0,p=0:", [tab[(0, h, 0, 0)] for h in range(8)])
print("nt -> (k,n):", [tab[(t, 0, 0, 0)] for t in range(4)])
