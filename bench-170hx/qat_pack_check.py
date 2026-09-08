import glob, json, torch
from safetensors import safe_open
F = "/models/gemma4-qat-w4a16/model.safetensors"
with safe_open(F, "pt", device="cpu") as sf:
    idx = {k: "model.safetensors" for k in sf.keys()}
names = [k for k in idx if "layers.0.mlp.down_proj" in k]
print("layer0 down_proj tensors:", names)
suffixes = sorted({k.rsplit(".", 1)[-1] for k in idx if "language_model.layers" in k})
print("all suffixes:", suffixes)
for n in names:
    with safe_open("/models/gemma4-qat-w4a16/" + idx[n], "pt", device="cpu") as sf:
        t = sf.get_tensor(n); print(n, tuple(t.shape), t.dtype, "" if t.numel() > 8 else t.tolist())
        if n.endswith("weight_packed"):
            w = t
        if n.endswith("weight_scale"):
            s = t
# unpack nibbles along last dim: word j holds k = 8j+i in bits 4i
nib = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(w.shape[0], -1)  # [N, K]
print("nibble range", nib.min().item(), nib.max().item(), "mean %.3f" % nib.float().mean().item(), "K =", nib.shape[1])
# signed interpretation: q = nib - 8 (uint4b8) vs two's complement
q_off = nib.to(torch.int16) - 8
q_tc = torch.where(nib >= 8, nib.to(torch.int16) - 16, nib.to(torch.int16))
print("offset-8: mean %.4f  |  two's-complement: mean %.4f" % (q_off.float().mean().item(), q_tc.float().mean().item()))
# which one makes rows look like zero-mean weights per group? compare abs mean of group means
g = 32
for name, q in (("offset-8", q_off), ("twos", q_tc)):
    gm = q.float().reshape(q.shape[0], -1, g).mean(-1).abs().mean().item()
    print(f"{name}: mean |group mean| = {gm:.4f}")
print("scale shape", tuple(s.shape), s.dtype, "scale>0 frac %.4f" % (s.float() > 0).float().mean().item())
