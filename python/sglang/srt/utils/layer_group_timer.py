"""Layer-group split of forward GPU time: attn / mlp / lm_head / other.

Forward hooks on every decoder layer's attention and MLP module and on the
logits processor record *external* CUDA events (cudaEventRecordExternal), so
the records survive cuda-graph capture and are re-issued on every replay.
Because a replay re-records the same event objects, a reading is taken only
every N wrapped forwards, right after a stream synchronize -- race-free at the
cost of one sync per N passes. Enable with SGLANG_DEVICE_TIMER_LAYER_GROUPS=N
(plus SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 and --enable-metrics); results are
exported as sglang:forward_layer_group_seconds_total{category, group}.
"""

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

# attn_core is the attention *kernel* module nested inside the attention
# block (RadixAttention); it is reported separately but is a subset of attn,
# so "other" is computed from the top-level groups only.
GROUPS = ("attn", "attn_core", "mlp", "lm_head")
TOP_LEVEL_GROUPS = ("attn", "mlp", "lm_head")
_LEAF_TO_GROUP = {
    "self_attn": "attn",
    "attn": "attn",
    "attention": "attn",
    "mlp": "mlp",
    "block_sparse_moe": "mlp",
    "feed_forward": "mlp",
    "logits_processor": "lm_head",
}
# Forward categories whose model actually carries the hooks (the target
# model). Draft-model forwards run un-hooked modules and would only re-read
# stale events.
SAMPLED_CATEGORIES = ("decode", "target_verify", "extend")


class LayerGroupTimer:
    def __init__(self, sample_every: int):
        self.sample_every = max(1, int(sample_every))
        self._pairs: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            g: [] for g in GROUPS
        }
        self._count = 0

    @staticmethod
    def _event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True, external=True)

    def install(self, model: torch.nn.Module) -> "LayerGroupTimer":
        registered: List[str] = []
        for name, mod in model.named_modules():
            group = _LEAF_TO_GROUP.get(name.rsplit(".", 1)[-1])
            if group is None:
                continue
            if group != "lm_head" and (
                ".layers." not in name or "vision" in name or "audio" in name
            ):
                continue
            # named_modules yields parents first: a match nested inside a
            # registered attention block is the attention kernel module.
            if any(name.startswith(r + ".") for r in registered):
                if group != "attn":
                    continue
                group = "attn_core"
            else:
                registered.append(name)
            start, end = self._event(), self._event()
            mod.register_forward_pre_hook(lambda m, args, ev=start: ev.record())
            mod.register_forward_hook(lambda m, args, out, ev=end: ev.record())
            self._pairs[group].append((start, end))
        logger.info(
            "Layer-group timer installed (%s), sampling every %d forwards",
            ", ".join(f"{g}={len(p)}" for g, p in self._pairs.items()),
            self.sample_every,
        )
        return self

    def wants_sample(self, category) -> bool:
        if category not in SAMPLED_CATEGORIES:
            return False
        self._count += 1
        return self._count % self.sample_every == 0

    def read(self) -> Dict[str, float]:
        """Seconds per group for the most recent forward. Call after a
        torch.cuda.synchronize(); pairs whose module did not run are skipped."""
        out = {}
        for group, pairs in self._pairs.items():
            total_ms = 0.0
            for start, end in pairs:
                try:
                    total_ms += start.elapsed_time(end)
                except Exception:
                    pass
            out[group] = total_ms / 1000.0
        return out
