"""Per-step warmup + cosine LR schedule -- the reference python multiplier (verbatim from
pipeline.py) and a spec object each port turns into its own `where`-based tensor version.

    lr(step) = peak * where(e < warmup, e / warmup,
                            floor + (1 - floor) * 0.5 * (1 + cos(pi * min((e - warmup)/decay, 1))))
    with e = step / steps_per_epoch, decay = max(total - warmup, 1), floor = 0.02.
"""
import math
from dataclasses import dataclass

import numpy as np


def make_lr_factor(warmup_epochs, total_epochs, steps_per_epoch, peak_mult=1.0, floor_mult=0.02):
    """Reference python multiplier (verbatim from pipeline.py)."""
    decay_epochs = max(total_epochs - warmup_epochs, 1)

    def factor(step):
        e = step / max(steps_per_epoch, 1)  # fractional epoch
        if e < warmup_epochs:
            return peak_mult * e / max(warmup_epochs, 1e-9)
        t = min((e - warmup_epochs) / decay_epochs, 1.0)  # 0..1
        return floor_mult + (peak_mult - floor_mult) * 0.5 * (1.0 + math.cos(math.pi * t))

    return factor


@dataclass(frozen=True)
class WarmupCosineSpec:
    peak_lr: float
    warmup_epochs: float
    total_epochs: float
    steps_per_epoch: int
    floor_mult: float = 0.02

    @property
    def decay_epochs(self):
        return max(float(self.total_epochs) - float(self.warmup_epochs), 1.0)

    @property
    def warm_div(self):
        return max(float(self.warmup_epochs), 1e-9)

    @property
    def step_div(self):
        return max(int(self.steps_per_epoch), 1)


def lr_numpy(spec, steps):
    """Vectorised numpy version of the same `where` formulation the ports implement."""
    e = np.asarray(steps, dtype=np.float32) / spec.step_div
    warm = e / spec.warm_div
    t = np.minimum((e - spec.warmup_epochs) / spec.decay_epochs, 1.0)
    cos = spec.floor_mult + (1.0 - spec.floor_mult) * 0.5 * (1.0 + np.cos(np.pi * t))
    return spec.peak_lr * np.where(e < spec.warmup_epochs, warm, cos)
