"""
signal_utils.py — pure-Python signal gating + alert-rate helpers.

Deliberately free of torch/transformers so the demo's detection logic can be
unit-tested without loading the ~350 MB AST model. numpy is used when it is
installed (the real server has it) and falls back to stdlib math otherwise
(test sandbox / CI), so the same code runs in both places.
"""
import math

try:
    import numpy as _np
except Exception:  # numpy not installed (e.g. the test sandbox)
    _np = None


def rms(samples):
    """Root-mean-square amplitude of a frame of float samples in [-1, 1]."""
    if _np is not None:
        a = _np.asarray(samples, dtype=_np.float64)
        if a.size == 0:
            return 0.0
        return float(_np.sqrt(_np.mean(a * a)))
    total, n = 0.0, 0
    for s in samples:
        v = float(s)
        total += v * v
        n += 1
    return math.sqrt(total / n) if n else 0.0


def dbfs(samples):
    """RMS level in dBFS (0 dB = full scale). Silence -> very negative."""
    return 20.0 * math.log10(rms(samples) + 1e-9)


def is_silent(samples, floor_dbfs):
    """True when the frame is quieter than `floor_dbfs` — skip classification."""
    return dbfs(samples) < floor_dbfs


class CooldownGate:
    """
    Per-category alert rate limiter. A sustained sound (a held alarm, a long
    cry) would otherwise fire one alert per analysis window; this collapses
    repeats of the SAME category within `cooldown_s` into a single alert.

    Time is injected via `allow(category, now)` so the logic is deterministic
    and unit-testable without a real clock.
    """

    def __init__(self, cooldown_s):
        self.cooldown_s = cooldown_s
        self._last = {}

    def allow(self, category, now):
        last = self._last.get(category)
        if last is not None and (now - last) < self.cooldown_s:
            return False
        self._last[category] = now
        return True
