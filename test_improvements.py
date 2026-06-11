"""
Unit tests for the demo's signal-gating logic.

These exercise the parts of the improvements that don't need the AST model:
the RMS silence gate and the per-category alert cooldown. Run with:

    python3 test_improvements.py

(No torch/transformers/numpy required — signal_utils falls back to stdlib.)
"""
from signal_utils import rms, dbfs, is_silent, CooldownGate

FLOOR = -45.0
N = 16000  # one 1 s window @ 16 kHz


def test_silence_is_gated():
    assert is_silent([0.0] * N, FLOOR)


def test_quiet_noise_floor_is_gated():
    # ~-66 dBFS hum should be treated as silence (the old peak-norm bug would
    # have amplified this to full scale and classified it).
    quiet = [0.0005 if i % 2 else -0.0005 for i in range(N)]
    assert is_silent(quiet, FLOOR)


def test_loud_tone_passes_gate():
    loud = [0.5 if i % 2 else -0.5 for i in range(N)]  # ~-6 dBFS
    assert not is_silent(loud, FLOOR)


def test_rms_and_dbfs_basic():
    assert abs(rms([1.0, -1.0, 1.0, -1.0]) - 1.0) < 1e-9
    assert rms([]) == 0.0
    assert dbfs([0.0] * 100) < -100  # ~full-scale silence


def test_cooldown_blocks_repeats_then_reopens():
    gate = CooldownGate(3.0)
    assert gate.allow("Coughing", 0.0) is True   # first fires
    assert gate.allow("Coughing", 1.0) is False  # within 3 s -> blocked
    assert gate.allow("Coughing", 2.9) is False  # still within -> blocked
    assert gate.allow("Coughing", 3.5) is True   # cooldown elapsed -> fires


def test_cooldown_is_per_category():
    gate = CooldownGate(3.0)
    assert gate.allow("Coughing", 0.0) is True
    assert gate.allow("Fall / Heavy Impact", 0.1) is True  # independent category


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
