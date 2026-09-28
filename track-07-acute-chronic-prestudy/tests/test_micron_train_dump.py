import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import micron_train_dump as M  # noqa: E402


def test_hysteresis_decode_keeps_middle_band():
    state = np.array([1, 1, 0, 0, 1], dtype=float)
    prob = np.array([0.9, 0.5, 0.85, 0.1, 0.15])
    out = M.hysteresis_decode(state, prob, 0.8, 0.2)
    assert out.tolist() == [1, 1, 1, 0, 0]          # 0.5 keeps previous 1; 0.15 drops previous 1
    assert state.tolist() == [1, 1, 0, 0, 1]        # input untouched


def test_select_thresholds_clips_and_skips_degenerate_drugs():
    rng = np.random.default_rng(0)
    n = 500
    labels = np.zeros((n, 3)); probs = rng.random((n, 3))
    labels[:, 0] = (probs[:, 0] > 0.6).astype(float)           # informative drug
    labels[:, 1] = (rng.random(n) < 0.3).astype(float)          # random drug
    # drug 2 has no positives -> skipped
    t1, t2, skipped = M.select_thresholds(labels, probs)
    assert skipped == 1 and 0.5 <= t1 <= 0.9 and 0.1 <= t2 <= 0.5


def test_official_module_is_the_downloaded_micron():
    assert "SOTA\\MICRON\\src" in M.micron_models.__file__ or "SOTA/MICRON/src" in M.micron_models.__file__
    assert M.LAMBDAS == (0.25, 0.25, 0.25, 0.25) and M.EPOCHS == 40 and M.LR == 2e-4
