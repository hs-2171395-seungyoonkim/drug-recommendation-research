import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_ddi_arms_size_matched as M  # noqa: E402


def test_topk_rows_picks_largest_and_respects_zero():
    prob = np.array([[0.1, 0.9, 0.5, 0.7], [0.3, 0.2, 0.1, 0.0]])
    out = M.topk_rows(prob, np.array([2, 0]))
    assert out.tolist() == [[0, 1, 0, 1], [0, 0, 0, 0]]
    assert out.sum(1).tolist() == [2, 0]


def test_threshold_set_equals_own_topk_of_its_size():
    rng = np.random.default_rng(0)
    prob = rng.random((50, 30)).astype(np.float32)
    thr = M.threshold_rows(prob, 0.5)
    assert np.array_equal(M.topk_rows(prob, thr.sum(1)), thr)


def test_find_threshold_matches_target_mean_size():
    rng = np.random.default_rng(1)
    prob = rng.random((400, 40))
    target = 12.0
    t = M.find_threshold_for_mean_size(prob, target)
    achieved = M.threshold_rows(prob, t).sum(1).mean()
    assert abs(achieved - target) < 0.15          # discrete counts: within a fraction of a drug
    assert 0.0 < t < 1.0
