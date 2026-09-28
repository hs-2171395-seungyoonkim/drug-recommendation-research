import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prev_meds_dependence as P  # noqa: E402


def test_dissect_hand_computed():
    prob = np.array([0.9, 0.8, 0.2, 0.6, 0.1, 0.55])
    prev, cur = {0, 1, 2}, {0, 1, 3, 5}          # added {3,5}, stopped {2}, cont {0,1}
    pred = {i for i in range(6) if prob[i] >= 0.5}   # {0,1,3,5}
    freq = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    d = P.dissect(prev, cur, pred, prob, freq)
    assert d["n_prev"] == 3 and d["n_cur"] == 4 and d["n_added"] == 2 and d["n_stopped"] == 1
    assert d["n_pred"] == 4 and d["size_gap"] == 0 and d["n_pred_in_prev"] == 2 and d["n_pred_added"] == 2
    assert d["miss_rate"] == 0.0 and d["cont_recall"] == 1.0 and d["stopped_drop_rate"] == 1.0
    # oracle = top-4 = {0,1,3,5} -> same; oracle at prev size (3) = {0,1,3}
    assert d["miss_rate_oracle"] == 0.0
    assert d["prob_added_mean"] == pytest.approx((0.6 + 0.55) / 2)
    assert d["rank_added_mean"] == pytest.approx((3 + 4) / 2)
    assert d["train_freq_added_mean"] == pytest.approx((0.2 + 0.05) / 2)
    assert d["miss_rate@0.3"] == 0.0 and d["n_pred@0.3"] == 4
    # a lower-probability added drug that 0.5 misses but oracle catches
    prob2 = np.array([0.9, 0.8, 0.2, 0.45, 0.1, 0.55])
    pred2 = {0, 1, 5}
    d2 = P.dissect(prev, cur, pred2, prob2, freq)
    assert d2["miss_rate"] == pytest.approx(0.5) and d2["miss_rate_oracle"] == 0.0 and d2["size_gap"] == -1


def test_topk_set_and_train_frequencies():
    assert P.topk_set(np.array([0.1, 0.9, 0.5]), 2) == {1, 2}
    assert P.topk_set(np.array([0.1, 0.9]), 0) == set()
    records = [[[[0], [0], [0, 1]], [[0], [0], [1]]], [[[0], [0], [2]]], [[[0], [0], [0]]]]
    # split_patients on 3 patients: train = first 2 (int(3*2/3)=2)
    f = P.train_frequencies(records, 3)
    assert f.tolist() == pytest.approx([1 / 3, 2 / 3, 1 / 3])
