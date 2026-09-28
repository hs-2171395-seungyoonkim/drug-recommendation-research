import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cvsplit_lib as L  # noqa: E402


def test_jaccard_and_f1_conventions():
    assert L.jaccard(set(), set()) == 1.0
    assert L.f1(set(), set()) == 1.0
    assert L.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert L.f1({1, 2}, {2, 3}) == pytest.approx(0.5)
    assert L.jaccard({1}, {2}) == 0.0


def test_change_bins_cover_the_unit_interval():
    assert L.change_bin(1.0) == "continuation (J=1)"
    assert L.change_bin(0.95) == "minor change [0.8,1)"
    assert L.change_bin(0.8) == "minor change [0.8,1)"
    assert L.change_bin(0.6) == "moderate [0.6,0.8)"
    assert L.change_bin(0.4) == "major [0.4,0.6)"
    assert L.change_bin(0.0) == "overhaul [0,0.4)"


def _toy():
    # patient 0: visits 0,1,2 ; patient 1: visits 0,1 ; 4 codes
    patient = np.array([0, 0, 0, 1, 1])
    visit = np.array([0, 1, 2, 0, 1])
    y_gt = np.array([
        [1, 1, 0, 0],  # p0 v0 {0,1}
        [1, 1, 0, 0],  # p0 v1 {0,1}  -> continuation
        [1, 0, 1, 0],  # p0 v2 {0,2}  -> change: +2, -1
        [0, 0, 1, 1],  # p1 v0 {2,3}
        [0, 0, 0, 1],  # p1 v1 {3}    -> change: -2
    ], dtype=np.uint8)
    y_pred = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],  # pred {0}   vs true {0,1}: J=0.5 ; copy {0,1}: J=1
        [1, 0, 1, 0],  # pred {0,2} vs true {0,2}: J=1   ; copy {0,1}: J=1/3
        [0, 0, 1, 1],
        [0, 0, 1, 1],  # pred {2,3} vs true {3}: J=0.5   ; copy {2,3}: J=0.5
    ], dtype=np.uint8)
    return patient, visit, y_gt, y_pred


def test_build_transitions_skips_first_visits_and_orders_by_visit():
    p, v, gt, pr = _toy()
    # shuffle row order to make sure sorting is used
    perm = np.array([4, 2, 0, 3, 1])
    rows = L.build_transitions(p[perm], v[perm], gt[perm], pr[perm], seed="s")
    assert [(r["patient"], r["visit"]) for r in rows] == [(0, 1), (0, 2), (1, 1)]
    assert rows[1]["prev"] == {0, 1} and rows[1]["cur"] == {0, 2} and rows[1]["pred"] == {0, 2}


def test_build_transitions_rejects_visit_gaps():
    p = np.array([0, 0]); v = np.array([0, 2])
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError):
        L.build_transitions(p, v, gt, gt, seed=0)


def test_keep_mask_restricts_scored_visits_but_previous_still_available():
    p, v, gt, pr = _toy()
    keep = np.array([False, False, True, False, True])  # score p0v2 and p1v1 only
    rows = L.build_transitions(p, v, gt, pr, seed=0, keep_mask=keep)
    assert [(r["patient"], r["visit"]) for r in rows] == [(0, 2), (1, 1)]


def test_score_and_summarize_strata():
    p, v, gt, pr = _toy()
    rows = L.build_transitions(p, v, gt, pr, seed=0)
    scored = L.score_rows(rows, constant_set={0, 3})
    by = {(r["patient"], r["visit"]): r for r in scored}
    assert by[(0, 1)]["is_change"] is False and by[(0, 1)]["model_j"] == 0.5 and by[(0, 1)]["copy_j"] == 1.0
    assert by[(0, 2)]["is_change"] is True and by[(0, 2)]["n_added"] == 1 and by[(0, 2)]["n_stopped"] == 1
    assert by[(0, 2)]["model_j"] == 1.0 and by[(0, 2)]["copy_j"] == pytest.approx(1 / 3)
    assert by[(0, 2)]["model_added_j"] == 1.0 and by[(0, 2)]["model_stopped_j"] == 1.0
    assert by[(1, 1)]["model_stopped_j"] == 0.0  # true stopped {2}, pred stopped {}
    assert by[(1, 1)]["const_j"] == pytest.approx(0.5)  # {0,3} vs {3}

    s = L.summarize(scored, n_boot=50)
    assert s["n_transitions"] == 3 and s["change_share"] == pytest.approx(2 / 3)
    assert s["strata"]["continuation"]["n_transitions"] == 1
    assert s["strata"]["change"]["n_transitions"] == 2
    assert s["strata"]["change"]["model_jaccard"] == pytest.approx(0.75)
    assert s["strata"]["change"]["copy_previous_jaccard"] == pytest.approx((1 / 3 + 0.5) / 2)
    assert s["strata"]["bin:continuation (J=1)"]["n_transitions"] == 1
    assert "model_minus_copy" in s["strata"]["change"]


def test_average_over_seeds_averages_metrics_only():
    p, v, gt, pr = _toy()
    a = L.score_rows(L.build_transitions(p, v, gt, pr, seed="a"), None)
    pr2 = pr.copy(); pr2[1] = [1, 1, 0, 0]  # seed b predicts p0v1 perfectly
    b = L.score_rows(L.build_transitions(p, v, gt, pr2, seed="b"), None)
    avg = L.average_over_seeds({"a": a, "b": b})
    by = {(r["patient"], r["visit"]): r for r in avg}
    assert by[(0, 1)]["model_j"] == pytest.approx(0.75)
    assert by[(0, 1)]["copy_j"] == 1.0 and by[(0, 1)]["is_change"] is False
