import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_group_threshold import (
    THRESHOLD_GRID,
    main,
    rethreshold_test_split,
    row_jaccard_batch,
    select_threshold,
)


def test_row_jaccard_batch_hand_computed():
    y_gt = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 0]])
    out = row_jaccard_batch(y_gt, y_pred)
    assert out == pytest.approx([0.5, 1.0, 0.0])


def test_select_threshold_finds_interior_peak():
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    y_gt = np.array([[1, 1, 1, 0, 0]])
    y_prob = np.array([[0.9, 0.6, 0.55, 0.45, 0.2]])
    best_t, table = select_threshold(y_gt, y_prob, thresholds)
    assert best_t == pytest.approx(0.5)
    assert table["mean_jaccard"].tolist() == pytest.approx(
        [0.75, 0.75, 1.0, 0.6666666667, 0.3333333333]
    )


def test_select_threshold_monotonic_case_prefers_lowest_threshold():
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    y_gt = np.array([[1, 1, 0, 0, 0]])
    y_prob = np.array([[0.9, 0.35, 0.2, 0.1, 0.05]])
    best_t, table = select_threshold(y_gt, y_prob, thresholds)
    assert best_t == pytest.approx(0.3)
    assert table["mean_jaccard"].tolist() == pytest.approx([1.0, 0.5, 0.5, 0.5, 0.5])


def test_select_threshold_pooled_global_control():
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    y_gt = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    y_prob = np.array([[0.9, 0.6, 0.55, 0.45, 0.2], [0.9, 0.35, 0.2, 0.1, 0.05]])
    best_t, table = select_threshold(y_gt, y_prob, thresholds)
    assert best_t == pytest.approx(0.3)
    assert table["mean_jaccard"].tolist() == pytest.approx(
        [0.875, 0.625, 0.75, 0.5833333333, 0.4166666667]
    )


def test_select_threshold_tie_breaks_toward_0_5():
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    y_gt = np.array([[1]])
    y_prob = np.array([[0.45]])
    best_t, table = select_threshold(y_gt, y_prob, thresholds)
    assert table["mean_jaccard"].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])
    assert best_t == pytest.approx(0.4)  # tied with 0.3; 0.4 is closer to 0.5


def test_select_threshold_rejects_empty_input():
    with pytest.raises(ValueError):
        select_threshold(np.zeros((0, 3)), np.zeros((0, 3)), [0.3, 0.5])


def test_rethreshold_test_split_only_touches_test_rows():
    npz = {
        "y_prob": np.array([[0.9, 0.2], [0.9, 0.2], [0.1, 0.6]], dtype=np.float32),
        "y_pred": np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8),
        "split": np.array(["test", "eval", "test"], dtype="<U8"),
        "HADM_ID": np.array([1, 2, 3], dtype=np.int64),
    }
    thresholds_by_row = np.array([0.5, 0.99, 0.15])  # eval row's threshold is irrelevant
    out = rethreshold_test_split(npz, thresholds_by_row)

    assert list(out["y_pred"][0]) == [1, 0]  # test row 0: prob>=0.5 -> [1,0] (unchanged here)
    assert list(out["y_pred"][1]) == [1, 0]  # eval row: untouched, original y_pred kept
    assert list(out["y_pred"][2]) == [0, 1]  # test row 2: prob>=0.15 -> [0,1] (unchanged here)
    assert list(out["y_prob"][0]) == pytest.approx([0.9, 0.2])  # y_prob never touched
    assert list(out["HADM_ID"]) == [1, 2, 3]


def test_rethreshold_test_split_actually_changes_predictions_at_a_new_threshold():
    npz = {
        "y_prob": np.array([[0.4, 0.6]], dtype=np.float32),
        "y_pred": np.array([[0, 1]], dtype=np.uint8),  # thresholded at 0.5 originally
        "split": np.array(["test"], dtype="<U8"),
        "HADM_ID": np.array([1], dtype=np.int64),
    }
    out = rethreshold_test_split(npz, np.array([0.35]))
    assert list(out["y_pred"][0]) == [1, 1]  # both meds now clear the lower threshold


def _write_label_fixture(tmp_path):
    dxtext = pd.DataFrame({"HADM_ID": [100, 101, 102, 103, 104]})
    dxtext_path = tmp_path / "dxtext.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "labels.npz"
    np.savez(
        labels_path,
        long_k10=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        long_k25=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        short_k10=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        concise_k10=np.array([0, 1, 0, 1, 0], dtype=np.int32),
    )
    ccs_path = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": [100, 101, 102, 103, 104], "group": ["A"] * 5}).to_csv(
        ccs_path, index=False, encoding="utf-8-sig"
    )
    return dxtext_path, labels_path, ccs_path


def _write_eval_dir(eval_dir: Path):
    eval_dir.mkdir(parents=True, exist_ok=True)
    # 5 visits, 3 meds: HADM 100 (eval, group0), 101 (eval, group1),
    # 102 (test, group0), 103 (test, group1), 104 (test, NO label -- HADM_ID
    # 104 has a long_k10 entry in the fixture above, so redefine it as
    # unlabeled here by using a 6th HADM_ID absent from the label fixture.
    arrays = {
        "patient_index": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "visit_index": np.array([0, 0, 0, 0, 0], dtype=np.int64),
        "HADM_ID": np.array([100, 101, 102, 103, 999], dtype=np.int64),
        "SUBJECT_ID": np.array([10, 11, 12, 13, 14], dtype=np.int64),
        "split": np.array(["eval", "eval", "test", "test", "test"], dtype="<U8"),
        "n_diag": np.array([2, 2, 2, 2, 2], dtype=np.int64),
        "n_proc": np.array([0, 0, 0, 0, 0], dtype=np.int64),
        "n_med_gt": np.array([2, 2, 2, 2, 2], dtype=np.int64),
        "y_gt": np.array(
            [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=np.uint8
        ),
        "y_pred": np.array(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8
        ),
        "y_prob": np.array(
            [
                [0.9, 0.55, 0.1],
                [0.9, 0.2, 0.1],
                [0.9, 0.55, 0.1],
                [0.9, 0.2, 0.1],
                [0.9, 0.55, 0.1],
            ],
            dtype=np.float32,
        ),
    }
    np.savez(eval_dir / "per_visit_predictions.npz", **arrays)
    manifest = {
        "official_metrics": {"test": {"ja": 0.5, "prauc": 0.5, "avg_f1": 0.5,
                                       "ddi_rate": 0.0, "avg_med": 2.0}},
        "best_epoch": 10, "best_eval_jaccard": 0.5,
    }
    (eval_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return eval_dir


def test_group_tuned_thresholds_fallback_to_global_for_unlabeled(tmp_path, monkeypatch):
    import safedrug_group_threshold as sgt

    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    monkeypatch.setattr(sgt, "DXTEXT_CSV", dxtext_path)
    monkeypatch.setattr(sgt, "LABELS_NPZ", labels_path)
    monkeypatch.setattr(sgt, "CCS_CSV", ccs_path)

    eval_dir = _write_eval_dir(tmp_path / "eval")
    out_dir = tmp_path / "out"
    main(["--eval-dir", str(eval_dir), "--out-dir", str(out_dir)])

    table = pd.read_csv(out_dir / "table_thresholds.csv")
    t_star = float(table.loc[table["group"] == "GLOBAL", "threshold"].iloc[0])

    group_npz = np.load(out_dir / "group_tuned" / "per_visit_predictions.npz", allow_pickle=False)
    # HADM 999 (row 4) has no long_k10 label -- must be rethresholded at t_star.
    row = list(group_npz["HADM_ID"]).index(999)
    expected_pred = (group_npz["y_prob"][row] >= t_star).astype(np.uint8)
    assert list(group_npz["y_pred"][row]) == list(expected_pred)


def test_main_writes_three_variant_dirs_and_table_thresholds_csv(tmp_path, monkeypatch):
    import safedrug_group_threshold as sgt

    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    monkeypatch.setattr(sgt, "DXTEXT_CSV", dxtext_path)
    monkeypatch.setattr(sgt, "LABELS_NPZ", labels_path)
    monkeypatch.setattr(sgt, "CCS_CSV", ccs_path)

    eval_dir = _write_eval_dir(tmp_path / "eval")
    out_dir = tmp_path / "out"
    main(["--eval-dir", str(eval_dir), "--out-dir", str(out_dir)])

    for variant in ("global_0.5", "global_tuned", "group_tuned"):
        assert (out_dir / variant / "per_visit_predictions.npz").exists()
        manifest = json.loads((out_dir / variant / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["threshold_policy"]["variant"] == variant
    assert (out_dir / "table_thresholds.csv").exists()

    baseline_npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    copy_npz = np.load(out_dir / "global_0.5" / "per_visit_predictions.npz", allow_pickle=False)
    assert list(copy_npz["y_pred"].flatten()) == list(baseline_npz["y_pred"].flatten())
