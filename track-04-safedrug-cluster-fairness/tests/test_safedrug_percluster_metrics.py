import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_percluster.metrics import attach_labels, patient_bootstrap_ci, visit_metrics


DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


def test_visit_metrics_normal_row():
    y_gt = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[1, 0, 0, 1]])
    y_prob = np.array([[0.9, 0.4, 0.1, 0.6]])
    out = visit_metrics(y_gt, y_pred, y_prob, DDI_ADJ)
    row = out.iloc[0]
    assert row["jaccard"] == pytest.approx(1 / 3)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(0.5)
    assert row["ddi_rate_visit"] == pytest.approx(1.0)
    assert row["n_med_pred"] == 2
    assert row["n_med_gt"] == 2
    from sklearn.metrics import average_precision_score

    assert row["prauc"] == pytest.approx(average_precision_score(y_gt[0], y_prob[0]))


def test_visit_metrics_no_ground_truth_positives_gives_nan_prauc():
    y_gt = np.array([[0, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 0]])
    y_prob = np.array([[0.7, 0.2, 0.1, 0.05]])
    out = visit_metrics(y_gt, y_pred, y_prob, DDI_ADJ)
    row = out.iloc[0]
    assert np.isnan(row["prauc"])
    assert row["jaccard"] == 0.0
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["f1"] == 0.0
    assert np.isnan(row["ddi_rate_visit"])  # only 1 predicted med
    assert row["n_med_pred"] == 1
    assert row["n_med_gt"] == 0


def test_visit_metrics_empty_prediction_gives_nan_ddi_rate():
    y_gt = np.array([[1, 0, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0]])
    y_prob = np.array([[0.1, 0.2, 0.3, 0.05]])
    out = visit_metrics(y_gt, y_pred, y_prob, DDI_ADJ)
    row = out.iloc[0]
    assert row["jaccard"] == 0.0
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert np.isnan(row["ddi_rate_visit"])
    assert row["n_med_pred"] == 0
    assert row["n_med_gt"] == 2
    from sklearn.metrics import average_precision_score

    assert row["prauc"] == pytest.approx(average_precision_score(y_gt[0], y_prob[0]))


def test_visit_metrics_perfect_prediction():
    y_gt = np.array([[1, 1, 1, 1]])
    y_pred = np.array([[1, 1, 1, 1]])
    y_prob = np.array([[0.9, 0.8, 0.7, 0.6]])
    out = visit_metrics(y_gt, y_pred, y_prob, DDI_ADJ)
    row = out.iloc[0]
    assert row["jaccard"] == 1.0
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert row["ddi_rate_visit"] == pytest.approx(1 / 6)  # 1 DDI pair of C(4,2)=6
    assert row["prauc"] == pytest.approx(1.0)


def test_visit_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        visit_metrics(np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 4)), DDI_ADJ)


def test_visit_metrics_ddi_adj_shape_mismatch_raises():
    with pytest.raises(ValueError):
        visit_metrics(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((3, 3)))


def _write_label_fixture(tmp_path):
    dxtext = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4, 5],
            "HADM_ID": [100, 101, 102, 103, 104],
            "n_dx": [5, 6, 7, 8, 9],
        }
    )
    dxtext_path = tmp_path / "dxtext_cluster_assignments.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "42_dxtext_labels.npz"
    np.savez(
        labels_path,
        long_k10=np.array([0, 1, 2, 0, 1], dtype=np.int32),
        long_k25=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        short_k10=np.array([1, 1, 0, 0, 1], dtype=np.int32),
        concise_k10=np.array([0, 0, 1, 1, 1], dtype=np.int32),
    )

    ccs = pd.DataFrame({"HADM_ID": [100, 101, 102, 103], "group": ["A", "B", "A", "C"]})
    ccs_path = tmp_path / "ccs_assignments.csv"
    ccs.to_csv(ccs_path, index=False, encoding="utf-8-sig")
    return dxtext_path, labels_path, ccs_path


def test_attach_labels_positional_alignment_and_missing_hadm(tmp_path):
    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    df = pd.DataFrame({"HADM_ID": [100, 101, 102, 103, 104, 999]})

    out = attach_labels(df, dxtext_path, labels_path, ccs_path)

    row104 = out[out["HADM_ID"] == 104].iloc[0]
    assert row104["long_k10"] == 1
    assert row104["has_label"] == True  # noqa: E712
    assert pd.isna(row104["ccs_group"])

    row999 = out[out["HADM_ID"] == 999].iloc[0]
    assert pd.isna(row999["long_k10"])
    assert row999["has_label"] == False  # noqa: E712
    assert pd.isna(row999["ccs_group"])

    row100 = out[out["HADM_ID"] == 100].iloc[0]
    assert row100["ccs_group"] == "A"


def test_attach_labels_planted_length_mismatch_raises(tmp_path):
    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    # Overwrite labels_npz with a mismatched (4-row) long_k10 array.
    np.savez(
        labels_path,
        long_k10=np.array([0, 1, 2, 0], dtype=np.int32),  # 4, not 5
        long_k25=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        short_k10=np.array([1, 1, 0, 0, 1], dtype=np.int32),
        concise_k10=np.array([0, 0, 1, 1, 1], dtype=np.int32),
    )
    df = pd.DataFrame({"HADM_ID": [100, 101, 102, 103, 104]})
    with pytest.raises(ValueError, match="alignment"):
        attach_labels(df, dxtext_path, labels_path, ccs_path)


def test_attach_labels_missing_hadm_id_column_raises(tmp_path):
    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    with pytest.raises(ValueError):
        attach_labels(pd.DataFrame({"not_hadm": [1]}), dxtext_path, labels_path, ccs_path)


def _bootstrap_fixture():
    return pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2, 2, 3, 3, 4, 4],
            "grp": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "val": [0.5, 0.6, 0.4, 0.3, 0.9, 0.8, 0.7, 0.6],
        }
    )


def test_patient_bootstrap_ci_shape():
    df = _bootstrap_fixture()
    out = patient_bootstrap_ci(df, "grp", "val", n_boot=200, seed=0)
    assert set(out.columns) == {"grp", "n_visits", "n_patients", "mean", "ci_low", "ci_high"}
    assert sorted(out["grp"]) == ["a", "b"]
    a = out[out["grp"] == "a"].iloc[0]
    assert a["n_visits"] == 4
    assert a["n_patients"] == 2
    assert a["mean"] == pytest.approx((0.5 + 0.6 + 0.4 + 0.3) / 4)
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]


def test_patient_bootstrap_ci_deterministic():
    df = _bootstrap_fixture()
    out1 = patient_bootstrap_ci(df, "grp", "val", n_boot=200, seed=0)
    out2 = patient_bootstrap_ci(df, "grp", "val", n_boot=200, seed=0)
    pd.testing.assert_frame_equal(
        out1.sort_values("grp").reset_index(drop=True),
        out2.sort_values("grp").reset_index(drop=True),
    )
