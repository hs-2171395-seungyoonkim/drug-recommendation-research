import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_excess_ddi import (
    excess_ddi_group_table,
    excess_ddi_permutation_table,
    per_visit_excess_ddi,
)

# Meds 0 and 3 are the one DDI pair -- the repo's existing DDI test fixture
# (same matrix tests/test_safedrug_group_ddi_targets.py and
# tests/test_safedrug_mitigation_compare.py use).
DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


def test_per_visit_excess_ddi_hand_computed():
    # row0: pred {0,3} (DDI pair -> 1.0), gt {1,2} (not DDI -> 0.0) -> excess 1.0
    # row1: pred {0} (< 2 -> NaN), gt {0,3} (DDI -> 1.0) -> excess NaN (pred side)
    # row2: pred {0,3} (DDI -> 1.0), gt {1} (< 2 -> NaN) -> excess NaN (gt side)
    # row3: pred {1} (< 2 -> NaN), gt {2} (< 2 -> NaN) -> excess NaN (both sides)
    y_pred = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
    ])
    y_gt = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

    out = per_visit_excess_ddi(y_gt, y_pred, DDI_ADJ)

    assert out.loc[0, "ddi_pred"] == pytest.approx(1.0)
    assert out.loc[0, "ddi_true"] == pytest.approx(0.0)
    assert out.loc[0, "excess_ddi"] == pytest.approx(1.0)

    assert np.isnan(out.loc[1, "ddi_pred"])
    assert out.loc[1, "ddi_true"] == pytest.approx(1.0)
    assert np.isnan(out.loc[1, "excess_ddi"])

    assert out.loc[2, "ddi_pred"] == pytest.approx(1.0)
    assert np.isnan(out.loc[2, "ddi_true"])
    assert np.isnan(out.loc[2, "excess_ddi"])

    assert np.isnan(out.loc[3, "ddi_pred"])
    assert np.isnan(out.loc[3, "ddi_true"])
    assert np.isnan(out.loc[3, "excess_ddi"])


def test_per_visit_excess_ddi_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        per_visit_excess_ddi(np.zeros((2, 4)), np.zeros((3, 4)), DDI_ADJ)


def test_excess_ddi_group_table_means_on_tiny_fixture():
    df = pd.DataFrame({
        "grp": ["g0", "g0", "g1"],
        "SUBJECT_ID": [1, 2, 3],
        "ddi_pred": [0.5, 0.7, 0.9],
        "ddi_true": [0.2, 0.4, 0.5],
        "excess_ddi": [0.3, 0.3, 0.4],
    })
    out = excess_ddi_group_table(df, "grp").set_index("group")

    assert out.loc["g0", "n_visits"] == 2
    assert out.loc["g0", "n_patients"] == 2
    assert out.loc["g0", "mean_ddi_pred"] == pytest.approx(0.6)
    assert out.loc["g0", "mean_ddi_true"] == pytest.approx(0.3)
    assert out.loc["g0", "mean_excess_ddi"] == pytest.approx(0.3)
    assert out.loc["g0", "ci_low_ddi_pred"] <= out.loc["g0", "mean_ddi_pred"] <= out.loc["g0", "ci_high_ddi_pred"]

    assert out.loc["g1", "n_visits"] == 1
    assert out.loc["g1", "n_patients"] == 1
    assert out.loc["g1", "mean_ddi_pred"] == pytest.approx(0.9)
    assert out.loc["g1", "mean_ddi_true"] == pytest.approx(0.5)
    assert out.loc["g1", "mean_excess_ddi"] == pytest.approx(0.4)


def test_excess_ddi_group_table_drops_nan_partition_rows():
    df = pd.DataFrame({
        "grp": ["g0", float("nan")],
        "SUBJECT_ID": [1, 2],
        "ddi_pred": [0.5, 0.9],
        "ddi_true": [0.2, 0.1],
        "excess_ddi": [0.3, 0.8],
    })
    out = excess_ddi_group_table(df, "grp")
    assert list(out["group"]) == ["g0"]


def test_excess_ddi_permutation_table_runs_with_tiny_n_perm():
    # Two groups, 30 visits each (MIN_VISITS=30), one patient per visit,
    # constant excess_ddi within each group -> exact observed range 0.3.
    rows = []
    sid = 0
    for g, val in enumerate([0.1, 0.4]):
        for _ in range(30):
            rows.append({"grp": g, "SUBJECT_ID": sid, "excess_ddi": val})
            sid += 1
    df = pd.DataFrame(rows)

    out = excess_ddi_permutation_table(df, "grp", n_perm=5, seed=0)

    assert list(out.columns) == ["partition", "statistic", "observed", "null_mean", "null_p95", "z", "p_raw"]
    assert set(out["statistic"]) == {"range", "weighted_sd"}
    range_row = out[out["statistic"] == "range"].iloc[0]
    assert range_row["observed"] == pytest.approx(0.3)
    assert not pd.isna(range_row["p_raw"])
    assert (out["partition"] == "grp").all()


def test_excess_ddi_permutation_table_below_min_visits_is_nan():
    df = pd.DataFrame({
        "grp": [0, 0, 1, 1],
        "SUBJECT_ID": [1, 2, 3, 4],
        "excess_ddi": [0.1, 0.1, 0.4, 0.4],
    })
    out = excess_ddi_permutation_table(df, "grp", n_perm=5, seed=0)
    range_row = out[out["statistic"] == "range"].iloc[0]
    assert np.isnan(range_row["observed"])


def _write_synthetic_eval_dir(eval_dir: Path) -> Path:
    """A tiny per_visit_predictions.npz with 60 test-split visits, 4
    medications (DDI_ADJ above), one patient per visit, split evenly across
    two long_k10/ccs_group groups (30 each, meeting MIN_VISITS=30)."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    n = 60
    hadm_ids = np.arange(2000, 2000 + n, dtype=np.int64)

    y_gt = np.zeros((n, 4), dtype=np.uint8)
    y_pred = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        if i % 2 == 0:
            y_pred[i, [0, 3]] = 1  # DDI pair -> ddi_pred = 1.0
            y_gt[i, [1, 2]] = 1  # not a DDI pair -> ddi_true = 0.0
        else:
            y_pred[i, [1, 2]] = 1  # ddi_pred = 0.0
            y_gt[i, [0, 3]] = 1  # ddi_true = 1.0

    npz_arrays = {
        "HADM_ID": hadm_ids,
        "SUBJECT_ID": hadm_ids.copy(),
        "split": np.array(["test"] * n),
        "y_gt": y_gt,
        "y_pred": y_pred,
        "y_prob": y_pred.astype(np.float32),
    }
    np.savez(eval_dir / "per_visit_predictions.npz", **npz_arrays)
    return eval_dir


def test_main_end_to_end_writes_all_outputs(tmp_path):
    eval_dir = _write_synthetic_eval_dir(tmp_path / "eval")
    out_dir = tmp_path / "excess_ddi_out"

    n = 60
    hadm_ids = list(range(2000, 2000 + n))
    dxtext_path = tmp_path / "dxtext.csv"
    pd.DataFrame({"HADM_ID": hadm_ids}).to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "labels.npz"
    long_k10 = np.array([0] * 30 + [1] * 30, dtype=np.int32)
    np.savez(
        labels_path,
        long_k10=long_k10, long_k25=long_k10, short_k10=long_k10, concise_k10=long_k10,
    )

    ccs_path = tmp_path / "ccs.csv"
    ccs_group = ["A"] * 30 + ["B"] * 30
    pd.DataFrame({"HADM_ID": hadm_ids, "group": ccs_group}).to_csv(
        ccs_path, index=False, encoding="utf-8-sig"
    )

    ddi_adj_path = tmp_path / "ddi_adj.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(DDI_ADJ, fh)

    import safedrug_excess_ddi as sed

    backup = sed.DXTEXT_CSV, sed.LABELS_NPZ, sed.CCS_CSV
    sed.DXTEXT_CSV, sed.LABELS_NPZ, sed.CCS_CSV = dxtext_path, labels_path, ccs_path
    try:
        sed.main([
            "--eval-dir", str(eval_dir),
            "--out-dir", str(out_dir),
            "--partitions", "long_k10", "ccs_group",
            "--n-perm", "5",
            "--seed", "0",
            "--ddi-adj", str(ddi_adj_path),
        ])
    finally:
        sed.DXTEXT_CSV, sed.LABELS_NPZ, sed.CCS_CSV = backup

    per_visit = pd.read_csv(out_dir / "per_visit_excess_ddi.csv")
    assert len(per_visit) == n
    for col in ("HADM_ID", "SUBJECT_ID", "split", "ddi_pred", "ddi_true", "excess_ddi", "long_k10", "ccs_group"):
        assert col in per_visit.columns
    assert (per_visit["split"] == "test").all()

    groups = pd.read_csv(out_dir / "table_excess_ddi_groups.csv")
    assert set(groups["partition"]) == {"long_k10", "ccs_group"}
    for col in ("mean_ddi_pred", "ci_low_ddi_pred", "ci_high_ddi_pred",
                "mean_ddi_true", "ci_low_ddi_true", "ci_high_ddi_true",
                "mean_excess_ddi", "ci_low_excess_ddi", "ci_high_excess_ddi"):
        assert col in groups.columns

    perm = pd.read_csv(out_dir / "table_excess_ddi_permutation.csv")
    assert set(perm["partition"]) == {"long_k10", "ccs_group"}
    assert set(perm["statistic"]) == {"range", "weighted_sd"}
