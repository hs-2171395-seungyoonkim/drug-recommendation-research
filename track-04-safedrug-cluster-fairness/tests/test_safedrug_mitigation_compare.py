import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_mitigation_compare import (
    build_mitigation_tables,
    flat_test_metrics,
    global_ddi_rate,
    joint_patient_bootstrap_delta_range,
    mitigation_group_deltas,
    variant_gap_table,
    weighted_gap_range,
)

DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


def test_global_ddi_rate_hand_computed():
    y_pred = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    assert global_ddi_rate(y_pred, DDI_ADJ) == pytest.approx(0.5)


def test_global_ddi_rate_no_multi_med_visits_returns_zero():
    y_pred = np.array([[1, 0, 0, 0], [0, 0, 0, 0]])
    assert global_ddi_rate(y_pred, DDI_ADJ) == pytest.approx(0.0)


def test_weighted_gap_range_hand_computed():
    group_codes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    values = np.array([0.4, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8])
    weights = np.array([1, 1, 1, 2, 2, 1, 1, 1, 1])
    out = weighted_gap_range(group_codes, values, weights, n_groups=3, min_visits=3)
    assert out == pytest.approx(0.4)


def test_weighted_gap_range_returns_nan_below_min_visits():
    group_codes = np.array([0, 0, 1, 1])
    values = np.array([0.4, 0.4, 0.8, 0.8])
    weights = np.array([1, 1, 1, 1])
    out = weighted_gap_range(group_codes, values, weights, n_groups=2, min_visits=30)
    assert np.isnan(out)


def test_mitigation_group_deltas_hand_computed():
    baseline = pd.DataFrame({"group": [0, 1], "raw_jaccard": [0.4, 0.6]})
    variant = pd.DataFrame({"group": [0, 1], "raw_jaccard": [0.5, 0.55]})
    out = mitigation_group_deltas(baseline, variant, ["raw_jaccard"]).set_index("group")
    assert out.loc[0, "delta_raw_jaccard"] == pytest.approx(0.1)
    assert out.loc[1, "delta_raw_jaccard"] == pytest.approx(-0.05)


def _deterministic_bootstrap_fixture(n_per_group: int = 30):
    # 3 groups, n_per_group one-visit patients each, within-group CONSTANT
    # value -- so any patient-level resample leaves each group's weighted
    # mean exactly unchanged (see design doc D-D3). baseline group means
    # 0.40/0.55/0.70 (range 0.30); variant = 0.5 + 0.3*(baseline-0.5)
    # applied per-row -> 0.47/0.515/0.56 (range 0.09) EXACTLY on every draw.
    rows_base, rows_var = [], []
    sid = 0
    for g, base_val in enumerate([0.40, 0.55, 0.70]):
        var_val = 0.5 + 0.3 * (base_val - 0.5)
        for _ in range(n_per_group):
            rows_base.append({"HADM_ID": sid, "SUBJECT_ID": sid, "long_k10": g, "y": base_val})
            rows_var.append({"HADM_ID": sid, "SUBJECT_ID": sid, "long_k10": g, "y": var_val})
            sid += 1
    baseline_df = pd.DataFrame(rows_base).sort_values("HADM_ID").reset_index(drop=True)
    variant_df = pd.DataFrame(rows_var).sort_values("HADM_ID").reset_index(drop=True)
    return baseline_df, variant_df


def test_joint_patient_bootstrap_delta_range_deterministic_shrinkage():
    baseline_df, variant_df = _deterministic_bootstrap_fixture()
    group_code_map = {0: 0, 1: 1, 2: 2}
    observed_delta, ci_low, ci_high = joint_patient_bootstrap_delta_range(
        baseline_df, variant_df, "y", n_groups=3, group_code_map=group_code_map,
        min_visits=3, n_boot=200, seed=0,
    )
    assert observed_delta == pytest.approx(-0.21, abs=1e-9)
    assert ci_low == pytest.approx(-0.21, abs=0.02)
    assert ci_high == pytest.approx(-0.21, abs=0.02)


def test_joint_patient_bootstrap_delta_range_rejects_misaligned_hadm_ids():
    baseline_df, variant_df = _deterministic_bootstrap_fixture(n_per_group=5)
    variant_df = variant_df.iloc[::-1].reset_index(drop=True)  # reversed order
    with pytest.raises(ValueError):
        joint_patient_bootstrap_delta_range(
            baseline_df, variant_df, "y", n_groups=3, group_code_map={0: 0, 1: 1, 2: 2},
            min_visits=1, n_boot=10, seed=0,
        )


def _write_synthetic_eval_dir(eval_dir: Path, jaccard_shift: float = 0.0):
    eval_dir.mkdir(parents=True, exist_ok=True)
    n = 80
    rng = np.random.default_rng(0)
    hadm_ids = np.arange(1000, 1000 + n)
    subject_ids = hadm_ids
    groups = np.array([0] * 40 + [1] * 40)
    split = np.array(["test"] * n)

    y_gt = np.zeros((n, 4), dtype=np.uint8)
    y_gt[:, [0, 1]] = 1
    base_prob = np.where(groups[:, None] == 0, 0.6, 0.4) + jaccard_shift
    y_prob = np.clip(base_prob + rng.normal(0, 0.01, size=(n, 4)), 0.01, 0.99).astype(np.float32)
    y_pred = (y_prob >= 0.5).astype(np.uint8)

    npz_arrays = {
        "patient_index": np.arange(n, dtype=np.int64), "visit_index": np.zeros(n, dtype=np.int64),
        "HADM_ID": hadm_ids.astype(np.int64), "SUBJECT_ID": subject_ids.astype(np.int64),
        "split": split, "n_diag": np.full(n, 3, dtype=np.int64),
        "n_proc": np.zeros(n, dtype=np.int64), "n_med_gt": np.full(n, 2, dtype=np.int64),
        "y_gt": y_gt, "y_pred": y_pred, "y_prob": y_prob,
    }
    np.savez(eval_dir / "per_visit_predictions.npz", **npz_arrays)

    precision = (y_gt & y_pred).sum(axis=1) / np.maximum(y_pred.sum(axis=1), 1)
    recall = (y_gt & y_pred).sum(axis=1) / y_gt.sum(axis=1)
    union = np.logical_or(y_gt, y_pred).sum(axis=1)
    inter = np.logical_and(y_gt, y_pred).sum(axis=1)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    # n_dx/n_med_gt/visit_index must vary across rows -- adjusted_group_means's
    # cell-means design has no separate intercept, so a constant covariate
    # column is an exact linear combination of the group-dummy columns (whose
    # row sum is always 1) and makes cluster_robust_ols's X'X singular (see
    # tests/test_safedrug_cluster_gap.py:98's identical fixture comment).
    n_dx_vals = rng.integers(3, 15, size=n)
    n_med_gt_vals = rng.integers(1, 6, size=n)
    visit_index_vals = rng.integers(0, 4, size=n)
    per_visit = pd.DataFrame({
        "HADM_ID": hadm_ids, "SUBJECT_ID": subject_ids, "split": split,
        "long_k10": groups.astype(float), "has_label": True, "n_dx": n_dx_vals,
        "n_med_gt": n_med_gt_vals, "visit_index": visit_index_vals, "jaccard": jaccard,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / np.maximum(precision + recall, 1e-9),
        "prauc": 0.5, "ddi_rate_visit": 0.0, "n_med_pred": y_pred.sum(axis=1),
    })
    per_visit.to_csv(eval_dir / "per_visit_metrics.csv", index=False)

    perm_rows = []
    for outcome in ("jaccard", "ddi_rate_visit"):
        for source in ("raw", "residual"):
            perm_rows.append({
                "partition": "long_k10", "scope": "test", "outcome": outcome,
                "source": source, "statistic": "range", "observed": 0.1,
                "null_mean": 0.05, "null_p95": 0.08, "z": 2.0, "p_raw": 0.01, "p_bonferroni": 0.05,
            })
    pd.DataFrame(perm_rows).to_csv(eval_dir / "table_permutation.csv", index=False)

    manifest = {
        "official_metrics": {"test": {"ja": 0.5, "prauc": 0.5, "avg_f1": 0.5,
                                       "ddi_rate": 0.0, "avg_med": 2.0}},
        "best_epoch": 10, "best_eval_jaccard": 0.5,
    }
    (eval_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return eval_dir


def test_flat_test_metrics_reads_manifest_and_recomputes_ddi(tmp_path):
    eval_dir = _write_synthetic_eval_dir(tmp_path / "baseline")
    ddi_adj = DDI_ADJ
    out = flat_test_metrics(eval_dir, ddi_adj)
    assert out["n_test_visits"] == 80
    assert out["official_jaccard"] == pytest.approx(0.5)
    assert out["official_metrics_reflect_variant"] is True
    assert 0.0 <= out["jaccard_mean"] <= 1.0


def test_flat_test_metrics_flags_threshold_variant_manifests(tmp_path):
    eval_dir = _write_synthetic_eval_dir(tmp_path / "variant")
    manifest = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["threshold_policy"] = {"variant": "group_tuned"}
    (eval_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = flat_test_metrics(eval_dir, DDI_ADJ)
    assert out["official_metrics_reflect_variant"] is False


def test_variant_gap_table_combines_permutation_csv_and_precision_recall(tmp_path):
    eval_dir = _write_synthetic_eval_dir(tmp_path / "baseline")
    out = variant_gap_table(eval_dir)
    assert set(out["outcome"]) == {"jaccard", "ddi_rate_visit", "precision", "recall"}
    jac_raw = out[(out["outcome"] == "jaccard") & (out["source"] == "raw")]
    assert jac_raw["observed"].iloc[0] == pytest.approx(0.1)
    pr = out[out["outcome"] == "precision"]
    assert (pr["source"] == "raw").all()


def test_build_mitigation_tables_writes_all_outputs_and_figures(tmp_path):
    baseline_dir = _write_synthetic_eval_dir(tmp_path / "baseline", jaccard_shift=0.0)
    variant_dir = _write_synthetic_eval_dir(tmp_path / "variant", jaccard_shift=0.05)
    out_dir = tmp_path / "compare"

    build_mitigation_tables(
        baseline_dir, {"my_variant": variant_dir}, out_dir, n_boot=20, seed=0, lang="ko",
    )

    assert (out_dir / "table_mitigation_overall.csv").exists()
    assert (out_dir / "table_mitigation_groups.csv").exists()
    assert (out_dir / "table_mitigation_gap.csv").exists()
    assert (out_dir / "figs" / "fig_mitigation_gap.png").exists()
    assert (out_dir / "figs" / "fig_mitigation_groups.png").exists()
    assert (out_dir / "REPORT_MITIGATION_KO.md").exists()

    overall = pd.read_csv(out_dir / "table_mitigation_overall.csv")
    assert set(overall["variant"]) == {"baseline", "my_variant"}
    gap = pd.read_csv(out_dir / "table_mitigation_gap.csv")
    my_variant_jac_range = gap[
        (gap["variant"] == "my_variant") & (gap["outcome"] == "jaccard")
        & (gap["source"] == "raw") & (gap["statistic"] == "range")
    ]
    assert len(my_variant_jac_range) == 1
    assert not pd.isna(my_variant_jac_range["delta_vs_baseline"].iloc[0])
