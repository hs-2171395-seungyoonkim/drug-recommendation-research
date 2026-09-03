import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_k_sweep import (
    _build_permutation_target,
    attach_variant_label,
    config_gap_stats,
    run_sweep,
)


def test_attach_variant_label_maps_by_hadm_id(tmp_path):
    dxtext = pd.DataFrame({"HADM_ID": [1, 2, 3, 4]})
    dxtext_path = tmp_path / "dxtext.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "labels.npz"
    np.savez(labels_path, long_k5=np.array([0, 1, 2, 0], dtype=np.int32))

    df = pd.DataFrame({"HADM_ID": [4, 3, 1, 999]})
    out = attach_variant_label(df, dxtext_path, labels_path, "long_k5")
    assert out.tolist()[:3] == [0, 2, 0]
    assert pd.isna(out.tolist()[3])


def _fake_df(n_groups=3, per_group=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    hid = 0
    for g in range(n_groups):
        for _ in range(per_group):
            rows.append({
                "HADM_ID": hid, "SUBJECT_ID": hid // 2,
                "n_dx": rng.integers(3, 15), "n_med_gt": rng.integers(1, 20),
                "visit_index": rng.integers(0, 4),
                "jaccard": rng.normal(0.3 + 0.1 * g, 0.05),
                "cfg": g,
            })
            hid += 1
    return pd.DataFrame(rows)


def test_config_gap_stats_shape_and_values():
    df = _fake_df()
    stats = config_gap_stats(df, "cfg")
    assert stats["n_groups_total"] == 3
    assert stats["min_group_size"] == 50
    assert stats["raw_range"] > 0
    assert stats["n_groups_used_raw"] == 3


def test_build_permutation_target_quota_sums_to_patients():
    df = _fake_df()
    stats = config_gap_stats(df, "cfg")
    target, patients, patient_of_visit = _build_permutation_target(
        stats["group_codes"], stats["subject"], stats["n_groups"]
    )
    assert target.sum() == len(patients)
    assert len(patient_of_visit) == len(df)


def test_run_sweep_selection_corrected_p_is_at_least_raw_p(tmp_path):
    dxtext = pd.DataFrame({"HADM_ID": list(range(300))})
    dxtext_path = tmp_path / "dxtext.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(1)
    labels_path = tmp_path / "labels.npz"
    np.savez(
        labels_path,
        cfgA_k2=rng.integers(0, 2, 300).astype(np.int32),
        cfgB_k2=rng.integers(0, 2, 300).astype(np.int32),
        cfgC_k2=rng.integers(0, 2, 300).astype(np.int32),
    )

    import safedrug_k_sweep as sks
    sks_dxtext_backup, sks_labels_backup = sks.DXTEXT_CSV, sks.LABELS_NPZ
    sks.DXTEXT_CSV = dxtext_path
    sks.LABELS_NPZ = labels_path
    try:
        df = pd.DataFrame({
            "HADM_ID": range(300), "SUBJECT_ID": np.arange(300) // 2,
            "n_dx": rng.integers(3, 15, 300), "n_med_gt": rng.integers(1, 20, 300),
            "visit_index": rng.integers(0, 4, 300), "jaccard": rng.normal(0.5, 0.1, 300),
            "ccs_group": rng.choice(["A", "B"], 300),
            "long_k10": rng.integers(0, 3, 300), "long_k25": rng.integers(0, 3, 300),
            "short_k10": rng.integers(0, 3, 300), "concise_k10": rng.integers(0, 3, 300),
        })
        configs = [("cfgA", 2, "cfgA_k2"), ("cfgB", 2, "cfgB_k2"), ("cfgC", 2, "cfgC_k2")]
        quality = pd.DataFrame({"변형": [], "k": [], "실루엣": [], "ARI_시드간": [],
                                "최소군집": [], "평균순위": []})
        table = run_sweep({"seed0": df}, configs, ["long_k10"], quality, n_perm=100, seed=0)
    finally:
        sks.DXTEXT_CSV, sks.LABELS_NPZ = sks_dxtext_backup, sks_labels_backup

    dx = table[~table["is_reference"]]
    for _, row in dx.iterrows():
        if not np.isnan(row["p_raw_range_adj"]) and not np.isnan(row["p_selcorr_range_adj"]):
            assert row["p_selcorr_range_adj"] >= row["p_raw_range_adj"] - 1e-9


def test_run_sweep_qualifies_flag_planted_pass_and_fail(tmp_path):
    dxtext = pd.DataFrame({"HADM_ID": list(range(200))})
    dxtext_path = tmp_path / "dxtext.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "labels.npz"
    # cfgPass: 2 big well-separated groups (100 each). cfgFail: one tiny group (<30).
    pass_labels = np.array([0] * 100 + [1] * 100, dtype=np.int32)
    fail_labels = np.array([0] * 190 + [1] * 10, dtype=np.int32)
    np.savez(labels_path, cfgPass_k2=pass_labels, cfgFail_k2=fail_labels)

    import safedrug_k_sweep as sks
    backup = sks.DXTEXT_CSV, sks.LABELS_NPZ
    sks.DXTEXT_CSV, sks.LABELS_NPZ = dxtext_path, labels_path
    try:
        rng = np.random.default_rng(2)
        jaccard = np.concatenate([rng.normal(0.2, 0.02, 100), rng.normal(0.8, 0.02, 100)])
        df = pd.DataFrame({
            "HADM_ID": range(200), "SUBJECT_ID": np.arange(200) // 2,
            "n_dx": rng.integers(3, 15, 200), "n_med_gt": rng.integers(1, 20, 200),
            "visit_index": rng.integers(0, 4, 200), "jaccard": jaccard,
        })
        configs = [("cfgPass", 2, "cfgPass_k2"), ("cfgFail", 2, "cfgFail_k2")]
        quality = pd.DataFrame({
            "변형": ["cfgPass", "cfgFail"], "k": [2, 2], "실루엣": [0.5, 0.5],
            "ARI_시드간": [0.95, 0.95], "최소군집": [100, 10], "평균순위": [1.0, 2.0],
        })
        table = run_sweep({"seed0": df}, configs, [], quality, n_perm=200, seed=0)
    finally:
        sks.DXTEXT_CSV, sks.LABELS_NPZ = backup

    row_pass = table[table["variant"] == "cfgPass"].iloc[0]
    row_fail = table[table["variant"] == "cfgFail"].iloc[0]
    assert bool(row_pass["meets_min_group"]) is True
    assert bool(row_fail["meets_min_group"]) is False
    assert bool(row_fail["qualifies"]) is False


def test_run_sweep_reference_rows_have_nan_selection_corrected_p():
    df = _fake_df(n_groups=2, per_group=40)
    df = df.rename(columns={"cfg": "long_k10"})
    quality = pd.DataFrame({"변형": [], "k": [], "실루엣": [], "ARI_시드간": [],
                            "최소군집": [], "평균순위": []})
    table = run_sweep({"seed0": df}, [], ["long_k10"], quality, n_perm=50, seed=0)
    ref = table[table["is_reference"]].iloc[0]
    assert np.isnan(ref["p_selcorr_range_adj"])
