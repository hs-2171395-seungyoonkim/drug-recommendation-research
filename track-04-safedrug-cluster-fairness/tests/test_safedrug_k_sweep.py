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
    make_ksweep_figure,
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


def test_run_sweep_reference_row_k_matches_partition_own_k():
    """D-D5: reference rows report the partition's own k (long_k10/short_k10/
    concise_k10 -> 10, long_k25 -> 25), NaN only for ccs_group."""
    df = _fake_df(n_groups=2, per_group=40)
    df = df.rename(columns={"cfg": "long_k10"})
    df["long_k25"] = df["long_k10"]
    df["short_k10"] = df["long_k10"]
    df["concise_k10"] = df["long_k10"]
    df["ccs_group"] = df["long_k10"].map({0: "A", 1: "B"})
    quality = pd.DataFrame({"변형": [], "k": [], "실루엣": [], "ARI_시드간": [],
                            "최소군집": [], "평균순위": []})
    partitions = ["long_k10", "short_k10", "concise_k10", "long_k25", "ccs_group"]
    table = run_sweep({"seed0": df}, [], partitions, quality, n_perm=20, seed=0)
    ref = table[table["is_reference"]].set_index("variant")
    assert ref.loc["long_k10", "k"] == 10
    assert ref.loc["short_k10", "k"] == 10
    assert ref.loc["concise_k10", "k"] == 10
    assert ref.loc["long_k25", "k"] == 25
    assert np.isnan(ref.loc["ccs_group", "k"])


def test_attach_variant_label_raises_on_length_mismatch(tmp_path):
    """Same positional-alignment invariant as
    safedrug_percluster/metrics.py::attach_labels: dxtext_csv and
    labels_npz[key] must have equal length, since the mapping is built by
    zipping them positionally. A silent length mismatch would mis-map
    HADM_IDs to the wrong labels rather than raising."""
    dxtext = pd.DataFrame({"HADM_ID": [1, 2, 3, 4, 5]})
    dxtext_path = tmp_path / "dxtext.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "labels.npz"
    # Planted mismatch: only 4 labels for 5 dxtext rows.
    np.savez(labels_path, long_k5=np.array([0, 1, 2, 0], dtype=np.int32))

    df = pd.DataFrame({"HADM_ID": [1, 2, 3]})
    with pytest.raises(ValueError):
        attach_variant_label(df, dxtext_path, labels_path, "long_k5")


def test_run_sweep_z_standardized_selection_correction_recovers_dominated_small_k_signal(tmp_path):
    """Controller ruling (Task D fix round 1): the raw max-statistic
    correction is dominated by high-k configs (range grows with k under the
    null too, since more groups means smaller, noisier per-group means), so
    a real small-k effect can be swamped by a noisy large-k config's null.
    Standardizing each config's null to its own mean/sd before taking the
    cross-config max removes that scale dependence.

    Deterministic construction (fixed seeds): a genuine, modest 2-group
    effect ("small", gap=0.15 against a sigma=0.8 noise floor) vs. a
    pure-noise 20-group config ("large", label unrelated to jaccard) whose
    own null -- inflated by splitting the same N into far more, smaller
    groups -- sits at or above "small"'s real observed range on almost every
    permutation round. That swamps the raw max-statistic correction
    (p_selcorr_range_raw -> ~1.0) while the per-config standardized
    correction (p_selcorr_z_range_raw) recovers "small"'s real significance.
    """
    N = 900
    dxtext = pd.DataFrame({"HADM_ID": list(range(N))})
    dxtext_path = tmp_path / "dxtext.csv"
    dxtext.to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    K_LARGE = 20
    rng = np.random.default_rng(7)
    small_labels = np.array([0] * (N // 2) + [1] * (N - N // 2), dtype=np.int32)
    large_labels = rng.integers(0, K_LARGE, N).astype(np.int32)
    labels_path = tmp_path / "labels.npz"
    np.savez(labels_path, small_k2=small_labels, large_klarge=large_labels)

    import safedrug_k_sweep as sks
    backup = sks.DXTEXT_CSV, sks.LABELS_NPZ
    sks.DXTEXT_CSV, sks.LABELS_NPZ = dxtext_path, labels_path
    try:
        small_gap, sigma, mid = 0.15, 0.8, 0.5
        jaccard = np.concatenate([
            rng.normal(mid - small_gap / 2, sigma, N // 2),
            rng.normal(mid + small_gap / 2, sigma, N - N // 2),
        ])
        df = pd.DataFrame({
            "HADM_ID": range(N), "SUBJECT_ID": np.arange(N) // 2,
            "n_dx": rng.integers(3, 15, N), "n_med_gt": rng.integers(1, 20, N),
            "visit_index": rng.integers(0, 4, N), "jaccard": jaccard,
        })
        configs = [("small", 2, "small_k2"), ("large", K_LARGE, "large_klarge")]
        quality = pd.DataFrame({"변형": [], "k": [], "실루엣": [], "ARI_시드간": [],
                                "최소군집": [], "평균순위": []})
        table = run_sweep({"seed0": df}, configs, [], quality, n_perm=300, seed=0)
    finally:
        sks.DXTEXT_CSV, sks.LABELS_NPZ = backup

    small = table[table["variant"] == "small"].iloc[0]
    # Raw max-statistic correction is swamped by "large"'s inflated null.
    assert small["p_selcorr_range_raw"] > 0.9
    # The z-standardized correction recovers "small"'s real signal.
    assert small["p_selcorr_z_range_raw"] < 0.05
    assert small["p_selcorr_z_range_raw"] <= small["p_selcorr_range_raw"]

    # p_selcorr_z >= p_raw by construction, for every (non-reference) config
    # and every stat: the standardized cross-config max (which includes that
    # config's own standardized null) can only be >= that config's own
    # standardized value -- same logic as the existing raw
    # p_selcorr >= p_raw property, one level up on the standardized scale.
    dx = table[~table["is_reference"]]
    for _, row in dx.iterrows():
        for name in ["range_raw", "wsd_raw", "range_adj", "wsd_adj"]:
            p_raw = row[f"p_raw_{name}"]
            p_sel_z = row[f"p_selcorr_z_{name}"]
            if not np.isnan(p_raw) and not np.isnan(p_sel_z):
                assert p_sel_z >= p_raw - 1e-9


def test_make_ksweep_figure_smoke_lang_ko_writes_ko_suffixed_png(tmp_path):
    # Task G item 3: tiny synthetic sweep table with every column
    # make_ksweep_figure reads (built directly, not via run_sweep, to keep
    # this fast and independent of the permutation machinery), lang="ko" --
    # checks the _ko-suffixed file is written and the plain (English
    # filename) file is not.
    table = pd.DataFrame(
        {
            "source": ["seed0", "seed0", "seed0", "seed0"],
            "variant": ["short", "short", "long", "long"],
            "k": [2, 4, 2, 4],
            "is_reference": [False, False, False, False],
            "adj_range": [0.10, 0.12, 0.15, 0.18],
            "meets_significance": [False, True, False, True],
            "ARI_시드간": [0.9, 0.88, 0.92, 0.86],
        }
    )
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    make_ksweep_figure(table, figs_dir, lang="ko")

    assert (figs_dir / "fig_k_sweep_ko.png").exists()
    assert not (figs_dir / "fig_k_sweep.png").exists()
