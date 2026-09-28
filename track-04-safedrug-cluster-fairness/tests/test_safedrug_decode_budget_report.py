import json
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_decode_budget_report import (
    build_table_e_excess_gap,
    build_table_e_groups,
    build_table_e_jaccard_gap,
    build_table_e_overall,
    main,
)
from safedrug_mitigation_compare import _is_posthoc, flat_test_metrics

DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


# ======================================================= R5: _is_posthoc
def test_is_posthoc_true_for_decode_policy():
    assert _is_posthoc({"decode_policy": {"variant": "ccs"}}) is True


def test_is_posthoc_true_for_threshold_policy():
    assert _is_posthoc({"threshold_policy": {"variant": "group_tuned"}}) is True


def test_is_posthoc_false_for_plain_manifest():
    assert _is_posthoc({"best_epoch": 10}) is False


def _write_flat_metrics_fixture(d: Path, jaccard: float, decode_policy: dict = None):
    d.mkdir(parents=True, exist_ok=True)
    n = 10
    y_pred = np.zeros((n, 4), dtype=np.uint8)
    y_pred[:, [0, 1]] = 1
    split = np.array(["test"] * n)
    np.savez(d / "per_visit_predictions.npz", split=split, y_pred=y_pred)

    per_visit = pd.DataFrame({
        "split": split, "jaccard": [jaccard] * n, "f1": [jaccard] * n,
        "prauc": [0.5] * n, "n_med_pred": [2] * n,
    })
    per_visit.to_csv(d / "per_visit_metrics.csv", index=False)

    manifest = {"official_metrics": {"test": {
        "ja": 0.5, "prauc": 0.5, "avg_f1": 0.5, "ddi_rate": 0.0, "avg_med": 2.0,
    }}}
    if decode_policy is not None:
        manifest["decode_policy"] = decode_policy
    (d / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_flat_test_metrics_flags_decode_policy_variant_manifests(tmp_path):
    d = tmp_path / "variant"
    _write_flat_metrics_fixture(d, jaccard=0.6, decode_policy={"variant": "ccs", "p_floor": 0.35})
    out = flat_test_metrics(d, DDI_ADJ)
    assert out["official_metrics_reflect_variant"] is False


# ======================================================= build_table_e_overall
def test_build_table_e_overall_reads_flat_metrics_and_p_floor(tmp_path):
    baseline_dir = tmp_path / "baseline"
    _write_flat_metrics_fixture(baseline_dir, jaccard=0.5)

    variant_root = tmp_path / "variants"
    _write_flat_metrics_fixture(variant_root / "seed0" / "ccs", jaccard=0.6,
                                 decode_policy={"variant": "ccs", "p_floor": 0.35})
    _write_flat_metrics_fixture(variant_root / "seed0" / "k10", jaccard=0.55,
                                 decode_policy={"variant": "k10", "p_floor": 0.40})
    _write_flat_metrics_fixture(variant_root / "seed0" / "global", jaccard=0.52,
                                 decode_policy={"variant": "global", "p_floor": 0.45})

    out = build_table_e_overall([baseline_dir], variant_root, DDI_ADJ)
    assert set(out["variant"]) == {"baseline", "ccs", "k10", "global"}
    row = out[(out["seed"] == 0) & (out["variant"] == "ccs")].iloc[0]
    assert row["jaccard_mean"] == pytest.approx(0.6)
    assert row["p_floor"] == pytest.approx(0.35)
    base_row = out[(out["seed"] == 0) & (out["variant"] == "baseline")].iloc[0]
    assert np.isnan(base_row["p_floor"])


# ======================================================= build_table_e_excess_gap
def _write_excess_perm_fixture(d: Path, values: dict):
    """values: {(axis, statistic): (observed, p_raw)}"""
    (d / "excess_ddi").mkdir(parents=True, exist_ok=True)
    rows = [
        {"partition": axis, "statistic": stat, "observed": obs, "null_mean": 0.0,
         "null_p95": 0.0, "z": 0.0, "p_raw": p}
        for (axis, stat), (obs, p) in values.items()
    ]
    pd.DataFrame(rows).to_csv(d / "excess_ddi" / "table_excess_ddi_permutation.csv", index=False)


def test_build_table_e_excess_gap_includes_mean_over_seeds_row(tmp_path):
    variant_root = tmp_path / "variants"
    baseline_dirs = []
    for seed, obs in enumerate([0.30, 0.20]):
        baseline_dir = tmp_path / f"baseline{seed}"
        values = {
            ("long_k10", "range"): (obs, 0.01), ("long_k10", "weighted_sd"): (obs / 2, 0.01),
            ("ccs_group", "range"): (obs, 0.01), ("ccs_group", "weighted_sd"): (obs / 2, 0.01),
        }
        _write_excess_perm_fixture(baseline_dir, values)
        baseline_dirs.append(baseline_dir)
        for variant in ("ccs", "k10", "global"):
            _write_excess_perm_fixture(variant_root / f"seed{seed}" / variant, values)

    out = build_table_e_excess_gap(baseline_dirs, variant_root)
    per_seed = out[out["seed"] != "mean"]
    assert len(per_seed) == 2 * 4 * 2 * 2  # seeds x variants x axes x statistics

    mean_row = out[
        (out["seed"] == "mean") & (out["variant"] == "baseline")
        & (out["axis"] == "long_k10") & (out["statistic"] == "range")
    ].iloc[0]
    assert mean_row["observed"] == pytest.approx((0.30 + 0.20) / 2)


# ======================================================= build_table_e_jaccard_gap
def _write_jaccard_perm_fixture(d: Path, observed: float, p_bonf: float):
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {"partition": "long_k10", "scope": "test", "outcome": "jaccard", "source": src,
         "statistic": "range", "observed": observed, "null_mean": 0.0, "null_p95": 0.0,
         "z": 0.0, "p_raw": p_bonf, "p_bonferroni": p_bonf}
        for src in ("raw", "residual")
    ]
    pd.DataFrame(rows).to_csv(d / "table_permutation.csv", index=False)


def test_build_table_e_jaccard_gap_includes_mean_over_seeds_row(tmp_path):
    variant_root = tmp_path / "variants"
    baseline_dirs = []
    for seed, obs in enumerate([0.10, 0.14]):
        baseline_dir = tmp_path / f"baseline{seed}"
        _write_jaccard_perm_fixture(baseline_dir, obs, 0.04)
        baseline_dirs.append(baseline_dir)
        for variant in ("ccs", "k10", "global"):
            _write_jaccard_perm_fixture(variant_root / f"seed{seed}" / variant, obs - 0.02, 0.06)

    out = build_table_e_jaccard_gap(baseline_dirs, variant_root)
    per_seed = out[out["seed"] != "mean"]
    assert len(per_seed) == 2 * 4 * 2  # seeds x variants x sources(raw,residual)

    mean_row = out[
        (out["seed"] == "mean") & (out["variant"] == "baseline") & (out["source"] == "raw")
    ].iloc[0]
    assert mean_row["observed"] == pytest.approx((0.10 + 0.14) / 2)


# ======================================================= build_table_e_groups
def _write_groups_fixture(d: Path):
    (d / "excess_ddi").mkdir(parents=True, exist_ok=True)
    # long_k10 group written the way a real attach_labels float-upcast +
    # pd.concat's mixed-partition "group" column actually round-trips it
    # ("6.0", not 6) -- see safedrug_cluster_gap._normalize_group_id's own
    # docstring (safedrug_cluster_gap.py:325-360).
    excess = pd.DataFrame({
        "partition": ["long_k10", "ccs_group"], "group": ["6.0", "Acute MI"],
        "n_visits": [40, 40], "n_patients": [40, 40],
        "mean_ddi_pred": [0.08, 0.09], "ci_low_ddi_pred": [0.06, 0.07], "ci_high_ddi_pred": [0.10, 0.11],
        "mean_ddi_true": [0.05, 0.05], "ci_low_ddi_true": [0.03, 0.03], "ci_high_ddi_true": [0.07, 0.07],
        "mean_excess_ddi": [0.03, 0.04], "ci_low_excess_ddi": [0.01, 0.02], "ci_high_excess_ddi": [0.05, 0.06],
    })
    excess.to_csv(d / "excess_ddi" / "table_excess_ddi_groups.csv", index=False)

    adjusted = pd.DataFrame({
        "partition": ["long_k10", "ccs_group"], "scope": ["test", "test"],
        "outcome": ["jaccard", "jaccard"], "group": ["6.0", "Acute MI"],
        "raw_mean": [0.5, 0.5], "adjusted_mean": [0.48, 0.51],
    })
    adjusted.to_csv(d / "table_adjusted_means.csv", index=False)


def test_build_table_e_groups_normalizes_mixed_dtype_group_ids_and_averages_seeds(tmp_path):
    variant_root = tmp_path / "variants"
    baseline_dirs = []
    for seed in (0, 1):
        baseline_dir = tmp_path / f"baseline{seed}"
        _write_groups_fixture(baseline_dir)
        baseline_dirs.append(baseline_dir)
        for variant in ("ccs", "k10", "global"):
            _write_groups_fixture(variant_root / f"seed{seed}" / variant)

    out = build_table_e_groups(baseline_dirs, variant_root)

    k10_row = out[(out["variant"] == "baseline") & (out["axis"] == "long_k10")].iloc[0]
    assert k10_row["group"] == 6  # normalized to a plain int, not the string "6.0"
    assert k10_row["adj_jaccard"] == pytest.approx(0.48)  # correctly joined despite the CSV dtype mismatch
    assert k10_row["n_seeds"] == 2
    assert k10_row["mean_excess_ddi"] == pytest.approx(0.03)
    assert k10_row["min_excess_ddi"] == pytest.approx(0.03)
    assert k10_row["max_excess_ddi"] == pytest.approx(0.03)

    ccs_row = out[(out["variant"] == "baseline") & (out["axis"] == "ccs_group")].iloc[0]
    assert ccs_row["group"] == "Acute MI"
    assert ccs_row["adj_jaccard"] == pytest.approx(0.51)


def _write_groups_fixture_mixed_dtypes(d: Path):
    """Same shape as _write_groups_fixture, but the two source tables carry
    DIFFERENT raw encodings of the same long_k10 id -- unlike
    _write_groups_fixture, which puts the identical string "6.0" in both,
    this cannot be joined by plain string equality.

    excess_ddi's "group" column here is [6, "Acute MI"] (a Python int mixed
    with a CCS name): pandas upcasts the column to object dtype, and to_csv
    then writes the int as the bare text "6" (verified: pd.DataFrame({
    "group": [6, "Acute MI"]}).to_csv(...) round-trips through read_csv as
    the string "6", not "6.0"). table_adjusted_means's "group" column is
    ["6.0", "Acute MI"] -- already the string "6.0", the float-upcast
    round-trip _write_groups_fixture's own docstring describes. So after
    read_csv, the excess side's long_k10 id is the string "6" and the
    adjusted side's is the string "6.0": two different raw strings for the
    same id, both of which _normalize_group_id must map to the plain int 6
    for build_table_e_groups's per-row match() lookup to find any match at
    all.
    """
    (d / "excess_ddi").mkdir(parents=True, exist_ok=True)
    excess = pd.DataFrame({
        "partition": ["long_k10", "ccs_group"], "group": [6, "Acute MI"],
        "n_visits": [40, 40], "n_patients": [40, 40],
        "mean_ddi_pred": [0.08, 0.09], "ci_low_ddi_pred": [0.06, 0.07], "ci_high_ddi_pred": [0.10, 0.11],
        "mean_ddi_true": [0.05, 0.05], "ci_low_ddi_true": [0.03, 0.03], "ci_high_ddi_true": [0.07, 0.07],
        "mean_excess_ddi": [0.03, 0.04], "ci_low_excess_ddi": [0.01, 0.02], "ci_high_excess_ddi": [0.05, 0.06],
    })
    excess.to_csv(d / "excess_ddi" / "table_excess_ddi_groups.csv", index=False)

    adjusted = pd.DataFrame({
        "partition": ["long_k10", "ccs_group"], "scope": ["test", "test"],
        "outcome": ["jaccard", "jaccard"], "group": ["6.0", "Acute MI"],
        "raw_mean": [0.5, 0.5], "adjusted_mean": [0.48, 0.51],
    })
    adjusted.to_csv(d / "table_adjusted_means.csv", index=False)


def test_build_table_e_groups_joins_across_differently_encoded_numeric_group_ids(tmp_path):
    # Regression guard for a normalization bug _write_groups_fixture's own
    # identical-string ("6.0" on both sides) fixture cannot catch: here the
    # excess-ddi table's long_k10 id round-trips as the bare string "6"
    # while the adjusted-means table's round-trips as "6.0" -- genuinely
    # different raw encodings of the same id. If _normalize_group_id ever
    # stopped mapping both consistently to the same plain int, the two
    # sides would fail to match and adj_jaccard would silently fall back to
    # NaN instead of raising -- so this asserts a single joined row with
    # both sides' values present (no NaN).
    variant_root = tmp_path / "variants"
    baseline_dirs = []
    for seed in (0, 1):
        baseline_dir = tmp_path / f"baseline{seed}"
        _write_groups_fixture_mixed_dtypes(baseline_dir)
        baseline_dirs.append(baseline_dir)
        for variant in ("ccs", "k10", "global"):
            _write_groups_fixture_mixed_dtypes(variant_root / f"seed{seed}" / variant)

    out = build_table_e_groups(baseline_dirs, variant_root)

    k10_rows = out[(out["variant"] == "baseline") & (out["axis"] == "long_k10")]
    assert len(k10_rows) == 1  # excess side's "6" and adjusted side's "6.0" joined into ONE row
    k10_row = k10_rows.iloc[0]
    assert k10_row["group"] == 6  # normalized to a plain int on both sides
    assert not pd.isna(k10_row["adj_jaccard"])  # the join found the adjusted-means row, not a NaN fallback
    assert k10_row["adj_jaccard"] == pytest.approx(0.48)
    assert k10_row["n_seeds"] == 2
    assert not pd.isna(k10_row["mean_excess_ddi"])
    assert k10_row["mean_excess_ddi"] == pytest.approx(0.03)


# ======================================================= main() smoke test
def _write_ddi_adj_pkl(tmp_path) -> Path:
    p = tmp_path / "ddi_adj.pkl"
    with p.open("wb") as fh:
        dill.dump(DDI_ADJ, fh)
    return p


def test_main_writes_all_outputs_and_figure(tmp_path):
    variant_root = tmp_path / "variants"
    baseline_dirs = []
    for seed in (0, 1):
        baseline_dir = tmp_path / f"baseline{seed}"
        _write_flat_metrics_fixture(baseline_dir, jaccard=0.5)
        _write_excess_perm_fixture(baseline_dir, {
            ("long_k10", "range"): (0.20, 0.03), ("long_k10", "weighted_sd"): (0.10, 0.03),
            ("ccs_group", "range"): (0.20, 0.03), ("ccs_group", "weighted_sd"): (0.10, 0.03),
        })
        _write_jaccard_perm_fixture(baseline_dir, 0.10, 0.04)
        _write_groups_fixture(baseline_dir)
        baseline_dirs.append(baseline_dir)

        for variant, p_floor in (("ccs", 0.35), ("k10", 0.40), ("global", 0.45)):
            d = variant_root / f"seed{seed}" / variant
            _write_flat_metrics_fixture(d, jaccard=0.55, decode_policy={"variant": variant, "p_floor": p_floor})
            _write_excess_perm_fixture(d, {
                ("long_k10", "range"): (0.15, 0.05), ("long_k10", "weighted_sd"): (0.08, 0.05),
                ("ccs_group", "range"): (0.15, 0.05), ("ccs_group", "weighted_sd"): (0.08, 0.05),
            })
            _write_jaccard_perm_fixture(d, 0.08, 0.06)
            _write_groups_fixture(d)

    out_dir = tmp_path / "report"
    main([
        "--baseline-dirs", *[str(d) for d in baseline_dirs],
        "--variant-root", str(variant_root), "--out-dir", str(out_dir),
        "--lang", "ko", "--ddi-adj", str(_write_ddi_adj_pkl(tmp_path)),
    ])

    for name in ("table_E_overall.csv", "table_E_excess_gap.csv", "table_E_jaccard_gap.csv",
                 "table_E_groups.csv", "REPORT_DECODE_BUDGET_KO.md"):
        assert (out_dir / name).exists()
    assert (out_dir / "figs" / "fig_E_excess_ddi.png").exists()
