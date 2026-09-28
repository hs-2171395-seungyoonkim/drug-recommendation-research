import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Meds 0 and 3 are the one DDI pair -- the repo's standard DDI test fixture.
DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


# ======================================================= compute_per_visit_table / overall_table
def test_compute_per_visit_table_hand_computed():
    from safedrug_ddi_pair_analysis import compute_per_visit_table

    y_gt = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    y_pred = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])
    out = compute_per_visit_table(y_gt, y_pred, DDI_ADJ)
    # row 0: gt={0,3}, pred={0} -> jaccard 1/2, precision 1/1, recall 1/2;
    #        ddi_true=1.0 (the only true pair IS DDI), ddi_pred=NaN (<2 preds) -> excess NaN.
    assert out.loc[0, "jaccard"] == pytest.approx(0.5)
    assert out.loc[0, "precision"] == pytest.approx(1.0)
    assert out.loc[0, "recall"] == pytest.approx(0.5)
    assert out.loc[0, "ddi_true"] == pytest.approx(1.0)
    assert np.isnan(out.loc[0, "ddi_pred"])
    assert np.isnan(out.loc[0, "excess"])
    # row 1: gt={1,2}, pred={1,2,3} -> jaccard 2/3; ddi_true=0.0 (pair {1,2} not DDI);
    #        ddi_pred: 3 pairs, none DDI -> 0.0; excess 0.0.
    assert out.loc[1, "jaccard"] == pytest.approx(2 / 3)
    assert out.loc[1, "ddi_true"] == pytest.approx(0.0)
    assert out.loc[1, "ddi_pred"] == pytest.approx(0.0)
    assert out.loc[1, "excess"] == pytest.approx(0.0)


def test_overall_table_aggregates_by_variant():
    from safedrug_ddi_pair_analysis import overall_table

    per_visit = pd.DataFrame({
        "variant": ["baseline", "baseline", "W"],
        "jaccard": [0.5, 0.7, 0.9],
        "precision": [0.5, 0.5, 0.5],
        "recall": [0.5, 0.5, 0.5],
        "n_pred": [2, 3, 4],
        "ddi_pred": [0.1, 0.2, 0.0],
        "ddi_true": [0.1, 0.1, 0.1],
        "excess": [0.0, 0.1, -0.1],
    })
    out = overall_table(per_visit).set_index("variant")
    assert out.loc["baseline", "n_visits"] == 2
    assert out.loc["baseline", "jaccard_mean"] == pytest.approx(0.6)
    assert out.loc["W", "n_visits"] == 1
    assert out.loc["W", "excess_mean"] == pytest.approx(-0.1)


# ======================================================= load_whitelist_by_group
def test_load_whitelist_by_group_builds_set_per_category():
    from safedrug_ddi_pair_analysis import load_whitelist_by_group

    whitelist_pairs = pd.DataFrame({
        "ccs_group": ["A", "A", "B"], "idx_a": [0, 1, 0], "idx_b": [3, 2, 3],
    })
    out = load_whitelist_by_group(whitelist_pairs)
    assert out == {"A": {(0, 3), (1, 2)}, "B": {(0, 3)}}


# ======================================================= build_pair_events
def test_build_pair_events_detects_missed_and_covered_pairs():
    from safedrug_ddi_pair_analysis import build_pair_events

    df = pd.DataFrame({"HADM_ID": [100, 101], "long_k10": [0, 0], "ccs_group": ["A", "A"]})
    y_gt = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
    y_pred = np.array([[1, 0, 0, 1], [1, 0, 0, 0]])  # visit0 covers both, visit1 misses med 3
    idx2atc = {0: "N02B", 1: "A01A", 2: "A02B", 3: "B01A"}
    out = build_pair_events(df, y_gt, y_pred, DDI_ADJ, idx2atc, whitelist_by_group={})
    assert len(out) == 2  # one flagged pair per visit
    assert out.iloc[0]["missed"] == False
    assert out.iloc[1]["missed"] == True
    assert list(out["atc_a"]) == ["N02B", "N02B"]
    assert list(out["atc_b"]) == ["B01A", "B01A"]


def test_build_pair_events_marks_whitelisted_membership():
    from safedrug_ddi_pair_analysis import build_pair_events

    df = pd.DataFrame({"HADM_ID": [100, 101], "long_k10": [0, 0], "ccs_group": ["A", "B"]})
    y_gt = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
    y_pred = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
    idx2atc = {0: "N02B", 1: "A01A", 2: "A02B", 3: "B01A"}
    whitelist_by_group = {"A": {(0, 3)}}  # B has no whitelist
    out = build_pair_events(df, y_gt, y_pred, DDI_ADJ, idx2atc, whitelist_by_group).set_index("HADM_ID")
    assert out.loc[100, "whitelisted"] == True
    assert out.loc[101, "whitelisted"] == False


def test_build_pair_events_skips_non_ddi_gt_pairs():
    from safedrug_ddi_pair_analysis import build_pair_events

    df = pd.DataFrame({"HADM_ID": [100], "long_k10": [0], "ccs_group": ["A"]})
    y_gt = np.array([[0, 1, 1, 0]])  # meds {1,2} -- not a DDI pair
    y_pred = np.array([[0, 1, 1, 0]])
    idx2atc = {0: "N02B", 1: "A01A", 2: "A02B", 3: "B01A"}
    out = build_pair_events(df, y_gt, y_pred, DDI_ADJ, idx2atc, whitelist_by_group={})
    assert out.empty


# ======================================================= miss_rate_by_group / whitelist split
def test_miss_rate_by_group_hand_computed():
    from safedrug_ddi_pair_analysis import miss_rate_by_group

    pair_events = pd.DataFrame({
        "variant": ["baseline"] * 3 + ["W"] * 2,
        "long_k10": [0, 0, 1, 0, 0],
        "missed": [True, False, True, False, False],
    })
    out = miss_rate_by_group(pair_events, "long_k10").set_index(["variant", "long_k10"])
    assert out.loc[("baseline", 0), "miss_rate"] == pytest.approx(0.5)
    assert out.loc[("baseline", 1), "miss_rate"] == pytest.approx(1.0)
    assert out.loc[("W", 0), "miss_rate"] == pytest.approx(0.0)


def test_miss_rate_by_group_drops_unlabeled_rows():
    from safedrug_ddi_pair_analysis import miss_rate_by_group

    pair_events = pd.DataFrame({
        "variant": ["baseline", "baseline"],
        "ccs_group": ["A", float("nan")],
        "missed": [True, True],
    })
    out = miss_rate_by_group(pair_events, "ccs_group")
    assert len(out) == 1
    assert out.iloc[0]["ccs_group"] == "A"


def test_miss_rate_inside_outside_whitelist_hand_computed():
    from safedrug_ddi_pair_analysis import miss_rate_inside_outside_whitelist

    pair_events = pd.DataFrame({
        "variant": ["baseline"] * 4,
        "whitelisted": [True, True, False, False],
        "missed": [False, True, True, True],
    })
    out = miss_rate_inside_outside_whitelist(pair_events).set_index("whitelisted")
    assert out.loc[True, "miss_rate"] == pytest.approx(0.5)
    assert out.loc[False, "miss_rate"] == pytest.approx(1.0)


# ======================================================= guideline_pair_table
def test_guideline_pair_table_pools_across_categories_and_includes_extended_pairs():
    from safedrug_ddi_pair_analysis import guideline_pair_table

    pair_events = pd.DataFrame({
        "variant": ["baseline", "baseline", "baseline"],
        "ccs_group": ["A", "B", "A"],
        "atc_a": ["N02A", "N02A", "X01A"],
        "atc_b": ["B01A", "B01A", "X02A"],
        "missed": [True, False, True],
    })
    whitelist_pairs = pd.DataFrame({
        "ccs_group": ["A"], "idx_a": [10], "idx_b": [11],
        "atc_a": ["X01A"], "atc_b": ["X02A"], "share": [0.15], "n_visits": [40],
    })
    out = guideline_pair_table(
        pair_events, whitelist_pairs, fixed_pairs=[("N02A", "B01A")], share_threshold=0.10,
    ).set_index(["atc_a", "atc_b"])
    # N02A-B01A pooled over categories A and B -> 1 missed of 2 -> 0.5.
    assert out.loc[("N02A", "B01A"), "miss_rate"] == pytest.approx(0.5)
    assert out.loc[("N02A", "B01A"), "n_pairs"] == 2
    # X01A-X02A included via the extended whitelist share (0.15 >= 0.10).
    assert out.loc[("X01A", "X02A"), "miss_rate"] == pytest.approx(1.0)


# ======================================================= excess-DDI split
def test_per_visit_excess_split_formula_hand_computed():
    from safedrug_ddi_pair_analysis import per_visit_excess_split

    ddi_adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])  # two DDI pairs: (0,1) and (2,3)
    df = pd.DataFrame({"HADM_ID": [100], "long_k10": [0], "ccs_group": ["A"]})
    y_gt = np.array([[1, 1, 1, 1]])    # true={0,1,2,3}: C(4,2)=6 pairs, 2 DDI
    y_pred = np.array([[1, 1, 0, 0]])  # pred={0,1}: 1 pair, DDI
    whitelist_by_group = {"A": {(0, 1)}}  # (0,1) whitelisted, (2,3) is "other"
    row = per_visit_excess_split(df, y_gt, y_pred, ddi_adj, whitelist_by_group).iloc[0]
    assert row["rate_pred_wl"] == pytest.approx(1.0)     # (0,1) is DDI+whitelisted, the only pred pair
    assert row["rate_pred_other"] == pytest.approx(0.0)
    assert row["rate_true_wl"] == pytest.approx(1 / 6)   # (0,1) whitelisted DDI, out of 6 true pairs
    assert row["rate_true_other"] == pytest.approx(1 / 6)  # (2,3) other DDI, out of 6 true pairs
    assert row["excess_wl"] == pytest.approx(1.0 - 1 / 6)
    assert row["excess_other"] == pytest.approx(0.0 - 1 / 6)


def test_per_visit_excess_split_components_sum_to_original_excess():
    from safedrug_ddi_pair_analysis import per_visit_excess_split
    from safedrug_percluster.metrics import _row_ddi_rate

    ddi_adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    df = pd.DataFrame({"HADM_ID": [100], "long_k10": [0], "ccs_group": ["A"]})
    y_gt = np.array([[1, 1, 1, 1]])
    y_pred = np.array([[1, 1, 0, 0]])
    whitelist_by_group = {"A": {(0, 1)}}
    out = per_visit_excess_split(df, y_gt, y_pred, ddi_adj, whitelist_by_group).iloc[0]
    expected_excess = _row_ddi_rate({0, 1}, ddi_adj) - _row_ddi_rate({0, 1, 2, 3}, ddi_adj)
    assert out["excess_wl"] + out["excess_other"] == pytest.approx(expected_excess)


def test_excess_split_by_cluster_aggregates_mean():
    from safedrug_ddi_pair_analysis import excess_split_by_cluster

    per_visit_excess = pd.DataFrame({
        "variant": ["baseline", "baseline", "baseline"],
        "long_k10": [0, 0, 1],
        "excess_wl": [0.2, 0.4, 0.1],
        "excess_other": [-0.1, -0.3, 0.0],
    })
    out = excess_split_by_cluster(per_visit_excess, "long_k10").set_index(["variant", "long_k10"])
    assert out.loc[("baseline", 0), "mean_excess_wl"] == pytest.approx(0.3)
    assert out.loc[("baseline", 0), "mean_excess_other"] == pytest.approx(-0.2)
    assert out.loc[("baseline", 0), "n_visits"] == 2


# ======================================================= parse_args / main
def test_parse_args_defaults_and_variant_append():
    from safedrug_ddi_pair_analysis import parse_args

    args = parse_args([
        "--baseline-dir", "B", "--variant", "W", "Wdir", "--variant", "no_ddi", "Ndir",
        "--whitelist-pairs", "wp.csv", "--out-dir", "OUT",
    ])
    assert args.baseline_dir == "B"
    assert args.variant == [["W", "Wdir"], ["no_ddi", "Ndir"]]
    assert args.lang == "en"


def _write_label_fixture(tmp_path):
    hadm_ids = [100, 101]
    dxtext_path = tmp_path / "dxtext.csv"
    pd.DataFrame({"HADM_ID": hadm_ids}).to_csv(dxtext_path, index=False, encoding="utf-8-sig")
    labels_path = tmp_path / "labels.npz"
    long_k10 = np.array([0, 0], dtype=np.int32)
    np.savez(labels_path, long_k10=long_k10, long_k25=long_k10, short_k10=long_k10, concise_k10=long_k10)
    ccs_path = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": hadm_ids, "group": ["A", "A"]}).to_csv(ccs_path, index=False, encoding="utf-8-sig")
    return dxtext_path, labels_path, ccs_path


def _write_dump(eval_dir: Path, y_gt, y_pred):
    eval_dir.mkdir(parents=True, exist_ok=True)
    n = len(y_gt)
    arrays = {
        "HADM_ID": np.array([100, 101][:n], dtype=np.int64),
        "SUBJECT_ID": np.array([1, 2][:n], dtype=np.int64),
        "split": np.array(["test"] * n, dtype="<U8"),
        "y_gt": np.array(y_gt, dtype=np.uint8),
        "y_pred": np.array(y_pred, dtype=np.uint8),
    }
    np.savez(eval_dir / "per_visit_predictions.npz", **arrays)


def test_main_writes_all_outputs_end_to_end(tmp_path, monkeypatch):
    import safedrug_ddi_pair_analysis as sdpa

    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    monkeypatch.setattr(sdpa, "DXTEXT_CSV", dxtext_path)
    monkeypatch.setattr(sdpa, "LABELS_NPZ", labels_path)
    monkeypatch.setattr(sdpa, "CCS_CSV", ccs_path)

    ddi_path = tmp_path / "ddi.pkl"
    with ddi_path.open("wb") as fh:
        dill.dump(DDI_ADJ, fh)

    safedrug_dir = tmp_path / "safedrug_data"
    safedrug_dir.mkdir()

    class _Voc:
        idx2word = {0: "N02B", 1: "A01A", 2: "A02B", 3: "B01A"}

    with (safedrug_dir / "voc_final.pkl").open("wb") as fh:
        dill.dump({"med_voc": _Voc()}, fh)

    baseline_dir = tmp_path / "baseline"
    _write_dump(baseline_dir, y_gt=[[1, 0, 0, 1], [1, 0, 0, 1]], y_pred=[[1, 0, 0, 0], [1, 0, 0, 1]])
    variant_dir = tmp_path / "W"
    _write_dump(variant_dir, y_gt=[[1, 0, 0, 1], [1, 0, 0, 1]], y_pred=[[1, 0, 0, 1], [1, 0, 0, 1]])

    whitelist_csv = tmp_path / "whitelist_pairs.csv"
    pd.DataFrame({
        "ccs_group": ["A"], "idx_a": [0], "idx_b": [3], "atc_a": ["N02B"], "atc_b": ["B01A"],
        "share": [1.0], "n_visits": [2],
    }).to_csv(whitelist_csv, index=False)

    out_dir = tmp_path / "out"
    sdpa.main([
        "--baseline-dir", str(baseline_dir), "--variant", "W", str(variant_dir),
        "--whitelist-pairs", str(whitelist_csv), "--out-dir", str(out_dir),
        "--ddi-adj", str(ddi_path), "--safedrug-data-dir", str(safedrug_dir),
    ])

    for name in ("table_overall.csv", "table_missrate_by_long_k10.csv", "table_missrate_by_ccs.csv",
                 "table_missrate_whitelist.csv", "table_guideline_pairs.csv",
                 "table_excess_split_by_cluster.csv"):
        assert (out_dir / name).exists()
    assert (out_dir / "figs" / "fig_pair_missrate_by_cluster.png").exists()
    assert (out_dir / "REPORT_PAIR_ANALYSIS_KO.md").exists()

    overall = pd.read_csv(out_dir / "table_overall.csv").set_index("variant")
    assert overall.loc["baseline", "n_visits"] == 2
    assert overall.loc["W", "n_visits"] == 2

    missrate = pd.read_csv(out_dir / "table_missrate_by_long_k10.csv").set_index("variant")
    # baseline: visit0 misses the pair, visit1 covers it -> 0.5; W: both visits cover it -> 0.0.
    assert missrate.loc["baseline", "miss_rate"] == pytest.approx(0.5)
    assert missrate.loc["W", "miss_rate"] == pytest.approx(0.0)
