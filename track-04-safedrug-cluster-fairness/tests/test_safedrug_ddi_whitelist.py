import json
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


# ======================================================= count_flagged_pairs
def test_count_flagged_pairs_hand_computed():
    from safedrug_ddi_whitelist import count_flagged_pairs

    med_sets = [{0, 1, 3}, {0, 3}, {1, 2}, {0, 3}]
    counts = count_flagged_pairs(med_sets, DDI_ADJ)
    assert counts == {(0, 3): 3}


def test_count_flagged_pairs_no_flagged_pairs_returns_empty():
    from safedrug_ddi_whitelist import count_flagged_pairs

    counts = count_flagged_pairs([{1, 2}, {1, 2}], DDI_ADJ)
    assert counts == {}


def test_count_flagged_pairs_tracks_multiple_pairs_independently():
    from safedrug_ddi_whitelist import count_flagged_pairs

    ddi_adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    med_sets = [{0, 1, 2, 3}, {0, 1}, {2, 3}]
    counts = count_flagged_pairs(med_sets, ddi_adj)
    assert counts == {(0, 1): 2, (2, 3): 2}


# ======================================================= select_whitelist
def test_select_whitelist_threshold_edge_inclusive():
    from safedrug_ddi_whitelist import select_whitelist

    counts = {(0, 3): 5, (1, 2): 4}
    out = select_whitelist(counts, n_visits=20, threshold=0.25)
    assert list(out["idx_a"]) == [0]
    assert list(out["idx_b"]) == [3]
    assert out["share"].iloc[0] == pytest.approx(0.25)
    assert out["n_pair_visits"].iloc[0] == 5


def test_select_whitelist_all_below_threshold_returns_empty():
    from safedrug_ddi_whitelist import select_whitelist

    out = select_whitelist({(0, 3): 2}, n_visits=20, threshold=0.25)
    assert out.empty
    assert list(out.columns) == ["idx_a", "idx_b", "n_pair_visits", "share"]


def test_select_whitelist_zero_n_visits_returns_empty_not_error():
    from safedrug_ddi_whitelist import select_whitelist

    out = select_whitelist({(0, 3): 5}, n_visits=0, threshold=0.25)
    assert out.empty


# ======================================================= visit_mask
def test_visit_mask_basic_and_symmetric():
    from safedrug_ddi_whitelist import visit_mask

    mask = visit_mask([(0, 3), (1, 2)], n_med=4)
    expected = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.uint8)
    assert np.array_equal(mask, expected)
    assert mask.dtype == np.uint8


def test_visit_mask_none_returns_zero_matrix():
    from safedrug_ddi_whitelist import visit_mask

    assert np.array_equal(visit_mask(None, n_med=4), np.zeros((4, 4), dtype=np.uint8))


def test_visit_mask_empty_list_returns_zero_matrix():
    from safedrug_ddi_whitelist import visit_mask

    assert np.array_equal(visit_mask([], n_med=3), np.zeros((3, 3), dtype=np.uint8))


# ======================================================= build_whitelist_pairs
def test_build_whitelist_pairs_min_visits_gate_excludes_small_category():
    from safedrug_ddi_whitelist import build_whitelist_pairs

    train_df = pd.DataFrame({
        "HADM_ID": [1, 2, 3, 4, 5],
        "ccs_group": ["A", "A", "A", "B", "B"],  # A: 3 visits, B: 2 visits
        "med_set": [{0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}],
    })
    out = build_whitelist_pairs(train_df, DDI_ADJ, threshold=0.25, min_visits=3)
    assert set(out["ccs_group"]) == {"A"}  # B excluded: 2 < min_visits=3


def test_build_whitelist_pairs_concatenates_multiple_qualifying_categories():
    from safedrug_ddi_whitelist import build_whitelist_pairs

    ddi_adj = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    train_df = pd.DataFrame({
        "HADM_ID": [1, 2, 3, 4],
        "ccs_group": ["A", "A", "B", "B"],
        "med_set": [{0, 3}, {1, 2}, {0, 3}, {0, 3}],
    })
    out = build_whitelist_pairs(train_df, ddi_adj, threshold=0.5, min_visits=2)
    assert len(out) == 3
    a_rows = out[out["ccs_group"] == "A"].sort_values(["idx_a", "idx_b"])
    assert list(zip(a_rows["idx_a"], a_rows["idx_b"])) == [(0, 3), (1, 2)]
    b_rows = out[out["ccs_group"] == "B"]
    assert list(zip(b_rows["idx_a"], b_rows["idx_b"])) == [(0, 3)]
    assert b_rows["share"].iloc[0] == pytest.approx(1.0)


def test_build_whitelist_pairs_empty_when_no_category_qualifies():
    from safedrug_ddi_whitelist import build_whitelist_pairs

    train_df = pd.DataFrame({
        "HADM_ID": [1, 2],
        "ccs_group": ["A", "A"],
        "med_set": [{0, 3}, {0, 3}],
    })
    out = build_whitelist_pairs(train_df, DDI_ADJ, threshold=0.25, min_visits=5)
    assert out.empty
    assert list(out.columns) == ["ccs_group", "idx_a", "idx_b", "n_pair_visits", "share", "n_visits"]


# ======================================================= whitelist_coverage
def test_whitelist_coverage_hand_computed():
    from safedrug_ddi_whitelist import whitelist_coverage

    train_df = pd.DataFrame({
        "HADM_ID": [1, 2, 3],
        "ccs_group": ["A", "A", "A"],
        "med_set": [{0, 3}, {0, 3}, {1, 2}],
    })
    ddi_adj = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    whitelist_pairs = pd.DataFrame({
        "ccs_group": ["A"], "idx_a": [0], "idx_b": [3], "share": [0.667], "n_visits": [3],
    })
    out = whitelist_coverage(train_df, whitelist_pairs, ddi_adj)
    # 3 flagged pairs total across training (visits 1,2 give (0,3), visit 3 gives (1,2));
    # only (0,3) is whitelisted for A -> 2 of 3 covered.
    assert out == {"n_total_pairs": 3, "n_covered_pairs": 2, "coverage": pytest.approx(2 / 3)}


def test_whitelist_coverage_category_without_whitelist_never_covered():
    from safedrug_ddi_whitelist import whitelist_coverage

    train_df = pd.DataFrame({"HADM_ID": [1], "ccs_group": ["B"], "med_set": [{0, 3}]})
    whitelist_pairs = pd.DataFrame({
        "ccs_group": ["A"], "idx_a": [0], "idx_b": [3], "share": [1.0], "n_visits": [1],
    })
    out = whitelist_coverage(train_df, whitelist_pairs, DDI_ADJ)
    assert out == {"n_total_pairs": 1, "n_covered_pairs": 0, "coverage": pytest.approx(0.0)}


# ======================================================= build_whitelist_visits
def test_build_whitelist_visits_covers_full_cohort_including_unlabeled():
    from safedrug_ddi_whitelist import build_whitelist_visits

    all_hadm_ids = pd.Series([100, 101, 102])
    ccs_lookup = {100: "A", 101: "B"}  # 102 absent -> unlabeled
    category_pair_counts = {"A": 3}
    out = build_whitelist_visits(all_hadm_ids, ccs_lookup, category_pair_counts).set_index("HADM_ID")
    assert out.loc[100, "n_whitelisted_pairs"] == 3
    assert out.loc[101, "n_whitelisted_pairs"] == 0  # B has no whitelist entry
    assert pd.isna(out.loc[102, "ccs_group"])
    assert out.loc[102, "n_whitelisted_pairs"] == 0


def test_build_whitelist_visits_zero_when_no_category_has_any_whitelist():
    from safedrug_ddi_whitelist import build_whitelist_visits

    out = build_whitelist_visits(pd.Series([1, 2]), {1: "A", 2: "A"}, category_pair_counts={})
    assert (out["n_whitelisted_pairs"] == 0).all()


# ======================================================= load_training_visits_with_ccs
def test_load_training_visits_with_ccs_joins_correctly(tmp_path):
    from safedrug_ddi_whitelist import load_training_visits_with_ccs

    safedrug_dir = tmp_path / "safedrug_data"
    safedrug_dir.mkdir()
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()

    # 3 patients -> split_point = int(3*2/3) = 2 -- only patients 0,1 are "training".
    records = [
        [[[1], [], [0, 3]]],                    # patient 0, visit 0 -> HADM 100
        [[[1], [], [1]], [[2], [], [2]]],       # patient 1, visits 0,1 -> HADM 101, 102
        [[[1], [], [0]]],                        # patient 2 -- past split_point, never read
    ]
    with (safedrug_dir / "records_final.pkl").open("wb") as fh:
        dill.dump(records, fh)

    master = pd.DataFrame({
        "HADM_ID": [100, 101, 102, 200],
        "SUBJECT_ID": [1, 2, 2, 3],
        "safedrug_patient_index": [0, 1, 1, 2],
        "safedrug_visit_index": [0, 0, 1, 0],
    })
    master.to_csv(cohort_dir / "master_visits.csv", index=False)

    ccs_csv = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": [100, 101], "group": ["A", "B"]}).to_csv(
        ccs_csv, index=False, encoding="utf-8-sig"
    )  # 102 deliberately absent -> unlabeled

    out = load_training_visits_with_ccs(cohort_dir, safedrug_dir, ccs_csv).set_index("HADM_ID")
    assert set(out.index) == {100, 101, 102}  # only the 2 training patients' visits, not patient 2
    assert out.loc[100, "ccs_group"] == "A"
    assert out.loc[100, "med_set"] == {0, 3}
    assert out.loc[101, "ccs_group"] == "B"
    assert pd.isna(out.loc[102, "ccs_group"])


def test_load_training_visits_with_ccs_dedups_duplicate_meds_before_pair_counting(tmp_path):
    """Regression for task-A-review.md finding 1: a raw adm[2] med list
    with a duplicate index (e.g. [0, 3, 3]) must not let a flagged pair
    get double-counted once it flows through the orchestration path --
    proves load_training_visits_with_ccs's med_set = set(adm[2]) dedup
    survives end to end, not just count_flagged_pairs's own isolated
    dedup-safety on a hand-built set."""
    from safedrug_ddi_whitelist import count_flagged_pairs, load_training_visits_with_ccs

    safedrug_dir = tmp_path / "safedrug_data"
    safedrug_dir.mkdir()
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()

    # 2 patients -> split_point = int(2*2/3) = 1 -- only patient 0's one
    # visit is "training". Its raw med list [0, 3, 3] duplicates index 3,
    # the flagged pair's own second member.
    records = [
        [[[1], [], [0, 3, 3]]],   # patient 0, visit 0 -> HADM 100, duplicate med index 3
        [[[1], [], [0]]],          # patient 1 -- past split_point, never read
    ]
    with (safedrug_dir / "records_final.pkl").open("wb") as fh:
        dill.dump(records, fh)

    master = pd.DataFrame({
        "HADM_ID": [100, 200],
        "SUBJECT_ID": [1, 2],
        "safedrug_patient_index": [0, 1],
        "safedrug_visit_index": [0, 0],
    })
    master.to_csv(cohort_dir / "master_visits.csv", index=False)

    ccs_csv = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": [100], "group": ["A"]}).to_csv(
        ccs_csv, index=False, encoding="utf-8-sig"
    )

    out = load_training_visits_with_ccs(cohort_dir, safedrug_dir, ccs_csv).set_index("HADM_ID")
    assert set(out.index) == {100}
    assert out.loc[100, "med_set"] == {0, 3}  # duplicate 3 collapsed by set(adm[2])

    counts = count_flagged_pairs([out.loc[100, "med_set"]], DDI_ADJ)
    assert counts == {(0, 3): 1}  # flagged pair counted exactly once for this one visit


# ======================================================= parse_args / main
def test_parse_args_defaults():
    from safedrug_ddi_whitelist import COHORT_DIR_DEFAULT, SAFEDRUG_DATA_DEFAULT, parse_args

    args = parse_args(["--out-dir", "out"])
    assert args.threshold == 0.25
    assert args.min_visits == 30
    assert args.cohort_dir == str(COHORT_DIR_DEFAULT)
    assert args.safedrug_data_dir == str(SAFEDRUG_DATA_DEFAULT)


def test_main_writes_three_files_end_to_end(tmp_path):
    from safedrug_ddi_whitelist import main

    safedrug_dir = tmp_path / "safedrug_data"
    safedrug_dir.mkdir()
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()

    ddi_adj = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    with (safedrug_dir / "ddi_A_final.pkl").open("wb") as fh:
        dill.dump(ddi_adj, fh)

    class _Voc:
        idx2word = {0: "N02B", 1: "A01A", 2: "A02B", 3: "B01A"}

    with (safedrug_dir / "voc_final.pkl").open("wb") as fh:
        dill.dump({"med_voc": _Voc()}, fh)

    # 6 patients -> split_point = int(6*2/3) = 4 (patients 0-3 are training).
    records = [
        [[[1], [], [0, 3]]],  # patient 0 -> HADM 100, group A
        [[[1], [], [0, 3]]],  # patient 1 -> HADM 101, group A
        [[[1], [], [0, 3]]],  # patient 2 -> HADM 102, group A
        [[[1], [], [1, 2]]],  # patient 3 -> HADM 103, group B -- only 1 training visit, below min_visits=3
        [[[1], [], [0]]],      # patient 4 -- past split_point, never read
        [[[1], [], [0]]],      # patient 5 -- past split_point, never read
    ]
    with (safedrug_dir / "records_final.pkl").open("wb") as fh:
        dill.dump(records, fh)

    master = pd.DataFrame({
        "HADM_ID": [100, 101, 102, 103, 200, 201],
        "SUBJECT_ID": [1, 2, 3, 4, 5, 6],
        "safedrug_patient_index": [0, 1, 2, 3, 4, 5],
        "safedrug_visit_index": [0, 0, 0, 0, 0, 0],
    })
    master.to_csv(cohort_dir / "master_visits.csv", index=False)

    ccs_csv = tmp_path / "ccs.csv"
    pd.DataFrame({
        "HADM_ID": [100, 101, 102, 103],
        "group": ["A", "A", "A", "B"],
    }).to_csv(ccs_csv, index=False, encoding="utf-8-sig")

    out_dir = tmp_path / "out"
    main([
        "--cohort-dir", str(cohort_dir), "--safedrug-data-dir", str(safedrug_dir),
        "--ccs-csv", str(ccs_csv), "--threshold", "0.5", "--min-visits", "3",
        "--out-dir", str(out_dir),
    ])

    pairs = pd.read_csv(out_dir / "whitelist_pairs.csv")
    assert list(pairs.columns) == ["ccs_group", "idx_a", "idx_b", "atc_a", "atc_b", "share", "n_visits"]
    assert len(pairs) == 1  # only A's (0,3) pair -- B is below min_visits
    row = pairs.iloc[0]
    assert row["ccs_group"] == "A"
    assert (row["idx_a"], row["idx_b"]) == (0, 3)
    assert row["atc_a"] == "N02B" and row["atc_b"] == "B01A"
    assert row["share"] == pytest.approx(1.0)
    assert row["n_visits"] == 3

    visits = pd.read_csv(out_dir / "whitelist_visits.csv")
    assert len(visits) == 6  # every cohort HADM_ID, not just training
    visits = visits.set_index("HADM_ID")
    assert visits.loc[100, "n_whitelisted_pairs"] == 1
    assert visits.loc[103, "n_whitelisted_pairs"] == 0  # B: below min_visits
    assert visits.loc[200, "n_whitelisted_pairs"] == 0  # unlabeled

    meta = json.loads((out_dir / "whitelist_meta.json").read_text(encoding="utf-8"))
    assert meta["threshold"] == 0.5
    assert meta["min_visits"] == 3
    assert meta["n_categories_with_whitelist"] == 1
    assert meta["n_pairs_total"] == 1
    assert meta["training_coverage"]["coverage"] == pytest.approx(1.0)
