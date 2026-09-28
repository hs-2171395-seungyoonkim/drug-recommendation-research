import json
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_decode_budget import (
    build_visit_budgets,
    compute_r_global,
    decode_visit,
    main,
    select_pfloor,
    _write_decode_variant,
)

# Meds 0 and 3 are the one DDI pair -- the repo's existing DDI test fixture
# (same matrix every other SafeDrug test file in this repo uses).
DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


# ======================================================= decode_visit
def test_decode_visit_remove_phase_picks_lowest_p_stops_at_budget():
    # 5 meds: pairs (0,1) and (2,3) are DDI, med 4 is isolated. All 5 clear
    # base_threshold, so S starts as {0,1,2,3,4}: 10 pairs, 2 are DDI -> 0.2.
    ddi_adj = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55])
    # budget=0.05: first removes med 3 (lowest p among the 4 DDI-involved
    # meds {0,1,2,3}), leaving {0,1,2,4} at 1/6 DDI (>0.05); then removes
    # med 1 (lowest p among the now-involved {0,1}), leaving {0,2,4} at 0
    # DDI (<=0.05) -- loop stops there. p_floor=0.9 excludes both removed
    # meds from the ADD phase (neither clears 0.9).
    out = decode_visit(p, budget=0.05, ddi_adj=ddi_adj, p_floor=0.9)
    assert set(np.flatnonzero(out).tolist()) == {0, 2, 4}


def test_decode_visit_treats_fewer_than_two_meds_as_zero_ddi():
    ddi_adj = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    p = np.array([0.9, 0.3, 0.3, 0.3, 0.3])  # only med 0 clears base_threshold
    out = decode_visit(p, budget=0.0, ddi_adj=ddi_adj, p_floor=1.1)  # p_floor unreachable
    assert list(out) == [1, 0, 0, 0, 0]  # |S|=1 < 2 -> REMOVE loop never runs


def test_decode_visit_add_phase_single_pass_no_reconsideration():
    # Only pair: (0,1). p=[0.45,0.44,0.43,0.42], all below base_threshold=0.5
    # -> S starts empty. p_floor=0.4 makes all 4 meds ADD candidates, tried
    # in descending-p order: 0 (S={0}, |S|<2 -> ddi=0, admit), 1 (S={0,1}
    # would be 1 pair, all DDI -> ddi=1.0 > budget 0.2, REJECT), 2 (S={0,2},
    # 1 pair, 0 DDI -> ddi=0 <= 0.2, admit), 3 (S={0,2,3}, 3 pairs, 0 DDI ->
    # admit). Final S={0,2,3}. A SECOND pass would now admit med 1 too
    # (S={0,2,3,1}: 6 pairs, 1 DDI -> 1/6=0.167 <= 0.2) -- single-pass must
    # NOT do that.
    ddi_adj = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    p = np.array([0.45, 0.44, 0.43, 0.42])
    out = decode_visit(p, budget=0.2, ddi_adj=ddi_adj, p_floor=0.4)
    assert set(np.flatnonzero(out).tolist()) == {0, 2, 3}


def test_decode_visit_p_floor_excludes_low_probability_candidates():
    ddi_adj = np.zeros((2, 2))
    p = np.array([0.6, 0.3])  # base_threshold=0.5 -> S starts {0}
    out = decode_visit(p, budget=1.0, ddi_adj=ddi_adj, p_floor=0.5)  # 0.3 < 0.5
    assert list(out) == [1, 0]


def test_decode_visit_is_deterministic():
    ddi_adj = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55])
    out1 = decode_visit(p, budget=0.05, ddi_adj=ddi_adj, p_floor=0.65)
    out2 = decode_visit(p, budget=0.05, ddi_adj=ddi_adj, p_floor=0.65)
    assert np.array_equal(out1, out2)


def test_decode_visit_reproduces_baseline_when_budget_sufficient_and_no_add_candidates():
    ddi_adj = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55])  # baseline S = all 5, ddi=0.2
    out = decode_visit(p, budget=1.0, ddi_adj=ddi_adj, p_floor=0.99)  # budget never binds, no candidates
    assert list(out) == list((p >= 0.5).astype(np.int8))


def test_decode_visit_asserts_on_nan_budget():
    ddi_adj = np.zeros((2, 2))
    p = np.array([0.6, 0.4])
    with pytest.raises(AssertionError):
        decode_visit(p, budget=float("nan"), ddi_adj=ddi_adj, p_floor=0.5)


def test_decode_visit_remove_phase_tie_break_lowest_index_removed_first():
    # 3 meds, pair (0,1) is the only DDI pair, med 2 isolated. p=[0.5,0.5,0.9]
    # all clear base_threshold=0.5 -> S starts {0,1,2}: 3 pairs, 1 DDI (0,1)
    # -> rate=1/3. budget=0.2 < 1/3 forces exactly one removal. Both
    # DDI-involved meds (0 and 1) tie at p=0.5 -- lowest index (0) must be
    # removed first. After removing 0, S={1,2}: 1 pair, 0 DDI -> rate=0 <=
    # 0.2, loop stops (exactly one removal). p_floor=0.6 keeps the removed
    # med 0 (p=0.5) out of the ADD phase, so the REMOVE-phase choice stays
    # observable in the final output: {1,2} means med 0 (not med 1) was the
    # one removed.
    ddi_adj = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    p = np.array([0.5, 0.5, 0.9])
    out = decode_visit(p, budget=0.2, ddi_adj=ddi_adj, p_floor=0.6)
    assert set(np.flatnonzero(out).tolist()) == {1, 2}


def test_decode_visit_add_phase_tie_break_lowest_index_admitted_first():
    # 2 meds, the only pair (0,1) is DDI. p=[0.45,0.45] genuinely tied, both
    # below base_threshold=0.5 -> S starts empty (REMOVE phase never runs,
    # |S|=0 < 2). p_floor=0.4 makes both ADD candidates, tried in
    # descending-p order with ties broken by lowest index: med 0 first ->
    # |S|=0 before admitting it, so ddi(S)=0 <= budget, admit -> S={0}. Then
    # med 1: trial S={0,1} is the one DDI pair -> rate=1.0 > budget=0.5,
    # reject. Final S={0}: the lower-index tied candidate is safe on its
    # own and the higher-index one interacts with it, so admitting 0 first
    # is exactly what blocks 1 from ever getting in.
    ddi_adj = np.array([
        [0, 1],
        [1, 0],
    ])
    p = np.array([0.45, 0.45])
    out = decode_visit(p, budget=0.5, ddi_adj=ddi_adj, p_floor=0.4)
    assert set(np.flatnonzero(out).tolist()) == {0}


# ======================================================= compute_r_global
def test_compute_r_global_matches_internal_pairwise_rate():
    # 3 labeled training visits: {0,3} DDI, {1,2} not, {0,3} DDI -> 2 DDI
    # pairs out of 3 total pairs pooled = 2/3 -- same fixture
    # tests/test_safedrug_group_ddi_targets.py's own shrink-k1 test uses for
    # this exact r_global value (see that test's comment).
    train_labeled = pd.DataFrame({"HADM_ID": [100, 101, 102], "long_k10": [0, 1, 1]})
    gt_med_sets = {100: {0, 3}, 101: {1, 2}, 102: {0, 3}}
    r_global = compute_r_global(train_labeled, gt_med_sets, DDI_ADJ, "long_k10")
    assert r_global == pytest.approx(2.0 / 3.0)


def test_compute_r_global_drops_unlabeled_rows_first():
    train_labeled = pd.DataFrame({"HADM_ID": [100, 101], "long_k10": [0.0, float("nan")]})
    gt_med_sets = {100: {0, 3}, 101: {1, 2}}
    r_global = compute_r_global(train_labeled, gt_med_sets, DDI_ADJ, "long_k10")
    assert r_global == pytest.approx(1.0)  # only visit 100 counted -> its 1 pair is DDI


# ======================================================= build_visit_budgets
def test_build_visit_budgets_uses_group_target_and_r_global_fallback():
    df = pd.DataFrame({"HADM_ID": [1, 2, 3], "long_k10": [0.0, 1.0, float("nan")]})
    group_table = pd.DataFrame({"group": [0, 1], "target_ddi": [0.10, 0.03]})
    out = build_visit_budgets(df, group_table, r_global=0.06, partition="long_k10")
    assert list(out) == pytest.approx([0.10, 0.03, 0.06])


def test_build_visit_budgets_ccs_group_string_keys():
    df = pd.DataFrame({"HADM_ID": [1, 2, 3], "ccs_group": ["A", "B", float("nan")]})
    group_table = pd.DataFrame({"group": ["A", "B"], "target_ddi": [0.08, 0.05]})
    out = build_visit_budgets(df, group_table, r_global=0.06, partition="ccs_group")
    assert list(out) == pytest.approx([0.08, 0.05, 0.06])


# ======================================================= select_pfloor
def test_select_pfloor_planted_probabilities_yield_expected_choice_and_tie_break():
    # 1 eval visit, 2 meds, no possible DDI pair (2x2 zero matrix). y_gt =
    # [1,1]. y_prob=[0.6,0.3]. base_threshold=0.5 -> baseline S={0}. Any
    # p_floor <= 0.3 admits med 1 too (Jaccard 1.0); p_floor > 0.3 leaves
    # S={0} (Jaccard 0.5). Grid values 0.25 and 0.30 tie at 1.0 -- 0.30 must
    # win (closer to 0.5), matching safedrug_group_threshold.select_threshold's
    # own tie rule exactly.
    y_gt = np.array([[1, 1]])
    y_prob = np.array([[0.6, 0.3]])
    budgets = np.array([1.0])
    ddi_adj = np.zeros((2, 2))
    grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_pf, table = select_pfloor(y_gt, y_prob, budgets, ddi_adj, grid)
    assert list(table.columns) == ["p_floor", "eval_mean_jaccard", "eval_mean_ddi"]
    assert table.loc[table["p_floor"] <= 0.30, "eval_mean_jaccard"].tolist() == pytest.approx([1.0, 1.0])
    assert table.loc[table["p_floor"] > 0.30, "eval_mean_jaccard"].tolist() == pytest.approx([0.5] * 4)
    assert best_pf == pytest.approx(0.30)


# ======================================================= _write_decode_variant
def test_write_decode_variant_writes_npz_and_manifest_with_decode_policy(tmp_path):
    variant_dir = tmp_path / "ccs"
    arrays = {"HADM_ID": np.array([1, 2]), "y_pred": np.array([[1, 0], [0, 1]], dtype=np.uint8)}
    manifest_src = {"best_epoch": 5}
    decode_policy = {"variant": "ccs", "p_floor": 0.35}
    train_log_src = tmp_path / "no_such_train_log.csv"  # deliberately absent

    _write_decode_variant(variant_dir, arrays, manifest_src, decode_policy, train_log_src)

    npz = np.load(variant_dir / "per_visit_predictions.npz", allow_pickle=False)
    assert list(npz["HADM_ID"]) == [1, 2]
    manifest = json.loads((variant_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["best_epoch"] == 5  # manifest_src carried through
    assert manifest["decode_policy"] == decode_policy
    assert "threshold_policy" not in manifest
    assert not (variant_dir / "train_log.csv").exists()


def test_write_decode_variant_copies_train_log_when_present(tmp_path):
    variant_dir = tmp_path / "ccs"
    train_log_src = tmp_path / "train_log.csv"
    train_log_src.write_text("epoch,eval_ja\n1,0.5\n", encoding="utf-8")

    _write_decode_variant(variant_dir, {"HADM_ID": np.array([1])}, {}, {}, train_log_src)

    assert (variant_dir / "train_log.csv").read_text(encoding="utf-8") == "epoch,eval_ja\n1,0.5\n"


# ======================================================= main() integration
def _write_cohort_fixture(cohort_dir: Path):
    cohort_dir.mkdir(parents=True, exist_ok=True)
    records = [
        [[[1], [], [0, 3]]],  # patient 0 -> HADM 100 (group A / long_k10=0), DDI pair
        [[[1], [], [0, 3]]],  # patient 1 -> HADM 101 (group A / long_k10=0), DDI pair
        [[[2], [], [1, 2]]],  # patient 2 -> HADM 102 (group B / long_k10=1), not DDI
        [[[2], [], [1, 2]]],  # patient 3 -> HADM 103 (group B / long_k10=1), not DDI
        [[[0], [], []]],       # patient 4 -- past split_point, never read
        [[[0], [], []]],       # patient 5 -- past split_point, never read
    ]
    with (cohort_dir / "records_reconstructed.pkl").open("wb") as fh:
        dill.dump(records, fh)
    master = pd.DataFrame({
        "HADM_ID": [100, 101, 102, 103], "SUBJECT_ID": [1, 2, 3, 4],
        "safedrug_patient_index": [0, 1, 2, 3], "safedrug_visit_index": [0, 0, 0, 0],
    })
    master.to_csv(cohort_dir / "master_visits.csv", index=False)


def _write_label_fixture(tmp_path):
    # HADM 301 is deliberately absent from every label file -> unlabeled on
    # both axes, exercising R1's r_global fallback end-to-end.
    hadm_ids = [100, 101, 102, 103, 200, 201, 300]
    dxtext_path = tmp_path / "dxtext.csv"
    pd.DataFrame({"HADM_ID": hadm_ids}).to_csv(dxtext_path, index=False, encoding="utf-8-sig")

    labels_path = tmp_path / "labels.npz"
    long_k10 = np.array([0, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    np.savez(labels_path, long_k10=long_k10, long_k25=long_k10, short_k10=long_k10, concise_k10=long_k10)

    ccs_path = tmp_path / "ccs.csv"
    ccs_group = ["A", "A", "B", "B", "A", "B", "A"]
    pd.DataFrame({"HADM_ID": hadm_ids, "group": ccs_group}).to_csv(
        ccs_path, index=False, encoding="utf-8-sig"
    )
    return dxtext_path, labels_path, ccs_path


def _write_eval_fixture(eval_dir: Path):
    eval_dir.mkdir(parents=True, exist_ok=True)
    n = 4
    y_prob = np.array([[0.9, 0.6, 0.2, 0.9]] * n, dtype=np.float32)
    arrays = {
        "patient_index": np.array([10, 11, 12, 13], dtype=np.int64),
        "visit_index": np.zeros(n, dtype=np.int64),
        "HADM_ID": np.array([200, 201, 300, 301], dtype=np.int64),
        "SUBJECT_ID": np.array([20, 21, 30, 31], dtype=np.int64),
        "split": np.array(["eval", "eval", "test", "test"], dtype="<U8"),
        "n_diag": np.full(n, 2, dtype=np.int64),
        "n_proc": np.zeros(n, dtype=np.int64),
        "n_med_gt": np.full(n, 2, dtype=np.int64),
        "y_gt": np.array([[1, 1, 0, 0]] * n, dtype=np.uint8),
        "y_prob": y_prob,
        "y_pred": (y_prob >= 0.5).astype(np.uint8),
    }
    np.savez(eval_dir / "per_visit_predictions.npz", **arrays)

    manifest = {"best_epoch": 1, "official_metrics": {"test": {
        "ja": 0.5, "prauc": 0.5, "avg_f1": 0.5, "ddi_rate": 0.0, "avg_med": 3.0,
    }}}
    (eval_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (eval_dir / "train_log.csv").write_text("epoch,eval_ja,eval_ddi_rate\n1,0.5,0.06\n", encoding="utf-8")
    return eval_dir


DDI_ADJ4 = DDI_ADJ  # alias for readability at the call site below


def test_main_writes_three_variant_dirs_and_budget_tables(tmp_path, monkeypatch):
    import safedrug_decode_budget as sdb
    import safedrug_group_threshold as sgt
    import safedrug_mechanism as sm

    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    for mod in (sgt, sm):
        monkeypatch.setattr(mod, "DXTEXT_CSV", dxtext_path)
        monkeypatch.setattr(mod, "LABELS_NPZ", labels_path)
        monkeypatch.setattr(mod, "CCS_CSV", ccs_path)

    cohort_dir = tmp_path / "cohort"
    _write_cohort_fixture(cohort_dir)
    eval_dir = _write_eval_fixture(tmp_path / "eval")
    out_dir = tmp_path / "out"

    ddi_adj_path = tmp_path / "ddi_adj.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(DDI_ADJ4, fh)

    main([
        "--eval-dir", str(eval_dir), "--out-dir", str(out_dir),
        "--ddi-adj", str(ddi_adj_path), "--cohort-dir", str(cohort_dir),
        "--shrink-k", "25",
    ])

    base_npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    for variant in ("ccs", "k10", "global"):
        variant_dir = out_dir / variant
        assert (variant_dir / "per_visit_predictions.npz").exists()
        assert (variant_dir / "train_log.csv").exists()
        manifest = json.loads((variant_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["decode_policy"]["variant"] == variant
        assert manifest["best_epoch"] == 1  # manifest_src carried through

        vnpz = np.load(variant_dir / "per_visit_predictions.npz", allow_pickle=False)
        assert set(vnpz.files) == set(base_npz.files)
        # eval rows (0, 1) are byte-identical to the source dump.
        assert list(vnpz["y_pred"][0]) == list(base_npz["y_pred"][0])
        assert list(vnpz["y_pred"][1]) == list(base_npz["y_pred"][1])

    assert (out_dir / "budgets_ccs.csv").exists()
    assert (out_dir / "budgets_k10.csv").exists()
    assert (out_dir / "budget_global.json").exists()
    assert (out_dir / "table_pfloor_selection.csv").exists()
    assert (out_dir / "per_visit_budgets.csv").exists()

    pfloor_table = pd.read_csv(out_dir / "table_pfloor_selection.csv")
    assert set(pfloor_table["variant"]) == {"ccs", "k10", "global"}
    assert len(pfloor_table) == 3 * len(sdb.PFLOOR_GRID)

    per_visit = pd.read_csv(out_dir / "per_visit_budgets.csv")
    r_global = json.loads((out_dir / "budget_global.json").read_text(encoding="utf-8"))["r_global"]
    assert r_global == pytest.approx(0.5)  # 2 DDI pairs / 4 total, pooled over all 4 training visits
    row_301 = per_visit[per_visit["HADM_ID"] == 301].iloc[0]
    assert row_301["budget_ccs"] == pytest.approx(r_global)
    assert row_301["budget_k10"] == pytest.approx(r_global)
    assert row_301["budget_global"] == pytest.approx(r_global)
    # global variant is r_global for EVERY row, not just the unlabeled one.
    assert np.allclose(per_visit["budget_global"].to_numpy(), r_global)

    for col in ("HADM_ID", "split", "ccs_group", "long_k10", "budget_ccs", "budget_k10",
                "budget_global", "n_pred_baseline", "n_pred_ccs", "n_pred_k10",
                "n_pred_global", "n_removed_ccs", "n_added_ccs"):
        assert col in per_visit.columns

    # R7: "shrink without clipping" hand-computed on this tiny fixture.
    # Group A/0's ground-truth sets are both {0,3} (the one DDI pair) ->
    # raw_ddi_rate=1.0; group B/1's are both {1,2} (not DDI) -> raw=0.0.
    # shrink_k=25, clip [0,1] (a no-op) -> target_ddi = shrunk_ddi_rate
    # exactly = (n_g*raw + 25*r_global)/(n_g+25) with n_g=2, r_global=0.5.
    budgets_ccs_table = pd.read_csv(out_dir / "budgets_ccs.csv").set_index("group")
    assert budgets_ccs_table.loc["A", "raw_ddi_rate"] == pytest.approx(1.0)
    assert budgets_ccs_table.loc["B", "raw_ddi_rate"] == pytest.approx(0.0)
    assert budgets_ccs_table.loc["A", "target_ddi"] == pytest.approx((2 * 1.0 + 25 * 0.5) / 27)
    assert budgets_ccs_table.loc["B", "target_ddi"] == pytest.approx((2 * 0.0 + 25 * 0.5) / 27)
    assert budgets_ccs_table.loc["A", "shrunk_ddi_rate"] == budgets_ccs_table.loc["A", "target_ddi"]  # no clip applied

    budgets_k10_table = pd.read_csv(out_dir / "budgets_k10.csv").set_index("group")
    assert budgets_k10_table.loc[0, "raw_ddi_rate"] == pytest.approx(1.0)
    assert budgets_k10_table.loc[1, "raw_ddi_rate"] == pytest.approx(0.0)
    assert budgets_k10_table.loc[0, "target_ddi"] == pytest.approx((2 * 1.0 + 25 * 0.5) / 27)
    assert budgets_k10_table.loc[1, "target_ddi"] == pytest.approx((2 * 0.0 + 25 * 0.5) / 27)


def test_main_variant_npz_keys_other_than_y_pred_are_byte_identical_to_source(tmp_path, monkeypatch):
    # arrays = dict(npz_dict); arrays["y_pred"] = variant_pred (see
    # scripts/safedrug_decode_budget.py main()) means every OTHER key is the
    # exact same numpy array object read straight from the source dump --
    # this pins that down as a byte-for-byte, dtype-preserving guarantee
    # (not just "same shape" or "same values after casting") for all three
    # written variants, plus the y_pred rows for the non-test ("eval")
    # split, which decode_visit never touches (main() only overwrites
    # test_mask rows -- see the `for i in np.flatnonzero(test_mask)` loop).
    import safedrug_group_threshold as sgt
    import safedrug_mechanism as sm

    dxtext_path, labels_path, ccs_path = _write_label_fixture(tmp_path)
    for mod in (sgt, sm):
        monkeypatch.setattr(mod, "DXTEXT_CSV", dxtext_path)
        monkeypatch.setattr(mod, "LABELS_NPZ", labels_path)
        monkeypatch.setattr(mod, "CCS_CSV", ccs_path)

    cohort_dir = tmp_path / "cohort"
    _write_cohort_fixture(cohort_dir)
    eval_dir = _write_eval_fixture(tmp_path / "eval")
    out_dir = tmp_path / "out"

    ddi_adj_path = tmp_path / "ddi_adj.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(DDI_ADJ4, fh)

    main([
        "--eval-dir", str(eval_dir), "--out-dir", str(out_dir),
        "--ddi-adj", str(ddi_adj_path), "--cohort-dir", str(cohort_dir),
        "--shrink-k", "25",
    ])

    base_npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    non_test_rows = np.flatnonzero(base_npz["split"] != "test")
    assert len(non_test_rows) > 0  # sanity: the fixture actually has eval rows to check

    for variant in ("ccs", "k10", "global"):
        vnpz = np.load(out_dir / variant / "per_visit_predictions.npz", allow_pickle=False)
        assert set(vnpz.files) == set(base_npz.files)
        for key in base_npz.files:
            if key == "y_pred":
                continue
            base_arr = base_npz[key]
            v_arr = vnpz[key]
            assert v_arr.dtype == base_arr.dtype, f"{variant}/{key}: dtype changed ({v_arr.dtype} != {base_arr.dtype})"
            assert np.array_equal(v_arr, base_arr), f"{variant}/{key}: not byte-identical to source"

        assert np.array_equal(
            vnpz["y_pred"][non_test_rows], base_npz["y_pred"][non_test_rows]
        ), f"{variant}/y_pred: non-test rows diverged from source"
