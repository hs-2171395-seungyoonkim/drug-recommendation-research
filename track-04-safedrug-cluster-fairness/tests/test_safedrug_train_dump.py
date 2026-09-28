import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")

pytestmark = pytest.mark.skipif(
    not SAFEDRUG_SRC.exists(),
    reason=f"SafeDrug repo not found at {SAFEDRUG_SRC}",
)


def test_dnc_stub_lets_models_import():
    import safedrug_train_dump  # noqa: F401  -- import alone exercises the dnc stub + sys.path insert

    assert "dnc" in sys.modules
    assert sys.modules["dnc"].DNC is object


def test_assemble_dump_arrays_row_alignment():
    from safedrug_train_dump import assemble_dump_arrays

    y_gt_a = np.zeros(5, dtype=np.uint8)
    y_gt_a[[0, 2]] = 1
    y_gt_b = np.zeros(5, dtype=np.uint8)
    y_gt_b[[1]] = 1

    rows = [
        {
            "patient_index": 10, "visit_index": 0, "HADM_ID": 555, "SUBJECT_ID": 42,
            "split": "test", "n_diag": 3, "n_proc": 1, "n_med_gt": 2,
            "y_gt": y_gt_a, "y_pred": y_gt_a.copy(), "y_prob": y_gt_a.astype(np.float32),
        },
        {
            "patient_index": 10, "visit_index": 1, "HADM_ID": 556, "SUBJECT_ID": 42,
            "split": "test", "n_diag": 4, "n_proc": 0, "n_med_gt": 1,
            "y_gt": y_gt_b, "y_pred": np.zeros(5, dtype=np.uint8),
            "y_prob": np.full(5, 0.1, dtype=np.float32),
        },
    ]
    out = assemble_dump_arrays(rows)
    assert out["y_gt"].shape == (2, 5)
    assert list(out["HADM_ID"]) == [555, 556]
    assert list(out["n_med_gt"]) == [2, 1]
    assert int(out["y_gt"][0].sum()) == 2
    assert int(out["y_gt"][1].sum()) == 1


def test_assemble_dump_arrays_rejects_n_med_gt_mismatch():
    from safedrug_train_dump import assemble_dump_arrays

    y_gt = np.zeros(4, dtype=np.uint8)
    y_gt[[0]] = 1
    rows = [
        {
            "patient_index": 0, "visit_index": 0, "HADM_ID": 1, "SUBJECT_ID": 1,
            "split": "test", "n_diag": 1, "n_proc": 0,
            "n_med_gt": 2,  # deliberately wrong -- y_gt only has 1 positive
            "y_gt": y_gt, "y_pred": y_gt.copy(), "y_prob": y_gt.astype(np.float32),
        }
    ]
    with pytest.raises(ValueError):
        assemble_dump_arrays(rows)


def test_assemble_dump_arrays_rejects_empty_input():
    from safedrug_train_dump import assemble_dump_arrays

    with pytest.raises(ValueError):
        assemble_dump_arrays([])


def test_resolve_seeds_none_returns_existing_defaults_unchanged():
    """--seed defaults to None; resolve_seeds(None) must reproduce the exact
    values SafeDrug.py itself hard-codes (lines 14-15: torch.manual_seed(1203),
    np.random.seed(2048); PY_RANDOM_SEED is a wrapper-only addition), so a run
    without --seed is byte-for-byte identical to before --seed existed."""
    from safedrug_train_dump import resolve_seeds

    seeds = resolve_seeds(None)
    assert seeds == {"torch": 1203, "numpy": 2048, "python": 1203}


def test_resolve_seeds_int_overrides_all_three():
    from safedrug_train_dump import resolve_seeds

    seeds = resolve_seeds(7)
    assert seeds == {"torch": 7, "numpy": 7, "python": 7}


def test_parse_args_seed_defaults_to_none():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5"])
    assert args.seed is None


def test_parse_args_accepts_seed_flag():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5", "--seed", "7"])
    assert args.seed == 7


def test_select_best_state_snapshots_epoch_zero_trained_weights_as_fallback():
    """Fix round 1: the fallback best_state must be epoch 0's *trained* weights
    (design D1a), not weights snapshotted before any training happened. Drives
    select_best_state over a 3-epoch sequence with a stub snapshot_fn (no real
    model, no training) so the fix is exercised directly and cheaply."""
    from safedrug_train_dump import select_best_state

    best_epoch, best_ja, best_state = 0, 0, None

    # Epoch 0: even though its own ja (0.3979) can never satisfy `epoch != 0` (the
    # original's own quirk, unchanged), best_state must still become epoch 0's
    # *trained* snapshot -- not stay at the pre-loop, never-trained None/init.
    best_epoch, best_ja, best_state = select_best_state(
        epoch=0, ja=0.3979, best_epoch=best_epoch, best_ja=best_ja,
        best_state=best_state, snapshot_fn=lambda: "epoch0_trained_state",
    )
    assert (best_epoch, best_ja) == (0, 0)  # unchanged -- SafeDrug's own quirk
    assert best_state == "epoch0_trained_state"  # the actual fix

    # Epoch 1: does not improve on best_ja=0 is false here since any ja > 0 beats
    # the floor of 0 -- use ja=0 itself to exercise the "no improvement" branch and
    # confirm epoch 0's snapshot survives untouched, and snapshot_fn is not called
    # (lazy -- no wasted deep copy).
    snapshot_calls = []
    best_epoch, best_ja, best_state = select_best_state(
        epoch=1, ja=0.0, best_epoch=best_epoch, best_ja=best_ja,
        best_state=best_state,
        snapshot_fn=lambda: snapshot_calls.append(1) or "epoch1_state",
    )
    assert (best_epoch, best_ja) == (0, 0)
    assert best_state == "epoch0_trained_state"  # still epoch 0's, not overwritten
    assert snapshot_calls == []  # no improvement -> snapshot_fn never invoked

    # Epoch 2: improves on best_ja=0 -- best_state must advance to epoch 2's own
    # snapshot (the ordinary, always-worked improvement path, unaffected by the fix).
    best_epoch, best_ja, best_state = select_best_state(
        epoch=2, ja=0.41, best_epoch=best_epoch, best_ja=best_ja,
        best_state=best_state, snapshot_fn=lambda: "epoch2_state",
    )
    assert (best_epoch, best_ja) == (2, 0.41)
    assert best_state == "epoch2_state"


import torch
import torch.nn.functional as F


def test_flatten_med_sets_builds_per_visit_sets():
    from safedrug_train_dump import flatten_med_sets

    # 2 patients, patient 0 has 2 visits, patient 1 has 1 -- adm[2] is the
    # medication index list, matching records_final.pkl's own visit shape
    # [diag_idx_list, proc_idx_list, med_idx_list].
    data_train = [
        [[[1], [2], [0, 1]], [[3], [4], [0, 2]]],
        [[[5], [6], [3]]],
    ]
    out = flatten_med_sets(data_train)
    assert out == [{0, 1}, {0, 2}, {3}]


def test_drug_training_frequency_hand_computed():
    from safedrug_train_dump import drug_training_frequency

    train_med_sets = [{0, 1}, {0, 1}, {0, 2}, {3}]
    freq = drug_training_frequency(train_med_sets, n_med=4)
    assert freq == pytest.approx([0.75, 0.5, 0.25, 0.25])


def test_drug_weight_vector_hand_computed_no_cap_binding():
    from safedrug_train_dump import drug_weight_vector

    freq = np.array([0.75, 0.5, 0.25, 0.25])
    w = drug_weight_vector(freq, cap=10.0)
    assert w == pytest.approx([0.5833333333, 0.875, 1.75, 1.75])


def test_drug_weight_vector_cap_binds():
    from safedrug_train_dump import drug_weight_vector

    freq = np.array([0.01, 0.3, 0.3, 0.39])
    w = drug_weight_vector(freq, cap=10.0)
    assert w == pytest.approx([10.0, 0.8333333333, 0.8333333333, 0.6410256410])


def test_build_drug_weight_vector_none_mode_returns_none():
    from safedrug_train_dump import build_drug_weight_vector

    data_train = [[[[1], [2], [0, 1]]]]
    assert build_drug_weight_vector(data_train, n_med=4, mode="none", cap=10.0) is None


def test_build_drug_weight_vector_inverse_freq_matches_manual_pipeline():
    from safedrug_train_dump import (
        build_drug_weight_vector,
        drug_training_frequency,
        drug_weight_vector,
        flatten_med_sets,
    )

    data_train = [
        [[[1], [2], [0, 1]], [[3], [4], [0, 1]]],
        [[[5], [6], [0, 2]], [[7], [8], [3]]],
    ]
    out = build_drug_weight_vector(data_train, n_med=4, mode="inverse_freq", cap=10.0)
    expected = drug_weight_vector(
        drug_training_frequency(flatten_med_sets(data_train), 4), 10.0
    )
    assert out == pytest.approx(expected)


def test_build_drug_weight_vector_rejects_unknown_mode():
    from safedrug_train_dump import build_drug_weight_vector

    with pytest.raises(ValueError):
        build_drug_weight_vector([[[[1], [2], [0]]]], n_med=4, mode="bogus", cap=10.0)


def test_weighted_bce_loss_none_matches_plain_call():
    from safedrug_train_dump import weighted_bce_loss

    result = torch.tensor([[2.0, -1.0, 0.5]])
    target = torch.tensor([[1.0, 0.0, 1.0]])
    plain = F.binary_cross_entropy_with_logits(result, target)
    via_helper = weighted_bce_loss(result, target, None)
    assert torch.equal(plain, via_helper)


def test_weighted_bce_loss_weighted_matches_hand_computation():
    from safedrug_train_dump import weighted_bce_loss

    result = torch.tensor([[2.0, -1.0, 0.5]])
    target = torch.tensor([[1.0, 0.0, 1.0]])
    weight = torch.tensor([2.0, 1.0, 0.5])

    unweighted = weighted_bce_loss(result, target, None)
    weighted = weighted_bce_loss(result, target, weight)

    assert unweighted.item() == pytest.approx(0.3047555983066559, abs=1e-9)
    assert weighted.item() == pytest.approx(0.2680521011352539, abs=1e-9)
    assert weighted.item() != pytest.approx(unweighted.item())

    elementwise = F.binary_cross_entropy_with_logits(result, target, reduction="none")
    manual = (elementwise[0] * weight).mean().item()
    assert weighted.item() == pytest.approx(manual, abs=1e-9)


def test_parse_args_drug_weight_defaults_to_none():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5"])
    assert args.drug_weight == "none"
    assert args.drug_weight_cap == 10.0


def test_parse_args_drug_weight_accepts_inverse_freq_and_cap():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5", "--drug-weight", "inverse_freq", "--drug-weight-cap", "5.5"])
    assert args.drug_weight == "inverse_freq"
    assert args.drug_weight_cap == 5.5


import pandas as pd


def test_load_ddi_target_csv_parses_and_validates_columns(tmp_path):
    from safedrug_train_dump import load_ddi_target_csv

    csv_path = tmp_path / "targets.csv"
    pd.DataFrame({"HADM_ID": [100, 101], "target_ddi": [0.1, 0.03]}).to_csv(csv_path, index=False)
    out = load_ddi_target_csv(csv_path)
    assert out == {100: 0.1, 101: 0.03}


def test_load_ddi_target_csv_rejects_wrong_header(tmp_path):
    from safedrug_train_dump import load_ddi_target_csv

    csv_path = tmp_path / "targets.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_ddi_target_csv(csv_path)


def test_load_ddi_target_csv_rejects_duplicate_hadm_id(tmp_path):
    from safedrug_train_dump import load_ddi_target_csv

    csv_path = tmp_path / "targets.csv"
    pd.DataFrame({"HADM_ID": [100, 100], "target_ddi": [0.1, 0.03]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_ddi_target_csv(csv_path)


def test_build_visit_target_ddi_uses_csv_value_and_falls_back_to_default():
    from safedrug_train_dump import build_visit_target_ddi

    data_train = [
        [[[1], [], [0]], [[2], [], [1]]],  # patient 0: 2 visits
        [[[3], [], [2]]],                  # patient 1: 1 visit
    ]
    hadm_lookup = {
        (0, 0): (100, 1), (0, 1): (101, 1),
        (1, 0): (999, 2),  # 999 absent from hadm_to_target -> fallback
    }
    hadm_to_target = {100: 0.10, 101: 0.03}
    out = build_visit_target_ddi(data_train, hadm_lookup, hadm_to_target, default_target=0.06)
    assert out == [[0.10, 0.03], [0.06]]


def test_parse_args_ddi_target_mode_defaults_to_global():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5"])
    assert args.ddi_target_mode == "global"
    assert args.ddi_target_csv is None


def test_parse_args_ddi_target_mode_accepts_group_and_csv():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5", "--ddi-target-mode", "group",
                        "--ddi-target-csv", "targets.csv"])
    assert args.ddi_target_mode == "group"
    assert args.ddi_target_csv == "targets.csv"


def test_validate_ddi_target_args_requires_csv_for_group_mode():
    from safedrug_train_dump import validate_ddi_target_args

    with pytest.raises(ValueError):
        validate_ddi_target_args("group", None)
    validate_ddi_target_args("group", "targets.csv")  # no raise
    validate_ddi_target_args("global", None)  # no raise


import dill


# ======================================================= Intervention W (R2): DDI whitelist
def test_parse_args_ddi_whitelist_defaults_to_none():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5"])
    assert args.ddi_whitelist_pairs is None
    assert args.ddi_whitelist_visits is None


def test_parse_args_ddi_whitelist_accepts_both_paths():
    from safedrug_train_dump import parse_args

    args = parse_args([
        "--epochs", "5", "--ddi-whitelist-pairs", "wp.csv", "--ddi-whitelist-visits", "wv.csv",
    ])
    assert args.ddi_whitelist_pairs == "wp.csv"
    assert args.ddi_whitelist_visits == "wv.csv"


def test_validate_ddi_whitelist_args_requires_both_or_neither():
    from safedrug_train_dump import validate_ddi_whitelist_args

    validate_ddi_whitelist_args(None, None)  # no raise
    validate_ddi_whitelist_args("pairs.csv", "visits.csv")  # no raise
    with pytest.raises(ValueError):
        validate_ddi_whitelist_args("pairs.csv", None)
    with pytest.raises(ValueError):
        validate_ddi_whitelist_args(None, "visits.csv")


def test_visit_mask_local_helper_basic_and_symmetric():
    from safedrug_train_dump import _visit_mask

    mask = _visit_mask([(0, 3)], n_med=4)
    assert mask[0, 3] == 1 and mask[3, 0] == 1
    assert mask.sum() == 2
    assert mask.dtype == np.uint8


def test_visit_mask_local_helper_none_returns_zero():
    from safedrug_train_dump import _visit_mask

    assert np.array_equal(_visit_mask(None, n_med=3), np.zeros((3, 3), dtype=np.uint8))


def test_masked_ddi_rate_matches_ddi_rate_score_when_unmasked(tmp_path):
    from safedrug_train_dump import _masked_ddi_rate
    from util import ddi_rate_score  # noqa: E402 -- sys.path already carries SAFEDRUG_SRC

    ddi_adj = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    ddi_path = tmp_path / "ddi.pkl"
    with ddi_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    y_label = np.array([0, 1, 3])
    from_wrapper = _masked_ddi_rate(y_label, ddi_adj)
    from_file = ddi_rate_score([[y_label]], path=str(ddi_path))
    assert from_wrapper == pytest.approx(from_file)
    assert from_wrapper == pytest.approx(1 / 3)  # (0,3) is the only DDI pair among C(3,2)=3 pairs


def test_masked_ddi_rate_reflects_masking_reduces_rate():
    from safedrug_train_dump import _masked_ddi_rate

    ddi_adj = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    masked = ddi_adj * (1 - np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]))  # whitelists (0,3) -> masked is all-zero
    y_label = np.array([0, 1, 3])
    assert _masked_ddi_rate(y_label, ddi_adj) == pytest.approx(1 / 3)
    assert _masked_ddi_rate(y_label, masked) == pytest.approx(0.0)


def test_load_ddi_whitelist_parses_pairs_and_builds_group_lookup(tmp_path):
    from safedrug_train_dump import load_ddi_whitelist

    pairs_csv = tmp_path / "whitelist_pairs.csv"
    pd.DataFrame({
        "ccs_group": ["A", "A", "B"], "idx_a": [0, 1, 0], "idx_b": [3, 2, 3],
        "atc_a": ["x", "y", "x"], "atc_b": ["z", "w", "z"], "share": [0.5, 0.3, 0.4], "n_visits": [10, 10, 5],
    }).to_csv(pairs_csv, index=False)
    visits_csv = tmp_path / "whitelist_visits.csv"
    pd.DataFrame({
        "HADM_ID": [100, 101, 102], "ccs_group": ["A", "B", float("nan")],
        "n_whitelisted_pairs": [2, 1, 0],
    }).to_csv(visits_csv, index=False)

    pairs_by_group, hadm_to_group = load_ddi_whitelist(pairs_csv, visits_csv)
    assert pairs_by_group["A"] == [(0, 3), (1, 2)]
    assert pairs_by_group["B"] == [(0, 3)]
    assert hadm_to_group == {100: "A", 101: "B"}


def test_load_ddi_whitelist_drops_unlabeled_visits_from_hadm_to_group(tmp_path):
    from safedrug_train_dump import load_ddi_whitelist

    pairs_csv = tmp_path / "whitelist_pairs.csv"
    pd.DataFrame({"ccs_group": ["A"], "idx_a": [0], "idx_b": [3], "atc_a": ["x"], "atc_b": ["z"],
                  "share": [0.5], "n_visits": [10]}).to_csv(pairs_csv, index=False)
    visits_csv = tmp_path / "whitelist_visits.csv"
    pd.DataFrame({"HADM_ID": [200], "ccs_group": [float("nan")], "n_whitelisted_pairs": [0]}).to_csv(
        visits_csv, index=False
    )
    _pairs_by_group, hadm_to_group = load_ddi_whitelist(pairs_csv, visits_csv)
    assert hadm_to_group == {}


def test_build_visit_ddi_masks_applies_group_pairs():
    from safedrug_train_dump import build_visit_ddi_masks

    data_train = [[[[0], [], [0, 3]]]]  # 1 patient, 1 visit
    hadm_lookup = {(0, 0): (100, 1)}
    out = build_visit_ddi_masks(data_train, hadm_lookup, {"A": [(0, 3)]}, {100: "A"}, n_med=4)
    assert out[0][0] is not None
    assert out[0][0][0, 3] == 1 and out[0][0][3, 0] == 1


def test_build_visit_ddi_masks_none_when_no_group():
    from safedrug_train_dump import build_visit_ddi_masks

    data_train = [[[[0], [], [0, 3]]]]
    hadm_lookup = {(0, 0): (999, 1)}  # not in hadm_to_group
    out = build_visit_ddi_masks(data_train, hadm_lookup, {"A": [(0, 3)]}, {}, n_med=4)
    assert out[0][0] is None


def test_build_visit_ddi_masks_none_when_group_has_no_pairs():
    from safedrug_train_dump import build_visit_ddi_masks

    data_train = [[[[0], [], [0, 3]]]]
    hadm_lookup = {(0, 0): (100, 1)}
    out = build_visit_ddi_masks(data_train, hadm_lookup, {}, {100: "B"}, n_med=4)
    assert out[0][0] is None


def test_summarize_visit_ddi_masks_none_input_returns_none():
    from safedrug_train_dump import _summarize_visit_ddi_masks

    assert _summarize_visit_ddi_masks(None) is None


def test_summarize_visit_ddi_masks_hand_computed():
    from safedrug_train_dump import _summarize_visit_ddi_masks, _visit_mask

    mask_a = _visit_mask([(0, 3)], n_med=4)          # 1 pair
    mask_b = _visit_mask([(0, 3), (1, 2)], n_med=4)  # 2 pairs
    visit_ddi_masks = [[mask_a, None], [mask_b]]  # patient0: 2 visits, patient1: 1 -> 3 total, 2 masked
    out = _summarize_visit_ddi_masks(visit_ddi_masks)
    assert out["n_training_visits"] == 3
    assert out["n_visits_with_mask"] == 2
    assert out["mean_pairs_per_visit"] == pytest.approx((1 + 2) / 3)
    assert out["mean_pairs_per_masked_visit"] == pytest.approx((1 + 2) / 2)


class _StubDDIModel(torch.nn.Module):
    """Minimal stand-in for SafeDrugModel exposing exactly what
    train_one_epoch touches: .train() (inherited from nn.Module),
    .tensor_ddi_adj (a plain, swappable attribute), and forward(seq_input)
    -> (result, loss_ddi), loss_ddi computed from self.tensor_ddi_adj
    exactly like the real model's batch_neg (models.py:247-250). Records
    the adjacency it saw at every forward() call so a test can assert on
    it directly."""

    def __init__(self, ddi_adj: np.ndarray, n_med: int):
        super().__init__()
        self.tensor_ddi_adj = torch.tensor(ddi_adj, dtype=torch.float32)
        self.w = torch.nn.Parameter(torch.zeros(n_med))
        self.forward_adj_seen = []

    def forward(self, seq_input):
        self.forward_adj_seen.append(self.tensor_ddi_adj.detach().clone())
        result = self.w.unsqueeze(0)  # (1, n_med)
        prob = torch.sigmoid(result)
        neg_pred_prob = prob.t() * prob
        loss_ddi = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, loss_ddi


def test_train_one_epoch_default_path_unchanged_when_visit_ddi_masks_none(tmp_path):
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModel(ddi_adj, n_med=2)
    data_train = [[[[0], [], [0, 1]]]]  # 1 patient, 1 visit, meds {0,1}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_one_epoch(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05,
    )

    assert len(model.forward_adj_seen) == 1
    assert torch.equal(model.forward_adj_seen[0], torch.tensor(ddi_adj, dtype=torch.float32))
    assert torch.equal(model.tensor_ddi_adj, torch.tensor(ddi_adj, dtype=torch.float32))


def test_train_one_epoch_applies_mask_and_restores_before_next_visit(tmp_path):
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModel(ddi_adj, n_med=2)
    # 1 patient, 2 visits -- visit 0 has a mask whitelisting the only pair,
    # visit 1 has no mask (None).
    data_train = [[[[0], [], [0, 1]], [[0], [], [0, 1]]]]
    mask0 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    visit_ddi_masks = [[mask0, None]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_one_epoch(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05, visit_ddi_masks=visit_ddi_masks,
    )

    assert len(model.forward_adj_seen) == 2
    # visit 0: the masked (all-zero) adjacency was live during its forward() call.
    assert torch.equal(model.forward_adj_seen[0], torch.zeros(2, 2))
    # visit 1: restored to the original, unmasked adjacency BEFORE its own
    # forward() call -- the regression the restore-after-step rule guards against.
    assert torch.equal(model.forward_adj_seen[1], torch.tensor(ddi_adj, dtype=torch.float32))
    # and the model's own attribute is back to base_adj once training ends.
    assert torch.equal(model.tensor_ddi_adj, torch.tensor(ddi_adj, dtype=torch.float32))


class _StubDDIModelRaisesOnSecondForward(_StubDDIModel):
    """_StubDDIModel variant whose forward() raises on its second call, so a
    test can prove train_one_epoch's `finally: model.tensor_ddi_adj =
    base_adj` restore fires even when an exception propagates out of the
    training loop mid-epoch (not just the normal in-loop restore after
    optimizer.step())."""

    def forward(self, seq_input):
        if len(self.forward_adj_seen) == 1:
            raise RuntimeError("boom on second visit")
        return super().forward(seq_input)


def test_train_one_epoch_finally_restores_base_adj_when_forward_raises(tmp_path):
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModelRaisesOnSecondForward(ddi_adj, n_med=2)
    # 1 patient, 2 visits -- both masked, so tensor_ddi_adj is still swapped
    # to the masked matrix at the moment the second visit's forward() raises;
    # only the finally block (not the normal post-optimizer.step() inline
    # restore, which is never reached) can put base_adj back.
    data_train = [[[[0], [], [0, 1]], [[0], [], [0, 1]]]]
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    visit_ddi_masks = [[mask, mask]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(RuntimeError, match="boom on second visit"):
        train_one_epoch(
            model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
            target_ddi=0.06, kp=0.05, visit_ddi_masks=visit_ddi_masks,
        )

    assert torch.equal(model.tensor_ddi_adj, torch.tensor(ddi_adj, dtype=torch.float32))


def test_train_one_epoch_combined_with_target_ddi_per_visit_runs_without_error(tmp_path):
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModel(ddi_adj, n_med=2)
    data_train = [[[[0], [], [0, 1]], [[0], [], [0, 1]]]]
    mask0 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    visit_ddi_masks = [[mask0, None]]
    target_ddi_per_visit = [[0.5, 0.5]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05, target_ddi_per_visit=target_ddi_per_visit,
        visit_ddi_masks=visit_ddi_masks,
    )
    assert isinstance(loss, float)
    assert len(model.forward_adj_seen) == 2


# ================================================= W2: always-on pair-level DDI penalty
def test_parse_args_ddi_pair_penalty_defaults_to_none():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5"])
    assert args.ddi_pair_penalty is None


def test_parse_args_ddi_pair_penalty_accepts_value():
    from safedrug_train_dump import parse_args

    args = parse_args(["--epochs", "5", "--ddi-pair-penalty", "0.4"])
    assert args.ddi_pair_penalty == pytest.approx(0.4)


def test_ddi_pair_penalty_manifest_block_none_when_unset():
    from safedrug_train_dump import _ddi_pair_penalty_manifest_block

    assert _ddi_pair_penalty_manifest_block(None) is None


def test_ddi_pair_penalty_manifest_block_hand_computed_when_set():
    from safedrug_train_dump import _ddi_pair_penalty_manifest_block

    out = _ddi_pair_penalty_manifest_block(0.4)
    assert out == {"lambda": 0.4, "mode": "always_on", "rate_gate": False}


def _expected_base_loss_for_zero_init_stub(ddi_adj_for_loss_ddi=None):
    """Independently reconstructs the stub's per-visit loss components for
    data_train=[[[[0], [], [0, 1]]]] (n_med=2, meds {0,1}), using _StubDDIModel's
    own zero-initialized weights (w=zeros(2) -> result=[[0,0]] on the first,
    only forward() call of a 1-visit epoch, before optimizer.step() ever
    mutates it). Returns (bce, multi, loss_ddi) as plain floats so a test can
    assemble whatever weighted sum it needs without re-deriving these by hand
    at every call site. loss_ddi is 0.0 when ddi_adj_for_loss_ddi is None
    (the all-zero/no-adjacency case), matching _StubDDIModel.forward's own
    0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum() formula otherwise."""
    result = torch.zeros(1, 2)
    target = torch.tensor([[1.0, 1.0]])
    bce = F.binary_cross_entropy_with_logits(result, target).item()
    multi_target = torch.tensor([[0, 1]])
    multi = F.multilabel_margin_loss(torch.sigmoid(result), multi_target).item()
    if ddi_adj_for_loss_ddi is None:
        loss_ddi = 0.0
    else:
        prob = torch.sigmoid(result)
        neg_pred_prob = prob.t() * prob
        adj_t = torch.tensor(ddi_adj_for_loss_ddi, dtype=torch.float32)
        loss_ddi = (0.0005 * neg_pred_prob.mul(adj_t).sum()).item()
    return bce, multi, loss_ddi


def test_train_one_epoch_pair_penalty_bypasses_gate_and_adds_scaled_loss_ddi(tmp_path):
    """Planted visit where the (unmasked) rate is AT the target (1.0 <= 1.0,
    the old gate's own <= comparison), so the pre-W2 gate would never fire
    and loss_ddi would never enter the loss. With pair_penalty set, the term
    must appear anyway -- proving the gate is bypassed, not just relaxed."""
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModel(ddi_adj, n_med=2)
    data_train = [[[[0], [], [0, 1]]]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lam = 0.7
    loss = train_one_epoch(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=1.0, kp=0.05, pair_penalty=lam,
    )

    bce, multi, loss_ddi = _expected_base_loss_for_zero_init_stub(ddi_adj)
    expected = 0.95 * bce + 0.05 * multi + lam * loss_ddi
    assert loss == pytest.approx(expected, abs=1e-6)


def test_train_one_epoch_pair_penalty_zero_equals_ungated_base_loss(tmp_path):
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModel(ddi_adj, n_med=2)
    data_train = [[[[0], [], [0, 1]]]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=1.0, kp=0.05, pair_penalty=0.0,
    )

    bce, multi, _loss_ddi = _expected_base_loss_for_zero_init_stub(ddi_adj)
    expected = 0.95 * bce + 0.05 * multi
    assert loss == pytest.approx(expected, abs=1e-6)


def test_train_one_epoch_pair_penalty_uses_masked_adjacency_when_combined_with_whitelist(tmp_path):
    """Combined with visit_ddi_masks: loss_ddi must be computed on the SAME
    masked adjacency the stub records at forward() time (Intervention W2's
    "on the CURRENT model.tensor_ddi_adj" contract), not on the original
    unmasked one."""
    from safedrug_train_dump import train_one_epoch

    ddi_adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    ddi_adj_path = tmp_path / "ddi.pkl"
    with ddi_adj_path.open("wb") as fh:
        dill.dump(ddi_adj, fh)

    model = _StubDDIModel(ddi_adj, n_med=2)
    data_train = [[[[0], [], [0, 1]]]]
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # whitelists the only pair
    visit_ddi_masks = [[mask]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lam = 0.4
    loss = train_one_epoch(
        model, data_train, optimizer, torch.device("cpu"), (1, 1, 2), ddi_adj_path,
        target_ddi=0.06, kp=0.05, pair_penalty=lam, visit_ddi_masks=visit_ddi_masks,
    )

    # The stub saw the masked (all-zero) adjacency at forward() time.
    assert len(model.forward_adj_seen) == 1
    assert torch.equal(model.forward_adj_seen[0], torch.zeros(2, 2))
    # model.tensor_ddi_adj is restored to the original after the visit.
    assert torch.equal(model.tensor_ddi_adj, torch.tensor(ddi_adj, dtype=torch.float32))

    bce, multi, loss_ddi = _expected_base_loss_for_zero_init_stub(np.zeros((2, 2)))
    expected = 0.95 * bce + 0.05 * multi + lam * loss_ddi  # loss_ddi is 0.0 -- fully masked
    assert loss == pytest.approx(expected, abs=1e-6)


def test_train_one_epoch_pair_penalty_none_default_unchanged():
    """Signature guard: pair_penalty defaults to None, so every pre-W2 call
    site (with no pair_penalty kwarg at all) keeps taking the gated branch --
    this is what makes the whole existing test suite above a byte-identical
    regression guard for the default path."""
    import inspect

    from safedrug_train_dump import train_one_epoch

    sig = inspect.signature(train_one_epoch)
    assert sig.parameters["pair_penalty"].default is None
