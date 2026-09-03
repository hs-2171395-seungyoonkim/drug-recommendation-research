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
