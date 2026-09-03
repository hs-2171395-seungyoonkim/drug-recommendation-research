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
