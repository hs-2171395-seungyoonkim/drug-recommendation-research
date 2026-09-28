import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_mimic3_chrono_records as B  # noqa: E402


def test_chronological_order_sorts_by_time_then_hadm():
    t = pd.to_datetime(["2100-03-01", "2100-01-01", "2100-03-01"])
    assert B.chronological_order(list(t), [30, 10, 20]) == [1, 2, 0]


def test_reorder_records_permutes_visits_only_and_keeps_patient_order():
    v00, v01, v02, v10 = [[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]], [[0], [0], [0]]
    records = [[v00, v01, v02], [v10]]
    master = pd.DataFrame({
        "SUBJECT_ID": [17, 17, 17, 99],
        "HADM_ID": [100, 200, 300, 400],
        "safedrug_patient_index": [0, 0, 0, 1],
        "safedrug_visit_index": [0, 1, 2, 0],
        "ADMITTIME": pd.to_datetime(["2100-05-01", "2099-12-01", "2100-02-01", "2101-01-01"]),
    })
    new_records, new_hadm, table = B.reorder_records(records, master)
    assert new_records[0] == [v01, v02, v00] and new_records[0][0] is v01  # same objects, re-ordered
    assert new_records[1] == [v10]
    assert new_hadm == [[200, 300, 100], [400]]
    row = table[(table.patient_index == 0) & (table.HADM_ID == 100)].iloc[0]
    assert row.old_visit_index == 0 and row.new_visit_index == 2
    admit_of = dict(zip(master.HADM_ID, master.ADMITTIME))
    assert B.backwards_transition_share([[100, 200, 300]], admit_of) == 0.5
    assert B.backwards_transition_share(new_hadm, admit_of) == 0.0


def test_reorder_records_rejects_misaligned_sidecar():
    records = [[[[1], [1], [1]], [[2], [2], [2]]]]
    master = pd.DataFrame({"SUBJECT_ID": [1], "HADM_ID": [5], "safedrug_patient_index": [0],
                           "safedrug_visit_index": [0], "ADMITTIME": pd.to_datetime(["2100-01-01"])})
    try:
        B.reorder_records(records, master)
    except AssertionError:
        return
    raise AssertionError("expected misalignment to be rejected")
