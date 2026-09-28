import pickle
import sys
from pathlib import Path

import dill
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_acute_novelty_features import (
    admittime_or_raise,
    compute_novelty_for_patient,
    count_chronology_violations,
    main,
    split_boundaries,
    split_of_patient,
)


def test_compute_novelty_first_visit_is_all_no_history():
    diag_lists = [[1, 2, 3]]
    proc_lists = [[9]]
    diag_out, proc_out = compute_novelty_for_patient(diag_lists, proc_lists)
    assert diag_out == [[0, 0, 0]]
    assert proc_out == [[0]]


def test_compute_novelty_second_visit_marks_seen_and_new():
    diag_lists = [[1, 2], [1, 3]]  # visit 1: code 1 seen before, code 3 new
    proc_lists = [[5], [5, 6]]
    diag_out, proc_out = compute_novelty_for_patient(diag_lists, proc_lists)
    assert diag_out[0] == [0, 0]
    assert diag_out[1] == [1, 2]
    assert proc_out[0] == [0]
    assert proc_out[1] == [1, 2]


def test_compute_novelty_modalities_do_not_cross_contaminate():
    # code index 1 appears in visit 0's diag list and visit 1's proc list --
    # a proc code must never be marked "seen" because of diag history.
    diag_lists = [[1], [2]]
    proc_lists = [[9], [1]]
    diag_out, proc_out = compute_novelty_for_patient(diag_lists, proc_lists)
    assert proc_out[1] == [2]  # code 1 is new to proc history, despite being an old diag code


def test_compute_novelty_third_visit_uses_union_of_both_earlier_visits():
    diag_lists = [[1], [2], [1, 2, 3]]
    proc_lists = [[], [], []]
    diag_out, _ = compute_novelty_for_patient(diag_lists, proc_lists)
    assert diag_out[2] == [1, 1, 2]  # 1 and 2 seen (from visits 0/1), 3 new


def test_compute_novelty_uses_admittime_chronology_not_records_order():
    # Controller override: novelty is defined over TRUE chronology
    # (ADMITTIME), not records order. records order is [B, A] (position 0 is
    # visit "B", position 1 is visit "A") but ADMITTIME orders A before B --
    # so A is chronologically first ("no history") and B's codes that
    # appear in A are "seen before", even though B is stored first.
    diag_lists = [[1, 2], [1, 3]]  # position 0 = B: codes 1,2 -- position 1 = A: codes 1,3
    proc_lists = [[], []]
    admittimes = [
        pd.Timestamp("2250-06-01 08:00:00"),  # B (position 0) -- LATER
        pd.Timestamp("2250-01-01 08:00:00"),  # A (position 1) -- EARLIER
    ]
    diag_out, _ = compute_novelty_for_patient(diag_lists, proc_lists, admittimes=admittimes)
    # A (position 1) is chronologically first -> all "no history"
    assert diag_out[1] == [0, 0]
    # B (position 0) is chronologically second -> code 1 (in A) is "seen
    # before", code 2 (not in A) is "new"
    assert diag_out[0] == [1, 2]


def test_compute_novelty_admittime_ties_broken_by_records_order():
    # Two visits share the same ADMITTIME -- the tie is broken by records
    # order, so position 0 is treated as chronologically first.
    diag_lists = [[1], [1, 2]]
    proc_lists = [[], []]
    same_time = pd.Timestamp("2250-03-01 12:00:00")
    diag_out, _ = compute_novelty_for_patient(diag_lists, proc_lists, admittimes=[same_time, same_time])
    assert diag_out[0] == [0]  # position 0 wins the tie -> no history
    assert diag_out[1] == [1, 2]  # position 1 sees position 0's code 1 as "seen before"


def test_split_boundaries_matches_known_cohort_size():
    # verified against the real records_final.pkl (6,350 patients) during
    # planning: split_point=4233, eval_len=1058.
    assert split_boundaries(6350) == (4233, 1058)


def test_split_of_patient_boundaries():
    split_point, eval_len = 4233, 1058
    assert split_of_patient(0, split_point, eval_len) == "train"
    assert split_of_patient(4232, split_point, eval_len) == "train"
    assert split_of_patient(4233, split_point, eval_len) == "test"
    assert split_of_patient(5290, split_point, eval_len) == "test"
    assert split_of_patient(5291, split_point, eval_len) == "eval"
    assert split_of_patient(6349, split_point, eval_len) == "eval"


def test_count_chronology_violations_detects_reversed_pair():
    hadm_ids = [[100, 200]]  # one patient, two visits
    master = pd.DataFrame(
        {
            "HADM_ID": [100, 200],
            "ADMITTIME": ["2250-06-01 08:00:00", "2250-01-01 08:00:00"],  # visit 1 is EARLIER
        }
    )
    result = count_chronology_violations(hadm_ids, master)
    assert result["total_adjacent_pairs"] == 1
    assert result["violations"] == 1
    assert result["violating_patients"] == 1
    assert result["n_patients"] == 1


def test_count_chronology_violations_forward_order_has_no_violations():
    hadm_ids = [[100, 200]]
    master = pd.DataFrame(
        {
            "HADM_ID": [100, 200],
            "ADMITTIME": ["2250-01-01 08:00:00", "2250-06-01 08:00:00"],
        }
    )
    result = count_chronology_violations(hadm_ids, master)
    assert result["violations"] == 0
    assert result["violating_patients"] == 0


def test_count_chronology_violations_single_visit_patient_contributes_no_pairs():
    hadm_ids = [[100], [200, 300]]
    master = pd.DataFrame(
        {
            "HADM_ID": [100, 200, 300],
            "ADMITTIME": ["2250-01-01", "2250-01-01", "2250-02-01"],
        }
    )
    result = count_chronology_violations(hadm_ids, master)
    assert result["total_adjacent_pairs"] == 1
    assert result["n_patients"] == 2


def test_admittime_or_raise_names_the_missing_hadm_id():
    with pytest.raises(KeyError, match="999"):
        admittime_or_raise({100: pd.Timestamp("2250-01-01")}, 999)


def test_admittime_or_raise_returns_value_when_present():
    ts = pd.Timestamp("2250-01-01")
    assert admittime_or_raise({100: ts}, 100) == ts


def test_main_end_to_end_uses_admittime_chronology_not_records_order(tmp_path):
    # Controller override, exercised end to end through main(): records
    # order is [B, A] (position 0 = B, position 1 = A) but ADMITTIME(A) <
    # ADMITTIME(B) -- the written novelty.pkl must reflect chronological
    # order, not records order.
    records_final = [
        [
            [[1, 2], [], []],  # position 0 = B: diag codes 1, 2
            [[1, 3], [], []],  # position 1 = A: diag codes 1, 3
        ]
    ]
    hadm_ids = [[200, 100]]  # B = HADM_ID 200 (position 0), A = HADM_ID 100 (position 1)

    records_path = tmp_path / "records_final.pkl"
    with open(records_path, "wb") as fh:
        dill.dump(records_final, fh)
    hadm_ids_path = tmp_path / "records_final_hadm_ids.pkl"
    with open(hadm_ids_path, "wb") as fh:
        dill.dump(hadm_ids, fh)

    admittimes = pd.DataFrame(
        {
            "HADM_ID": [100, 200],
            "ADMITTIME": ["2250-01-01 08:00:00", "2250-06-01 08:00:00"],  # A earlier than B
        }
    )
    master_csv = tmp_path / "master_visits.csv"
    admittimes.to_csv(master_csv, index=False)
    admissions_csv = tmp_path / "ADMISSIONS.csv"
    admittimes.to_csv(admissions_csv, index=False)

    out_dir = tmp_path / "out"
    main(
        [
            "--records-final", str(records_path),
            "--hadm-ids-pkl", str(hadm_ids_path),
            "--master-visits-csv", str(master_csv),
            "--admissions-csv", str(admissions_csv),
            "--out-dir", str(out_dir),
        ]
    )

    with open(out_dir / "novelty.pkl", "rb") as fh:
        novelty = pickle.load(fh)

    assert novelty[100]["diag"].tolist() == [0, 0]  # A: chronologically first -> no history
    assert novelty[200]["diag"].tolist() == [1, 2]  # B: code 1 seen (in A), code 2 new
