from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from safedrug_mimic3_cohort import (  # noqa: E402
    assert_records_semantically_equal,
    build_cohort_views,
    build_records_and_hadm_ids,
    combine_visit_tables,
    filter_codes_to_reference_vocabulary,
    retain_early_multi_visit_subjects,
    select_most_frequent_codes,
)


class TinyVoc:
    def __init__(self, words):
        self.word2idx = {word: idx for idx, word in enumerate(words)}


def test_record_equality_ignores_code_order_but_rejects_membership_changes():
    reference = [[[[0, 1], [2], [3, 4]]]]
    assert_records_semantically_equal([[[[1, 0], [2], [4, 3]]]], reference)

    try:
        assert_records_semantically_equal([[[[0], [2], [3, 4]]]], reference)
    except ValueError as exc:
        assert "code-set mismatch" in str(exc)
    else:
        raise AssertionError("a missing code was accepted")


def test_early_multi_visit_filter_is_not_reapplied_after_later_visit_loss():
    prescriptions = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2],
            "HADM_ID": [10, 11, 20],
            "ATC3": ["A", "B", "A"],
        }
    )

    eligible = retain_early_multi_visit_subjects(prescriptions)

    assert eligible[["SUBJECT_ID", "HADM_ID"]].values.tolist() == [[1, 10], [1, 11]]
    # A later filter may remove visit 11. Subject 1 must not be removed again.
    assert eligible[eligible.HADM_ID == 10].SUBJECT_ID.tolist() == [1]


def test_top_code_selection_uses_frequency_then_reference_order_for_ties():
    rows = pd.DataFrame({"code": ["B", "A", "C", "B", "A", "C", "D"]})

    selected = select_most_frequent_codes(rows, "code", limit=2)

    assert selected == ["B", "A"]


def test_reference_vocabulary_filter_removes_version_dependent_boundary_codes():
    rows = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2],
            "HADM_ID": [10, 10, 20],
            "ICD9_CODE": ["KEPT", "TIED_OUT", "TIED_OUT"],
        }
    )

    filtered = filter_codes_to_reference_vocabulary(
        rows, "ICD9_CODE", TinyVoc(["KEPT"])
    )

    assert filtered.ICD9_CODE.tolist() == ["KEPT"]


def test_visit_combination_keeps_only_three_table_intersection():
    med = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2],
            "HADM_ID": [10, 11, 20],
            "ATC3": ["M1", "M2", "M3"],
        }
    )
    diag = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2],
            "HADM_ID": [10, 11, 20],
            "ICD9_CODE": ["D1", "D2", "D3"],
        }
    )
    proc = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2],
            "HADM_ID": [10, 20],
            "PRO_CODE": ["P1", "P3"],
        }
    )

    combined = combine_visit_tables(med, diag, proc)

    assert combined.HADM_ID.tolist() == [10, 20]
    assert combined.loc[0, "ICD9_CODE"] == ["D1"]
    assert combined.loc[0, "PRO_CODE"] == ["P1"]
    assert combined.loc[0, "ATC3"] == ["M1"]


def test_records_and_hadm_sidecar_are_positionally_identical():
    visits = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2],
            "HADM_ID": [10, 11, 20],
            "ICD9_CODE": [["D1"], ["D2", "D1"], ["D2"]],
            "PRO_CODE": [["P1"], ["P2"], ["P1"]],
            "ATC3": [["M1"], ["M2"], ["M1", "M2"]],
        }
    )

    records, hadm_ids, subjects = build_records_and_hadm_ids(
        visits,
        TinyVoc(["D1", "D2"]),
        TinyVoc(["P1", "P2"]),
        TinyVoc(["M1", "M2"]),
    )

    assert records == [[[[0], [0], [0]], [[1, 0], [1], [1]]], [[[1], [0], [0, 1]]]]
    assert hadm_ids == [[10, 11], [20]]
    assert subjects == [1, 2]
    assert [len(patient) for patient in records] == [len(patient) for patient in hadm_ids]


def test_unknown_reference_vocabulary_code_fails_closed():
    visits = pd.DataFrame(
        {
            "SUBJECT_ID": [1],
            "HADM_ID": [10],
            "ICD9_CODE": [["UNKNOWN"]],
            "PRO_CODE": [["P1"]],
            "ATC3": [["M1"]],
        }
    )

    try:
        build_records_and_hadm_ids(
            visits, TinyVoc(["D1"]), TinyVoc(["P1"]), TinyVoc(["M1"])
        )
    except ValueError as exc:
        assert "reference vocabulary" in str(exc)
    else:
        raise AssertionError("unknown code was silently accepted")


def test_cohort_views_preserve_full_and_separate_elective_and_longitudinal():
    master = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 1, 2],
            "HADM_ID": [10, 11, 12, 20],
            "ADMITTIME": pd.to_datetime(
                ["2100-01-01", "2100-02-01", "2100-03-01", "2100-01-15"]
            ),
            "ADMISSION_TYPE": ["ELECTIVE", "EMERGENCY", "URGENT", "EMERGENCY"],
            "safedrug_patient_index": [0, 0, 0, 1],
            "safedrug_visit_index": [0, 1, 2, 0],
        }
    )

    full, acute, longitudinal = build_cohort_views(master)

    assert full.HADM_ID.tolist() == [10, 11, 12, 20]
    assert full.loc[full.HADM_ID == 10, "analysis_status"].item() == "planned/elective"
    assert acute.HADM_ID.tolist() == [11, 12, 20]
    assert longitudinal.HADM_ID.tolist() == [11, 12]
    assert longitudinal.previous_HADM_ID.tolist() == [10, 11]
    assert longitudinal.previous_ADMISSION_TYPE.tolist() == ["ELECTIVE", "EMERGENCY"]
