import pandas as pd

from mimic_iv_rebuild.records import build_records_and_hadm_ids, combine_admissions
from mimic_iv_rebuild.vocab import Voc


def _codes_df(rows):
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id", "code"])


def test_combine_admissions_inner_joins_and_drops_incomplete_admissions():
    diag_df = _codes_df([[1, 100, "D1"], [1, 100, "D2"], [1, 200, "D3"]])
    proc_df = _codes_df([[1, 100, "P1"], [1, 200, "P2"]])
    med_df = _codes_df([[1, 100, "M1"]])  # hadm 200 has no med -> dropped
    table = combine_admissions(diag_df, proc_df, med_df)
    assert list(table["hadm_id"]) == [100]
    assert table.iloc[0]["diag_codes"] == ["D1", "D2"]
    assert table.iloc[0]["proc_codes"] == ["P1"]
    assert table.iloc[0]["med_codes"] == ["M1"]


def test_combine_admissions_filters_patients_with_fewer_than_two_visits():
    diag_df = _codes_df([[1, 100, "D1"], [2, 200, "D1"], [2, 201, "D1"]])
    proc_df = _codes_df([[1, 100, "P1"], [2, 200, "P1"], [2, 201, "P1"]])
    med_df = _codes_df([[1, 100, "M1"], [2, 200, "M1"], [2, 201, "M1"]])
    table = combine_admissions(diag_df, proc_df, med_df)
    assert set(table["subject_id"]) == {2}  # subject 1 has only 1 visit
    assert len(table) == 2


def test_combine_admissions_sorts_by_subject_then_hadm_id():
    diag_df = _codes_df([[2, 300, "D"], [2, 100, "D"], [1, 500, "D"], [1, 200, "D"]])
    proc_df = _codes_df([[2, 300, "P"], [2, 100, "P"], [1, 500, "P"], [1, 200, "P"]])
    med_df = _codes_df([[2, 300, "M"], [2, 100, "M"], [1, 500, "M"], [1, 200, "M"]])
    table = combine_admissions(diag_df, proc_df, med_df)
    assert list(zip(table["subject_id"], table["hadm_id"])) == [
        (1, 200), (1, 500), (2, 100), (2, 300),
    ]


def test_build_records_and_hadm_ids_matches_shapes_and_index_mapping():
    diag_voc, pro_voc, med_voc = Voc(), Voc(), Voc()
    table = pd.DataFrame({
        "subject_id": [1, 1],
        "hadm_id": [100, 101],
        "diag_codes": [["D1"], ["D1", "D2"]],
        "proc_codes": [["P1"], []],
        "med_codes": [["M1"], ["M1"]],
    })
    records, hadm_ids = build_records_and_hadm_ids(table, diag_voc, pro_voc, med_voc)
    assert len(records) == 1  # one patient
    assert len(records[0]) == 2  # two visits
    assert hadm_ids == [[100, 101]]
    assert records[0][0] == [[diag_voc.word2idx["D1"]], [pro_voc.word2idx["P1"]], [med_voc.word2idx["M1"]]]
    assert records[0][1][0] == [diag_voc.word2idx["D1"], diag_voc.word2idx["D2"]]
    assert records[0][1][1] == []
