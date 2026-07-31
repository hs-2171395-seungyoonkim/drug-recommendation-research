import pandas as pd
import pytest

from mimic_iv_rebuild.records import (
    attach_and_sort_admissions,
    build_records_and_hadm_ids,
    combine_admissions,
    load_admissions,
)
from mimic_iv_rebuild.vocab import Voc, build_vocab_from_column


def _codes_df(rows):
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id", "code"])


def _complete_table(rows):
    return pd.DataFrame({
        "subject_id": [subject_id for subject_id, _ in rows],
        "hadm_id": [hadm_id for _, hadm_id in rows],
        "diag_codes": [["D"] for _ in rows],
        "proc_codes": [["P"] for _ in rows],
        "med_codes": [["M"] for _ in rows],
    })


def test_attach_and_sort_admissions_uses_admittime_not_hadm_id():
    table = _complete_table([[1, 200], [1, 100]])
    admissions = pd.DataFrame({
        "subject_id": [1, 1],
        "hadm_id": [100, 200],
        "admittime": ["2258-05-20 11:08:00", "2258-05-22 11:08:00"],
    })

    ordered = attach_and_sort_admissions(table, admissions)

    assert ordered["hadm_id"].tolist() == [100, 200]
    assert ordered["admittime"].is_monotonic_increasing


def test_attach_and_sort_admissions_rejects_unmatched_complete_admission():
    table = _complete_table([[1, 100], [1, 200]])
    admissions = pd.DataFrame({
        "subject_id": [1],
        "hadm_id": [100],
        "admittime": ["2258-05-20 11:08:00"],
    })

    with pytest.raises(ValueError, match="missing an admissions.admittime match"):
        attach_and_sort_admissions(table, admissions)


def test_attach_and_sort_admissions_rejects_duplicate_admission_keys():
    table = _complete_table([[1, 100], [1, 200]])
    admissions = pd.DataFrame({
        "subject_id": [1, 1, 1],
        "hadm_id": [100, 100, 200],
        "admittime": ["2258-05-20 11:08:00", "2258-05-21 11:08:00", "2258-05-22 11:08:00"],
    })

    with pytest.raises(ValueError, match="duplicate subject_id/hadm_id keys"):
        attach_and_sort_admissions(table, admissions)


def test_load_admissions_reads_gzip_and_parses_admittime(tmp_path):
    path = tmp_path / "admissions.csv.gz"
    pd.DataFrame({
        "subject_id": [1],
        "hadm_id": [100],
        "admittime": ["2258-05-20 11:08:00"],
        "unused": ["ignored"],
    }).to_csv(path, index=False, compression="gzip")

    admissions = load_admissions(path)

    assert admissions.columns.tolist() == ["subject_id", "hadm_id", "admittime"]
    assert admissions["admittime"].tolist() == [pd.Timestamp("2258-05-20 11:08:00")]


def test_load_admissions_rejects_missing_or_invalid_admittime(tmp_path):
    path = tmp_path / "admissions.csv.gz"
    pd.DataFrame({
        "subject_id": [1, 1],
        "hadm_id": [100, 200],
        "admittime": ["2258-05-20 11:08:00", "not-a-time"],
    }).to_csv(path, index=False, compression="gzip")

    with pytest.raises(ValueError, match="missing or invalid admittime"):
        load_admissions(path)


def test_combine_admissions_inner_joins_and_drops_incomplete_admissions():
    # Subject 1 has 3 diagnosis-level admissions (100, 200, 300); hadm 200
    # has no medication row so it gets dropped by the inner join, but the
    # subject still legitimately has 2 COMPLETE surviving admissions (100,
    # 300), so this isolates "does the inner join drop incomplete
    # admissions" from the separate ">=2 complete visits" filter.
    diag_df = _codes_df([[1, 100, "D1"], [1, 100, "D2"], [1, 200, "D3"], [1, 300, "D4"]])
    proc_df = _codes_df([[1, 100, "P1"], [1, 200, "P2"], [1, 300, "P3"]])
    med_df = _codes_df([[1, 100, "M1"], [1, 300, "M2"]])  # hadm 200 has no med -> dropped
    table = combine_admissions(diag_df, proc_df, med_df)
    assert list(table["hadm_id"]) == [100, 300]
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


def test_build_records_and_hadm_ids_pre_built_and_auto_built_vocs_agree():
    table = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "hadm_id": [100, 101, 200],
        "diag_codes": [["D1"], ["D1", "D2"], ["D3"]],
        "proc_codes": [["P1"], [], ["P1", "P2"]],
        "med_codes": [["M1"], ["M1", "M2"], ["M3"]],
    })

    pre_diag_voc = build_vocab_from_column(table["diag_codes"])
    pre_pro_voc = build_vocab_from_column(table["proc_codes"])
    pre_med_voc = build_vocab_from_column(table["med_codes"])
    pre_records, pre_hadm_ids = build_records_and_hadm_ids(table, pre_diag_voc, pre_pro_voc, pre_med_voc)

    auto_diag_voc, auto_pro_voc, auto_med_voc = Voc(), Voc(), Voc()
    auto_records, auto_hadm_ids = build_records_and_hadm_ids(table, auto_diag_voc, auto_pro_voc, auto_med_voc)

    assert pre_records == auto_records
    assert pre_hadm_ids == auto_hadm_ids
    assert pre_diag_voc.word2idx == auto_diag_voc.word2idx
    assert pre_pro_voc.word2idx == auto_pro_voc.word2idx
    assert pre_med_voc.word2idx == auto_med_voc.word2idx
