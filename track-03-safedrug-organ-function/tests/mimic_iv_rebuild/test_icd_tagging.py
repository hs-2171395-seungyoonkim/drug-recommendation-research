import pandas as pd

from mimic_iv_rebuild.icd_tagging import (
    load_and_tag_diagnoses,
    load_and_tag_procedures,
    tag_icd,
)

DIAG_HEADER = "subject_id,hadm_id,seq_num,icd_code,icd_version\n"
PRO_HEADER = "subject_id,hadm_id,seq_num,chartdate,icd_code,icd_version\n"


def test_tag_icd_keeps_icd9_and_icd10_of_same_code_string_distinct():
    assert tag_icd("5990", 9) == "5990_9"
    assert tag_icd("5990", 10) == "5990_10"
    assert tag_icd("5990", 9) != tag_icd("5990", 10)


def test_load_and_tag_diagnoses_keeps_both_icd_versions(tmp_path):
    path = tmp_path / "diagnoses_icd.csv"
    path.write_text(
        DIAG_HEADER
        + "1,100,1,5990,9\n"
        + "1,100,2,I10,10\n"
        + "2,200,1,5990,9\n"
    )
    result = load_and_tag_diagnoses(str(path), top_k=2000)
    codes = set(result["code"])
    assert "5990_9" in codes
    assert "I10_10" in codes
    assert len(result) == 3


def test_load_and_tag_diagnoses_applies_top_k_frequency_filter(tmp_path):
    path = tmp_path / "diagnoses_icd.csv"
    rows = [DIAG_HEADER]
    for hadm in range(10):
        rows.append(f"1,{hadm},1,COMMON,9\n")
    rows.append("1,999,1,RARE,9\n")
    path.write_text("".join(rows))
    result = load_and_tag_diagnoses(str(path), top_k=1)
    assert set(result["code"]) == {"COMMON_9"}
    assert "RARE_9" not in set(result["code"])


def test_load_and_tag_procedures_keeps_both_icd_versions(tmp_path):
    path = tmp_path / "procedures_icd.csv"
    path.write_text(
        PRO_HEADER
        + "1,100,1,2024-01-01,4562,9\n"
        + "1,100,2,2024-01-01,0DTJ4ZZ,10\n"
    )
    result = load_and_tag_procedures(str(path), top_k=1000)
    codes = set(result["code"])
    assert "4562_9" in codes
    assert "0DTJ4ZZ_10" in codes


def test_load_and_tag_diagnoses_against_real_data_has_both_icd_versions():
    result = load_and_tag_diagnoses("data/raw_mimic_iv/diagnoses_icd.csv", top_k=2000)
    has_icd9 = any(c.endswith("_9") for c in result["code"])
    has_icd10 = any(c.endswith("_10") for c in result["code"])
    assert has_icd9 is True
    assert has_icd10 is True
