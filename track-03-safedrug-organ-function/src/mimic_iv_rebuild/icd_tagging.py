"""
Tags ICD diagnosis/procedure codes with their version (DrugRec's approach,
github.com/ssshddd/DrugRec, data/mimic-iv/processing_iv.py) instead of the
icd_version==9 filter found in
HI-DR/HEIDR/service_personalization/rebuild_records_with_hadm_id.py:36-37,
which silently drops 35.6% of MIMIC-IV admissions (185,733 of 521,111
hadm_ids have ONLY ICD-10 diagnoses and zero ICD-9 ones - verified against
the real data/raw_mimic_iv/diagnoses_icd.csv, see plan Global Constraints).
"""
import pandas as pd


def tag_icd(icd_code: str, icd_version) -> str:
    """DrugRec's tagging scheme: "5990_9", "I10_10" - keeps ICD-9 and
    ICD-10 codes as distinct vocabulary tokens rather than conflating or
    dropping either version."""
    return f"{icd_code}_{icd_version}"


def _load_and_tag(path: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"icd_code": str})
    df = df.dropna(subset=["icd_code", "icd_version"])
    df["code"] = [tag_icd(c, v) for c, v in zip(df["icd_code"], df["icd_version"])]
    df = df[["subject_id", "hadm_id", "code"]].drop_duplicates()

    top_codes = df["code"].value_counts().nlargest(top_k).index
    return df[df["code"].isin(top_codes)].reset_index(drop=True)


def load_and_tag_diagnoses(path: str, top_k: int = 2000) -> pd.DataFrame:
    """path: data/raw_mimic_iv/diagnoses_icd.csv
    ([subject_id, hadm_id, seq_num, icd_code, icd_version]).
    Returns [subject_id, hadm_id, code], code = tag_icd(...), kept only for
    the top_k most frequent tagged codes (matches original SafeDrug's
    filter_2000_most_diag, applied over the combined ICD-9+10 code space)."""
    return _load_and_tag(path, top_k)


def load_and_tag_procedures(path: str, top_k: int = 1000) -> pd.DataFrame:
    """path: data/raw_mimic_iv/procedures_icd.csv
    ([subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version]).
    Same tagging/top-K logic as load_and_tag_diagnoses, with SafeDrug's
    procedure cutoff (1000)."""
    return _load_and_tag(path, top_k)
