"""
records_final2.pkl / records_final2_hadm_ids.pkl never stored subject_id.
This recovers it via diagnoses_icd.csv's hadm_id -> subject_id mapping, which
is a verified total, 1:1 mapping for every hadm_id referenced by
records_final2_hadm_ids.pkl (see plan Background).
"""
import pandas as pd


def build_hadm_to_subject(diag_df: pd.DataFrame) -> pd.Series:
    """diag_df: DataFrame with at least [subject_id, hadm_id] columns
    (data/raw_mimic_iv/diagnoses_icd.csv). Returns a Series indexed by
    hadm_id, values subject_id. Raises ValueError if any hadm_id maps to more
    than one subject_id (verified zero such cases on the real
    diagnoses_icd.csv, but drop_duplicates would otherwise silently keep an
    arbitrary row and produce a wrong subject_id)."""
    per_hadm_subjects = diag_df.groupby("hadm_id")["subject_id"].nunique()
    ambiguous = per_hadm_subjects[per_hadm_subjects > 1]
    if len(ambiguous):
        raise ValueError(
            "hadm_id maps to multiple subject_ids: "
            + ", ".join(f"{h} -> {n} distinct subject_ids" for h, n in ambiguous.items())
        )
    return diag_df.drop_duplicates("hadm_id").set_index("hadm_id")["subject_id"]


def subject_id_for_patient(hadm_to_subject: pd.Series, patient_hadm_ids: list[int]) -> int:
    """patient_hadm_ids: one patient's entry from records_final2_hadm_ids.pkl."""
    subject_ids = {hadm_to_subject.loc[h] for h in patient_hadm_ids}
    if len(subject_ids) != 1:
        raise ValueError(
            f"inconsistent subject_id across patient's hadm_ids: {patient_hadm_ids} -> {subject_ids}"
        )
    return subject_ids.pop()
