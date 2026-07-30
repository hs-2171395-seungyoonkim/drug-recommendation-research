"""
Real ADMITTIME/admissions.csv is unavailable (verified absent on this
machine — see plan Background). MIN(starttime) per hadm_id from
prescriptions.csv is used as a proxy: every hadm_id in records_final2.pkl
already has >=1 prescription row by construction, and only 1 of 452,115
hadm_ids in the real prescriptions.csv has every starttime null.
"""
import pandas as pd


def build_admission_times(presc_df: pd.DataFrame) -> pd.Series:
    """presc_df: DataFrame with [hadm_id, starttime] columns (subset of
    data/raw_mimic_iv/prescriptions.csv). Returns a Series indexed by
    hadm_id, values = earliest parsed starttime. hadm_ids with every
    starttime null map to NaT."""
    df = presc_df.copy()
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    return df.groupby("hadm_id")["starttime"].min()
