"""
Per-visit organ function feature computation. See plan Background for the
temporal-leakage rule (index_time is strict-exclusive) and the deviation
formula's rationale.
"""
import numpy as np
import pandas as pd


def compute_lab_features(subject_labs: pd.DataFrame, itemid: int, index_time: pd.Timestamp) -> dict:
    """subject_labs: rows for ONE subject_id (any order), with columns
    [subject_id, itemid, charttime, valuenum, ref_range_lower,
    ref_range_upper] (organ_function.lab_subset.LAB_SUBSET_COLUMNS).
    index_time: only lab values with charttime STRICTLY BEFORE this may be
    used (avoids leaking values from the visit being predicted).
    Returns dict with keys: value, missing, deviation, delta."""
    prior = subject_labs[
        (subject_labs["itemid"] == itemid) & (subject_labs["charttime"] < index_time)
    ].sort_values("charttime")

    if len(prior) == 0:
        return {"value": np.nan, "missing": True, "deviation": np.nan, "delta": np.nan}

    latest = prior.iloc[-1]
    value = float(latest["valuenum"])
    lower, upper = latest["ref_range_lower"], latest["ref_range_upper"]

    if pd.isna(lower) or pd.isna(upper) or upper <= lower:
        deviation = np.nan
    elif value > upper:
        deviation = (value - upper) / (upper - lower)
    elif value < lower:
        deviation = (value - lower) / (upper - lower)
    else:
        deviation = 0.0

    if len(prior) >= 2:
        delta = value - float(prior.iloc[-2]["valuenum"])
    else:
        delta = np.nan

    return {"value": value, "missing": False, "deviation": deviation, "delta": delta}


from organ_function.lab_config import LAB_ITEMIDS
from organ_function.new_diagnosis import has_new_diagnosis
from organ_function.subject_lookup import subject_id_for_patient


def build_visit_features(
    subject_labs: pd.DataFrame,
    index_time: pd.Timestamp,
    current_diag_ids: list[int],
    prior_diag_ids_seen: set[int],
) -> dict:
    """subject_labs: all labs for the ONE subject_id this visit belongs to
    (or an empty frame with LAB_SUBSET_COLUMNS if the subject has none).
    index_time: this visit's proxy admission time, or NaT (see
    build_feature_table's NaT handling below)."""
    features: dict = {}
    for name, itemid in LAB_ITEMIDS.items():
        if pd.isna(index_time):
            lab_result = {"value": np.nan, "missing": True, "deviation": np.nan, "delta": np.nan}
        else:
            lab_result = compute_lab_features(subject_labs, itemid, index_time)
        features[f"{name}_value"] = lab_result["value"]
        features[f"{name}_deviation"] = lab_result["deviation"]
        features[f"{name}_delta"] = lab_result["delta"]
        features[f"{name}_missing"] = lab_result["missing"]

    features["new_diagnosis_flag"] = has_new_diagnosis(current_diag_ids, prior_diag_ids_seen)
    return features


def build_feature_table(
    records: list,
    hadm_ids: list,
    lab_subset: pd.DataFrame,
    admission_times: pd.Series,
    hadm_to_subject: pd.Series,
) -> list:
    """records/hadm_ids: records_final2.pkl / records_final2_hadm_ids.pkl
    (same length, same per-patient visit order). lab_subset: output of
    organ_function.lab_subset.extract_lab_subset(). admission_times: output
    of organ_function.admission_time.build_admission_times(). hadm_to_subject:
    output of organ_function.subject_lookup.build_hadm_to_subject().
    Returns list[patient] of list[visit feature dict], same shape as
    records."""
    empty_labs = lab_subset.iloc[0:0]
    table = []
    for patient_records, patient_hadm_ids in zip(records, hadm_ids):
        subject_id = subject_id_for_patient(hadm_to_subject, patient_hadm_ids)
        subject_labs = lab_subset[lab_subset["subject_id"] == subject_id]
        if len(subject_labs) == 0:
            subject_labs = empty_labs

        prior_diag_ids_seen: set = set()
        patient_features = []
        for visit, hadm_id in zip(patient_records, patient_hadm_ids):
            current_diag_ids = visit[0]
            index_time = admission_times.get(hadm_id, np.nan)
            visit_features = build_visit_features(
                subject_labs, index_time, current_diag_ids, prior_diag_ids_seen
            )
            patient_features.append(visit_features)
            prior_diag_ids_seen |= set(current_diag_ids)

        table.append(patient_features)
    return table
