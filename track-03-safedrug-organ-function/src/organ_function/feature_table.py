"""
Per-visit organ function feature computation.

Temporal-leakage rule (exact semantics)
---------------------------------------
A lab value may be used for a visit only if its ``charttime`` is STRICTLY
BEFORE this visit's first medication order time (the ``prescriptions.starttime``
proxy for admission time, see ``organ_function.admission_time``). This can
include labs drawn earlier in the SAME encounter, before any medication was
ordered, which is clinically appropriate since those results are genuinely
available to the prescriber at decision time. It is therefore NOT true that
"labs during the visit are never used" — the boundary is the first order time,
not the encounter boundary. What is excluded is any result the prescriber could
not have seen when writing the first order.

See plan Background for the deviation formula's rationale.
"""
import numpy as np
import pandas as pd

from organ_function.lab_config import LAB_ITEMIDS
from organ_function.new_diagnosis import has_new_diagnosis
from organ_function.subject_lookup import subject_id_for_patient

# Feature dict returned for a lab that has no usable prior reading at all
# (either the subject has no such lab strictly before index_time, or
# index_time itself is unresolvable).
_ALL_MISSING_LAB_RESULT = {
    "value": np.nan,
    "missing": True,
    "deviation": np.nan,
    "delta": np.nan,
    "delta_missing": True,
    "age_days": np.nan,
}


def compute_lab_features(subject_labs: pd.DataFrame, itemid: int, index_time: pd.Timestamp) -> dict:
    """subject_labs: rows for ONE subject_id (any order), with columns
    [subject_id, itemid, charttime, valuenum, ref_range_lower,
    ref_range_upper] (organ_function.lab_subset.LAB_SUBSET_COLUMNS).
    index_time: only lab values with charttime STRICTLY BEFORE this may be
    used. index_time is the visit's first medication order time, so labs drawn
    earlier in the same encounter (before any order) ARE eligible — they are
    genuinely on the prescriber's screen at decision time.
    Returns dict with keys: value, missing, deviation, delta, delta_missing,
    age_days.
      - delta is NaN when fewer than 2 eligible prior readings exist;
        delta_missing is True in exactly that case (so a downstream imputer can
        tell "no trend available" apart from "trend happened to be 0.0").
      - age_days is (index_time - latest charttime) in days, i.e. how stale the
        value backing this feature is; NaN when there is no prior value."""
    prior = subject_labs[
        (subject_labs["itemid"] == itemid) & (subject_labs["charttime"] < index_time)
    ].sort_values("charttime")

    if len(prior) == 0:
        return dict(_ALL_MISSING_LAB_RESULT)

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
        delta_missing = False
    else:
        delta = np.nan
        delta_missing = True

    age_days = (index_time - latest["charttime"]).total_seconds() / 86400.0

    return {
        "value": value,
        "missing": False,
        "deviation": deviation,
        "delta": delta,
        "delta_missing": delta_missing,
        "age_days": age_days,
    }


def build_visit_features(
    subject_labs: pd.DataFrame,
    index_time: pd.Timestamp,
    current_diag_ids: list[int],
    prior_diag_ids_seen: set[int],
) -> dict:
    """subject_labs: all labs for the ONE subject_id this visit belongs to
    (or an empty frame with LAB_SUBSET_COLUMNS if the subject has none).
    index_time: this visit's first medication order time (proxy admission
    time), or NaT — see build_feature_table's NaT handling below. Every lab is
    resolved strictly before index_time (labs earlier in the same encounter but
    before the first order are eligible; see module docstring).
    Returns 6 keys per lab ({name}_value/_deviation/_delta/_delta_missing/
    _age_days/_missing) plus new_diagnosis_flag."""
    features: dict = {}
    # Hoisted: index_time does not change across labs, so branch once.
    index_time_missing = pd.isna(index_time)
    for name, itemid in LAB_ITEMIDS.items():
        if index_time_missing:
            lab_result = _ALL_MISSING_LAB_RESULT
        else:
            lab_result = compute_lab_features(subject_labs, itemid, index_time)
        features[f"{name}_value"] = lab_result["value"]
        features[f"{name}_deviation"] = lab_result["deviation"]
        features[f"{name}_delta"] = lab_result["delta"]
        features[f"{name}_delta_missing"] = lab_result["delta_missing"]
        features[f"{name}_age_days"] = lab_result["age_days"]
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
    Returns list[patient] of list[visit feature dict], same shape AND same
    per-patient visit order as records/hadm_ids.

    Visit ordering: records_final2.pkl stores each patient's visits in hadm_id
    order, which is NOT chronological (verified on the real data: 50.1% of
    adjacent visit pairs are inverted). new_diagnosis_flag is a strictly
    historical feature, so prior_diag_ids_seen is accumulated in CHRONOLOGICAL
    order (by each visit's proxy admission time) while each visit's features
    are written back to its ORIGINAL list position — downstream consumers rely
    on positional alignment with records_final2.pkl. A visit whose index_time
    is unresolvable (NaT / absent from admission_times) sorts LAST, since it
    cannot be confidently placed earlier; its real NaT index_time is still what
    gets passed to build_visit_features, so its labs are all-missing."""
    assert len(records) == len(hadm_ids), (
        f"records/hadm_ids length mismatch: {len(records)} patients in records "
        f"vs {len(hadm_ids)} in hadm_ids — zip() would silently truncate"
    )

    empty_labs = lab_subset.iloc[0:0]
    # One groupby pass instead of a full-frame scan per patient. .indices maps
    # subject_id -> positional row indices; .take() then slices just that
    # subject's rows. (Cheaper in memory than materializing every group frame.)
    subject_row_indices = lab_subset.groupby("subject_id", sort=False).indices

    table = []
    for patient_idx, (patient_records, patient_hadm_ids) in enumerate(zip(records, hadm_ids)):
        assert len(patient_records) == len(patient_hadm_ids), (
            f"patient {patient_idx}: {len(patient_records)} visits in records "
            f"but {len(patient_hadm_ids)} hadm_ids — zip() would silently truncate"
        )

        subject_id = subject_id_for_patient(hadm_to_subject, patient_hadm_ids)
        row_idx = subject_row_indices.get(subject_id)
        subject_labs = empty_labs if row_idx is None else lab_subset.take(row_idx)

        patient_times = [admission_times.get(hadm_id, pd.NaT) for hadm_id in patient_hadm_ids]
        if any(pd.isna(t) for t in patient_times):
            raise ValueError(f"patient {patient_idx} is missing official admission time")
        if any(later < earlier for earlier, later in zip(patient_times, patient_times[1:])):
            raise ValueError(f"patient {patient_idx} is not in nondecreasing admission-time order")

        prior_diag_ids_seen: set = set()
        patient_features: list = []
        for i in range(len(patient_hadm_ids)):
            visit = patient_records[i]
            hadm_id = patient_hadm_ids[i]
            current_diag_ids = visit[0]
            index_time = patient_times[i]
            visit_features = build_visit_features(
                subject_labs, index_time, current_diag_ids, prior_diag_ids_seen
            )
            patient_features.append(visit_features)
            prior_diag_ids_seen |= set(current_diag_ids)

        table.append(patient_features)
    return table
