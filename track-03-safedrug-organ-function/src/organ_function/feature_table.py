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
