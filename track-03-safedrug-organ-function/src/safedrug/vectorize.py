"""
Converts organ_function_features.pkl's per-visit dicts (73 keys each) into
a fixed-order float32 vector for model input. Missing numeric values are
imputed with TRAIN-SPLIT means (frozen at fit time, applied unchanged to
test/eval - no leakage), then z-score standardized using TRAIN-SPLIT std.
Binary flags pass through as 0.0/1.0 unchanged - no imputation, no
standardization. See design spec §4.
"""
import numpy as np

from organ_function.lab_config import LAB_NAMES

NUMERIC_SUFFIXES = ["value", "deviation", "delta", "age_days"]
BINARY_SUFFIXES = ["missing", "delta_missing"]

NUMERIC_COLUMNS = [f"{name}_{suffix}" for name in LAB_NAMES for suffix in NUMERIC_SUFFIXES]
BINARY_COLUMNS = [
    f"{name}_{suffix}" for name in LAB_NAMES for suffix in BINARY_SUFFIXES
] + ["new_diagnosis_flag"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS


def fit_impute_stats(train_visit_features: list) -> dict:
    """train_visit_features: flat list of per-visit feature dicts from the
    TRAIN split only. Returns {column: {"mean": float, "std": float}} for
    each column in NUMERIC_COLUMNS (std=1.0 substituted when the true std
    is 0, so standardization becomes a no-op rather than a divide-by-zero)."""
    stats = {}
    for col in NUMERIC_COLUMNS:
        values = np.array([v[col] for v in train_visit_features], dtype=np.float64)
        values = values[~np.isnan(values)]
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 1.0
        if std == 0.0:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def to_vector(visit_features: dict, impute_stats: dict) -> np.ndarray:
    """visit_features: one visit's dict from organ_function_features.pkl.
    impute_stats: fit_impute_stats()'s output (from the TRAIN split).
    Returns a FEATURE_COLUMNS-ordered float32 array of length 73."""
    values = []
    for col in NUMERIC_COLUMNS:
        raw = visit_features[col]
        if np.isnan(raw):
            raw = impute_stats[col]["mean"]
        standardized = (raw - impute_stats[col]["mean"]) / impute_stats[col]["std"]
        values.append(standardized)
    for col in BINARY_COLUMNS:
        values.append(1.0 if visit_features[col] else 0.0)
    return np.array(values, dtype=np.float32)
