import numpy as np

from organ_function.lab_config import LAB_NAMES
from safedrug.vectorize import (
    BINARY_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    fit_impute_stats,
    to_vector,
)


def _make_visit(value=1.0, deviation=0.0, delta=0.0, age_days=1.0,
                 missing=False, delta_missing=False, new_diag=False):
    v = {}
    for name in LAB_NAMES:
        v[f"{name}_value"] = value
        v[f"{name}_deviation"] = deviation
        v[f"{name}_delta"] = delta
        v[f"{name}_age_days"] = age_days
        v[f"{name}_missing"] = missing
        v[f"{name}_delta_missing"] = delta_missing
    v["new_diagnosis_flag"] = new_diag
    return v


def test_feature_columns_has_73_entries_in_numeric_then_binary_order():
    assert len(FEATURE_COLUMNS) == 73
    assert len(NUMERIC_COLUMNS) == 48
    assert len(BINARY_COLUMNS) == 25
    assert FEATURE_COLUMNS == NUMERIC_COLUMNS + BINARY_COLUMNS


def test_fit_impute_stats_computes_train_mean_and_std():
    visits = [_make_visit(value=v) for v in [1.0, 2.0, 3.0]]
    stats = fit_impute_stats(visits)
    col = f"{LAB_NAMES[0]}_value"
    assert stats[col]["mean"] == 2.0
    assert stats[col]["std"] == np.array([1.0, 2.0, 3.0]).std()


def test_fit_impute_stats_zero_variance_substitutes_std_one():
    visits = [_make_visit(value=5.0), _make_visit(value=5.0)]
    stats = fit_impute_stats(visits)
    col = f"{LAB_NAMES[0]}_value"
    assert stats[col]["mean"] == 5.0
    assert stats[col]["std"] == 1.0


def test_to_vector_imputes_nan_with_train_mean_to_zero_after_standardization():
    visits = [_make_visit(value=v) for v in [1.0, 2.0, 3.0]]
    stats = fit_impute_stats(visits)
    visit = _make_visit(value=float("nan"))
    vec = to_vector(visit, stats)
    idx = FEATURE_COLUMNS.index(f"{LAB_NAMES[0]}_value")
    assert vec[idx] == 0.0


def test_to_vector_standardizes_real_value():
    visits = [_make_visit(value=v) for v in [1.0, 2.0, 3.0]]
    stats = fit_impute_stats(visits)
    visit = _make_visit(value=4.0)
    vec = to_vector(visit, stats)
    idx = FEATURE_COLUMNS.index(f"{LAB_NAMES[0]}_value")
    expected = (4.0 - 2.0) / np.array([1.0, 2.0, 3.0]).std()
    assert abs(vec[idx] - expected) < 1e-6


def test_to_vector_binary_fields_pass_through_as_zero_one():
    visits = [_make_visit()]
    stats = fit_impute_stats(visits)
    visit = _make_visit(missing=True, new_diag=True)
    vec = to_vector(visit, stats)
    missing_idx = FEATURE_COLUMNS.index(f"{LAB_NAMES[0]}_missing")
    flag_idx = FEATURE_COLUMNS.index("new_diagnosis_flag")
    assert vec[missing_idx] == 1.0
    assert vec[flag_idx] == 1.0


def test_to_vector_returns_float32_array_of_length_73():
    visits = [_make_visit()]
    stats = fit_impute_stats(visits)
    vec = to_vector(_make_visit(), stats)
    assert vec.dtype == np.float32
    assert len(vec) == 73
