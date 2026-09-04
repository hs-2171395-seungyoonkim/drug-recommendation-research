import pickle

import pytest

from organ_function.lab_config import LAB_NAMES
from safedrug.data import (
    build_baseline_dataset,
    build_organ_function_dataset,
    load_records_and_features,
    split_patients,
)
from safedrug.vectorize import FEATURE_COLUMNS, fit_impute_stats


def _make_visit_features(value=1.0):
    v = {}
    for name in LAB_NAMES:
        v[f"{name}_value"] = value
        v[f"{name}_deviation"] = 0.0
        v[f"{name}_delta"] = 0.0
        v[f"{name}_age_days"] = 1.0
        v[f"{name}_missing"] = False
        v[f"{name}_delta_missing"] = False
    v["new_diagnosis_flag"] = False
    return v


def test_split_patients_matches_two_thirds_train_split():
    records = [[["v"]]] * 9  # 9 patients: split_point=int(9*2/3)=6, eval_len=int(3/2)=1
    train_idx, test_idx, eval_idx = split_patients(records)
    assert train_idx == [0, 1, 2, 3, 4, 5]
    assert test_idx == [6]
    assert eval_idx == [7, 8]


def test_build_baseline_dataset_preserves_original_visit_triples():
    records = [[[[1], [2], [3]], [[4], [5], [6]]]]
    dataset = build_baseline_dataset(records, [0])
    assert dataset == [[[[1], [2], [3]], [[4], [5], [6]]]]


def test_build_organ_function_dataset_appends_vectorized_organ_vec():
    records = [[[[1], [2], [3]]]]
    organ_features = [[_make_visit_features(value=5.0)]]
    impute_stats = fit_impute_stats([_make_visit_features(value=v) for v in [1.0, 5.0, 9.0]])
    dataset = build_organ_function_dataset(records, organ_features, [0], impute_stats)
    assert len(dataset[0][0]) == 4
    assert dataset[0][0][0] == [1]
    assert len(dataset[0][0][3]) == len(FEATURE_COLUMNS)


def test_load_records_and_features_raises_on_patient_count_mismatch(tmp_path):
    records_path = tmp_path / "records.pkl"
    features_path = tmp_path / "features.pkl"
    with open(records_path, "wb") as f:
        pickle.dump([[["v"]], [["v"]]], f)  # 2 patients
    with open(features_path, "wb") as f:
        pickle.dump([[{}]], f)  # 1 patient
    with pytest.raises(ValueError, match="length mismatch"):
        load_records_and_features(str(records_path), str(features_path))


def test_load_records_and_features_raises_on_per_patient_visit_mismatch(tmp_path):
    records_path = tmp_path / "records2.pkl"
    features_path = tmp_path / "features2.pkl"
    with open(records_path, "wb") as f:
        pickle.dump([[["v1"], ["v2"]]], f)  # patient 0 has 2 visits
    with open(features_path, "wb") as f:
        pickle.dump([[{}]], f)  # patient 0 has 1 visit
    with pytest.raises(ValueError, match="visit-count mismatch"):
        load_records_and_features(str(records_path), str(features_path))


def test_load_records_and_features_rejects_non_safedrug_visits(tmp_path):
    records_path = tmp_path / "records_final4.pkl"
    features_path = tmp_path / "organ_function_features_final4.pkl"
    with open(records_path, "wb") as f:
        pickle.dump([[[[0], [0], [0], 12.0]]], f)
    with open(features_path, "wb") as f:
        pickle.dump([[{}]], f)

    with pytest.raises(ValueError, match="three-item SafeDrug visits"):
        load_records_and_features(str(records_path), str(features_path))
