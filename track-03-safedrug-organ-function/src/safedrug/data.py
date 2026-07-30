"""
Loads records_final2.pkl (visits) aligned with organ_function_features.pkl
(same list[patient] of list[visit] shape - verified 1:1 positional parity
in Plan A's final review), builds train/test/eval splits matching
SOTA/SafeDrug/src/SafeDrug.py exactly, and produces two parallel dataset
variants: baseline (3-tuple visits) and +organ_function (4-tuple visits
with a standardized organ vector appended).
"""
import pickle

from safedrug.vectorize import to_vector


def load_records_and_features(records_path: str, organ_features_path: str):
    with open(records_path, "rb") as f:
        records = pickle.load(f)
    with open(organ_features_path, "rb") as f:
        organ_features = pickle.load(f)
    if len(records) != len(organ_features):
        raise ValueError(
            f"records/organ_features length mismatch: {len(records)} vs {len(organ_features)}"
        )
    for i, (patient_visits, patient_features) in enumerate(zip(records, organ_features)):
        if len(patient_visits) != len(patient_features):
            raise ValueError(f"patient {i} visit-count mismatch between records and organ_features")
    return records, organ_features


def split_patients(records: list):
    """Matches SOTA/SafeDrug/src/SafeDrug.py's split exactly: 2/3 train,
    remaining half/half test/eval, by patient list order (no shuffling)."""
    split_point = int(len(records) * 2 / 3)
    train_idx = list(range(0, split_point))
    eval_len = int((len(records) - split_point) / 2)
    test_idx = list(range(split_point, split_point + eval_len))
    eval_idx = list(range(split_point + eval_len, len(records)))
    return train_idx, test_idx, eval_idx


def build_baseline_dataset(records: list, indices: list) -> list:
    """Returns list[patient] of list[visit] using only the original
    [diag_ids, proc_ids, med_ids] triple - the baseline (no organ function)
    experiment arm."""
    return [records[i] for i in indices]


def build_organ_function_dataset(
    records: list, organ_features: list, indices: list, impute_stats: dict
) -> list:
    """Returns list[patient] of list[[diag_ids, proc_ids, med_ids, organ_vec]]
    - the +OrganFunction experiment arm. organ_vec is vectorize.to_vector()'s
    output, using impute_stats fit on the TRAIN split only."""
    dataset = []
    for i in indices:
        patient_visits = []
        for visit, visit_features in zip(records[i], organ_features[i]):
            organ_vec = to_vector(visit_features, impute_stats)
            patient_visits.append(list(visit) + [organ_vec])
        dataset.append(patient_visits)
    return dataset
