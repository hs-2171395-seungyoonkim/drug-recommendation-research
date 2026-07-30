import pandas as pd
import pytest

from organ_function.subject_lookup import build_hadm_to_subject, subject_id_for_patient


def test_builds_one_to_one_hadm_to_subject_mapping():
    diag_df = pd.DataFrame({
        "subject_id": [10, 10, 20],
        "hadm_id": [100, 101, 200],
    })
    mapping = build_hadm_to_subject(diag_df)
    assert mapping.loc[100] == 10
    assert mapping.loc[101] == 10
    assert mapping.loc[200] == 20


def test_subject_id_for_patient_uses_first_hadm_id():
    mapping = pd.Series({100: 10, 101: 10, 200: 20})
    assert subject_id_for_patient(mapping, [100, 101]) == 10
    assert subject_id_for_patient(mapping, [200]) == 20


def test_subject_id_for_patient_raises_on_inconsistent_hadm_ids():
    mapping = pd.Series({100: 10, 200: 20})
    with pytest.raises(ValueError, match="inconsistent subject_id"):
        subject_id_for_patient(mapping, [100, 200])
