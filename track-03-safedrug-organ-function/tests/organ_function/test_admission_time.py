import pandas as pd

from organ_function.admission_time import build_admission_times


def test_takes_earliest_starttime_per_hadm_id():
    df = pd.DataFrame({
        "hadm_id": [1, 1, 1, 2],
        "starttime": [
            "2239-12-25 15:14:00",
            "2239-12-24 13:14:00",
            "2239-12-26 14:14:00",
            "2240-04-22 05:14:00",
        ],
    })
    result = build_admission_times(df)
    assert result.loc[1] == pd.Timestamp("2239-12-24 13:14:00")
    assert result.loc[2] == pd.Timestamp("2240-04-22 05:14:00")


def test_ignores_null_starttime_rows_when_others_present():
    df = pd.DataFrame({
        "hadm_id": [1, 1],
        "starttime": [None, "2239-12-24 13:14:00"],
    })
    result = build_admission_times(df)
    assert result.loc[1] == pd.Timestamp("2239-12-24 13:14:00")


def test_all_null_starttime_yields_nat():
    df = pd.DataFrame({
        "hadm_id": [3, 3],
        "starttime": [None, None],
    })
    result = build_admission_times(df)
    assert pd.isna(result.loc[3])
