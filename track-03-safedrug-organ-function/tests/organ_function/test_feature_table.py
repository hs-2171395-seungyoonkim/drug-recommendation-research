import math
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from organ_function.feature_table import compute_lab_features
# Imported rather than hand-copied: this is the Task 5 -> Task 6/7 seam, so a
# column-shape change in extract_lab_subset's output must break these tests.
from organ_function.lab_subset import LAB_SUBSET_COLUMNS, extract_lab_subset


def _labs(rows):
    return pd.DataFrame(rows, columns=LAB_SUBSET_COLUMNS)


def test_no_prior_labs_returns_missing():
    labs = _labs([])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-05 10:24:00"))
    assert result["missing"] is True
    assert math.isnan(result["value"])
    assert math.isnan(result["deviation"])
    assert math.isnan(result["delta"])


def test_single_prior_lab_has_value_but_no_delta():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["missing"] is False
    assert result["value"] == 0.8
    assert math.isnan(result["delta"])


def test_two_prior_labs_computes_delta_as_latest_minus_previous():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-06 18:24:00"), 1.3, 0.4, 1.1],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-07 10:24:00"))
    assert result["value"] == 1.3
    assert result["delta"] == 1.3 - 0.8


def test_value_within_range_has_zero_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.7, 0.4, 1.1]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["deviation"] == 0.0


def test_value_above_range_has_positive_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 2.1, 0.4, 1.1]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["deviation"] == (2.1 - 1.1) / (1.1 - 0.4)


def test_value_below_range_has_negative_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.1, 0.4, 1.1]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["deviation"] == (0.1 - 0.4) / (1.1 - 0.4)


def test_missing_ref_range_gives_nan_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, float("nan"), float("nan")]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert math.isnan(result["deviation"])
    assert result["value"] == 0.8


def test_excludes_labs_at_or_after_index_time():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-06 18:24:00"), 9.9, 0.4, 1.1],  # in/after the visit
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 18:24:00"))
    assert result["value"] == 0.8  # the 9.9 reading is not strictly before index_time


def test_ignores_rows_for_other_itemids():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1],
        [100, 51006, pd.Timestamp("2240-11-05 10:24:00"), 15.0, 7, 20],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["value"] == 0.8


def test_delta_missing_is_true_with_one_prior_reading_and_false_with_two():
    one = _labs([[100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1]])
    result_one = compute_lab_features(one, 50912, pd.Timestamp("2240-11-07 10:24:00"))
    assert result_one["delta_missing"] is True
    assert math.isnan(result_one["delta"])

    two = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-06 18:24:00"), 1.3, 0.4, 1.1],
    ])
    result_two = compute_lab_features(two, 50912, pd.Timestamp("2240-11-07 10:24:00"))
    assert result_two["delta_missing"] is False
    assert result_two["delta"] == 1.3 - 0.8


def test_delta_missing_true_when_second_reading_is_after_index_time():
    # 2 rows total, but only 1 is eligible -> delta_missing must track
    # eligibility, not raw row count.
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-09 18:24:00"), 1.3, 0.4, 1.1],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["delta_missing"] is True
    assert result["value"] == 0.8


def test_age_days_is_index_time_minus_latest_charttime():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-07 22:24:00"), 1.3, 0.4, 1.1],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-10 10:24:00"))
    # latest eligible reading is 11-07 22:24 -> 2.5 days before index_time
    assert result["age_days"] == 2.5


def test_age_days_is_nan_when_no_prior_reading():
    result = compute_lab_features(_labs([]), 50912, pd.Timestamp("2240-11-05 10:24:00"))
    assert math.isnan(result["age_days"])


from organ_function.feature_table import build_visit_features, build_feature_table
from organ_function.lab_config import LAB_NAMES


ORGAN_BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_organ_function_features.py"


def test_organ_feature_build_targets_time_ordered_final4_artifacts():
    spec = importlib.util.spec_from_file_location("build_organ_function_features", ORGAN_BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.RECORDS_PATH.name == "records_final4.pkl"
    assert module.HADM_IDS_PATH.name == "records_final4_hadm_ids.pkl"
    assert module.OUTPUT_PATH.name == "organ_function_features_final4.pkl"


def test_build_visit_features_has_all_expected_keys():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1]])
    result = build_visit_features(
        subject_labs=labs,
        index_time=pd.Timestamp("2240-11-06 10:24:00"),
        current_diag_ids=[1, 2],
        prior_diag_ids_seen=set(),
    )
    for name in LAB_NAMES:
        assert f"{name}_value" in result
        assert f"{name}_deviation" in result
        assert f"{name}_delta" in result
        assert f"{name}_delta_missing" in result
        assert f"{name}_age_days" in result
        assert f"{name}_missing" in result
    assert "new_diagnosis_flag" in result
    # 12 labs x 6 fields + new_diagnosis_flag
    assert len(result) == len(LAB_NAMES) * 6 + 1
    assert result["creatinine_value"] == 0.8
    assert result["bun_missing"] is True
    assert result["new_diagnosis_flag"] is False  # empty prior history


def test_build_feature_table_matches_records_shape_and_uses_previous_visit_only():
    # Two patients: patient 0 has 2 visits, patient 1 has 1 visit.
    records = [
        [
            [[1], [], [0]],       # patient 0, visit 0: diag=[1]
            [[1, 2], [], [0]],    # patient 0, visit 1: diag gains code 2 (new)
        ],
        [
            [[9], [], [0]],       # patient 1, visit 0
        ],
    ]
    hadm_ids = [[100, 101], [200]]

    lab_subset = _labs([
        # subject 5 (patient 0): one creatinine reading before visit 0's index time
        [5, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1],
        # and a later one, before visit 1's index time but after visit 0's
        [5, 50912, pd.Timestamp("2240-11-14 10:24:00"), 1.5, 0.4, 1.1],
    ])

    hadm_to_subject = pd.Series({100: 5, 101: 5, 200: 7})
    admission_times = pd.Series({
        100: pd.Timestamp("2240-11-09 10:24:00"),
        101: pd.Timestamp("2240-11-19 10:24:00"),
        200: pd.Timestamp("2240-12-06 10:24:00"),
    })

    table = build_feature_table(records, hadm_ids, lab_subset, admission_times, hadm_to_subject)

    assert len(table) == len(records)
    assert len(table[0]) == 2
    assert len(table[1]) == 1

    # visit 0 (index_time 11-09 10:24): only the 11-05 10:24 reading is prior -> value 0.8
    assert table[0][0]["creatinine_value"] == 0.8
    # visit 1 (index_time 11-19 10:24): both readings are prior -> latest is 1.5
    assert table[0][1]["creatinine_value"] == 1.5
    assert table[0][1]["new_diagnosis_flag"] is True  # code 2 is new vs. {1}

    # patient 1 has no labs for subject 7 -> missing
    assert table[1][0]["creatinine_missing"] is True


def test_build_visit_features_all_missing_when_index_time_is_nat():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1]])
    result = build_visit_features(
        subject_labs=labs,
        index_time=pd.NaT,
        current_diag_ids=[1],
        prior_diag_ids_seen=set(),
    )
    for name in LAB_NAMES:
        assert result[f"{name}_missing"] is True
        assert result[f"{name}_delta_missing"] is True
        assert math.isnan(result[f"{name}_value"])
        assert math.isnan(result[f"{name}_deviation"])
        assert math.isnan(result[f"{name}_delta"])
        assert math.isnan(result[f"{name}_age_days"])


def test_build_feature_table_rejects_non_chronological_input_order():
    # Regression for the whole-branch review finding: records_final2.pkl stores
    # visits in hadm_id order, which is NOT chronological. hadm 100 sits at
    # list position 0 but happened AFTER hadm 101 at position 1.
    records = [[
        [[1, 2], [], [0]],   # position 0, hadm 100, admitted 2241-04-05 10:24:00 (LATER)
        [[1], [], [0]],      # position 1, hadm 101, admitted 2240-11-05 10:24:00 (EARLIER)
    ]]
    hadm_ids = [[100, 101]]
    admission_times = pd.Series({
        100: pd.Timestamp("2241-04-05 10:24:00"),
        101: pd.Timestamp("2240-11-05 10:24:00"),
    })
    hadm_to_subject = pd.Series({100: 5, 101: 5})

    with pytest.raises(ValueError, match="nondecreasing admission-time order"):
        build_feature_table(records, hadm_ids, _labs([]), admission_times, hadm_to_subject)


def test_build_feature_table_rejects_missing_official_admission_time():
    # hadm 102 has NaT: it must not poison the earlier visits' history order,
    # and its own labs must all be missing (real NaT is still passed through).
    records = [[
        [[1], [], [0]],      # position 0, hadm 100, NaT
        [[1], [], [0]],      # position 1, hadm 101, 2240-11-05 10:24:00
        [[1, 2], [], [0]],   # position 2, hadm 102, 2240-12-06 10:24:00
    ]]
    hadm_ids = [[100, 101, 102]]
    admission_times = pd.Series({
        100: pd.NaT,
        101: pd.Timestamp("2240-11-05 10:24:00"),
        102: pd.Timestamp("2240-12-06 10:24:00"),
    })
    hadm_to_subject = pd.Series({100: 5, 101: 5, 102: 5})
    labs = _labs([[5, 50912, pd.Timestamp("2240-10-05 10:24:00"), 0.8, 0.4, 1.1]])

    with pytest.raises(ValueError, match="missing official admission time"):
        build_feature_table(records, hadm_ids, labs, admission_times, hadm_to_subject)


def test_build_feature_table_rejects_absent_official_admission_time():
    labs = _labs([[5, 50912, pd.Timestamp("2240-10-05 10:24:00"), 0.8, 0.4, 1.1]])
    hadm_to_subject = pd.Series({100: 5, 200: 5})
    records = [[[[1], [], [0]]]]

    with pytest.raises(ValueError, match="missing official admission time"):
        build_feature_table(records, [[200]], labs, pd.Series({100: pd.Timestamp("2240-11-05 10:24:00")}), hadm_to_subject)


def test_build_feature_table_raises_on_patient_count_mismatch():
    records = [[[[1], [], [0]]], [[[2], [], [0]]]]
    hadm_ids = [[100]]
    with pytest.raises(AssertionError, match="records/hadm_ids length mismatch"):
        build_feature_table(
            records, hadm_ids, _labs([]), pd.Series({100: pd.Timestamp("2240-11-05 10:24:00")}),
            pd.Series({100: 5}),
        )


def test_build_feature_table_raises_on_per_patient_visit_count_mismatch():
    records = [[[[1], [], [0]]], [[[2], [], [0]], [[3], [], [0]]]]
    hadm_ids = [[100], [200]]
    with pytest.raises(AssertionError, match="patient 1"):
        build_feature_table(
            records, hadm_ids, _labs([]),
            pd.Series({100: pd.Timestamp("2240-11-05 10:24:00"), 200: pd.Timestamp("2240-12-06 10:24:00")}),
            pd.Series({100: 5, 200: 7}),
        )


LABEVENTS_HEADER = (
    "labevent_id,subject_id,hadm_id,specimen_id,itemid,charttime,storetime,"
    "value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag,priority,comments\n"
)


def test_end_to_end_extract_lab_subset_into_build_feature_table(tmp_path):
    """Task 5 -> Task 6/7 seam: a real extract_lab_subset() frame must be
    directly consumable by build_feature_table() with no reshaping."""
    csv_path = tmp_path / "labevents_e2e.csv"
    csv_path.write_text(
        LABEVENTS_HEADER
        # subject 5, creatinine: two readings before visit 0's index time
        + "1,5,,1,50912,2240-11-05 10:24:00,2240-11-05 11:24:00,0.8,0.8,mg/dL,0.4,1.1,,STAT,\n"
        + "2,5,,2,50912,2240-11-07 10:24:00,2240-11-07 11:24:00,1.3,1.3,mg/dL,0.4,1.1,,STAT,\n"
        # subject 5, bun: one reading only -> delta_missing True
        + "3,5,,3,51006,2240-11-06 10:24:00,2240-11-06 11:24:00,15,15,mg/dL,7,20,,STAT,\n"
        # not a target itemid -> must be dropped by extract_lab_subset
        + "4,5,,4,99999,2240-11-06 10:24:00,2240-11-06 11:24:00,5.0,5.0,x,1.0,2.0,,STAT,\n"
        # null valuenum -> must be dropped
        + "5,5,,5,50912,2240-11-08 10:24:00,2240-11-08 11:24:00,,,mg/dL,0.4,1.1,,STAT,\n"
    )
    lab_subset = extract_lab_subset(str(csv_path), chunksize=2)
    assert list(lab_subset.columns) == LAB_SUBSET_COLUMNS

    records = [[[[1], [], [0]]]]
    hadm_ids = [[100]]
    admission_times = pd.Series({100: pd.Timestamp("2240-11-09 10:24:00")})
    hadm_to_subject = pd.Series({100: 5})

    table = build_feature_table(records, hadm_ids, lab_subset, admission_times, hadm_to_subject)
    visit = table[0][0]

    assert visit["creatinine_missing"] is False
    assert visit["creatinine_value"] == 1.3
    assert visit["creatinine_delta"] == pytest.approx(0.5)
    assert visit["creatinine_delta_missing"] is False
    assert visit["creatinine_age_days"] == 2.0            # 11-07 10:24 -> 11-09 10:24
    assert visit["creatinine_deviation"] == pytest.approx((1.3 - 1.1) / (1.1 - 0.4))

    assert visit["bun_missing"] is False
    assert visit["bun_value"] == 15.0
    assert visit["bun_delta_missing"] is True
    assert visit["bun_age_days"] == 3.0                   # 11-06 10:24 -> 11-09 10:24

    # a lab with no rows at all for this subject
    assert visit["alt_missing"] is True
    assert visit["new_diagnosis_flag"] is False
    assert len(visit) == len(LAB_NAMES) * 6 + 1
