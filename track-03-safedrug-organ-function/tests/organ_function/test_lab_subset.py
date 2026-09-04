from organ_function.lab_subset import extract_lab_subset, LAB_SUBSET_COLUMNS

HEADER = (
    "labevent_id,subject_id,hadm_id,specimen_id,itemid,charttime,storetime,"
    "value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag,priority,comments\n"
)


def test_filters_to_target_itemids_and_drops_null_valuenum(tmp_path):
    csv_path = tmp_path / "labevents_sample.csv"
    csv_path.write_text(
        HEADER
        + "1,100,,1,50912,2228-04-22 11:53:00,2228-04-22 12:53:00,1.0,1.0,mg/dL,0.4,1.1,,STAT,\n"
        + "2,100,5001,2,50912,2228-04-23 11:53:00,2228-04-23 12:53:00,,,mg/dL,0.4,1.1,,STAT,\n"
        + "3,100,,3,99999,2228-04-22 10:53:00,2228-04-22 11:23:00,5.0,5.0,mmol/L,1.0,2.0,,STAT,\n"
        + "4,200,,4,51006,2228-05-23 09:53:00,2228-05-23 10:53:00,20,20,mg/dL,7,20,,STAT,\n"
    )
    result = extract_lab_subset(str(csv_path), chunksize=2)
    assert list(result.columns) == LAB_SUBSET_COLUMNS
    assert len(result) == 2
    assert set(result["itemid"]) == {50912, 51006}
    row100 = result[result["subject_id"] == 100].iloc[0]
    assert row100["valuenum"] == 1.0
    assert row100["ref_range_lower"] == 0.4
    assert row100["ref_range_upper"] == 1.1


def test_returns_empty_frame_with_correct_columns_when_no_matches(tmp_path):
    csv_path = tmp_path / "labevents_none.csv"
    csv_path.write_text(
        HEADER
        + "1,100,,1,99999,2228-04-22 11:53:00,2228-04-22 12:53:00,1.0,1.0,mg/dL,0.4,1.1,,STAT,\n"
    )
    result = extract_lab_subset(str(csv_path), chunksize=2)
    assert len(result) == 0
    assert list(result.columns) == LAB_SUBSET_COLUMNS


def test_charttime_is_parsed_to_datetime(tmp_path):
    csv_path = tmp_path / "labevents_dt.csv"
    csv_path.write_text(
        HEADER
        + "1,100,,1,50912,2228-04-22 11:53:00,2228-04-22 12:53:00,1.0,1.0,mg/dL,0.4,1.1,,STAT,\n"
    )
    result = extract_lab_subset(str(csv_path), chunksize=2)
    assert result["charttime"].iloc[0] == __import__("pandas").Timestamp("2228-04-22 11:53:00")
