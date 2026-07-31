import pandas as pd

from mimic_iv_rebuild.medication_mapping import (
    build_medication_table,
    ndc_to_atc3,
    normalize_ndc,
)


def test_normalize_ndc_strips_dashes_and_zero_pads_to_11_digits():
    assert normalize_ndc("59-1083-960") == "00591083960"
    assert normalize_ndc("591083960") == "00591083960"


def test_normalize_ndc_handles_missing_values():
    assert normalize_ndc("nan") == ""
    assert normalize_ndc("") == ""


def test_ndc_to_atc3_returns_none_when_ndc_not_in_crosswalk():
    assert ndc_to_atc3("00000000000", {}, {}) is None


def test_ndc_to_atc3_maps_through_rxcui():
    ndc2rxcui = {"00591083960": "12345"}
    rxcui2atc3 = {"12345": "A01A"}
    assert ndc_to_atc3("59-1083-960", ndc2rxcui, rxcui2atc3) == "A01A"


def test_build_medication_table_drops_unmapped_ndcs(tmp_path):
    path = tmp_path / "prescriptions.csv"
    path.write_text(
        "subject_id,hadm_id,ndc\n"
        "1,100,591083960\n"
        "1,100,99999999999\n"
    )
    ndc2rxcui = {"00591083960": "12345"}
    rxcui2atc3 = {"12345": "A01A"}
    result = build_medication_table(str(path), ndc2rxcui, rxcui2atc3)
    assert len(result) == 1
    assert result.iloc[0]["code"] == "A01A"
