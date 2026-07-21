import pandas as pd

from HEIDR.service_personalization.build_dataset import build_order_table


def test_build_order_table_joins_and_maps_atc3():
    pharmacy_df = pd.DataFrame({
        "subject_id": [1, 1],
        "hadm_id": [10, 10],
        "pharmacy_id": [100, 101],
        "starttime": ["2245-05-25 15:37:00", "2245-05-25 19:37:00"],
    })
    presc_df = pd.DataFrame({
        "subject_id": [1, 1],
        "hadm_id": [10, 10],
        "pharmacy_id": [100, 101],
        "ndc": ["00591083960", "00000000000"],
        "manufacturer": ["Actavis Pharma, Inc.", "Unknown Co."],
    })
    services_df = pd.DataFrame({
        "subject_id": [1],
        "hadm_id": [10],
        "transfertime": ["2245-05-24 07:37:00"],
        "curr_service": ["MED"],
    })
    ndc2rxcui = {"00591083960": "12345"}
    rxcui2atc3 = {"12345": "A10B"}

    result = build_order_table(pharmacy_df, presc_df, services_df, ndc2rxcui, rxcui2atc3)

    assert list(result.columns) == [
        "subject_id", "hadm_id", "pharmacy_id", "order_time",
        "curr_service", "atc3", "manufacturer",
    ]
    # 두 번째 처방 건(pharmacy_id 101)은 ndc가 매핑 안 되므로 제외되어야 함
    assert list(result["pharmacy_id"]) == [100]
    assert result.iloc[0]["atc3"] == "A10B"
    assert result.iloc[0]["curr_service"] == "MED"
    assert result.iloc[0]["manufacturer"] == "Actavis Pharma, Inc."


def test_build_order_table_drops_orders_without_prescription_match():
    pharmacy_df = pd.DataFrame({
        "subject_id": [1],
        "hadm_id": [10],
        "pharmacy_id": [999],
        "starttime": ["2245-05-25 15:37:00"],
    })
    presc_df = pd.DataFrame({
        "subject_id": [], "hadm_id": [], "pharmacy_id": [], "ndc": [], "manufacturer": [],
    })
    services_df = pd.DataFrame({
        "subject_id": [], "hadm_id": [], "transfertime": [], "curr_service": [],
    })

    result = build_order_table(pharmacy_df, presc_df, services_df, {}, {})

    assert len(result) == 0
