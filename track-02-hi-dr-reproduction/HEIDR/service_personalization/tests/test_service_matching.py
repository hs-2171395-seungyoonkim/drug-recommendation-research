import pandas as pd

from HEIDR.service_personalization.service_matching import match_service


def test_match_service_picks_latest_transfertime_before_order():
    orders = pd.DataFrame({
        "hadm_id": [1, 1, 2],
        "order_time": pd.to_datetime([
            "2256-08-09 21:53:00",
            "2256-08-11 21:53:00",
            "2256-08-09 21:53:00",
        ]),
    })
    services = pd.DataFrame({
        "hadm_id": [1, 1, 2],
        "transfertime": pd.to_datetime([
            "2256-08-08 11:53:00",
            "2256-08-10 11:53:00",
            "2256-08-09 20:53:00",
        ]),
        "curr_service": ["MED", "SURG", "OBS"],
    })

    result = match_service(orders, services)

    assert list(result["curr_service"]) == ["MED", "SURG", "OBS"]


def test_match_service_falls_back_to_unknown_when_order_before_any_transfer():
    orders = pd.DataFrame({
        "hadm_id": [3],
        "order_time": pd.to_datetime(["2256-08-03 11:53:00"]),
    })
    services = pd.DataFrame({
        "hadm_id": [3],
        "transfertime": pd.to_datetime(["2256-08-07 11:53:00"]),
        "curr_service": ["MED"],
    })

    result = match_service(orders, services)

    assert result["curr_service"].iloc[0] == "UNKNOWN"


def test_match_service_falls_back_to_unknown_when_hadm_id_missing_from_services():
    orders = pd.DataFrame({
        "hadm_id": [999],
        "order_time": pd.to_datetime(["2256-08-03 11:53:00"]),
    })
    services = pd.DataFrame({
        "hadm_id": [3],
        "transfertime": pd.to_datetime(["2256-08-07 11:53:00"]),
        "curr_service": ["MED"],
    })

    result = match_service(orders, services)

    assert result["curr_service"].iloc[0] == "UNKNOWN"
