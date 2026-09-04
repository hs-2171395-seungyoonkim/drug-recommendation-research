import pandas as pd


def match_service(orders: pd.DataFrame, services: pd.DataFrame) -> pd.DataFrame:
    """
    orders의 각 행(hadm_id, order_time)에 대해, 같은 hadm_id 내에서
    transfertime <= order_time 인 것 중 가장 최근 curr_service를 붙인다.
    매칭이 없으면 'UNKNOWN'.
    """
    orders = orders.reset_index(drop=True)
    orders = orders.sort_values("order_time")

    services = services.dropna(subset=["transfertime"])
    services = services.sort_values("transfertime")

    merged = pd.merge_asof(
        orders,
        services[["hadm_id", "transfertime", "curr_service"]],
        left_on="order_time",
        right_on="transfertime",
        by="hadm_id",
        direction="backward",
    )
    merged["curr_service"] = merged["curr_service"].fillna("UNKNOWN")
    merged = merged.drop(columns=["transfertime"])
    # Restore original index order
    merged.index = orders.index
    return merged.sort_index()
