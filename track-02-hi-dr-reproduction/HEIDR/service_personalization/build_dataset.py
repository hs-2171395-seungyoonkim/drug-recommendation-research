import pandas as pd

from HEIDR.service_personalization.ndc_atc_mapping import ndc_to_atc3
from HEIDR.service_personalization.service_matching import match_service


def build_order_table(pharmacy_df, presc_df, services_df, ndc2rxcui, rxcui2atc3) -> pd.DataFrame:
    """
    처방 건(pharmacy_id) 단위 조인 테이블 생성.

    pharmacy_df: [subject_id, hadm_id, pharmacy_id, starttime]
    presc_df:    [subject_id, hadm_id, pharmacy_id, ndc, manufacturer]
    services_df: [subject_id, hadm_id, transfertime, curr_service]

    ATC3 매핑 실패 행은 제외한다.
    """
    merged = pharmacy_df.merge(
        presc_df, on=["subject_id", "hadm_id", "pharmacy_id"], how="inner"
    )
    if len(merged) == 0:
        return pd.DataFrame(columns=[
            "subject_id", "hadm_id", "pharmacy_id", "order_time",
            "curr_service", "atc3", "manufacturer",
        ])

    merged["atc3"] = merged["ndc"].map(lambda n: ndc_to_atc3(n, ndc2rxcui, rxcui2atc3))
    merged = merged.dropna(subset=["atc3"]).reset_index(drop=True)
    merged = merged.rename(columns={"starttime": "order_time"})
    merged["order_time"] = pd.to_datetime(merged["order_time"])

    services_df = services_df.copy()
    services_df["transfertime"] = pd.to_datetime(services_df["transfertime"])

    matched = match_service(
        merged[["subject_id", "hadm_id", "pharmacy_id", "order_time"]],
        services_df,
    )
    result = matched.merge(
        merged[["subject_id", "hadm_id", "pharmacy_id", "atc3", "manufacturer"]],
        on=["subject_id", "hadm_id", "pharmacy_id"],
        how="left",
    )
    return result[[
        "subject_id", "hadm_id", "pharmacy_id", "order_time",
        "curr_service", "atc3", "manufacturer",
    ]]
