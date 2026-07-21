import sys

import pandas as pd
import torch

sys.path.insert(0, ".")
from HEIDR.service_personalization.ndc_atc_mapping import load_ndc2rxcui, load_rxcui2atc3
from HEIDR.service_personalization.build_dataset import build_order_table
from HEIDR.service_personalization.vocab import build_vocab, load_atc3_vocab, filter_to_known_atc3


def to_records(order_table: pd.DataFrame, atc3_vocab: dict, service_vocab: dict,
               manufacturer_vocab: dict) -> list:
    records = []
    for _, row in order_table.iterrows():
        records.append({
            "hadm_id": int(row["hadm_id"]),
            "service_id": service_vocab[row["curr_service"]],
            "atc3_id": atc3_vocab[row["atc3"]],
            "manufacturer_id": manufacturer_vocab[row["manufacturer"]],
        })
    return records


def main():
    ndc2rxcui = load_ndc2rxcui("data/ndc2rxnorm_mapping.txt")
    rxcui2atc3 = load_rxcui2atc3("data/ndc2atc_level4.csv")

    pharmacy_df = pd.read_csv(
        "pharmacy.csv", usecols=["subject_id", "hadm_id", "pharmacy_id", "starttime"]
    )
    # A small number of rows (~17.5k / 14.7M) have null/unparseable starttime,
    # which becomes NaT after build_order_table's pd.to_datetime and breaks
    # service_matching.match_service's pd.merge_asof (null merge keys are not
    # allowed on the left side). Drop them here rather than in Task 5's code.
    valid_starttime = pd.to_datetime(pharmacy_df["starttime"], errors="coerce").notna()
    pharmacy_df = pharmacy_df[valid_starttime].reset_index(drop=True)
    presc_df = pd.read_csv(
        "prescriptions_with_manufacturer.csv",
        usecols=["subject_id", "hadm_id", "pharmacy_id", "ndc", "manufacturer"],
        dtype={"ndc": str},
    )
    services_df = pd.read_csv(
        "services.csv", usecols=["subject_id", "hadm_id", "transfertime", "curr_service"]
    )

    order_table = build_order_table(pharmacy_df, presc_df, services_df, ndc2rxcui, rxcui2atc3)
    order_table = order_table.dropna(subset=["manufacturer"])

    atc3_vocab = load_atc3_vocab("data/mimic-iv/voc_final2.pkl")
    order_table = filter_to_known_atc3(order_table, atc3_vocab)

    service_vocab = build_vocab(list(order_table["curr_service"]) + ["UNKNOWN"])
    manufacturer_vocab = build_vocab(order_table["manufacturer"])

    records = to_records(order_table, atc3_vocab, service_vocab, manufacturer_vocab)

    print(f"orders: {len(records)}")
    print(f"services: {len(service_vocab)}, manufacturers: {len(manufacturer_vocab)}, atc3: {len(atc3_vocab)}")

    torch.save({
        "records": records,
        "atc3_vocab": atc3_vocab,
        "service_vocab": service_vocab,
        "manufacturer_vocab": manufacturer_vocab,
    }, "HEIDR/service_personalization/assembled_dataset.pt")


if __name__ == "__main__":
    main()
