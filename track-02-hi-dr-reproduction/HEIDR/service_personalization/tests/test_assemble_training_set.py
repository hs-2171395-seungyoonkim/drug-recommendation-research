import pandas as pd

from HEIDR.service_personalization.assemble_training_set import to_records


def test_to_records_maps_codes_to_vocab_ids():
    order_table = pd.DataFrame({
        "subject_id": [1, 1],
        "hadm_id": [10, 10],
        "pharmacy_id": [100, 101],
        "curr_service": ["MED", "SURG"],
        "atc3": ["A10B", "C01B"],
        "manufacturer": ["Pfizer", "Actavis"],
    })
    atc3_vocab = {"A10B": 0, "C01B": 1}
    service_vocab = {"MED": 0, "SURG": 1, "UNKNOWN": 2}
    manufacturer_vocab = {"Pfizer": 0, "Actavis": 1}

    records = to_records(order_table, atc3_vocab, service_vocab, manufacturer_vocab)

    assert records == [
        {"hadm_id": 10, "service_id": 0, "atc3_id": 0, "manufacturer_id": 0},
        {"hadm_id": 10, "service_id": 1, "atc3_id": 1, "manufacturer_id": 1},
    ]
