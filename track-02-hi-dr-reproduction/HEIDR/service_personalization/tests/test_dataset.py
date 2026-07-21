import torch

from HEIDR.service_personalization.dataset import OrderPersonalizationDataset


def test_dataset_len_and_getitem():
    records = [
        {"hadm_id": 10, "service_id": 0, "atc3_id": 3, "manufacturer_id": 7},
        {"hadm_id": 11, "service_id": 1, "atc3_id": 4, "manufacturer_id": 8},
    ]
    visit_embeddings = {
        10: torch.ones(64),
        11: torch.zeros(64),
    }

    ds = OrderPersonalizationDataset(records, visit_embeddings)

    assert len(ds) == 2
    item = ds[0]
    assert torch.equal(item["visit_emb"], torch.ones(64))
    assert item["service_id"].item() == 0
    assert item["atc3_id"].item() == 3
    assert item["manufacturer_id"].item() == 7
    assert item["service_id"].dtype == torch.long


def test_dataset_filters_records_missing_embedding():
    records = [
        {"hadm_id": 10, "service_id": 0, "atc3_id": 3, "manufacturer_id": 7},
        {"hadm_id": 999, "service_id": 1, "atc3_id": 4, "manufacturer_id": 8},
    ]
    visit_embeddings = {10: torch.ones(64)}

    ds = OrderPersonalizationDataset(records, visit_embeddings)

    assert len(ds) == 1
    assert ds[0]["service_id"].item() == 0
