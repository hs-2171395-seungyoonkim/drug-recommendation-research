import torch

from HEIDR.drug_filter.dataset import DrugFilterDataset


def test_flattens_visits_into_candidate_pairs():
    visit_records = [
        {
            "visit_emb": torch.ones(4),
            "candidates": [(0, 0.9), (1, 0.2)],
            "gt_ids": [0],
        },
        {
            "visit_emb": torch.zeros(4),
            "candidates": [(2, 0.7)],
            "gt_ids": [2, 3],
        },
    ]
    drug_memory = torch.arange(4 * 4, dtype=torch.float).view(4, 4)

    ds = DrugFilterDataset(visit_records, drug_memory)

    assert len(ds) == 3

    item0 = ds[0]
    assert torch.equal(item0["visit_emb"], torch.ones(4))
    assert torch.equal(item0["drug_emb"], drug_memory[0])
    assert item0["hidr_prob"].item() == 0.9
    assert item0["label"].item() == 1.0

    item1 = ds[1]
    assert item1["label"].item() == 0.0  # drug_id 1은 gt_ids=[0]에 없음

    item2 = ds[2]
    assert torch.equal(item2["visit_emb"], torch.zeros(4))
    assert item2["label"].item() == 1.0  # drug_id 2는 gt_ids=[2, 3]에 있음


def test_empty_candidates_produce_no_samples():
    visit_records = [{"visit_emb": torch.ones(4), "candidates": [], "gt_ids": [0]}]
    drug_memory = torch.zeros(1, 4)

    ds = DrugFilterDataset(visit_records, drug_memory)

    assert len(ds) == 0
