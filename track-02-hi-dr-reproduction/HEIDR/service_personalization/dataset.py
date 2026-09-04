import torch
from torch.utils.data import Dataset


class OrderPersonalizationDataset(Dataset):
    """처방 건 레코드 + hadm_id별 frozen 방문 임베딩을 묶는 Dataset."""

    def __init__(self, records: list, visit_embeddings: dict):
        self.visit_embeddings = visit_embeddings
        self.records = [r for r in records if r["hadm_id"] in visit_embeddings]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "visit_emb": self.visit_embeddings[r["hadm_id"]],
            "service_id": torch.tensor(r["service_id"], dtype=torch.long),
            "atc3_id": torch.tensor(r["atc3_id"], dtype=torch.long),
            "manufacturer_id": torch.tensor(r["manufacturer_id"], dtype=torch.long),
        }
