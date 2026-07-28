import torch
from torch.utils.data import Dataset

from HEIDR.drug_filter.ddi_features import compute_ddi_conflict_features


class DrugFilterDataset(Dataset):
    """방문별 (후보 약물, HI-DR 확률, DDI 충돌 피처) 리스트를 방문x후보 쌍 샘플로 펼치는 Dataset."""

    def __init__(self, visit_records: list, drug_memory: torch.Tensor, ddi_A):
        self.drug_memory = drug_memory
        self.samples = []
        for rec in visit_records:
            gt_set = set(rec["gt_ids"])
            ddi_feats = compute_ddi_conflict_features(rec["candidates"], ddi_A)
            for (drug_id, prob), (conflict_sum, conflict_max) in zip(rec["candidates"], ddi_feats):
                label = 1.0 if drug_id in gt_set else 0.0
                self.samples.append((rec["visit_emb"], drug_id, prob, conflict_sum, conflict_max, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        visit_emb, drug_id, prob, conflict_sum, conflict_max, label = self.samples[idx]
        return {
            "visit_emb": visit_emb,
            "drug_emb": self.drug_memory[drug_id],
            "hidr_prob": torch.tensor(prob, dtype=torch.float64),
            "ddi_conflict_sum": torch.tensor(conflict_sum, dtype=torch.float64),
            "ddi_conflict_max": torch.tensor(conflict_max, dtype=torch.float64),
            "label": torch.tensor(label, dtype=torch.float64),
        }
