import torch
from torch.utils.data import Dataset

from HEIDR.drug_filter.candidate_pool import expand_candidates
from HEIDR.drug_filter.history_features import compute_history_features


def build_scoring_inputs(visit_records: list, visit_histories: list) -> list:
    """방문별로 (확장된 후보 id, hidr_logprob, extra_features)를 만든다.
    학습(Dataset)과 추론(평가·threshold 선택)이 똑같은 입력을 보도록
    한 곳에서만 구성한다."""
    out = []
    for rec, hist in zip(visit_records, visit_histories):
        prev_sets = hist["prev_med_sets"]
        last = prev_sets[-1] if prev_sets else set()
        pool = expand_candidates(rec["candidates"], last)

        pool_ids = [d for d, _, _ in pool]
        logprobs = [p for _, p, _ in pool]
        hist_feats = compute_history_features(prev_sets, pool_ids)
        extras = [hf + (is_beam,) for hf, (_, _, is_beam) in zip(hist_feats, pool)]
        out.append((pool_ids, logprobs, extras))
    return out


class DrugFilterDataset(Dataset):
    """확장된 후보 풀을 (방문 x 후보) 샘플로 펼치는 Dataset."""

    def __init__(self, visit_records: list, drug_memory: torch.Tensor, visit_histories: list):
        self.drug_memory = drug_memory
        self.samples = []

        scoring_inputs = build_scoring_inputs(visit_records, visit_histories)
        for rec, (pool_ids, logprobs, extras) in zip(visit_records, scoring_inputs):
            gt_set = set(rec["gt_ids"])
            for drug_id, logprob, extra in zip(pool_ids, logprobs, extras):
                label = 1.0 if drug_id in gt_set else 0.0
                self.samples.append((rec["visit_emb"], drug_id, logprob, extra, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        visit_emb, drug_id, logprob, extra, label = self.samples[idx]
        return {
            "visit_emb": visit_emb,
            "drug_emb": self.drug_memory[drug_id],
            "hidr_logprob": torch.tensor(logprob, dtype=torch.float32),
            "extra_features": torch.tensor(extra, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }
