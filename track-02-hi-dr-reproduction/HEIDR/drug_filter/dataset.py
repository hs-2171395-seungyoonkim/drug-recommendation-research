import torch
from torch.utils.data import Dataset

from HEIDR.drug_filter.candidate_pool import expand_candidates
from HEIDR.drug_filter.history_features import compute_history_features


def build_scoring_inputs(visit_records: list, visit_histories: list) -> list:
    """방문별로 (확장된 후보 id, hidr_logprob, extra_features)를 만든다.
    학습(Dataset)과 추론(평가·threshold 선택)이 똑같은 입력을 보도록
    한 곳에서만 구성한다.

    visit_records와 visit_histories는 반드시 같은 split을, 같은 순서로
    가리켜야 한다(예: candidates_test.pt의 레코드에는 test split의 histories만
    짝지어야 한다). zip은 길이가 다르거나 순서가 어긋나도 조용히 잘못된
    쌍을 만들어내므로 — 다섯 개 스크립트가 각자 캐시와 histories를 손으로
    짝짓고 있어 이 실수가 실제로 재현된 적이 있다 — 여기서 길이와 내용을
    모두 검증한다."""
    if len(visit_records) != len(visit_histories):
        raise ValueError(
            f"visit_records({len(visit_records)}개)와 visit_histories({len(visit_histories)}개)의 "
            "길이가 다릅니다. 서로 다른 split의 캐시와 history를 짝짓지 않았는지 "
            "확인하세요(예: candidates_test.pt를 eval histories와 zip)."
        )
    if visit_records and visit_histories:
        check_positions = sorted({0, len(visit_records) // 2, len(visit_records) - 1})
        for i in check_positions:
            rec_gt = visit_records[i].get("gt_ids")
            hist_gt = visit_histories[i].get("gt_ids")
            if rec_gt != hist_gt:
                raise ValueError(
                    f"visit_records[{i}]와 visit_histories[{i}]의 gt_ids가 다릅니다 "
                    f"(record={rec_gt}, history={hist_gt}). 길이는 같지만 서로 다른 "
                    "split이거나 순서가 어긋난 캐시/history가 짝지어졌을 가능성이 높습니다."
                )
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
