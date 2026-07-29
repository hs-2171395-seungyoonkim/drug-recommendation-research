"""
DDI rate(dd_cnt/all_cnt, 비율)만 보면 필터링 후 오히려 오른 것처럼 보이지만, 이는
필터링이 전체 약물쌍(all_cnt)을 DDI쌍보다 더 빨리 줄이기 때문에 생기는 분모 효과다
(작업 보고서 §2). 여기서는 "방문당 평균 DDI쌍 개수"(절대량)를 §7-3과 동일하게
5개 seed(0~4)로 재학습·재평가해 측정한다 -- 절대량 기준으로도 필터링이 실제로
DDI 노출을 줄이는지, 그리고 그 결론이 재학습 노이즈에 흔들리지 않는지를 검증한다.

재학습마다 candidates_train.pt로 필터를 새로 학습하므로(각 17초 내외) 몇 분이면
끝난다 -- beam search(candidates_*.pt 생성)는 재실행하지 않는다.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import dill
import numpy as np
import torch

from HEIDR.drug_filter.candidate_pool import expand_candidates
from HEIDR.drug_filter.evaluate_filter import score_records
from HEIDR.drug_filter.history_features import build_patient_splits, iter_visit_histories
from HEIDR.drug_filter.select_threshold import (
    apply_filter_to_visits,
    compute_achieved_ddi_rate,
    compute_ddi_pair_stats,
    select_ddi_aware_threshold,
    select_threshold_f_beta,
)
from HEIDR.drug_filter.train_filter import train_model

N_SEEDS = 5
VARIANTS = ("before", "ddi_aware", "f1_only")
STAT_KEYS = ("rate", "avg_dd_per_visit", "avg_med", "dd_cnt_total")


def _last_prev_set(hist: dict) -> set:
    prev_sets = hist["prev_med_sets"]
    return prev_sets[-1] if prev_sets else set()


def _expanded_before_labels(records: list, histories: list) -> list:
    """'before' 베이스라인은 score_records가 실제로 채점하는 확장된 풀(빔 +
    직전 방문 처방, evaluate_filter.py의 _last_prev_set/_expanded_pools와 동일
    패턴)에서 뽑아야 필터링된 확장 풀인 'after'와 공정하게 비교된다.
    rec['candidates']만 쓰면 빔-only 풀(coverage 낮음)을 확장 풀 필터링
    결과와 비교하는 셈이 되어 두 값이 서로 다른 후보집합을 가리키게 된다."""
    return [
        [d for d, _, _ in expand_candidates(rec["candidates"], _last_prev_set(hist))]
        for rec, hist in zip(records, histories)
    ]


def _run_one_seed(seed: int, ddi_A, eval_cache, test_cache, eval_histories, test_histories) -> dict:
    model, device, _ = train_model(seed, verbose=False)
    model.eval()

    eval_scores = score_records(model, eval_cache["visit_records"], eval_cache["drug_memory"], eval_histories, device)
    all_labels, all_scores = [], []
    for rec, (candidate_ids, scores) in zip(eval_cache["visit_records"], eval_scores):
        gt_set = set(rec["gt_ids"])
        for cid, s in zip(candidate_ids, scores):
            all_labels.append(1 if cid in gt_set else 0)
            all_scores.append(s)
    all_labels, all_scores = np.array(all_labels), np.array(all_scores)

    gt_ddi_rate = compute_achieved_ddi_rate([rec["gt_ids"] for rec in eval_cache["visit_records"]], ddi_A)
    ddi_aware_info = select_ddi_aware_threshold(
        all_labels, all_scores, eval_scores, ddi_A, gt_ddi_rate, beta=1.0, margin=0.01,
    )
    f1_only_info = select_threshold_f_beta(all_labels, all_scores, beta=1.0)

    test_scores = score_records(model, test_cache["visit_records"], test_cache["drug_memory"], test_histories, device)
    before_labels = _expanded_before_labels(test_cache["visit_records"], test_histories)
    ddi_aware_labels = apply_filter_to_visits(test_scores, ddi_aware_info["threshold"])
    f1_only_labels = apply_filter_to_visits(test_scores, f1_only_info["threshold"])

    return {
        "before": compute_ddi_pair_stats(before_labels, ddi_A),
        "ddi_aware": compute_ddi_pair_stats(ddi_aware_labels, ddi_A),
        "f1_only": compute_ddi_pair_stats(f1_only_labels, ddi_A),
    }


def main():
    ddi_A = dill.load(open("data/ddi_A_final.pkl", "rb"))
    eval_cache = torch.load("HEIDR/drug_filter/candidates_eval.pt")
    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")

    splits = build_patient_splits()
    eval_histories = iter_visit_histories(splits["eval"])
    test_histories = iter_visit_histories(splits["test"])

    per_seed = []
    for seed in range(N_SEEDS):
        result = _run_one_seed(seed, ddi_A, eval_cache, test_cache, eval_histories, test_histories)
        per_seed.append(result)
        print(f"seed={seed}  " + "  ".join(
            f"{v}.avg_dd_per_visit={result[v]['avg_dd_per_visit']:.3f}" for v in VARIANTS
        ))

    print()
    for variant in VARIANTS:
        print(f"=== {variant} ===")
        for key in STAT_KEYS:
            vals = np.array([r[variant][key] for r in per_seed], dtype=float)
            print(f"  {key:<18} mean={vals.mean():.4f}  std={vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}"
                  f"  min={vals.min():.4f}  max={vals.max():.4f}")

    print()
    print("방문당 평균 DDI쌍(절대량), 필터 전 대비 변화율 -- seed별:")
    pct_ddi_aware, pct_f1_only = [], []
    for seed, r in enumerate(per_seed):
        before_dd = r["before"]["avg_dd_per_visit"]
        aware_pct = (r["ddi_aware"]["avg_dd_per_visit"] - before_dd) / before_dd * 100
        f1_pct = (r["f1_only"]["avg_dd_per_visit"] - before_dd) / before_dd * 100
        pct_ddi_aware.append(aware_pct)
        pct_f1_only.append(f1_pct)
        print(f"  seed={seed}: DDI-aware {aware_pct:+.1f}%   F1-only {f1_pct:+.1f}%")

    all_negative = all(p < 0 for p in pct_ddi_aware)
    print(f"\nDDI-aware가 모든 {N_SEEDS}개 seed에서 필터 전 대비 절대 DDI쌍을 줄임: {all_negative}")


if __name__ == "__main__":
    main()
