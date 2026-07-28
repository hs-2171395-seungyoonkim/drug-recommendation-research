import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import dill
import numpy as np
import torch
from scipy import stats

from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.select_threshold import (
    apply_filter_to_visits,
    compute_achieved_ddi_rate,
    select_ddi_aware_threshold,
    select_threshold_f_beta,
)
from HEIDR.drug_filter.evaluate_filter import score_records


def per_visit_scores(records: list, predicted_labels: list) -> dict:
    """정답 처방(gt_ids)과 예측 약물 리스트로부터 방문별 precision/recall/jaccard/f1
    배열을 계산한다 (HEIDR/util.py의 sequence_metric과 동일한 macro 방문별 평균 정의 —
    방문마다 하나의 스칼라 점수를 내고, 방문 간 paired 통계검정에 쓸 수 있도록
    평균이 아닌 배열을 반환한다)."""
    precisions, recalls, jaccards, f1s = [], [], [], []
    for rec, label in zip(records, predicted_labels):
        target = set(rec["gt_ids"])
        pred = set(label)
        inter = target & pred
        union = target | pred
        precision = len(inter) / len(pred) if pred else 0.0
        recall = len(inter) / len(target) if target else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        jaccard = len(inter) / len(union) if union else 0.0
        precisions.append(precision)
        recalls.append(recall)
        jaccards.append(jaccard)
        f1s.append(f1)
    return {
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "jaccard": np.array(jaccards),
        "f1": np.array(f1s),
    }


def bootstrap_ddi_rate_diff(
    predicted_a: list, predicted_b: list, ddi_A, n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """방문을 복원추출로 리샘플링해 achieved DDI rate(A) - achieved DDI rate(B)의
    부트스트랩 분포를 계산한다. DDI rate는 (util.py의 ddi_rate_score와 동일하게)
    방문별 평균이 아니라 전체 방문을 합친 pooled 비율이라, 방문별 paired test가
    아니라 방문 단위 리샘플링 부트스트랩으로 신뢰구간/유의성을 추정한다.
    predicted_a/predicted_b는 같은 순서(같은 방문)여야 한다."""
    rng = np.random.default_rng(seed)
    n = len(predicted_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled_a = [predicted_a[j] for j in idx]
        resampled_b = [predicted_b[j] for j in idx]
        diffs[i] = compute_achieved_ddi_rate(resampled_a, ddi_A) - compute_achieved_ddi_rate(resampled_b, ddi_A)
    return diffs


def _print_paired_test(name_a, name_b, scores_a, scores_b):
    diff = scores_a - scores_b
    wilcoxon = stats.wilcoxon(scores_a, scores_b) if np.any(diff != 0) else None
    ttest = stats.ttest_rel(scores_a, scores_b)
    print(
        f"  mean({name_a})={scores_a.mean():.4f} mean({name_b})={scores_b.mean():.4f} "
        f"diff={diff.mean():+.4f}  "
        f"wilcoxon p={'NA (all ties)' if wilcoxon is None else f'{wilcoxon.pvalue:.2e}'}  "
        f"paired-t p={ttest.pvalue:.2e}"
    )


def _print_bootstrap_ddi(name_a, name_b, predicted_a, predicted_b, ddi_A, n_boot=2000, seed=0):
    diffs = bootstrap_ddi_rate_diff(predicted_a, predicted_b, ddi_A, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_ge_zero = float(np.mean(diffs >= 0))
    print(
        f"  ddi_rate({name_a}) - ddi_rate({name_b}): mean_diff={diffs.mean():+.4f}  "
        f"95% CI=[{lo:+.4f}, {hi:+.4f}]  P(diff >= 0)={p_ge_zero:.4f} (n_boot={n_boot})"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("HEIDR/drug_filter/drug_filter.pt", map_location=device)
    model = DrugFilterHead(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    ddi_A = dill.load(open("data/ddi_A_final.pkl", "rb"))

    eval_cache = torch.load("HEIDR/drug_filter/candidates_eval.pt")
    eval_scores = score_records(model, eval_cache["visit_records"], eval_cache["drug_memory"], ddi_A, device)

    all_labels, all_scores = [], []
    for rec, (candidate_ids, scores) in zip(eval_cache["visit_records"], eval_scores):
        gt_set = set(rec["gt_ids"])
        for cid, s in zip(candidate_ids, scores):
            all_labels.append(1 if cid in gt_set else 0)
            all_scores.append(s)
    all_labels, all_scores = np.array(all_labels), np.array(all_scores)

    gt_ddi_rate = compute_achieved_ddi_rate(
        [rec["gt_ids"] for rec in eval_cache["visit_records"]], ddi_A
    )

    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")
    test_scores = score_records(model, test_cache["visit_records"], test_cache["drug_memory"], ddi_A, device)
    records = test_cache["visit_records"]

    before_labels = [[d for d, _ in rec["candidates"]] for rec in records]

    ddi_aware_info = select_ddi_aware_threshold(all_labels, all_scores, eval_scores, ddi_A, gt_ddi_rate, beta=1.0)
    ddi_aware_labels = apply_filter_to_visits(test_scores, ddi_aware_info["threshold"])

    # 같은(재학습된) 모델 위에, DDI를 전혀 모르는 plain F1-argmax threshold를 골라
    # "동일 모델, 다른 threshold 선택 알고리즘"으로 공정하게 비교한다.
    f1_only_info = select_threshold_f_beta(all_labels, all_scores, beta=1.0)
    f1_only_labels = apply_filter_to_visits(test_scores, f1_only_info["threshold"])

    print(f"thresholds: DDI-aware={ddi_aware_info['threshold']:.4f}  F1-only(same model)={f1_only_info['threshold']:.4f}")
    print()

    before_pv = per_visit_scores(records, before_labels)
    ddi_pv = per_visit_scores(records, ddi_aware_labels)
    f1_pv = per_visit_scores(records, f1_only_labels)

    print(f"n visits = {len(records)}")
    print()

    for metric in ("precision", "recall", "jaccard", "f1"):
        print(f"[{metric}] paired tests (Wilcoxon signed-rank + paired t-test), n={len(records)} visits:")
        print(" before vs DDI-aware:")
        _print_paired_test("before", "DDI-aware", before_pv[metric], ddi_pv[metric])
        print(" before vs F1-only(same model):")
        _print_paired_test("before", "F1-only", before_pv[metric], f1_pv[metric])
        print(" F1-only(same model) vs DDI-aware:")
        _print_paired_test("F1-only", "DDI-aware", f1_pv[metric], ddi_pv[metric])
        print()

    print("[ddi_rate] bootstrap (visit-level resampling, 2000 reps):")
    print(" before vs DDI-aware:")
    _print_bootstrap_ddi("before", "DDI-aware", before_labels, ddi_aware_labels, ddi_A)
    print(" DDI-aware vs F1-only(same model):")
    _print_bootstrap_ddi("DDI-aware", "F1-only", ddi_aware_labels, f1_only_labels, ddi_A)


if __name__ == "__main__":
    main()
