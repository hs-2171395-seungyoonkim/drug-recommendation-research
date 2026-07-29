"""
5-seed 본실험 (설계 문서 §4 "5-seed, 제약 만족률"). seed 0~4로 필터를 재학습하고,
매 seed마다 eval split에서 quality floor(필터 없이 확장된 풀 전체의 Jaccard)를
측정해 그 이상을 유지하는 최소 AVG_MED threshold를 고른 뒤(select_min_avgmed_threshold),
test split에서 필터 전/후 지표, 풀 커버리지, 삭제된 정답 약물 수, 세 매칭 AVG_MED
지점(13.0/16.85/20.07)의 Jaccard, 그리고 "eval에서 정한 품질 하한이 test에서도
지켜지는가"(eval->test 제약 만족)를 보고한다.

재학습마다 candidates_train.pt로 필터를 새로 학습하므로(각 십수 초) 전체는 몇 분
내에 끝난다 -- beam search(candidates_*.pt 생성)는 재실행하지 않는다.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import dill
import numpy as np
import torch

from HEIDR.drug_filter.candidate_pool import expand_candidates, pool_coverage
from HEIDR.drug_filter.dataset import build_scoring_inputs
from HEIDR.drug_filter.evaluate_filter import (
    MATCHED_AVG_MED_POINTS,
    compute_metrics,
    deleted_gt_per_visit,
    jaccard_at_avg_med,
    score_records,
)
from HEIDR.drug_filter.history_features import build_patient_splits, iter_visit_histories
from HEIDR.drug_filter.select_threshold import (
    apply_filter_to_visits,
    compute_ddi_pair_stats,
    select_min_avgmed_threshold,
    visit_jaccard,
)
from HEIDR.drug_filter.train_filter import train_model

N_SEEDS = 5
KEYS = ("precision", "recall", "jaccard", "f1", "ddi_rate", "avg_med")


def _last_prev_set(hist: dict) -> set:
    prev_sets = hist["prev_med_sets"]
    return prev_sets[-1] if prev_sets else set()


def _expanded_pools(records: list, histories: list) -> list:
    """레코드마다 expand_candidates가 반환하는 (drug_id, hidr_logprob, is_beam)
    3-tuple 풀을 그대로 재구성한다 (evaluate_filter.py의 _expanded_pools와 동일)."""
    return [
        expand_candidates(rec["candidates"], _last_prev_set(hist))
        for rec, hist in zip(records, histories)
    ]


def _mean_std(values: list) -> tuple:
    arr = np.array(values, dtype=float)
    if len(arr) > 1:
        return float(arr.mean()), float(arr.std(ddof=1))
    return float(arr.mean()), 0.0


def _run_one_seed(seed: int, eval_cache, test_cache, eval_histories, test_histories, ddi_A) -> dict:
    model, device, n_train = train_model(seed, verbose=False)
    model.eval()

    # --- eval split: quality floor + threshold 선택 ---
    eval_records = eval_cache["visit_records"]
    eval_scores = score_records(model, eval_records, eval_cache["drug_memory"], eval_histories, device)

    eval_scoring_inputs = build_scoring_inputs(eval_records, eval_histories)
    eval_no_filter_pools = [pool_ids for pool_ids, _, _ in eval_scoring_inputs]
    quality_floor = visit_jaccard(eval_records, eval_no_filter_pools)

    threshold_info = select_min_avgmed_threshold(eval_records, eval_scores, quality_floor)

    # --- test split: before/after 지표, 커버리지, 삭제된 정답, 매칭 AVG_MED 지점 ---
    test_records = test_cache["visit_records"]
    test_scores = score_records(model, test_records, test_cache["drug_memory"], test_histories, device)

    expanded_pools = _expanded_pools(test_records, test_histories)
    before_labels = [[d for d, _, _ in pool] for pool in expanded_pools]  # 확장된 풀 전체
    after_labels = apply_filter_to_visits(test_scores, threshold_info["threshold"])

    before_metrics = compute_metrics(test_records, before_labels)
    after_metrics = compute_metrics(test_records, after_labels)

    before_ddi = compute_ddi_pair_stats(before_labels, ddi_A)
    after_ddi = compute_ddi_pair_stats(after_labels, ddi_A)

    gt_id_lists = [rec["gt_ids"] for rec in test_records]
    expanded_coverage = pool_coverage(expanded_pools, gt_id_lists)
    beam_only_pools = [[(d, p, 1.0) for d, p in rec["candidates"]] for rec in test_records]
    beam_only_coverage = pool_coverage(beam_only_pools, gt_id_lists)

    deleted = deleted_gt_per_visit(test_records, expanded_pools, after_labels)

    matched_points = {}
    for target in MATCHED_AVG_MED_POINTS:
        row = jaccard_at_avg_med(test_records, test_scores, target)
        # jaccard_at_avg_med는 jaccard/avg_med만 낸다. 같은 threshold에 apply_filter_to_visits
        # + compute_metrics를 적용해 precision/recall도 같은 운영점 기준으로 함께 낸다
        # (검증 기준표에 recall@20.07도 포함돼 있어 필요).
        matched_labels = apply_filter_to_visits(test_scores, row["threshold"])
        matched_metrics = compute_metrics(test_records, matched_labels)
        matched_points[target] = {**row, "precision": matched_metrics["precision"], "recall": matched_metrics["recall"]}

    # eval->test 제약 만족: threshold를 eval의 quality_floor(+safety margin)로 골랐을 때,
    # 실제 test split에 적용한 필터 후 Jaccard가 그 eval quality_floor 이상을 유지하는가.
    # 이게 지켜지지 않으면 eval에서 고른 threshold가 test로 일반화되지 않는다는 뜻이다
    # (select_threshold.py의 SAFETY_MARGIN 주석 참고 -- winner's curse 방지용 마진).
    test_constraint_satisfied = bool(after_metrics["jaccard"] >= quality_floor)

    return {
        "seed": seed,
        "n_train": n_train,
        "quality_floor": quality_floor,
        "threshold_info": threshold_info,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_ddi": before_ddi,
        "after_ddi": after_ddi,
        "expanded_coverage": expanded_coverage,
        "beam_only_coverage": beam_only_coverage,
        "deleted_gt_per_visit": deleted,
        "matched_points": matched_points,
        "test_constraint_satisfied": test_constraint_satisfied,
    }


def main():
    ddi_A = dill.load(open("data/ddi_A_final.pkl", "rb"))

    splits = build_patient_splits()
    eval_histories = iter_visit_histories(splits["eval"])
    test_histories = iter_visit_histories(splits["test"])

    eval_cache = torch.load("HEIDR/drug_filter/candidates_eval.pt")
    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")

    results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed} ===")
        r = _run_one_seed(seed, eval_cache, test_cache, eval_histories, test_histories, ddi_A)
        results.append(r)

        print(f"  train samples (visit x candidate pairs): {r['n_train']}")
        print(f"  quality floor (eval split, no filter / full expanded pool jaccard): {r['quality_floor']:.4f}")
        print(f"  selected threshold (eval split, min AVG_MED at quality floor): {r['threshold_info']}")
        print(f"  eval feasible: {r['threshold_info']['feasible']}")

        print(f"  {'metric':<12}{'before':>10}{'after':>10}")
        for key in KEYS:
            print(f"  {key:<12}{r['before_metrics'][key]:>10.4f}{r['after_metrics'][key]:>10.4f}")

        print(f"  {'pool coverage':<20}{'expanded':>12}{'beam-only':>12}")
        print(f"  {'':<20}{r['expanded_coverage']:>12.4f}{r['beam_only_coverage']:>12.4f}")

        print(f"  deleted ground-truth drugs per visit (after filter): {r['deleted_gt_per_visit']:.4f}")
        print(
            f"  avg DDI pairs/visit: before={r['before_ddi']['avg_dd_per_visit']:.4f}"
            f"  after={r['after_ddi']['avg_dd_per_visit']:.4f}"
        )
        print(f"  ddi_rate: before={r['before_ddi']['rate']:.4f}  after={r['after_ddi']['rate']:.4f}")

        print(f"  {'target_avg_med':>15}{'achieved_avg_med':>18}{'jaccard':>10}{'precision':>11}{'recall':>10}")
        for target in MATCHED_AVG_MED_POINTS:
            row = r["matched_points"][target]
            print(f"  {target:>15.2f}{row['avg_med']:>18.4f}{row['jaccard']:>10.4f}{row['precision']:>11.4f}{row['recall']:>10.4f}")

        print(f"  eval->test quality constraint satisfied (test jaccard >= eval quality floor): {r['test_constraint_satisfied']}")
        print()

    # --- 5-seed 요약 ---
    print("=== 5-seed 요약 (mean +/- std, n=5) ===")
    print(f"{'metric':<14}{'before':>22}{'after':>22}")
    for key in KEYS:
        b_mean, b_std = _mean_std([r["before_metrics"][key] for r in results])
        a_mean, a_std = _mean_std([r["after_metrics"][key] for r in results])
        print(f"{key:<14}{b_mean:>14.4f} +/- {b_std:<6.4f}{a_mean:>14.4f} +/- {a_std:<6.4f}")

    bdd_mean, bdd_std = _mean_std([r["before_ddi"]["avg_dd_per_visit"] for r in results])
    add_mean, add_std = _mean_std([r["after_ddi"]["avg_dd_per_visit"] for r in results])
    print(f"{'avg_dd/visit':<14}{bdd_mean:>14.4f} +/- {bdd_std:<6.4f}{add_mean:>14.4f} +/- {add_std:<6.4f}")

    print()
    ec_mean, ec_std = _mean_std([r["expanded_coverage"] for r in results])
    bc_mean, bc_std = _mean_std([r["beam_only_coverage"] for r in results])
    print(f"pool coverage: expanded={ec_mean:.4f} +/- {ec_std:.4f}   beam-only={bc_mean:.4f} +/- {bc_std:.4f}")
    print("(풀 커버리지는 모델/seed와 무관하게 결정론적이어야 한다 -- 5-seed 전부 동일한지 위 seed별 값으로 확인)")

    del_mean, del_std = _mean_std([r["deleted_gt_per_visit"] for r in results])
    print(f"deleted ground-truth drugs/visit: {del_mean:.4f} +/- {del_std:.4f}")

    print()
    print(f"{'target_avg_med':>15}{'avg_med_mean':>15}{'avg_med_std':>13}{'jaccard_mean':>14}{'jaccard_std':>13}{'recall_mean':>13}{'recall_std':>12}")
    for target in MATCHED_AVG_MED_POINTS:
        am_mean, am_std = _mean_std([r["matched_points"][target]["avg_med"] for r in results])
        j_mean, j_std = _mean_std([r["matched_points"][target]["jaccard"] for r in results])
        rc_mean, rc_std = _mean_std([r["matched_points"][target]["recall"] for r in results])
        print(f"{target:>15.2f}{am_mean:>15.4f}{am_std:>13.4f}{j_mean:>14.4f}{j_std:>13.4f}{rc_mean:>13.4f}{rc_std:>12.4f}")

    n_satisfied = sum(1 for r in results if r["test_constraint_satisfied"])
    print()
    print(f"eval->test quality constraint satisfied: {n_satisfied}/{N_SEEDS} seeds")


if __name__ == "__main__":
    main()
