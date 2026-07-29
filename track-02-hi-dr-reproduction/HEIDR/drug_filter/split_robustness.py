import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import dill
import numpy as np
import torch

from HEIDR.drug_filter.candidate_pool import expand_candidates
from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.history_features import build_patient_splits, iter_visit_histories
from HEIDR.drug_filter.select_threshold import (
    apply_filter_to_visits,
    compute_achieved_ddi_rate,
    select_ddi_aware_threshold,
    select_threshold_f_beta,
)
from HEIDR.drug_filter.evaluate_filter import compute_metrics, score_records

N_REPEATS = 10


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


def resplit_indices(n_total: int, n_eval: int, seed: int) -> tuple:
    """전체 n_total개 인덱스를 무작위로 섞어 앞 n_eval개는 eval-role, 나머지는
    test-role로 배정한다. 같은 seed면 같은 분할을 재현한다. 모델 재학습이나
    beam search 없이, 이미 캐싱된 방문 풀에서 "어느 방문이 threshold 선택용
    eval-role에 들어가느냐"의 우연성만 검증하기 위한 용도."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    return idx[:n_eval], idx[n_eval:]


def _run_one_split(records, scored, histories, ddi_A, eval_idx, test_idx):
    eval_records = [records[i] for i in eval_idx]
    eval_scored = [scored[i] for i in eval_idx]
    test_records = [records[i] for i in test_idx]
    test_scored = [scored[i] for i in test_idx]
    test_histories = [histories[i] for i in test_idx]

    all_labels, all_scores = [], []
    for rec, (candidate_ids, scores) in zip(eval_records, eval_scored):
        gt_set = set(rec["gt_ids"])
        for cid, s in zip(candidate_ids, scores):
            all_labels.append(1 if cid in gt_set else 0)
            all_scores.append(s)

    gt_ddi_rate = compute_achieved_ddi_rate([rec["gt_ids"] for rec in eval_records], ddi_A)

    threshold_info = select_ddi_aware_threshold(
        np.array(all_labels), np.array(all_scores), eval_scored, ddi_A, gt_ddi_rate, beta=1.0,
    )
    f1_only_info = select_threshold_f_beta(np.array(all_labels), np.array(all_scores), beta=1.0)

    before_labels = _expanded_before_labels(test_records, test_histories)
    after_labels = apply_filter_to_visits(test_scored, threshold_info["threshold"])
    f1_only_labels = apply_filter_to_visits(test_scored, f1_only_info["threshold"])

    before_metrics = compute_metrics(test_records, before_labels)
    after_metrics = compute_metrics(test_records, after_labels)
    f1_only_metrics = compute_metrics(test_records, f1_only_labels)
    return threshold_info["threshold"], gt_ddi_rate, before_metrics, after_metrics, f1_only_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("HEIDR/drug_filter/drug_filter.pt", map_location=device)
    model = DrugFilterHead(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    ddi_A = dill.load(open("data/ddi_A_final.pkl", "rb"))

    splits = build_patient_splits()
    eval_histories = iter_visit_histories(splits["eval"])
    test_histories = iter_visit_histories(splits["test"])

    eval_cache = torch.load("HEIDR/drug_filter/candidates_eval.pt")
    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")

    eval_scored = score_records(model, eval_cache["visit_records"], eval_cache["drug_memory"], eval_histories, device)
    test_scored = score_records(model, test_cache["visit_records"], test_cache["drug_memory"], test_histories, device)

    # score_records를 eval/test 각각의 올바른 histories와 짝지어 먼저 호출한 뒤
    # 결과(scored)만 이어붙인다. eval_scored는 eval_cache["visit_records"]와,
    # test_scored는 test_cache["visit_records"]와 이미 같은 순서이므로, 이후
    # records = eval + test 로 이어붙이는 순서와 scored = eval_scored + test_scored
    # 순서가 원소 단위로 정확히 대응한다 — 스코어링 자체에는 histories를 직접
    # 풀링할 필요가 없다.
    #
    # 다만 재분할 후 '확장된 풀' before baseline(_expanded_before_labels)을
    # 계산하려면 재분할된 test_idx로 histories도 함께 인덱싱해야 하므로,
    # 여기서는 eval_histories + test_histories를 같은 순서로 풀링해둔다. 이
    # 풀링은 records/scored 풀링과 정확히 같은 순서(eval 다음 test)를 따르므로
    # (eval_histories는 build_patient_splits()["eval"]로 eval_cache와, test_histories는
    # ["test"]로 test_cache와 이미 각각 정렬돼 있음) records[i]/scored[i]/histories[i]가
    # 항상 같은 방문을 가리킨다.
    records = eval_cache["visit_records"] + test_cache["visit_records"]
    scored = eval_scored + test_scored
    histories = eval_histories + test_histories
    n_total = len(records)
    n_eval = len(eval_cache["visit_records"])

    print(f"pooled visits: {n_total} (original eval={n_eval}, test={len(test_cache['visit_records'])})")
    print()

    keys = ("precision", "recall", "jaccard", "f1", "ddi_rate", "avg_med")
    after_by_key = {k: [] for k in keys}
    before_by_key = {k: [] for k in keys}
    f1_only_by_key = {k: [] for k in keys}

    header = f"{'repeat':>6}{'threshold':>10}{'gt_ddi':>8}" + "".join(f"{k:>10}" for k in keys)
    print(header)
    for r in range(N_REPEATS):
        eval_idx, test_idx = resplit_indices(n_total, n_eval, seed=r)
        threshold, gt_ddi_rate, before_metrics, after_metrics, f1_only_metrics = _run_one_split(
            records, scored, histories, ddi_A, eval_idx, test_idx
        )
        row = f"{r:>6}{threshold:>10.4f}{gt_ddi_rate:>8.4f}" + "".join(f"{after_metrics[k]:>10.4f}" for k in keys)
        print(row)
        for k in keys:
            after_by_key[k].append(after_metrics[k])
            before_by_key[k].append(before_metrics[k])
            f1_only_by_key[k].append(f1_only_metrics[k])

    print()
    print(f"{'metric':<10}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    for k in keys:
        arr = np.array(after_by_key[k])
        print(f"{k:<10}{arr.mean():>8.4f}{arr.std(ddof=1):>8.4f}{arr.min():>8.4f}{arr.max():>8.4f}")

    print()
    print("directional consistency across all repeats (DDI-aware 'after' vs that repeat's own 'before'):")
    for k in keys:
        after_arr = np.array(after_by_key[k])
        before_arr = np.array(before_by_key[k])
        if k in ("recall", "avg_med"):
            consistent, direction = np.all(after_arr < before_arr), "<"
        elif k == "ddi_rate":
            continue  # ddi_rate before-vs-after direction isn't the interesting claim; see F1-only comparison below
        else:
            consistent, direction = np.all(after_arr > before_arr), ">"
        print(f"  DDI-aware {direction} before for '{k}' in all {N_REPEATS} repeats: {bool(consistent)}")

    print()
    print("directional consistency across all repeats (DDI-aware vs F1-only, same model, same split each repeat):")
    for k in keys:
        after_arr = np.array(after_by_key[k])
        f1_arr = np.array(f1_only_by_key[k])
        if k in ("precision",):
            consistent, direction = np.all(after_arr < f1_arr), "<"
        elif k == "ddi_rate":
            consistent, direction = np.all(after_arr < f1_arr), "<"
        else:
            consistent, direction = np.all(after_arr > f1_arr), ">"
        print(f"  DDI-aware {direction} F1-only for '{k}' in all {N_REPEATS} repeats: {bool(consistent)}")


if __name__ == "__main__":
    main()
