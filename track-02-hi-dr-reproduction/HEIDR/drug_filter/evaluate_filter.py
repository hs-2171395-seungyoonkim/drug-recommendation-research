import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import numpy as np
import torch

from HEIDR.drug_filter.candidate_pool import expand_candidates, pool_coverage
from HEIDR.drug_filter.dataset import build_scoring_inputs
from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.history_features import build_patient_splits, iter_visit_histories
from HEIDR.drug_filter.select_threshold import (
    N_THRESHOLDS,
    apply_filter_to_visits,
    select_min_avgmed_threshold,
    visit_jaccard,
)
from util import sequence_metric, ddi_rate_score

MED_NUM = 131

# 세 매칭 AVG_MED 운영점(설계 문서 §4). 20.07은 test split 정답 처방 자체의
# 평균 크기와 같아, 방식 간 비교에서 가장 우선하는 지점이다.
MATCHED_AVG_MED_POINTS = (13.0, 16.85, 20.07)


def score_records(model, records, drug_memory, visit_histories, device):
    """방문별로 확장된 풀의 각 후보에 필터 점수(0~1)를 매긴다.
    records와 build_scoring_inputs(records, visit_histories)의 출력을 함께
    zip해, 각 방문의 임베딩이 반드시 같은 방문의 풀에 쓰이도록 한다."""
    model.eval()
    scoring_inputs = build_scoring_inputs(records, visit_histories)
    per_visit_scores = []
    with torch.no_grad():
        for rec, (pool_ids, logprobs, extras) in zip(records, scoring_inputs):
            if not pool_ids:
                per_visit_scores.append(([], []))
                continue
            visit_emb = rec["visit_emb"].unsqueeze(0).repeat(len(pool_ids), 1).to(device)
            logits = model(
                visit_emb.float(),
                drug_memory[pool_ids].to(device).float(),
                torch.tensor(logprobs, dtype=torch.float32).to(device),
                torch.tensor(extras, dtype=torch.float32).to(device),
            )
            per_visit_scores.append((pool_ids, torch.sigmoid(logits).cpu().tolist()))
    return per_visit_scores


def deleted_gt_per_visit(records: list, pools: list, predicted_labels: list) -> float:
    """풀에는 들어 있었는데 필터가 잘라낸 정답 약물 수의 방문 평균.
    기존 필터가 방문당 1.38개를 지우면서 한 번도 보고하지 않았던 값이다."""
    counts = []
    for rec, pool, pred in zip(records, pools, predicted_labels):
        gt = set(rec["gt_ids"])
        in_pool = gt & {d for d, _, _ in pool}
        counts.append(len(in_pool - set(pred)))
    return float(np.mean(counts)) if counts else 0.0


def jaccard_at_avg_med(records: list, scores_per_visit: list, target_avg_med: float) -> dict:
    """threshold를 훑어 AVG_MED가 target에 가장 가까운 지점의 지표를 낸다.
    풀 크기가 달라지면 운영점이 달라지므로, 방식 간 비교는 반드시 동일
    AVG_MED에서 해야 한다(설계 문서 §4)."""
    all_scores = np.concatenate([np.asarray(s) for _, s in scores_per_visit if len(s) > 0])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, N_THRESHOLDS)))

    best = None
    for t in thresholds:
        predicted = apply_filter_to_visits(scores_per_visit, float(t))
        avg_med = float(np.mean([len(p) for p in predicted]))
        row = {
            "threshold": float(t),
            "avg_med": avg_med,
            "jaccard": visit_jaccard(records, predicted),
            "gap": abs(avg_med - target_avg_med),
        }
        if best is None or row["gap"] < best["gap"]:
            best = row
    return best


def compute_metrics(records, predicted_labels):
    y_gt, y_pred, y_prob, y_label, smm_record = [], [], [], [], []
    for rec, label in zip(records, predicted_labels):
        gt = np.zeros(MED_NUM)
        gt[rec["gt_ids"]] = 1
        pred = np.zeros(MED_NUM)
        prob = np.zeros(MED_NUM)
        prob_by_id = dict(rec["candidates"])
        for d in label:
            pred[d] = 1
            prob[d] = prob_by_id.get(d, 0.0)
        y_gt.append(gt)
        y_pred.append(pred)
        y_prob.append(prob)
        y_label.append(label)
        smm_record.append([label])

    ja, prauc, avg_p, avg_r, avg_f1 = sequence_metric(
        np.array(y_gt), np.array(y_pred), np.array(y_prob), np.array(y_label, dtype=object)
    )
    ddi = ddi_rate_score(smm_record, path="data/ddi_A_final.pkl")
    avg_med = float(np.mean([len(l) for l in y_label]))
    return {
        "precision": avg_p, "recall": avg_r, "jaccard": ja,
        "f1": avg_f1, "ddi_rate": ddi, "avg_med": avg_med,
    }


def _last_prev_set(hist: dict) -> set:
    prev_sets = hist["prev_med_sets"]
    return prev_sets[-1] if prev_sets else set()


def _expanded_pools(records: list, histories: list) -> list:
    """레코드마다 expand_candidates가 반환하는 (drug_id, hidr_logprob, is_beam)
    3-tuple 풀을 그대로 재구성한다. pool_coverage는 이 형태를 요구한다."""
    return [
        expand_candidates(rec["candidates"], _last_prev_set(hist))
        for rec, hist in zip(records, histories)
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("HEIDR/drug_filter/drug_filter.pt", map_location=device)
    model = DrugFilterHead(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    splits = build_patient_splits()
    eval_histories = iter_visit_histories(splits["eval"])
    test_histories = iter_visit_histories(splits["test"])

    # --- eval split: quality floor + threshold 선택 ---
    eval_cache = torch.load("HEIDR/drug_filter/candidates_eval.pt")
    eval_records = eval_cache["visit_records"]
    eval_scores = score_records(model, eval_records, eval_cache["drug_memory"], eval_histories, device)

    # quality floor = "필터 없이 확장된 풀을 통째로 남겼을 때"의 Jaccard.
    eval_scoring_inputs = build_scoring_inputs(eval_records, eval_histories)
    eval_no_filter_pools = [pool_ids for pool_ids, _, _ in eval_scoring_inputs]
    quality_floor = visit_jaccard(eval_records, eval_no_filter_pools)
    print(f"quality floor (eval split, no filter / full expanded pool jaccard): {quality_floor:.4f}")

    threshold_info = select_min_avgmed_threshold(eval_records, eval_scores, quality_floor)
    print(f"selected threshold (eval split, min AVG_MED at quality floor): {threshold_info}")
    print(f"  feasible: {threshold_info['feasible']}")

    # --- test split: before/after 지표, 커버리지, 삭제된 정답, 매칭 AVG_MED 지점 ---
    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")
    test_records = test_cache["visit_records"]
    test_scores = score_records(model, test_records, test_cache["drug_memory"], test_histories, device)

    expanded_pools = _expanded_pools(test_records, test_histories)
    before_labels = [[d for d, _, _ in pool] for pool in expanded_pools]  # 확장된 풀 전체
    after_labels = apply_filter_to_visits(test_scores, threshold_info["threshold"])

    before_metrics = compute_metrics(test_records, before_labels)
    after_metrics = compute_metrics(test_records, after_labels)

    print(f"{'metric':<12}{'before':>10}{'after':>10}")
    for key in ("precision", "recall", "jaccard", "f1", "ddi_rate", "avg_med"):
        print(f"{key:<12}{before_metrics[key]:>10.4f}{after_metrics[key]:>10.4f}")

    gt_id_lists = [rec["gt_ids"] for rec in test_records]
    expanded_coverage = pool_coverage(expanded_pools, gt_id_lists)
    beam_only_pools = [[(d, p, 1.0) for d, p in rec["candidates"]] for rec in test_records]
    beam_only_coverage = pool_coverage(beam_only_pools, gt_id_lists)
    print(f"{'pool coverage':<20}{'expanded':>12}{'beam-only':>12}")
    print(f"{'':<20}{expanded_coverage:>12.4f}{beam_only_coverage:>12.4f}")

    deleted = deleted_gt_per_visit(test_records, expanded_pools, after_labels)
    print(f"deleted ground-truth drugs per visit (after filter): {deleted:.4f}")

    print(f"{'target_avg_med':>15}{'achieved_avg_med':>18}{'jaccard':>10}")
    for target in MATCHED_AVG_MED_POINTS:
        row = jaccard_at_avg_med(test_records, test_scores, target)
        print(f"{target:>15.2f}{row['avg_med']:>18.4f}{row['jaccard']:>10.4f}")


if __name__ == "__main__":
    main()
