import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import dill
import numpy as np
import torch

from HEIDR.drug_filter.ddi_features import compute_ddi_conflict_features
from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.select_threshold import (
    apply_filter_to_visits,
    compute_achieved_ddi_rate,
    select_ddi_aware_threshold,
)
from util import sequence_metric, ddi_rate_score

MED_NUM = 131


def score_records(model, records, drug_memory, ddi_A, device):
    """방문별로 그 방문의 모든 후보 약물에 필터 점수(0~1)를 매긴다 (DDI 충돌 피처 포함)."""
    model.eval()
    per_visit_scores = []
    with torch.no_grad():
        for rec in records:
            candidate_ids = [d for d, _ in rec["candidates"]]
            if not candidate_ids:
                per_visit_scores.append(([], []))
                continue
            ddi_feats = compute_ddi_conflict_features(rec["candidates"], ddi_A)
            visit_emb = rec["visit_emb"].unsqueeze(0).repeat(len(candidate_ids), 1).to(device)
            drug_emb = drug_memory[candidate_ids].to(device)
            hidr_prob = torch.tensor([p for _, p in rec["candidates"]], dtype=torch.float).to(device)
            ddi_features = torch.tensor(ddi_feats, dtype=torch.float).to(device)
            logits = model(visit_emb, drug_emb, hidr_prob, ddi_features)
            scores = torch.sigmoid(logits).cpu().tolist()
            per_visit_scores.append((candidate_ids, scores))
    return per_visit_scores


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

    # eval split 정답(ground truth) 처방 자체의 DDI rate를 threshold 선택의 목표치로
    # 쓴다 — "DDI rate=0"이 아니라 "정답 수준으로 수렴"이 목표이기 때문이다
    # (docs/specs/2026-07-27-drug-recommendation-postfilter-design.md
    # Section 6.1 참고).
    gt_ddi_rate = compute_achieved_ddi_rate(
        [rec["gt_ids"] for rec in eval_cache["visit_records"]], ddi_A
    )
    print(f"eval split ground-truth DDI rate (threshold selection target): {gt_ddi_rate:.4f}")

    threshold_info = select_ddi_aware_threshold(
        np.array(all_labels), np.array(all_scores), eval_scores, ddi_A, gt_ddi_rate, beta=1.0,
    )
    print(f"selected threshold (eval split, DDI-aware F1-max): {threshold_info}")

    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")
    test_scores = score_records(model, test_cache["visit_records"], test_cache["drug_memory"], ddi_A, device)

    before_labels = [[d for d, _ in rec["candidates"]] for rec in test_cache["visit_records"]]
    after_labels = apply_filter_to_visits(test_scores, threshold_info["threshold"])

    before_metrics = compute_metrics(test_cache["visit_records"], before_labels)
    after_metrics = compute_metrics(test_cache["visit_records"], after_labels)

    print(f"{'metric':<12}{'before':>10}{'after':>10}")
    for key in ("precision", "recall", "jaccard", "f1", "ddi_rate", "avg_med"):
        print(f"{key:<12}{before_metrics[key]:>10.4f}{after_metrics[key]:>10.4f}")


if __name__ == "__main__":
    main()
