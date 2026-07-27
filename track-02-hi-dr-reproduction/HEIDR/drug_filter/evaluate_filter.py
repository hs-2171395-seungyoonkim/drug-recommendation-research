import sys

sys.path.insert(0, ".")
sys.path.insert(0, "HEIDR")

import numpy as np
import torch

from HEIDR.drug_filter.filter_model import DrugFilterHead
from HEIDR.drug_filter.select_threshold import select_threshold_f_beta, apply_filter
from util import sequence_metric, ddi_rate_score

MED_NUM = 131


def score_records(model, records, drug_memory, device):
    """방문별로 그 방문의 모든 후보 약물에 필터 점수(0~1)를 매긴다."""
    model.eval()
    per_visit_scores = []
    with torch.no_grad():
        for rec in records:
            candidate_ids = [d for d, _ in rec["candidates"]]
            if not candidate_ids:
                per_visit_scores.append(([], []))
                continue
            visit_emb = rec["visit_emb"].unsqueeze(0).repeat(len(candidate_ids), 1).to(device)
            drug_emb = drug_memory[candidate_ids].to(device)
            hidr_prob = torch.tensor([p for _, p in rec["candidates"]], dtype=torch.float).to(device)
            logits = model(visit_emb, drug_emb, hidr_prob)
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

    eval_cache = torch.load("HEIDR/drug_filter/candidates_eval.pt")
    eval_scores = score_records(model, eval_cache["visit_records"], eval_cache["drug_memory"], device)

    all_labels, all_scores = [], []
    for rec, (candidate_ids, scores) in zip(eval_cache["visit_records"], eval_scores):
        gt_set = set(rec["gt_ids"])
        for cid, s in zip(candidate_ids, scores):
            all_labels.append(1 if cid in gt_set else 0)
            all_scores.append(s)

    # beta=1.0 (F1): F0.5 over-weighted precision and picked a threshold that cost
    # ~33% relative recall for a marginal F0.5 gain (the F0.5 curve was flat across
    # thresholds 0.5-0.8). F1 lands on a threshold that improves precision/AVG_MED
    # substantially while keeping recall loss modest, matching the project's actual
    # goal (curb over-generation) better than the F0.5 pick did.
    threshold_info = select_threshold_f_beta(np.array(all_labels), np.array(all_scores), beta=1.0)
    print(f"selected threshold (eval split, F1-max): {threshold_info}")

    test_cache = torch.load("HEIDR/drug_filter/candidates_test.pt")
    test_scores = score_records(model, test_cache["visit_records"], test_cache["drug_memory"], device)

    before_labels = [[d for d, _ in rec["candidates"]] for rec in test_cache["visit_records"]]
    after_labels = [
        apply_filter(candidate_ids, scores, threshold_info["threshold"])
        for candidate_ids, scores in test_scores
    ]

    before_metrics = compute_metrics(test_cache["visit_records"], before_labels)
    after_metrics = compute_metrics(test_cache["visit_records"], after_labels)

    print(f"{'metric':<12}{'before':>10}{'after':>10}")
    for key in ("precision", "recall", "jaccard", "f1", "ddi_rate", "avg_med"):
        print(f"{key:<12}{before_metrics[key]:>10.4f}{after_metrics[key]:>10.4f}")


if __name__ == "__main__":
    main()
