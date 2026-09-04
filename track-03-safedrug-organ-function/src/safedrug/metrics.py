"""
Evaluation metrics ported from SOTA/SafeDrug/src/util.py (multi_label_metric,
ddi_rate_score, get_n_params). ddi_rate_score takes the ddi_adj array
directly rather than a pickle path, since callers already have it loaded.
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def multi_label_metric(y_gt, y_pred, y_prob):
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            score.append(0 if len(union) == 0 else len(inter) / len(union))
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            score.append(0 if len(out_list) == 0 else len(inter) / len(out_list))
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            score.append(0 if len(target) == 0 else len(inter) / len(target))
        return score

    def average_f1(prc, recall):
        score = []
        for p, r in zip(prc, recall):
            score.append(0 if (p + r) == 0 else 2 * p * r / (p + r))
        return score

    def f1(y_gt, y_pred):
        return np.mean(
            [f1_score(y_gt[b], y_pred[b], average="macro") for b in range(y_gt.shape[0])]
        )

    def precision_auc(y_gt, y_prob):
        return np.mean(
            [
                average_precision_score(y_gt[b], y_prob[b], average="macro")
                for b in range(y_gt.shape[0])
            ]
        )

    ja = jaccard(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1_scores = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1_scores)


def ddi_rate_score(records, ddi_adj):
    """records: list[patient] of list[visit] of predicted med index lists.
    ddi_adj: (n_drugs, n_drugs) 0/1 matrix."""
    all_cnt = 0
    dd_cnt = 0
    for patient in records:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def get_n_params(model):
    total = 0
    for p in model.parameters():
        n = 1
        for s in p.size():
            n *= s
        total += n
    return total
