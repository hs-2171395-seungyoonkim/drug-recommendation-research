import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold_f_beta(labels: np.ndarray, scores: np.ndarray, beta: float = 0.5) -> dict:
    """
    precision-recall curve를 스윕해 F-beta(기본 F0.5, precision에 4배 가중치)를
    최대화하는 threshold를 고른다.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]  # 마지막 항은 threshold=+inf sentinel

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    f_beta = np.nan_to_num(f_beta, nan=0.0)

    best_idx = int(np.argmax(f_beta))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f_beta": float(f_beta[best_idx]),
    }


def apply_filter(candidate_ids: list, scores: list, threshold: float) -> list:
    """threshold 이상인 후보만 남긴다. 전부 걸러지면 점수가 가장 높은 후보 1개는 강제로 남긴다."""
    kept = [cid for cid, s in zip(candidate_ids, scores) if s >= threshold]
    if kept:
        return kept
    if not candidate_ids:
        return []
    best_idx = int(np.argmax(scores))
    return [candidate_ids[best_idx]]
