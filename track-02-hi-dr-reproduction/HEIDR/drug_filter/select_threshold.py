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


def apply_filter_to_visits(scores_per_visit: list, threshold: float) -> list:
    """방문별 (candidate_ids, scores)에 apply_filter를 적용해 방문별 예측 라벨 리스트를 만든다."""
    return [apply_filter(cids, scores, threshold) for cids, scores in scores_per_visit]


def compute_achieved_ddi_rate(predicted_labels: list, ddi_A) -> float:
    """방문별 예측 약물 리스트에서 DDI rate(dd_cnt/all_cnt, HEIDR/util.py의
    ddi_rate_score와 동일 정의)를 계산한다."""
    all_cnt = 0
    dd_cnt = 0
    for label in predicted_labels:
        for i in range(len(label)):
            for j in range(i + 1, len(label)):
                a, b = label[i], label[j]
                all_cnt += 1
                if ddi_A[a, b] == 1 or ddi_A[b, a] == 1:
                    dd_cnt += 1
    return dd_cnt / all_cnt if all_cnt > 0 else 0.0


def _precision_recall_f_beta_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float, beta: float) -> tuple:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return precision, recall, f_beta


def select_ddi_aware_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    scores_per_visit: list,
    ddi_A,
    gt_ddi_rate: float,
    beta: float = 1.0,
    margin: float = 0.005,
    n_thresholds: int = 50,
) -> dict:
    """
    threshold 후보(scores의 분위수 n_thresholds개)마다 F-beta와, scores_per_visit에
    apply_filter를 적용했을 때의 achieved DDI rate를 함께 계산한다. 그중
    ddi_rate <= gt_ddi_rate + margin을 만족하는 후보(feasible)들 중 f_beta가 가장
    높은 threshold를 직접 고른다 (constrained argmax) — "DDI rate=0"이 아니라
    "정답 수준(gt_ddi_rate) 근처로 수렴"이 목표이므로, margin 안에서는 필터링
    효과(precision/AVG_MED 개선)를 최대화하는 지점을 우선한다. feasible한 후보가
    하나도 없으면 achieved ddi_rate가 가장 낮은 후보로 fallback한다.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds)))

    candidates = []
    for t in candidate_thresholds:
        precision, recall, f_beta = _precision_recall_f_beta_at_threshold(labels, scores, float(t), beta)
        predicted = apply_filter_to_visits(scores_per_visit, float(t))
        ddi_rate = compute_achieved_ddi_rate(predicted, ddi_A)
        candidates.append({
            "threshold": float(t), "precision": precision, "recall": recall,
            "f_beta": f_beta, "ddi_rate": ddi_rate,
        })

    feasible = [c for c in candidates if c["ddi_rate"] <= gt_ddi_rate + margin]
    if feasible:
        return max(feasible, key=lambda c: c["f_beta"])
    return min(candidates, key=lambda c: c["ddi_rate"])
