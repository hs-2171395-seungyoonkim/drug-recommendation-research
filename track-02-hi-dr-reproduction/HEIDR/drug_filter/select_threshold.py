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


def _prf_from_predictions(scores_per_visit: list, predicted: list, labels: np.ndarray, beta: float) -> tuple:
    """
    predicted(방문별 실제로 남은 약물 id 리스트, apply_filter의 fallback 포함)를
    scores_per_visit과 같은 순서로 평탄화해 labels와 정렬한 뒤, 실제로 적용된
    예측 집합 기준으로 precision/recall/f_beta를 계산한다. (naive score>=threshold
    컷오프로만 계산하면 fallback을 무시하게 되어, ddi_rate가 보는 예측 집합과
    f_beta가 보는 예측 집합이 서로 달라진다 — 이 함수로 통일한다.)
    """
    kept_mask = []
    for (cids, _), kept_ids in zip(scores_per_visit, predicted):
        kept_set = set(kept_ids)
        for cid in cids:
            kept_mask.append(cid in kept_set)
    kept_mask = np.array(kept_mask, dtype=bool)

    tp = int(np.sum(kept_mask & (labels == 1)))
    fp = int(np.sum(kept_mask & (labels == 0)))
    fn = int(np.sum((~kept_mask) & (labels == 1)))
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
    min_recall_ratio: float = 0.5,
) -> dict:
    """
    threshold 후보(scores의 분위수 n_thresholds개)마다, scores_per_visit에
    apply_filter_to_visits를 적용한 실제 예측 집합 기준으로 precision/recall/f_beta와
    achieved DDI rate를 함께 계산한다(둘 다 같은 예측 집합에서 계산되므로 서로
    어긋나지 않는다). 그 후보들 중,
      1) recall이 후보군 전체 최대 recall의 min_recall_ratio 미만인
         "퇴화(degenerate) threshold"(threshold가 너무 높아 거의 모든 방문이
         1개 약물로 주저앉아 약물쌍 자체가 사라지고 ddi_rate가 우연히 0에
         가까워지는 경우)를 먼저 제외하고,
      2) 남은 후보 중 ddi_rate <= gt_ddi_rate + margin을 만족하는(feasible)
         후보들 중 f_beta가 가장 높은 threshold를 고른다 (constrained argmax).
    feasible한 비퇴화 후보가 없으면 비퇴화 후보 중 ddi_rate가 가장 낮은 것으로,
    비퇴화 후보 자체가 없으면 전체 후보 중 ddi_rate가 가장 낮은 것으로 fallback한다.
    "DDI rate=0"이 아니라 "정답 수준(gt_ddi_rate) 근처로 수렴"이 목표다.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds)))

    candidates = []
    for t in candidate_thresholds:
        predicted = apply_filter_to_visits(scores_per_visit, float(t))
        precision, recall, f_beta = _prf_from_predictions(scores_per_visit, predicted, labels, beta)
        ddi_rate = compute_achieved_ddi_rate(predicted, ddi_A)
        candidates.append({
            "threshold": float(t), "precision": precision, "recall": recall,
            "f_beta": f_beta, "ddi_rate": ddi_rate,
        })

    max_recall = max(c["recall"] for c in candidates)
    non_degenerate = [c for c in candidates if c["recall"] >= min_recall_ratio * max_recall]

    feasible = [c for c in non_degenerate if c["ddi_rate"] <= gt_ddi_rate + margin]
    if feasible:
        return max(feasible, key=lambda c: c["f_beta"])
    if non_degenerate:
        return min(non_degenerate, key=lambda c: c["ddi_rate"])
    return min(candidates, key=lambda c: c["ddi_rate"])
