def compute_ddi_conflict_features(candidates: list, ddi_A) -> list:
    """
    한 방문의 후보 약물 리스트에서, 각 후보가 같은 방문의 다른 후보와 DDI로
    충돌하는 정도를 (conflict_sum_norm, conflict_max) 튜플로 계산한다.
    다른 후보의 hidr_prob로 가중해, 실제로 함께 남을 가능성이 높은 후보와의
    충돌만 위험 신호로 반영한다.
    """
    n = len(candidates)
    features = []
    for i, (drug_i, _) in enumerate(candidates):
        conflict_sum = 0.0
        conflict_max = 0.0
        for j, (drug_j, prob_j) in enumerate(candidates):
            if i == j:
                continue
            if ddi_A[drug_i, drug_j] == 1 or ddi_A[drug_j, drug_i] == 1:
                conflict_sum += prob_j
                conflict_max = max(conflict_max, prob_j)
        other_count = n - 1
        conflict_sum_norm = conflict_sum / other_count if other_count > 0 else 0.0
        features.append((conflict_sum_norm, conflict_max))
    return features
