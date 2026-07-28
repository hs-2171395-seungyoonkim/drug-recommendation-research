import numpy as np

from HEIDR.drug_filter.select_threshold import select_threshold_f_beta, apply_filter


def test_select_threshold_f_beta_separates_perfectly_separable_scores():
    # score >= 0.5 인 것만 label=1인 완벽히 분리 가능한 합성 데이터
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.1])

    result = select_threshold_f_beta(labels, scores, beta=0.5)

    assert 0.4 < result["threshold"] <= 0.6
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f_beta"] == 1.0


def test_apply_filter_keeps_only_above_threshold():
    kept = apply_filter([10, 20, 30], [0.9, 0.4, 0.6], threshold=0.5)
    assert kept == [10, 30]


def test_apply_filter_falls_back_to_best_single_candidate_when_all_below():
    kept = apply_filter([10, 20], [0.2, 0.4], threshold=0.5)
    assert kept == [20]  # 가장 점수 높은 후보 1개는 남는다


def test_apply_filter_returns_empty_when_no_candidates():
    kept = apply_filter([], [], threshold=0.5)
    assert kept == []


from HEIDR.drug_filter.select_threshold import (
    apply_filter_to_visits,
    compute_achieved_ddi_rate,
    select_ddi_aware_threshold,
)


def test_apply_filter_to_visits_applies_per_visit_threshold():
    scores_per_visit = [([10, 20], [0.9, 0.3]), ([30], [0.6])]

    result = apply_filter_to_visits(scores_per_visit, threshold=0.5)

    assert result == [[10], [30]]


def test_compute_achieved_ddi_rate_counts_flagged_pairs():
    ddi_A = np.zeros((4, 4))
    ddi_A[0, 1] = 1
    predicted_labels = [[0, 1], [2, 3]]  # visit1: 0-1 쌍(DDI), visit2: 2-3 쌍(비-DDI)

    rate = compute_achieved_ddi_rate(predicted_labels, ddi_A)

    assert rate == 0.5


def test_compute_achieved_ddi_rate_returns_zero_when_no_pairs():
    assert compute_achieved_ddi_rate([[0], [1]], np.zeros((4, 4))) == 0.0


def test_select_ddi_aware_threshold_penalizes_ddi_when_lambda_large():
    # 방문 하나: drug0(정답,0.9)+drug1(정답,0.8)이 서로 DDI로 충돌, drug2(오답,0.3)
    labels = np.array([1, 1, 0])
    scores = np.array([0.9, 0.8, 0.3])
    scores_per_visit = [([100, 101, 102], [0.9, 0.8, 0.3])]
    ddi_A = np.zeros((200, 200))
    ddi_A[100, 101] = 1

    # lambda=0(패널티 없음): F1만 최대화하는 지점을 고르는데, 그 지점은 정답인
    # drug0+drug1이 함께 남아 DDI가 발생한다.
    result_no_penalty = select_ddi_aware_threshold(
        labels, scores, scores_per_visit, ddi_A, gt_ddi_rate=0.0,
        lambdas=(0.0,), margin=0.0,
    )
    assert result_no_penalty["ddi_rate"] > 0.0

    # lambda 그리드에 충분히 큰 값을 포함하면, DDI를 피하기 위해 recall을 일부
    # 희생하는(F1이 내려가는) 더 높은 threshold를 선택해야 한다.
    result_with_penalty = select_ddi_aware_threshold(
        labels, scores, scores_per_visit, ddi_A, gt_ddi_rate=0.0,
        lambdas=(0.0, 1.0), margin=0.0,
    )
    assert result_with_penalty["ddi_rate"] == 0.0
    assert result_with_penalty["lambda"] == 1.0
    assert result_with_penalty["f_beta"] < 1.0
