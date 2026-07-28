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


def test_select_ddi_aware_threshold_picks_best_f_beta_among_feasible_thresholds():
    # 방문 하나: drug0(정답,0.9)+drug1(정답,0.8)이 서로 DDI로 충돌, drug2(오답,0.3)
    labels = np.array([1, 1, 0])
    scores = np.array([0.9, 0.8, 0.3])
    scores_per_visit = [([100, 101, 102], [0.9, 0.8, 0.3])]
    ddi_A = np.zeros((200, 200))
    ddi_A[100, 101] = 1

    # margin=0: ddi_rate=0을 만족하는 threshold만 feasible -> drug1까지 걸러내는
    # 지점(f_beta<1.0)을 골라야 한다 (F1이 더 높은 drug0+drug1 유지 지점은
    # ddi_rate=1.0이라 feasible하지 않음).
    strict = select_ddi_aware_threshold(
        labels, scores, scores_per_visit, ddi_A, gt_ddi_rate=0.0, margin=0.0,
    )
    assert strict["ddi_rate"] == 0.0
    assert strict["f_beta"] < 1.0

    # margin을 넉넉히 주면(예: 1.0) drug0+drug1을 유지하는 f_beta=1.0 지점도
    # feasible해지므로, 그중 F1이 가장 높은 지점을 골라야 한다 (무조건 ddi_rate만
    # 최소화하지 않는다는 걸 검증).
    loose = select_ddi_aware_threshold(
        labels, scores, scores_per_visit, ddi_A, gt_ddi_rate=0.0, margin=1.0,
    )
    assert loose["f_beta"] == 1.0
