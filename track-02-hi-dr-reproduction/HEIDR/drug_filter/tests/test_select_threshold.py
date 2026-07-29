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


from HEIDR.drug_filter.select_threshold import compute_ddi_pair_stats


def test_compute_ddi_pair_stats_counts_absolute_pairs_per_visit():
    ddi_A = np.zeros((4, 4))
    ddi_A[0, 1] = 1
    # visit1: 0-1 쌍(DDI) 1개, visit2: 2-3 쌍(비-DDI) 1개
    predicted_labels = [[0, 1], [2, 3]]

    stats = compute_ddi_pair_stats(predicted_labels, ddi_A)

    assert stats["rate"] == 0.5
    assert stats["dd_cnt_total"] == 1
    assert stats["avg_dd_per_visit"] == 0.5  # (1 + 0) / 2 visits
    assert stats["avg_med"] == 2.0


def test_compute_ddi_pair_stats_returns_zeros_for_empty_input():
    stats = compute_ddi_pair_stats([], np.zeros((4, 4)))
    assert stats == {"rate": 0.0, "dd_cnt_total": 0, "avg_dd_per_visit": 0.0, "avg_med": 0.0}


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


def test_select_ddi_aware_threshold_excludes_degenerate_high_threshold_candidates():
    # 방문 하나, 후보 4개(전부 정답), 600-601만 서로 DDI로 충돌.
    # threshold를 극단적으로 높이면 방문당 600 하나만 남아(recall=0.25) 쌍
    # 자체가 사라져 ddi_rate가 0이 되지만, 이런 퇴화 지점은 가드로 제외되고
    # 대신 recall을 훨씬 덜 희생하는 낮은 threshold(recall=1.0)가 선택돼야 한다.
    labels = np.array([1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    scores_per_visit = [([600, 601, 602, 603], [0.9, 0.8, 0.7, 0.6])]
    ddi_A = np.zeros((700, 700))
    ddi_A[600, 601] = 1

    result = select_ddi_aware_threshold(
        labels, scores, scores_per_visit, ddi_A, gt_ddi_rate=0.0, margin=0.0,
    )

    assert result["recall"] >= 0.5


from HEIDR.drug_filter.select_threshold import (
    SAFETY_MARGIN,
    select_min_avgmed_threshold,
    visit_jaccard,
)


def test_visit_jaccard_averages_per_visit_intersection_over_union():
    records = [{"gt_ids": [1, 2]}, {"gt_ids": [3]}]
    predicted = [[1, 2], [3, 4]]  # 1.0, 0.5

    assert abs(visit_jaccard(records, predicted) - 0.75) < 1e-9


def test_select_min_avgmed_picks_smallest_pool_meeting_quality_floor():
    # 후보 3개 중 2개가 정답. threshold를 올릴수록 AVG_MED가 준다.
    records = [{"gt_ids": [10, 11]}]
    scores_per_visit = [([10, 11, 12], [0.9, 0.8, 0.2])]

    # floor를 아주 낮게 두면 가장 공격적인(=AVG_MED 최소) 지점을 고른다
    result = select_min_avgmed_threshold(records, scores_per_visit, quality_floor=0.0, safety_margin=0.0)

    assert result["feasible"] is True
    assert result["avg_med"] <= 3.0


def test_select_min_avgmed_respects_the_quality_floor():
    records = [{"gt_ids": [10, 11]}]
    scores_per_visit = [([10, 11, 12], [0.9, 0.8, 0.2])]

    # 정답 2개를 모두 남겨야만 도달 가능한 floor
    result = select_min_avgmed_threshold(
        records, scores_per_visit, quality_floor=0.66, safety_margin=0.0
    )

    assert result["jaccard"] >= 0.66


def test_safety_margin_makes_selection_more_conservative():
    records = [{"gt_ids": [10, 11]}]
    scores_per_visit = [([10, 11, 12], [0.9, 0.8, 0.2])]

    loose = select_min_avgmed_threshold(records, scores_per_visit, 0.6, safety_margin=0.0)
    tight = select_min_avgmed_threshold(records, scores_per_visit, 0.6, safety_margin=0.3)

    assert tight["jaccard"] >= loose["jaccard"]


def test_falls_back_to_best_quality_when_floor_unreachable():
    records = [{"gt_ids": [10, 11]}]
    scores_per_visit = [([10, 12], [0.9, 0.8])]  # 정답 11이 후보에 없어 1.0 불가

    result = select_min_avgmed_threshold(records, scores_per_visit, quality_floor=1.0)

    assert result["feasible"] is False


def test_default_safety_margin_is_documented_value():
    assert SAFETY_MARGIN == 0.005
