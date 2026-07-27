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
