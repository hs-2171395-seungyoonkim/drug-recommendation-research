from HEIDR.drug_filter.evaluate_filter import deleted_gt_per_visit, jaccard_at_avg_med


def test_deleted_gt_counts_ground_truth_dropped_from_the_pool():
    records = [{"gt_ids": [1, 2, 3]}]
    pools = [[(1, -0.1, 1.0), (2, -0.2, 1.0), (9, -0.3, 1.0)]]  # 풀에 정답 1,2
    predicted = [[1, 9]]  # 정답 2를 잘라냄

    assert deleted_gt_per_visit(records, pools, predicted) == 1.0


def test_deleted_gt_is_zero_when_nothing_correct_is_removed():
    records = [{"gt_ids": [1]}]
    pools = [[(1, -0.1, 1.0), (9, -0.2, 1.0)]]
    predicted = [[1]]

    assert deleted_gt_per_visit(records, pools, predicted) == 0.0


def test_jaccard_at_avg_med_returns_the_closest_operating_point():
    records = [{"gt_ids": [10, 11]}]
    scores_per_visit = [([10, 11, 12], [0.9, 0.8, 0.2])]

    result = jaccard_at_avg_med(records, scores_per_visit, target_avg_med=2.0)

    assert abs(result["avg_med"] - 2.0) < 1.01
    assert 0.0 <= result["jaccard"] <= 1.0
