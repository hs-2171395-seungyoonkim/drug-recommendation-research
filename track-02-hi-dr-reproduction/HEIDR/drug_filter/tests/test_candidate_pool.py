from HEIDR.drug_filter.candidate_pool import (
    SENTINEL_LOGPROB,
    expand_candidates,
    pool_coverage,
)


def test_beam_candidates_are_preserved_with_is_beam_flag():
    out = expand_candidates([(10, -0.2), (11, -1.5)], prev_med_set=set())

    assert out == [(10, -0.2, 1.0), (11, -1.5, 1.0)]


def test_previous_visit_drugs_are_appended_with_sentinel_logprob():
    out = expand_candidates([(10, -0.2), (11, -1.5)], prev_med_set={11, 20})

    # 11은 이미 빔 후보라 중복 추가되지 않는다
    assert [d for d, _, _ in out] == [10, 11, 20]
    # 새로 추가된 20은 그 방문 빔 후보의 최소 logprob을 받고 is_beam=0
    assert out[2] == (20, -1.5, 0.0)


def test_sentinel_falls_back_to_constant_when_no_beam_candidates():
    out = expand_candidates([], prev_med_set={7})

    assert out == [(7, SENTINEL_LOGPROB, 0.0)]


def test_no_duplicates_even_if_previous_visit_repeats():
    out = expand_candidates([(10, -0.2)], prev_med_set={10})

    assert out == [(10, -0.2, 1.0)]


def test_pool_coverage_counts_ground_truth_found_in_pool():
    # Asymmetric GT sizes to distinguish per-visit averaging from global pooling:
    # per-visit mean(2/2, 1/3) = 2/3 ≈ 0.6667 vs global pooling (2+1)/(2+3) = 0.6
    pools = [[(1, -0.1, 1.0), (2, -0.2, 1.0)], [(3, -0.1, 1.0)]]
    gt_lists = [[1, 2], [3, 4, 5]]  # 첫 방문 2/2, 둘째 1/3

    assert abs(pool_coverage(pools, gt_lists) - (2/3)) < 1e-9


def test_pool_coverage_ignores_visits_without_ground_truth():
    pools = [[(1, -0.1, 1.0)], [(2, -0.1, 1.0)]]
    gt_lists = [[1], []]

    assert pool_coverage(pools, gt_lists) == 1.0
