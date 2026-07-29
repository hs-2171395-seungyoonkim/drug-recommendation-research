import torch

from HEIDR.drug_filter.history_features import (
    build_patient_splits,
    iter_visit_histories,
)

EXPECTED_COUNTS = {"train": 6141, "eval": 1286, "test": 1255}


def test_iter_visit_histories_orders_and_excludes_current_visit():
    # 환자 1명, 방문 3개. 캐시는 idx=1..2 방문만 레코드로 만든다.
    patients = [[
        [[0], [0], [10, 11]],
        [[1], [1], [20]],
        [[2], [2], [30, 31]],
    ]]

    out = iter_visit_histories(patients)

    assert len(out) == 2
    # 첫 레코드: 현재 방문 idx=1, 이전은 idx=0 하나
    assert out[0]["gt_ids"] == [20]
    assert out[0]["prev_med_sets"] == [{10, 11}]
    # 둘째 레코드: 현재 방문 idx=2, 이전은 idx=0,1 (오래된 것이 앞)
    assert out[1]["gt_ids"] == [30, 31]
    assert out[1]["prev_med_sets"] == [{10, 11}, {20}]


def test_iter_visit_histories_skips_single_visit_patients():
    patients = [[[[0], [0], [1]]]]  # 방문 1개짜리 환자는 레코드 없음
    assert iter_visit_histories(patients) == []


def test_splits_align_with_cached_candidates():
    """캐시의 gt_ids와 복원한 gt_ids가 순서까지 완전히 일치해야 한다.
    이 테스트가 깨지면 이후 모든 이력 피처가 잘못된 환자에 붙는다."""
    splits = build_patient_splits()

    for name, expected_n in EXPECTED_COUNTS.items():
        cache = torch.load(f"HEIDR/drug_filter/candidates_{name}.pt")
        records = cache["visit_records"]
        histories = iter_visit_histories(splits[name])

        assert len(records) == expected_n, f"{name}: 캐시 레코드 수가 {expected_n}이 아님"
        assert len(histories) == len(records), f"{name}: 복원 개수 불일치"
        for i, (rec, hist) in enumerate(zip(records, histories)):
            assert rec["gt_ids"] == hist["gt_ids"], f"{name} 레코드 {i}에서 gt_ids 불일치"


def test_build_patient_splits_matches_generate_candidates_split():
    """generate_candidates.build_splits의 분할과 동일해야 한다."""
    import sys
    sys.path.insert(0, "HEIDR")
    from HEIDR.drug_filter.generate_candidates import build_splits

    reference, _ = build_splits("data/records_final.pkl", "data/voc_final.pkl")
    ours = build_patient_splits()

    for name in ("train", "eval", "test"):
        assert ours[name] == reference[name][0], f"{name} 분할이 build_splits와 다름"


from HEIDR.drug_filter.history_features import (
    HISTORY_FEATURE_NAMES,
    compute_history_features,
)

MED_NUM = 131


def test_history_feature_names_are_six():
    assert len(HISTORY_FEATURE_NAMES) == 6


def test_in_prev_uses_only_the_immediately_previous_visit():
    # 방문 2회: 오래된 것에 10, 직전에 20
    prev = [{10}, {20}]
    feats = compute_history_features(prev, [10, 20, 30])
    names = HISTORY_FEATURE_NAMES

    in_prev = names.index("in_prev")
    assert feats[0][in_prev] == 0.0  # 10은 직전 방문에 없음
    assert feats[1][in_prev] == 1.0  # 20은 직전 방문에 있음
    assert feats[2][in_prev] == 0.0


def test_in_any_prev_and_frac_prev():
    prev = [{10}, {10, 20}]
    names = HISTORY_FEATURE_NAMES
    feats = compute_history_features(prev, [10, 20, 30])

    any_i, frac_i = names.index("in_any_prev"), names.index("frac_prev")
    assert feats[0][any_i] == 1.0
    assert feats[0][frac_i] == 1.0   # 10은 2회 중 2회
    assert feats[1][frac_i] == 0.5   # 20은 2회 중 1회
    assert feats[2][any_i] == 0.0
    assert feats[2][frac_i] == 0.0


def test_n_prev_and_prev_size_are_normalized():
    prev = [{1}, {2, 3, 4}]
    names = HISTORY_FEATURE_NAMES
    feats = compute_history_features(prev, [1])

    assert feats[0][names.index("n_prev")] == 2.0
    # prev_size는 직전 방문 약물 수를 MED_NUM으로 정규화
    assert abs(feats[0][names.index("prev_size")] - 3 / MED_NUM) < 1e-9


def test_recency_weighted_favors_recent_visits():
    names = HISTORY_FEATURE_NAMES
    rec_i = names.index("recency_weighted")

    # 약물 A는 오래된 방문에만, 약물 B는 직전 방문에만
    prev = [{10}, {20}]
    feats = compute_history_features(prev, [10, 20])

    assert feats[1][rec_i] > feats[0][rec_i]
    # 가중치 0.5**age를 정규화하므로 둘의 합은 1
    assert abs(feats[0][rec_i] + feats[1][rec_i] - 1.0) < 1e-9


def test_no_previous_visits_yields_all_zero_features():
    feats = compute_history_features([], [1, 2])
    assert feats == [(0.0,) * 6, (0.0,) * 6]


def test_features_never_read_current_visit():
    """현재 방문 정답을 인자로 받지 않으므로 구조적으로 누수가 불가능하다.
    시그니처가 바뀌어 정답이 흘러들어오면 이 테스트가 깨진다."""
    import inspect

    params = list(inspect.signature(compute_history_features).parameters)
    assert params == ["prev_med_sets", "candidate_ids"]
