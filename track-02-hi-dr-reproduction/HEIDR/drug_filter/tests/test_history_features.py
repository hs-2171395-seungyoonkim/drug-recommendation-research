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
