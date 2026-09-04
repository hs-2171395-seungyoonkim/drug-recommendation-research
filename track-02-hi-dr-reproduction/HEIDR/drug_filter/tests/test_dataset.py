import pytest
import torch

from HEIDR.drug_filter.dataset import DrugFilterDataset, build_scoring_inputs


def _records():
    return [
        {"visit_emb": torch.ones(4), "candidates": [(0, -0.1), (1, -1.2)], "gt_ids": [0]},
        {"visit_emb": torch.zeros(4), "candidates": [(2, -0.3)], "gt_ids": [2, 3]},
    ]


def _histories():
    return [
        {"prev_med_sets": [{0}], "gt_ids": [0]},
        {"prev_med_sets": [{3}], "gt_ids": [2, 3]},
    ]


def test_flattens_expanded_pool_into_candidate_pairs():
    drug_memory = torch.arange(4 * 4, dtype=torch.float).view(4, 4)

    ds = DrugFilterDataset(_records(), drug_memory, _histories())

    # 방문1: 빔 0,1 (0은 직전에도 있음 -> 중복 없음) = 2개
    # 방문2: 빔 2 + 직전 방문의 3 = 2개
    assert len(ds) == 4

    item0 = ds[0]
    assert torch.equal(item0["visit_emb"], torch.ones(4))
    assert torch.equal(item0["drug_emb"], drug_memory[0])
    assert abs(item0["hidr_logprob"].item() - (-0.1)) < 1e-6
    assert item0["label"].item() == 1.0
    assert item0["extra_features"].shape == (7,)
    assert item0["extra_features"].dtype == torch.float32


def test_expanded_candidate_is_labelled_and_flagged():
    drug_memory = torch.zeros(4, 4)

    ds = DrugFilterDataset(_records(), drug_memory, _histories())

    # 마지막 샘플 = 방문2에 추가된 drug 3 (정답에 있음, 빔이 낸 게 아님)
    item = ds[3]
    assert item["label"].item() == 1.0
    assert item["extra_features"][-1].item() == 0.0  # is_beam
    # sentinel logprob = 그 방문 빔 후보의 최소값 (-0.3)
    assert abs(item["hidr_logprob"].item() - (-0.3)) < 1e-6


def test_beam_candidate_has_is_beam_one():
    ds = DrugFilterDataset(_records(), torch.zeros(4, 4), _histories())

    assert ds[0]["extra_features"][-1].item() == 1.0


def test_empty_candidates_and_no_history_produce_no_samples():
    records = [{"visit_emb": torch.ones(4), "candidates": [], "gt_ids": [0]}]
    histories = [{"prev_med_sets": [set()], "gt_ids": [0]}]

    ds = DrugFilterDataset(records, torch.zeros(1, 4), histories)

    assert len(ds) == 0


def test_build_scoring_inputs_matches_dataset_pool():
    inputs = build_scoring_inputs(_records(), _histories())

    assert len(inputs) == 2
    pool_ids, logprobs, extras = inputs[1]
    assert pool_ids == [2, 3]
    assert len(logprobs) == 2
    assert len(extras) == 2 and len(extras[0]) == 7


def test_build_scoring_inputs_rejects_length_mismatch():
    # records/histories 캐시를 서로 다른 split끼리 잘못 짝지으면(예: test 캐시와
    # eval histories) 길이부터 어긋나는 게 흔한 실패 형태다. zip이 조용히 뒤쪽을
    # 잘라내며 삼켜버리지 않고 즉시 에러가 나야 한다.
    records = _records()
    histories = _histories()[:1]  # 하나 모자란 history 리스트

    with pytest.raises(ValueError, match="visit_records"):
        build_scoring_inputs(records, histories)


def test_build_scoring_inputs_rejects_crossed_pairing():
    # 길이는 같지만 순서가 한 칸 밀려 서로 다른 방문이 짝지어진 경우(크로스
    # 페어링). gt_ids가 위치별로 일치하지 않으면 잡아내야 한다 — 이런 경우
    # 길이 체크만으로는 통과해버리고, 커버리지 같은 지표가 그럴듯하지만 틀린
    # 값을 낸다.
    records = _records()
    histories = list(reversed(_histories()))  # 순서를 뒤집어 gt_ids를 어긋나게 함

    with pytest.raises(ValueError, match="gt_ids"):
        build_scoring_inputs(records, histories)
