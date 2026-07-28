import numpy as np

from HEIDR.drug_filter.ddi_features import compute_ddi_conflict_features


def test_no_conflicts_returns_zero_features():
    candidates = [(0, 0.9), (1, 0.5)]
    ddi_A = np.zeros((4, 4))

    features = compute_ddi_conflict_features(candidates, ddi_A)

    assert features == [(0.0, 0.0), (0.0, 0.0)]


def test_single_conflicting_partner_weighted_by_its_hidr_prob():
    # drug0-drug1만 상호작용. drug2는 아무와도 충돌하지 않음.
    candidates = [(0, 0.9), (1, 0.5), (2, 0.2)]
    ddi_A = np.zeros((4, 4))
    ddi_A[0, 1] = 1

    features = compute_ddi_conflict_features(candidates, ddi_A)

    # drug0: 충돌 상대는 drug1(prob 0.5) 하나, 다른 후보 수=2 -> sum_norm=0.5/2=0.25
    assert features[0] == (0.25, 0.5)
    # drug1: 충돌 상대는 drug0(prob 0.9) 하나, 다른 후보 수=2 -> sum_norm=0.9/2=0.45
    assert features[1] == (0.45, 0.9)
    # drug2: 충돌 없음
    assert features[2] == (0.0, 0.0)


def test_ddi_matrix_checked_symmetrically():
    # ddi_A의 하삼각만 채워져 있어도 상호작용으로 인식해야 함 (util.py의
    # ddi_rate_score/A[i,j]==1 or A[j,i]==1 규칙과 동일)
    candidates = [(0, 0.9), (1, 0.5)]
    ddi_A = np.zeros((4, 4))
    ddi_A[1, 0] = 1

    features = compute_ddi_conflict_features(candidates, ddi_A)

    assert features[0] == (0.5, 0.5)
    assert features[1] == (0.9, 0.9)


def test_single_candidate_has_no_other_to_conflict_with():
    candidates = [(0, 0.9)]
    ddi_A = np.zeros((4, 4))

    features = compute_ddi_conflict_features(candidates, ddi_A)

    assert features == [(0.0, 0.0)]


def test_empty_candidates_returns_empty_list():
    assert compute_ddi_conflict_features([], np.zeros((4, 4))) == []
