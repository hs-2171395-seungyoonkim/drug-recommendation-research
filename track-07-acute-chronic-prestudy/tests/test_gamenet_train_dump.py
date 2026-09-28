import sys
from pathlib import Path

import dill
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gamenet_train_dump as G  # noqa: E402


def test_ddi_rate_of_set_matches_util(tmp_path):
    from util import ddi_rate_score
    A = np.zeros((7, 7))
    for i, j in [(0, 1), (2, 5), (3, 4), (1, 6)]:
        A[i, j] = A[j, i] = 1
    p = tmp_path / "ddi.pkl"
    with p.open("wb") as f:
        dill.dump(A, f)
    for meds in [[0, 1, 2, 5], [3], [], [0, 2, 3, 4, 6], list(range(7))]:
        assert G.ddi_rate_of_set(meds, A) == pytest.approx(ddi_rate_score([[meds]], path=str(p)))


def test_build_ehr_adj_is_symmetric_cooccurrence():
    recs = [[[[0], [0], [0, 1, 2]], [[0], [0], [2, 3]]], [[[0], [0], [1]]]]
    adj = G.build_ehr_adj(recs, 4)
    assert adj.tolist() == [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
    assert np.array_equal(adj, adj.T) and np.trace(adj) == 0


def test_split_data_matches_safedrug_rule():
    data = list(range(10))
    tr, te, ev, sp, el = G.split_data(data)
    assert (tr, te, ev, sp, el) == ([0, 1, 2, 3, 4, 5], [6, 7], [8, 9], 6, 2)
