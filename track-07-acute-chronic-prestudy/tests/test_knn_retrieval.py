import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import knn_retrieval_mimic4 as K  # noqa: E402


def test_flatten_and_tfidf_rows_are_unit_norm():
    records = [[[[0, 1], [0], [0, 1]], [[1], [1], [1, 2]]], [[[2], [0], [2]]]]
    X, M, P, pi, vi = K.flatten(records, [0, 1], n_dx=3, n_proc=2, n_med=3)
    assert X.shape == (3, 5) and M.shape == (3, 3) and pi.tolist() == [0, 0, 1] and vi.tolist() == [0, 1, 0]
    assert P[1].toarray().tolist() == [[1, 1, 0]] and P[0].nnz == 0          # prev meds only from visit 2 on
    T, Q = K.tfidf(X, X)
    norms = np.sqrt(np.asarray(T.multiply(T).sum(1)).ravel())
    assert np.allclose(norms, 1.0)


def test_knn_scores_recover_neighbour_medication_frequency():
    # two train visits with identical features; query identical to them -> score = mean of their med rows
    Xtr = sp.csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    Mtr = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
    T, Q = K.tfidf(Xtr, sp.csr_matrix(np.array([[1, 0, 1]], dtype=np.float32)))
    s = K.knn_scores(Q, T, Mtr, ks=[2])[2]
    assert np.allclose(s[0], [1.0, 0.5, 0.0])


def test_jaccard_rows():
    pred = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool); gt = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)
    assert K.jaccard_rows(pred, gt).tolist() == pytest.approx([1 / 3, 1.0])
