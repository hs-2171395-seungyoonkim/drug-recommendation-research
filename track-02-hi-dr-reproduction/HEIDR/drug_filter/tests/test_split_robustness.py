import numpy as np

from HEIDR.drug_filter.split_robustness import resplit_indices


def test_resplit_indices_partitions_without_overlap_and_is_reproducible():
    eval_idx1, test_idx1 = resplit_indices(10, 4, seed=0)
    eval_idx2, test_idx2 = resplit_indices(10, 4, seed=0)

    assert len(eval_idx1) == 4
    assert len(test_idx1) == 6
    assert set(eval_idx1) | set(test_idx1) == set(range(10))
    assert set(eval_idx1) & set(test_idx1) == set()
    assert np.array_equal(eval_idx1, eval_idx2)
    assert np.array_equal(test_idx1, test_idx2)


def test_resplit_indices_differs_across_seeds():
    eval_idx_a, _ = resplit_indices(20, 8, seed=0)
    eval_idx_b, _ = resplit_indices(20, 8, seed=1)

    assert not np.array_equal(eval_idx_a, eval_idx_b)
