import numpy as np

from HEIDR.drug_filter.significance_test import bootstrap_ddi_rate_diff, per_visit_scores


def test_per_visit_scores_computes_precision_recall_jaccard_f1_per_visit():
    records = [
        {"gt_ids": [1, 2]},
        {"gt_ids": [3]},
    ]
    predicted_labels = [
        [1, 2, 4],  # inter={1,2}, pred={1,2,4}, target={1,2}, union={1,2,4}
        [],         # inter={}, pred={}, target={3}, union={3}
    ]

    result = per_visit_scores(records, predicted_labels)

    assert result["precision"][0] == 2 / 3
    assert result["recall"][0] == 1.0
    assert result["jaccard"][0] == 2 / 3
    expected_f1 = 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)
    assert abs(result["f1"][0] - expected_f1) < 1e-9

    assert result["precision"][1] == 0.0
    assert result["recall"][1] == 0.0
    assert result["jaccard"][1] == 0.0
    assert result["f1"][1] == 0.0


def test_bootstrap_ddi_rate_diff_is_zero_when_predictions_identical():
    predicted = [[0, 1], [2]]
    ddi_A = np.zeros((4, 4))
    ddi_A[0, 1] = 1

    diffs = bootstrap_ddi_rate_diff(predicted, predicted, ddi_A, n_boot=50, seed=0)

    assert np.allclose(diffs, 0.0)


def test_bootstrap_ddi_rate_diff_reproducible_with_same_seed():
    predicted_a = [[0, 1], [2], [3, 0]]
    predicted_b = [[0], [2], [3]]
    ddi_A = np.zeros((4, 4))
    ddi_A[0, 1] = 1

    diffs1 = bootstrap_ddi_rate_diff(predicted_a, predicted_b, ddi_A, n_boot=30, seed=42)
    diffs2 = bootstrap_ddi_rate_diff(predicted_a, predicted_b, ddi_A, n_boot=30, seed=42)

    assert np.array_equal(diffs1, diffs2)
    assert len(diffs1) == 30
