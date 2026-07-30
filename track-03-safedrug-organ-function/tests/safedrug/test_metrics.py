import numpy as np
import torch.nn as nn

from safedrug.metrics import ddi_rate_score, get_n_params, multi_label_metric


def test_multi_label_metric_perfect_prediction_gives_jaccard_one():
    y_gt = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 1], [0, 1, 0]])
    y_prob = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.1]])
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt, y_pred, y_prob)
    assert ja == 1.0
    assert avg_p == 1.0
    assert avg_r == 1.0


def test_multi_label_metric_no_overlap_gives_jaccard_zero():
    y_gt = np.array([[1, 0, 0]])
    y_pred = np.array([[0, 1, 0]])
    y_prob = np.array([[0.1, 0.9, 0.1]])
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt, y_pred, y_prob)
    assert ja == 0.0


def test_ddi_rate_score_counts_known_interacting_pair():
    ddi_adj = np.array([[0, 1], [1, 0]])
    records = [[[0, 1]]]  # one patient, one visit, prescribed drugs 0 and 1
    assert ddi_rate_score(records, ddi_adj) == 1.0


def test_ddi_rate_score_zero_when_no_pairs_interact():
    ddi_adj = np.array([[0, 0], [0, 0]])
    records = [[[0, 1]]]
    assert ddi_rate_score(records, ddi_adj) == 0.0


def test_ddi_rate_score_zero_when_no_admissions():
    ddi_adj = np.array([[0, 1], [1, 0]])
    assert ddi_rate_score([], ddi_adj) == 0


def test_get_n_params_counts_linear_layer_parameters():
    model = nn.Linear(4, 2)  # weight (2,4)=8 + bias (2,)=2 = 10
    assert get_n_params(model) == 10
