import numpy as np
import torch
from torch.optim import Adam

from safedrug.ddi_mask import build_ddi_mask_h, build_molecule_map
from safedrug.model import SafeDrugModel
from safedrug.mpnn import build_mpnn_set
from safedrug.train import evaluate, train_one_epoch


def _tiny_model(organ_dim=0):
    molecule = build_molecule_map({"A01A": {"CCO"}, "A02A": {"C"}})
    med_voc_idx2word = {0: "A01A", 1: "A02A"}
    ddi_mask_h = build_ddi_mask_h(molecule, med_voc_idx2word)
    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc_idx2word, radius=2, device="cpu"
    )
    ddi_adj = np.zeros((2, 2))
    vocab_size = (3, 3, 2)
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=organ_dim, device=torch.device("cpu"),
    )
    return model, vocab_size, ddi_adj


def test_train_one_epoch_runs_without_error_and_updates_weights():
    model, vocab_size, ddi_adj = _tiny_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    before = model.query[1].weight.clone().detach()
    data_train = [
        [[[0], [0], [0]], [[1], [0], [1]]],
        [[[0, 1], [0], [0]]],
    ]
    train_one_epoch(model, data_train, optimizer, vocab_size, ddi_adj, device=torch.device("cpu"))
    after = model.query[1].weight.clone().detach()
    assert not torch.allclose(before, after)


def test_evaluate_returns_expected_metric_keys_and_subgroup_counts():
    model, vocab_size, ddi_adj = _tiny_model()
    data_eval = [[[[0], [0], [0]], [[1], [0], [1]]]]
    organ_features_eval = [
        [
            {"creatinine_deviation": 0.5, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0},
            {"creatinine_deviation": -0.1, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0},
        ]
    ]
    metrics = evaluate(model, data_eval, organ_features_eval, vocab_size, ddi_adj)
    assert set(metrics.keys()) >= {
        "ddi_rate", "jaccard", "prauc", "f1",
        "renal_jaccard", "renal_n", "liver_jaccard", "liver_n",
    }
    assert metrics["renal_n"] == 1  # patient had >=1 renal-dysfunction visit
    assert metrics["liver_n"] == 0  # no liver-dysfunction visit


def test_evaluate_with_organ_function_model_and_4_tuple_visits():
    model, vocab_size, ddi_adj = _tiny_model(organ_dim=73)
    organ_vec = np.zeros(73, dtype=np.float32)
    data_eval = [[[[0], [0], [0], organ_vec], [[1], [0], [1], organ_vec]]]
    organ_features_eval = [
        [
            {"creatinine_deviation": 0.0, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0},
            {"creatinine_deviation": 0.0, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0},
        ]
    ]
    metrics = evaluate(model, data_eval, organ_features_eval, vocab_size, ddi_adj)
    assert metrics["renal_n"] == 0
