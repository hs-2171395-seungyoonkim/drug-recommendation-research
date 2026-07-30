import numpy as np
import torch
from torch.optim import Adam

import safedrug.train as train_mod
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


def test_evaluate_multi_patient_subgroup_alignment_matches_correct_patient():
    """Regression test for organ_features_eval[patient_idx] alignment: with 3
    patients where only the MIDDLE one has a renal-dysfunction visit, renal_n
    must be exactly 1 (not 0 or 2/3 from an off-by-one indexing bug)."""
    model, vocab_size, ddi_adj = _tiny_model()
    data_eval = [
        [[[0], [0], [0]]],
        [[[0], [0], [0]], [[1], [0], [1]]],
        [[[1], [0], [0]]],
    ]
    organ_features_eval = [
        [{"creatinine_deviation": -0.2, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0}],
        [
            {"creatinine_deviation": -0.2, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0},
            {"creatinine_deviation": 0.9, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0},
        ],
        [{"creatinine_deviation": 0.0, "albumin_deviation": 0.0, "bilirubin_total_deviation": 0.0}],
    ]
    metrics = evaluate(model, data_eval, organ_features_eval, vocab_size, ddi_adj)
    assert metrics["renal_n"] == 1
    assert metrics["liver_n"] == 0


def test_train_one_epoch_ddi_conditioned_branch_uses_min_beta_formula(monkeypatch):
    """The else-branch of train_one_epoch (current_ddi_rate > target_ddi) is
    never exercised by the other tests, which all use an all-zero ddi_adj so
    current_ddi_rate is always exactly 0. This test forces that branch (via a
    monkeypatched ddi_rate_score) and pins the exact loss formula
    (`beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)`, kept as
    original SafeDrug for baseline fidelity) so a future accidental change to
    it does not go unnoticed."""
    model, vocab_size, ddi_adj = _tiny_model()
    optimizer = Adam(model.parameters(), lr=1e-3)

    target_ddi = 0.06
    kp = 0.05
    fixed_ddi_rate = 0.5  # > target_ddi forces the else branch

    fixed_result = torch.zeros(1, vocab_size[2], requires_grad=True)
    fixed_ddi_loss = torch.tensor(5.0, requires_grad=True)
    fixed_bce = torch.tensor(2.0, requires_grad=True)
    fixed_multi = torch.tensor(3.0, requires_grad=True)

    monkeypatch.setattr(train_mod, "ddi_rate_score", lambda *a, **k: fixed_ddi_rate)
    monkeypatch.setattr(model, "forward", lambda seq_input: (fixed_result, fixed_ddi_loss))
    monkeypatch.setattr(
        train_mod.F, "binary_cross_entropy_with_logits", lambda *a, **k: fixed_bce
    )
    monkeypatch.setattr(train_mod.F, "multilabel_margin_loss", lambda *a, **k: fixed_multi)

    captured_losses = []
    orig_backward = torch.Tensor.backward

    def capture_backward(self, *args, **kwargs):
        captured_losses.append(self.detach().clone())
        return orig_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", capture_backward)

    data_train = [[[[0], [0], [0]]]]
    train_one_epoch(
        model, data_train, optimizer, vocab_size, ddi_adj,
        target_ddi=target_ddi, kp=kp, device=torch.device("cpu"),
    )

    assert len(captured_losses) == 1
    expected_beta = min(0, 1 + (target_ddi - fixed_ddi_rate) / kp)
    expected_loss = expected_beta * (0.95 * fixed_bce + 0.05 * fixed_multi) + (
        1 - expected_beta
    ) * fixed_ddi_loss
    assert torch.isclose(captured_losses[0], expected_loss.detach())
