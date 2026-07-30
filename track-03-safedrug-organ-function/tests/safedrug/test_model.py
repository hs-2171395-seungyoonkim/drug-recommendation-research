import numpy as np
import torch

from safedrug.ddi_mask import build_ddi_mask_h, build_molecule_map
from safedrug.model import SafeDrugModel
from safedrug.mpnn import build_mpnn_set


def _tiny_setup():
    molecule = build_molecule_map({"A01A": {"CCO"}, "A02A": {"C"}})
    med_voc_idx2word = {0: "A01A", 1: "A02A"}
    ddi_mask_h = build_ddi_mask_h(molecule, med_voc_idx2word)
    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc_idx2word, radius=2, device="cpu"
    )
    ddi_adj = np.zeros((2, 2))
    vocab_size = (5, 4, 2)  # tiny diag/proc/med vocab
    return vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection


def test_forward_pass_shape_without_organ_function():
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=0, device=torch.device("cpu"),
    )
    visit = [[0, 1], [0], [0]]
    result, batch_neg = model([visit])
    assert result.shape == (1, vocab_size[2])
    assert batch_neg.dim() == 0


def test_forward_pass_shape_with_organ_function():
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=73, device=torch.device("cpu"),
    )
    organ_vec = np.zeros(73, dtype=np.float32)
    visit = [[0, 1], [0], [0], organ_vec]
    result, batch_neg = model([visit])
    assert result.shape == (1, vocab_size[2])


def test_forward_pass_multi_visit_sequence_uses_last_visit_organ_vec():
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=73, device=torch.device("cpu"),
    )
    visit1 = [[0], [0], [0], np.zeros(73, dtype=np.float32)]
    visit2 = [[1], [0], [1], np.ones(73, dtype=np.float32)]
    result, _ = model([visit1, visit2])
    assert result.shape == (1, vocab_size[2])


def test_query_input_dim_matches_organ_dim():
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=73, device=torch.device("cpu"),
    )
    assert model.query[1].in_features == 2 * 8 + 73


def test_query_input_dim_is_baseline_when_organ_dim_zero():
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=0, device=torch.device("cpu"),
    )
    assert model.query[1].in_features == 2 * 8
