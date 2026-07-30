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
    """Changing ONLY the last visit's organ vector (holding every medical code
    and every other visit fixed) must change the model's output. This proves
    `input[-1][3]` — the LAST visit's organ vector — is what actually gets
    consumed, not e.g. `input[0][3]` or a value that gets ignored entirely.
    `.eval()` disables dropout so the two forward passes are deterministic
    given identical non-organ inputs, making the comparison meaningful.
    """
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=73, device=torch.device("cpu"),
    )
    model.eval()

    visit1 = [[0], [0], [0], np.zeros(73, dtype=np.float32)]
    visit2_a = [[1], [0], [1], np.zeros(73, dtype=np.float32)]
    visit2_b = [[1], [0], [1], np.ones(73, dtype=np.float32)]

    with torch.no_grad():
        result_a, _ = model([visit1, visit2_a])
        result_b, _ = model([visit1, visit2_b])

    assert result_a.shape == (1, vocab_size[2])
    assert not torch.allclose(result_a, result_b), (
        "changing only the last visit's organ vector should change the model's "
        "output; identical outputs mean the organ vector isn't being consumed "
        "from the last visit position"
    )


def test_organ_vector_changes_single_visit_output():
    """Dedicated regression guard for the +OrganFunction arm: if a future change
    dropped the `torch.cat` in the forward pass and zero-padded instead (or
    otherwise stopped threading the organ vector into `self.query`'s input),
    every other test in this file would stay green while the +OrganFunction
    arm silently became architecturally identical to the baseline arm — which
    would invalidate the whole ablation. This test fails loudly in that case:
    two single-visit inputs, identical except for the organ vector, must
    produce different outputs.
    """
    vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection = _tiny_setup()
    model = SafeDrugModel(
        vocab_size, ddi_adj, ddi_mask_h, mpnn_set, n_fingerprint, average_projection,
        emb_dim=8, organ_dim=73, device=torch.device("cpu"),
    )
    model.eval()

    visit_a = [[0, 1], [0], [0], np.zeros(73, dtype=np.float32)]
    visit_b = [[0, 1], [0], [0], np.ones(73, dtype=np.float32)]

    with torch.no_grad():
        result_a, _ = model([visit_a])
        result_b, _ = model([visit_b])

    assert not torch.allclose(result_a, result_b), (
        "changing the organ vector (all zeros vs. all ones) on an otherwise "
        "identical single visit should change the model's output"
    )


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
