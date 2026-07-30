import dill

from safedrug.ddi_mask import build_molecule_map
from safedrug.mpnn import build_mpnn_set


def test_build_mpnn_set_shapes_for_tiny_vocab():
    molecule = build_molecule_map({"A01A": {"CCO"}, "A02A": {"C"}})
    med_voc_idx2word = {0: "A01A", 1: "A02A"}
    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc_idx2word, radius=2, device="cpu"
    )
    assert len(mpnn_set) == 2  # one molecule instance per code (each has exactly 1 SMILES)
    assert average_projection.shape == (2, 2)
    assert n_fingerprint > 0


def test_build_mpnn_set_zero_row_for_code_without_smiles():
    molecule = build_molecule_map({"A01A": {"CCO"}})
    med_voc_idx2word = {0: "A01A", 1: "ZZZZ"}
    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc_idx2word, radius=2, device="cpu"
    )
    assert average_projection.shape[0] == 2
    assert average_projection[1].sum() == 0  # ZZZZ: no SMILES -> no instances -> zero row


def test_build_mpnn_set_against_real_vocab_matches_verified_shape():
    with open("data/mimic-iv/voc_final2.pkl", "rb") as f:
        voc = dill.load(f)
    med_voc = voc["med_voc"]
    with open("data/mappings/idx2drug.pkl", "rb") as f:
        idx2drug = dill.load(f)
    molecule = build_molecule_map(idx2drug)
    mpnn_set, n_fingerprint, average_projection = build_mpnn_set(
        molecule, med_voc.idx2word, radius=2, device="cpu"
    )
    assert average_projection.shape == (152, 276)
    assert n_fingerprint == 2134
