import dill
import numpy as np

from safedrug.ddi_mask import build_ddi_mask_h, build_molecule_map


def test_build_molecule_map_falls_back_to_empty_set_for_missing_code():
    m = build_molecule_map({"A01A": {"CCO"}})
    assert m["A01A"] == {"CCO"}
    assert m["ZZZZ"] == set()


def test_build_ddi_mask_h_shape_and_zero_row_for_missing_code():
    molecule = build_molecule_map({"A01A": {"CCO"}})  # ethanol, real SMILES
    med_voc_idx2word = {0: "A01A", 1: "ZZZZ"}  # ZZZZ has no SMILES
    mask = build_ddi_mask_h(molecule, med_voc_idx2word)
    assert mask.shape[0] == 2
    assert mask[0].sum() > 0  # A01A got at least one fragment
    assert mask[1].sum() == 0  # ZZZZ (no SMILES) is all-zero


def test_build_ddi_mask_h_against_real_vocab_matches_verified_shape():
    with open("data/mimic-iv/voc_final2.pkl", "rb") as f:
        voc = dill.load(f)
    med_voc = voc["med_voc"]
    with open("data/mappings/idx2drug.pkl", "rb") as f:
        idx2drug = dill.load(f)
    molecule = build_molecule_map(idx2drug)
    mask = build_ddi_mask_h(molecule, med_voc.idx2word)
    assert mask.shape == (152, 490)
    assert (mask.sum(axis=1) == 0).sum() == 22
