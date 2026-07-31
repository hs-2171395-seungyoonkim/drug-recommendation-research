"""Vocabulary-aligned molecular assets for the MIMIC-IV final4 baseline."""

from safedrug.ddi_mask import build_ddi_mask_h, build_molecule_map


def build_final4_assets(idx2drug: dict, med_voc_idx2word: dict):
    """Build molecule and DDI-mask assets without dropping uncovered ATC3 codes."""
    molecule = build_molecule_map(idx2drug)
    for atc3 in med_voc_idx2word.values():
        molecule[atc3]
    return molecule, build_ddi_mask_h(molecule, med_voc_idx2word)
