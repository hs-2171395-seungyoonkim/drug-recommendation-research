"""
Regenerates the DDI substructure mask for the current 152-code med_voc
(voc_final2.pkl), reusing the MIMIC-III SMILES map (idx2drug.pkl) since
ATC is a MIMIC-version-independent standard classification. The inherited
HI-DR ddi_mask_H2.pkl (124 rows) does not match voc_final2.pkl's 152-code
vocab (discovered during Plan A's final review) - this module produces a
fresh, self-consistent mask instead.

Verified against real data during planning: 130/152 codes have >=1 SMILES,
22 get an all-zero row (no substructure evidence - same convention
SafeDrug's own code uses for codes it can't resolve). 0/281 SMILES failed
RDKit parsing/BRICS decomposition.
"""
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS


def build_molecule_map(idx2drug: dict) -> "defaultdict[str, set]":
    """idx2drug: MIMIC-III ATC3->SMILES map (data/mappings/idx2drug.pkl).
    Returns a defaultdict(set) so codes absent from idx2drug (22 of our
    152) look up to an empty set instead of raising KeyError."""
    return defaultdict(set, idx2drug)


def build_ddi_mask_h(molecule: "defaultdict[str, set]", med_voc_idx2word: dict) -> np.ndarray:
    """molecule: build_molecule_map()'s output. med_voc_idx2word: voc_final2.pkl's
    med_voc.idx2word (152 entries: int index -> ATC3 string).
    Returns an (len(med_voc_idx2word), num_fragments) 0/1 matrix - row i is
    the BRICS fragment-membership indicator for med_voc index i's SMILES set.
    A code with no SMILES gets an all-zero row."""
    fraction = []
    for _, atc3 in med_voc_idx2word.items():
        frags = set()
        for smiles in molecule[atc3]:
            try:
                decomposed = BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles))
                frags.update(decomposed)
            except Exception:
                pass
        fraction.append(frags)

    frag_list = []
    for frags in fraction:
        frag_list += list(frags)
    frag_list = list(set(frag_list))

    ddi_mask_h = np.zeros((len(med_voc_idx2word), len(frag_list)))
    for i, frags in enumerate(fraction):
        for frag in frags:
            ddi_mask_h[i, frag_list.index(frag)] = 1
    return ddi_mask_h
