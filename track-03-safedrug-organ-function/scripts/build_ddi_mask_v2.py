"""
One-time real-data run: regenerates data/ddi_mask_H_v2.pkl and
data/atc3toSMILES_v2.pkl for the current 152-code med_voc (voc_final2.pkl),
reusing idx2drug.pkl's MIMIC-III SMILES map (ATC is MIMIC-version-independent).
Run from the ServerityMed repo root:
  C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/build_ddi_mask_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dill

from safedrug.ddi_mask import build_ddi_mask_h, build_molecule_map

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "data/mimic-iv/voc_final2.pkl", "rb") as f:
        voc = dill.load(f)
    med_voc = voc["med_voc"]

    with open(ROOT / "data/mappings/idx2drug.pkl", "rb") as f:
        idx2drug = dill.load(f)

    molecule = build_molecule_map(idx2drug)
    ddi_mask_h = build_ddi_mask_h(molecule, med_voc.idx2word)

    print(f"med_voc size: {len(med_voc.idx2word)}")
    print(f"ddi_mask_h shape: {ddi_mask_h.shape}")
    print(f"all-zero rows (no SMILES coverage): {(ddi_mask_h.sum(axis=1) == 0).sum()}")

    with open(ROOT / "data/ddi_mask_H_v2.pkl", "wb") as f:
        dill.dump(ddi_mask_h, f)
    with open(ROOT / "data/atc3toSMILES_v2.pkl", "wb") as f:
        dill.dump(dict(molecule), f)

    print("wrote data/ddi_mask_H_v2.pkl and data/atc3toSMILES_v2.pkl")


if __name__ == "__main__":
    main()
