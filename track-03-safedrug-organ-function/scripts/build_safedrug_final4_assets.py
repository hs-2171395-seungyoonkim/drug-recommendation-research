"""Build MPNN/DDI-mask inputs for the isolated MIMIC-IV final4 baseline."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dill

from safedrug.final4_assets import build_final4_assets

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATHS = [
    ROOT / "data/safedrug/final4/ddi_mask_H_final4.pkl",
    ROOT / "data/safedrug/final4/molecule_final4.pkl",
    ROOT / "data/safedrug/final4/final4_assets.meta.json",
]


def main():
    if any(path.exists() for path in OUTPUT_PATHS):
        raise FileExistsError("refusing to overwrite existing final4 baseline assets")
    OUTPUT_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/mimic-iv/voc_final4.pkl").open("rb") as source:
        voc = dill.load(source)
    with (ROOT / "data/mappings/idx2drug.pkl").open("rb") as source:
        idx2drug = dill.load(source)
    molecule, ddi_mask_h = build_final4_assets(idx2drug, voc["med_voc"].idx2word)
    with OUTPUT_PATHS[0].open("wb") as target:
        dill.dump(ddi_mask_h, target)
    with OUTPUT_PATHS[1].open("wb") as target:
        dill.dump(dict(molecule), target)
    with OUTPUT_PATHS[2].open("w", encoding="utf-8") as target:
        json.dump({"med_vocab_size": len(voc["med_voc"].idx2word), "ddi_mask_shape": list(ddi_mask_h.shape)}, target, indent=2)


if __name__ == "__main__":
    main()
