from safedrug.final4_assets import build_final4_assets
from pathlib import Path
import importlib.util


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_safedrug_final4_assets.py"


def test_build_final4_assets_preserves_every_medication_code():
    med_voc_idx2word = {0: "A01A", 1: "B01A"}
    idx2drug = {"A01A": {"CCO"}}

    molecule, ddi_mask_h = build_final4_assets(idx2drug, med_voc_idx2word)

    assert set(molecule) >= {"A01A", "B01A"}
    assert ddi_mask_h.shape[0] == 2


def test_final4_asset_script_uses_isolated_final4_output_paths():
    spec = importlib.util.spec_from_file_location("build_safedrug_final4_assets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert all("final4" in path.name for path in module.OUTPUT_PATHS)
