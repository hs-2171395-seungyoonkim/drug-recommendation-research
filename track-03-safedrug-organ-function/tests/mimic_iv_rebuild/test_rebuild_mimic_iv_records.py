import importlib.util
import hashlib
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rebuild_mimic_iv_records.py"
SPEC = importlib.util.spec_from_file_location("rebuild_mimic_iv_records", SCRIPT_PATH)
rebuild = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rebuild)


def test_rebuild_outputs_are_new_final4_artifacts():
    assert [path.name for path in rebuild.OUTPUT_PATHS] == [
        "records_final4.pkl",
        "records_final4_hadm_ids.pkl",
        "voc_final4.pkl",
        "ddi_A_final4.pkl",
        "records_final4.meta.json",
    ]


def test_sha256_hashes_file_content(tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"admission-time-input")

    assert rebuild._sha256(path) == hashlib.sha256(b"admission-time-input").hexdigest()
