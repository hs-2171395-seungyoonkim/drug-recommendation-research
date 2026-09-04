"""
One-time real-data run: builds data/mimic-iv/organ_function_features.pkl and a
git-committable provenance sidecar data/mimic-iv/organ_function_features.meta.json
(the .pkl itself is gitignored under the PhysioNet DUA, so the JSON is the only
record of this run that survives in git).
Run from the ServerityMed repo root:
  C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/build_organ_function_features.py
"""
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from organ_function.feature_table import build_feature_table
from organ_function.lab_config import LAB_ITEMIDS, LAB_NAMES
from organ_function.lab_subset import extract_lab_subset
from organ_function.subject_lookup import build_hadm_to_subject

ROOT = Path(__file__).resolve().parent.parent

RECORDS_PATH = ROOT / "data/mimic-iv/records_final4.pkl"
HADM_IDS_PATH = ROOT / "data/mimic-iv/records_final4_hadm_ids.pkl"
DIAGNOSES_PATH = ROOT / "data/raw_mimic_iv/diagnoses_icd.csv"
ADMISSIONS_PATH = ROOT / "data/raw_mimic_iv/admissions.csv.gz"
LABEVENTS_PATH = ROOT / "data/raw_labs/labevents.csv.gz"

INPUT_PATHS = [
    RECORDS_PATH,
    HADM_IDS_PATH,
    DIAGNOSES_PATH,
    ADMISSIONS_PATH,
    LABEVENTS_PATH,
]
OUTPUT_PATH = ROOT / "data/mimic-iv/organ_function_features_final4.pkl"
META_PATH = ROOT / "data/mimic-iv/organ_function_features_final4.meta.json"


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance must not kill the run
        return f"unavailable: {exc}"


def input_file_stats() -> list:
    stats = []
    for path in INPUT_PATHS:
        st = path.stat()
        stats.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        })
    return stats


def per_lab_missing_rates(table: list) -> dict:
    """Fraction of visits with {name}_missing == True, per lab."""
    counts = {name: 0 for name in LAB_NAMES}
    total = 0
    for patient in table:
        for visit in patient:
            total += 1
            for name in LAB_NAMES:
                if visit[f"{name}_missing"]:
                    counts[name] += 1
    if total == 0:
        return {name: None for name in LAB_NAMES}
    return {name: counts[name] / total for name in LAB_NAMES}


def main():
    run_started = time.time()
    if OUTPUT_PATH.exists() or META_PATH.exists():
        raise FileExistsError("refusing to overwrite existing final4 organ-function artifacts")

    print("loading records_final4.pkl / records_final4_hadm_ids.pkl ...")
    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)
    with open(HADM_IDS_PATH, "rb") as f:
        hadm_ids = pickle.load(f)
    patient_count = len(records)
    visit_count = sum(len(p) for p in records)
    print(f"  {patient_count} patients, {visit_count} visits")

    print("recovering subject_id per patient from diagnoses_icd.csv ...")
    diag_df = pd.read_csv(DIAGNOSES_PATH, usecols=["subject_id", "hadm_id"])
    hadm_to_subject = build_hadm_to_subject(diag_df)

    print("loading official admission times ...")
    admissions = pd.read_csv(ADMISSIONS_PATH, usecols=["hadm_id", "admittime"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    if admissions["hadm_id"].duplicated().any() or admissions["admittime"].isna().any():
        raise ValueError("official admissions contains duplicate or invalid admission times")
    admission_times = admissions.set_index("hadm_id")["admittime"]

    print("streaming labevents.csv.gz (this takes a few minutes) ...")
    tic = time.time()
    lab_subset = extract_lab_subset(str(LABEVENTS_PATH))
    extract_seconds = time.time() - tic
    lab_row_count = len(lab_subset)
    print(f"  extracted {lab_row_count} rows in {extract_seconds:.1f}s")

    print("building feature table ...")
    tic = time.time()
    table = build_feature_table(records, hadm_ids, lab_subset, admission_times, hadm_to_subject)
    build_seconds = time.time() - tic
    built_visits = sum(len(p) for p in table)
    print(f"  built {built_visits} visit feature rows in {build_seconds:.1f}s")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(table, f)
    print(f"wrote {OUTPUT_PATH}")

    print("computing per-lab missing rates ...")
    missing_rates = per_lab_missing_rates(table)

    total_seconds = time.time() - run_started
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "script": "scripts/build_organ_function_features.py",
        "output": "data/mimic-iv/organ_function_features_final4.pkl",
        "inputs": input_file_stats(),
        "patient_count": patient_count,
        "visit_count": built_visits,
        "extracted_lab_row_count": lab_row_count,
        "feature_keys_per_visit": len(table[0][0]) if table and table[0] else 0,
        "per_lab_missing_rate": missing_rates,
        "lab_itemids": LAB_ITEMIDS,
        "runtime_seconds": {
            "lab_extract": round(extract_seconds, 1),
            "feature_table_build": round(build_seconds, 1),
            "total": round(total_seconds, 1),
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {META_PATH}")

    print("\n--- sanity check ---")
    print(f"patients: {len(table)}  visits: {built_visits}")
    print(f"keys per visit dict: {len(table[0][0])}")
    print("\nper-lab missing rate:")
    for name, rate in missing_rates.items():
        print(f"  {name}: {rate:.4f}")
    print(f"\ntotal runtime: {total_seconds:.1f}s")


if __name__ == "__main__":
    main()
