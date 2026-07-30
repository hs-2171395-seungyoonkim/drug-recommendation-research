"""
One-time real-data run: builds data/mimic-iv/organ_function_features.pkl.
Run from the ServerityMed repo root:
  C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/build_organ_function_features.py
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from organ_function.admission_time import build_admission_times
from organ_function.feature_table import build_feature_table
from organ_function.lab_subset import extract_lab_subset
from organ_function.subject_lookup import build_hadm_to_subject

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("loading records_final2.pkl / records_final2_hadm_ids.pkl ...")
    with open(ROOT / "data/mimic-iv/records_final2.pkl", "rb") as f:
        records = pickle.load(f)
    with open(ROOT / "data/mimic-iv/records_final2_hadm_ids.pkl", "rb") as f:
        hadm_ids = pickle.load(f)
    print(f"  {len(records)} patients, {sum(len(p) for p in records)} visits")

    print("recovering subject_id per patient from diagnoses_icd.csv ...")
    diag_df = pd.read_csv(
        ROOT / "data/raw_mimic_iv/diagnoses_icd.csv", usecols=["subject_id", "hadm_id"]
    )
    hadm_to_subject = build_hadm_to_subject(diag_df)

    print("building proxy admission times from prescriptions.csv ...")
    presc_df = pd.read_csv(
        ROOT / "data/raw_mimic_iv/prescriptions.csv", usecols=["hadm_id", "starttime"]
    )
    admission_times = build_admission_times(presc_df)

    print("streaming labevents.csv.gz (this takes a few minutes) ...")
    tic = time.time()
    lab_subset = extract_lab_subset(str(ROOT / "data/raw_labs/labevents.csv.gz"))
    print(f"  extracted {len(lab_subset)} rows in {time.time() - tic:.1f}s")

    print("building feature table ...")
    tic = time.time()
    table = build_feature_table(records, hadm_ids, lab_subset, admission_times, hadm_to_subject)
    print(f"  built {sum(len(p) for p in table)} visit feature rows in {time.time() - tic:.1f}s")

    out_path = ROOT / "data/mimic-iv/organ_function_features.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(table, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
