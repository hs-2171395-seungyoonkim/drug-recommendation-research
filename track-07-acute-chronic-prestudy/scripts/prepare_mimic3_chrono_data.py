"""Assemble a SafeDrug data directory whose records are the chronological copy.

The track-04 training wrapper (unfair/.worktrees/admission-driver-consensus/
scripts/safedrug_train_dump.py) reads five pickles from one directory and the
HADM sidecar (master_visits.csv + records_final_hadm_ids.pkl) from another,
both as module constants. This builds one directory that serves both roles:

    out/mimic3_chrono_data/
        records_final.pkl            <- out/mimic3/records_final_chrono.pkl
        voc_final.pkl, ddi_A_final.pkl, ddi_mask_H.pkl, atc3toSMILES.pkl
                                     <- byte-identical copies of SOTA/SafeDrug/data/output
        master_visits.csv            <- visit index re-numbered to the chronological order
        records_final_hadm_ids.pkl   <- chronological HADM_ID lists (plain pickle)
        manifest.json

Row-level MIMIC-derived files: never committed.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import dill
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
ORIG_MASTER = Path(r"C:\Users\Administrator\Desktop\unfair\out\acute_driver_audit\safedrug_mimic3_cohort\master_visits.csv")
CHRONO_DIR = ROOT / "out" / "mimic3"
DATA_DIR = ROOT / "out" / "mimic3_chrono_data"
COPIED = ["voc_final.pkl", "ddi_A_final.pkl", "ddi_mask_H.pkl", "atc3toSMILES.pkl"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "files": {}}

    for name in COPIED:
        src, dst = SAFEDRUG_DATA / name, DATA_DIR / name
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise AssertionError(f"copy mismatch for {name}")
        manifest["files"][name] = {"source": str(src), "sha256": sha256(dst)}

    src = CHRONO_DIR / "records_final_chrono.pkl"
    dst = DATA_DIR / "records_final.pkl"
    shutil.copyfile(src, dst)
    manifest["files"]["records_final.pkl"] = {"source": str(src), "sha256": sha256(dst),
                                              "note": "chronological re-ordering, NOT the SafeDrug original"}

    # sidecars in the new visit order
    visits = pd.read_csv(CHRONO_DIR / "records_final_chrono_visits.csv", parse_dates=["ADMITTIME"])
    orig = pd.read_csv(ORIG_MASTER, usecols=["HADM_ID", "ADMISSION_TYPE"])
    master = visits.merge(orig, on="HADM_ID", how="left")
    master = master.rename(columns={"patient_index": "safedrug_patient_index", "new_visit_index": "safedrug_visit_index"})
    master = master[["SUBJECT_ID", "HADM_ID", "safedrug_patient_index", "safedrug_visit_index", "ADMITTIME", "ADMISSION_TYPE", "old_visit_index"]]
    master = master.sort_values(["safedrug_patient_index", "safedrug_visit_index"]).reset_index(drop=True)
    master.to_csv(DATA_DIR / "master_visits.csv", index=False)

    with (CHRONO_DIR / "records_final_chrono_hadm_ids.pkl").open("rb") as f:
        hadm = dill.load(f)
    with (DATA_DIR / "records_final_hadm_ids.pkl").open("wb") as f:
        pickle.dump(hadm, f)

    # cross-check: master csv order == hadm pkl order == records order
    with dst.open("rb") as f:
        records = dill.load(f)
    by_p = master.groupby("safedrug_patient_index", sort=True)
    for pi, g in by_p:
        if list(g["safedrug_visit_index"]) != list(range(len(records[pi]))):
            raise AssertionError(f"patient {pi}: visit index gap in master_visits.csv")
        if list(g["HADM_ID"].astype(int)) != [int(h) for h in hadm[pi]]:
            raise AssertionError(f"patient {pi}: master_visits.csv and hadm pkl disagree")
        if not g["ADMITTIME"].is_monotonic_increasing:
            raise AssertionError(f"patient {pi}: not chronological")
    manifest["files"]["master_visits.csv"] = {"rows": int(len(master)), "sha256": sha256(DATA_DIR / "master_visits.csv")}
    manifest["files"]["records_final_hadm_ids.pkl"] = {"sha256": sha256(DATA_DIR / "records_final_hadm_ids.pkl")}
    manifest["n_patients"] = len(records)
    manifest["n_visits"] = int(len(master))
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
