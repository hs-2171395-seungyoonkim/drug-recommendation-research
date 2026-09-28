"""Build a chronologically ordered copy of the SafeDrug MIMIC-III records.

SafeDrug's records_final.pkl keeps each patient's visits in HADM_ID order, which
is not chronological (50% of transitions go backwards in time). This script
re-orders the visits of each patient by (ADMITTIME, HADM_ID) using the aligned
sidecar the unfair track reconstructed, and leaves everything else untouched:

- patient order is unchanged, so SafeDrug's 2/3 : 1/6 : 1/6 patient split is
  identical;
- each visit's [diag, proc, med] index lists are the same objects, so
  voc_final.pkl, ddi_A_final.pkl, ddi_mask_H.pkl and ehr_adj_final.pkl
  (within-visit co-occurrence) stay valid.

Integrity checks before writing:
1. sidecar row count / patient count match records_final.pkl;
2. sidecar HADM_IDs equal the unfair track's records_final_hadm_ids.pkl;
3. for every visit, the diagnosis codes in records (via voc) are a subset of
   DIAGNOSES_ICD rows for that HADM_ID (records keep only the 2,000 most
   frequent codes, so subset is the right test);
4. after re-ordering, no patient has a backwards transition and each patient's
   multiset of visits is unchanged.

Outputs (row-level, MIMIC-derived, never committed):
    out/mimic3/records_final_chrono.pkl
    out/mimic3/records_final_chrono_hadm_ids.pkl
    out/mimic3/records_final_chrono_visits.csv
Aggregate manifest (safe to keep):
    results/mimic3_chrono_records_manifest.json

Run:  py -3.12 scripts/build_mimic3_chrono_records.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
RECORDS = SAFEDRUG_DATA / "records_final.pkl"
VOC = SAFEDRUG_DATA / "voc_final.pkl"
SIDECAR_DIR = Path(r"C:\Users\Administrator\Desktop\unfair\out\acute_driver_audit\safedrug_mimic3_cohort")
MASTER = SIDECAR_DIR / "master_visits.csv"
HADM_SIDECAR = SIDECAR_DIR / "records_final_hadm_ids.pkl"
DXICD = Path(r"C:\Users\Administrator\Desktop\unfair\DIAGNOSES_ICD.csv")
OUT_DIR = ROOT / "out" / "mimic3"
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chronological_order(admit_times, hadm_ids) -> list[int]:
    """Indices that sort one patient's visits by (ADMITTIME, HADM_ID)."""
    keys = list(zip(admit_times, hadm_ids))
    return sorted(range(len(keys)), key=lambda i: keys[i])


def reorder_records(records: list, master: pd.DataFrame) -> tuple[list, list, pd.DataFrame]:
    """Returns (new_records, new_hadm_ids, visit_table)."""
    by_patient = {p: g for p, g in master.groupby("safedrug_patient_index", sort=False)}
    new_records, new_hadm, rows = [], [], []
    for pi, patient in enumerate(records):
        g = by_patient[pi].sort_values("safedrug_visit_index")
        if len(g) != len(patient) or list(g["safedrug_visit_index"]) != list(range(len(patient))):
            raise AssertionError(f"patient {pi}: sidecar has {len(g)} visits, records has {len(patient)}")
        order = chronological_order(list(g["ADMITTIME"]), list(g["HADM_ID"]))
        new_records.append([patient[i] for i in order])
        new_hadm.append([int(g["HADM_ID"].iloc[i]) for i in order])
        for new_pos, old_pos in enumerate(order):
            rows.append({"SUBJECT_ID": int(g["SUBJECT_ID"].iloc[old_pos]), "HADM_ID": int(g["HADM_ID"].iloc[old_pos]),
                         "patient_index": pi, "old_visit_index": old_pos, "new_visit_index": new_pos,
                         "ADMITTIME": g["ADMITTIME"].iloc[old_pos]})
    return new_records, new_hadm, pd.DataFrame(rows)


def backwards_transition_share(records_hadm: list, admit_of: dict) -> float:
    n = back = 0
    for visits in records_hadm:
        for a, b in zip(visits[:-1], visits[1:]):
            n += 1
            back += admit_of[b] < admit_of[a]
    return back / n if n else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_records = OUT_DIR / "records_final_chrono.pkl"
    if out_records.exists():
        raise FileExistsError(f"refusing to overwrite {out_records}; delete it to rebuild")

    with RECORDS.open("rb") as f:
        records = dill.load(f)
    with VOC.open("rb") as f:
        voc = dill.load(f)
    dxw = voc["diag_voc"].idx2word
    master = pd.read_csv(MASTER, parse_dates=["ADMITTIME"])
    with HADM_SIDECAR.open("rb") as f:
        hadm_sidecar = dill.load(f)

    n_visits = sum(len(p) for p in records)
    checks = {}
    # 1. counts
    checks["sidecar_rows_match_visits"] = bool(len(master) == n_visits)
    checks["sidecar_patients_match"] = bool(master["safedrug_patient_index"].nunique() == len(records))
    # 2. sidecar HADM vs the pkl sidecar
    m_sorted = master.sort_values(["safedrug_patient_index", "safedrug_visit_index"])
    sidecar_hadm = [list(g["HADM_ID"].astype(int)) for _, g in m_sorted.groupby("safedrug_patient_index", sort=True)]
    checks["master_csv_equals_hadm_pkl"] = bool(sidecar_hadm == [[int(h) for h in p] for p in hadm_sidecar])
    if not all(checks.values()):
        raise AssertionError(f"sidecar alignment failed: {checks}")

    # 3. diagnosis subset check for every visit
    dx = pd.read_csv(DXICD, usecols=["HADM_ID", "ICD9_CODE"], dtype={"ICD9_CODE": str}).dropna()
    dx_by_hadm = dx.groupby("HADM_ID")["ICD9_CODE"].agg(set).to_dict()
    mismatches, empty_hadm = [], 0
    for pi, patient in enumerate(records):
        for vi, visit in enumerate(patient):
            h = sidecar_hadm[pi][vi]
            codes = {dxw[i] for i in visit[0]}
            raw = dx_by_hadm.get(h)
            if raw is None:
                empty_hadm += 1
                mismatches.append((pi, vi, h, "no DIAGNOSES_ICD rows"))
            elif not codes <= raw:
                mismatches.append((pi, vi, h, sorted(codes - raw)[:5]))
    checks["diag_subset_of_DIAGNOSES_ICD"] = {"n_visits_checked": n_visits, "n_mismatch": len(mismatches),
                                             "n_hadm_without_rows": empty_hadm}
    if mismatches:
        raise AssertionError(f"{len(mismatches)} visits whose diagnosis codes are not a subset of "
                             f"DIAGNOSES_ICD for their HADM_ID, e.g. {mismatches[:3]}")

    admit_of = dict(zip(master["HADM_ID"].astype(int), master["ADMITTIME"]))
    before = backwards_transition_share(sidecar_hadm, admit_of)
    new_records, new_hadm, table = reorder_records(records, master)
    after = backwards_transition_share(new_hadm, admit_of)
    checks["backwards_transition_share_before"] = before
    checks["backwards_transition_share_after"] = after
    if after != 0.0:
        raise AssertionError("re-ordered records still contain backwards transitions")
    # 4. same multiset of visits per patient (identity of the inner lists)
    same = all(sorted(map(id, a)) == sorted(map(id, b)) for a, b in zip(records, new_records))
    checks["visits_unchanged_per_patient"] = bool(same)
    changed_patients = sum(1 for a, b in zip(records, new_records) if [id(x) for x in a] != [id(x) for x in b])
    checks["patients_whose_order_changed"] = changed_patients
    checks["share_patients_order_changed"] = changed_patients / len(records)
    split_point = int(len(records) * 2 / 3)
    eval_len = int((len(records) - split_point) / 2)
    checks["split_unchanged"] = {"train": split_point, "test": eval_len, "eval": len(records) - split_point - eval_len,
                                 "note": "split is by patient position; patient order is untouched"}

    with out_records.open("wb") as f:
        dill.dump(new_records, f)
    with (OUT_DIR / "records_final_chrono_hadm_ids.pkl").open("wb") as f:
        dill.dump(new_hadm, f)
    table.to_csv(OUT_DIR / "records_final_chrono_visits.csv", index=False)

    # re-load and verify round trip
    with out_records.open("rb") as f:
        reloaded = dill.load(f)
    checks["round_trip_equal"] = bool(reloaded == new_records)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"records_final.pkl": {"path": str(RECORDS), "sha256": sha256(RECORDS)},
                   "voc_final.pkl": {"path": str(VOC), "sha256": sha256(VOC)},
                   "master_visits.csv": {"path": str(MASTER), "sha256": sha256(MASTER)},
                   "records_final_hadm_ids.pkl": {"path": str(HADM_SIDECAR), "sha256": sha256(HADM_SIDECAR)},
                   "DIAGNOSES_ICD.csv": {"path": str(DXICD), "sha256": sha256(DXICD)}},
        "outputs": {"records_final_chrono.pkl": {"path": str(out_records), "sha256": sha256(out_records)},
                    "records_final_chrono_hadm_ids.pkl": {"path": str(OUT_DIR / "records_final_chrono_hadm_ids.pkl"),
                                                          "sha256": sha256(OUT_DIR / "records_final_chrono_hadm_ids.pkl")},
                    "records_final_chrono_visits.csv": {"path": str(OUT_DIR / "records_final_chrono_visits.csv")}},
        "n_patients": len(records), "n_visits": n_visits, "n_transitions": n_visits - len(records),
        "ordering": "within patient by (ADMITTIME, HADM_ID); patient order unchanged",
        "checks": checks,
        "usage": "drop-in replacement for records_final.pkl; voc/ddi/ehr_adj files unchanged; "
                 "results are NOT comparable to literature numbers trained on the HADM_ID order",
    }
    (RESULTS / "mimic3_chrono_records_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("inputs", "outputs")}, indent=2, default=str))
    print(f"wrote {out_records} ({out_records.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
