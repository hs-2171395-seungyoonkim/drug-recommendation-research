"""
One-time real-data run: rebuilds records_final3.pkl, voc_final3.pkl,
records_final3_hadm_ids.pkl, and ddi_A_final3.pkl - a corrected version of
records_final2.pkl/voc_final2.pkl/records_final2_hadm_ids.pkl/ddi_A_final2.pkl
that includes BOTH ICD-9 and ICD-10 diagnoses/procedures (tagged by
version) instead of silently dropping all ICD-10 admissions (35.6% of all
MIMIC-IV admissions - see plan Global Constraints).

Writes to NEW filenames ("final3") - never overwrites records_final2.pkl
etc, which are hardlinked to Desktop/HI-DR's own copy (confirmed via
`stat`: same inode, Links: 2) - overwriting them in place would corrupt
HI-DR's project data too. Enforced in code below (not just by convention):
every output path is asserted to not reference "final2" before it's opened.

NOTE on voc_final3.pkl: unlike voc_final2.pkl (dumped from `__main__`, so
dill embedded the Voc class definition by value and it loads with plain
dill anywhere), voc_final3.pkl's Voc class lives in the importable
mimic_iv_rebuild.vocab module, so dill stores it by reference. Any
consumer must put this repo's `src/` on sys.path (exactly like this
script's own sys.path.insert call below) BEFORE calling dill.load() - dill
then imports mimic_iv_rebuild.vocab automatically to reconstruct the
object, no explicit import needed by the caller. This file is intended for
ServerityMed's own future scripts only (which already all follow this
convention), not for cross-project consumption by Desktop/HI-DR.

Run: C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/rebuild_mimic_iv_records.py
"""
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dill

from mimic_iv_rebuild.ddi_matrix import build_ddi_adjacency
from mimic_iv_rebuild.icd_tagging import load_and_tag_diagnoses, load_and_tag_procedures
from mimic_iv_rebuild.medication_mapping import (
    build_medication_table,
    load_ndc2rxcui,
    load_rxcui2atc3,
)
from mimic_iv_rebuild.records import build_records_and_hadm_ids, combine_admissions
from mimic_iv_rebuild.vocab import build_vocab_from_column

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATHS = [
    ROOT / "data/mimic-iv/records_final3.pkl",
    ROOT / "data/mimic-iv/records_final3_hadm_ids.pkl",
    ROOT / "data/mimic-iv/voc_final3.pkl",
    ROOT / "data/mimic-iv/ddi_A_final3.pkl",
    ROOT / "data/mimic-iv/records_final3.meta.json",
]


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    for p in OUTPUT_PATHS:
        assert "final2" not in str(p), f"refusing to write to a final2-named path: {p}"

    tic = time.time()
    print("loading and tagging diagnoses/procedures (ICD-9+10)...")
    diag_df = load_and_tag_diagnoses(str(ROOT / "data/raw_mimic_iv/diagnoses_icd.csv"), top_k=2000)
    # top_k=None: no frequency cutoff, matching the original SafeDrug/DrugRec
    # lineage (filter_1000_most_pro is defined but never called upstream -
    # see icd_tagging.load_and_tag_procedures's docstring). An earlier run
    # incorrectly passed top_k=1000 here, which cost 2,317 admissions their
    # entire procedure list.
    proc_df = load_and_tag_procedures(str(ROOT / "data/raw_mimic_iv/procedures_icd.csv"), top_k=None)
    print(f"  diag rows: {len(diag_df)}, proc rows: {len(proc_df)}")

    print("mapping medications NDC->ATC3...")
    ndc2rxcui = load_ndc2rxcui(str(ROOT / "data/mappings/ndc2rxnorm_mapping.txt"))
    rxcui2atc3 = load_rxcui2atc3(str(ROOT / "data/mappings/ndc2atc_level4.csv"))
    med_df = build_medication_table(
        str(ROOT / "data/raw_mimic_iv/prescriptions.csv"), ndc2rxcui, rxcui2atc3
    )
    print(f"  med rows: {len(med_df)}")

    print("combining admissions (inner join + >=2 visits filter)...")
    table = combine_admissions(diag_df, proc_df, med_df)
    print(f"  surviving admissions: {len(table)}, patients: {table['subject_id'].nunique()}")

    print("building vocabularies...")
    diag_voc = build_vocab_from_column(table["diag_codes"])
    pro_voc = build_vocab_from_column(table["proc_codes"])
    med_voc = build_vocab_from_column(table["med_codes"])
    print(f"  diag_voc: {len(diag_voc.idx2word)}, pro_voc: {len(pro_voc.idx2word)}, med_voc: {len(med_voc.idx2word)}")

    print("building records and hadm_ids...")
    records, hadm_ids = build_records_and_hadm_ids(table, diag_voc, pro_voc, med_voc)

    print("building DDI adjacency matrix from real TWOSIDES data...")
    ddi_adj = build_ddi_adjacency(
        med_voc.idx2word,
        str(ROOT / "data/mappings/drug-DDI.csv"),
        str(ROOT / "data/mappings/drug-atc.csv"),
        top_k=40,
    )
    print(f"  ddi_adj shape: {ddi_adj.shape}, known interacting pairs: {int(ddi_adj.sum() / 2)}")

    # --- statistics, compared against the buggy final2 files and published SafeDrug (MIMIC-III) numbers ---
    n_patients = len(records)
    n_visits = sum(len(p) for p in records)
    avg_visits = n_visits / n_patients
    avg_diag = sum(len(v[0]) for p in records for v in p) / n_visits
    avg_proc = sum(len(v[1]) for p in records for v in p) / n_visits
    avg_med = sum(len(v[2]) for p in records for v in p) / n_visits
    print("\n=== final3 (corrected, ICD-9+10) statistics ===")
    print(f"patients: {n_patients}, visits: {n_visits}, avg visits/patient: {avg_visits:.2f}")
    print(f"avg diag/visit: {avg_diag:.2f}, avg proc/visit: {avg_proc:.2f}, avg med/visit: {avg_med:.2f}")
    print("(compare against final2: 29518 patients, 95951 visits, 11.50 diag/visit, 2.53 proc/visit, 14.32 med/visit)")
    print("(compare against published SafeDrug MIMIC-III: 6350 patients, 15032 visits, 13.63 diag/visit, 4.54 proc/visit, 19.57 med/visit)")

    elapsed = time.time() - tic
    print(f"\ntotal time: {elapsed:.1f}s")

    with open(ROOT / "data/mimic-iv/records_final3.pkl", "wb") as f:
        dill.dump(records, f)
    with open(ROOT / "data/mimic-iv/records_final3_hadm_ids.pkl", "wb") as f:
        pickle.dump(hadm_ids, f)
    with open(ROOT / "data/mimic-iv/voc_final3.pkl", "wb") as f:
        dill.dump({"diag_voc": diag_voc, "pro_voc": pro_voc, "med_voc": med_voc}, f)
    with open(ROOT / "data/mimic-iv/ddi_A_final3.pkl", "wb") as f:
        dill.dump(ddi_adj, f)

    meta = {
        "git_commit_sha": _git_commit_sha(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "inputs": {
            "diagnoses_icd_rows": len(diag_df),
            "procedures_icd_rows": len(proc_df),
            "medications_rows": len(med_df),
            "diag_top_k": 2000,
            "proc_top_k": None,
        },
        "outputs": {
            "n_patients": n_patients,
            "n_visits": n_visits,
            "avg_visits_per_patient": round(avg_visits, 4),
            "avg_diag_per_visit": round(avg_diag, 4),
            "avg_proc_per_visit": round(avg_proc, 4),
            "avg_med_per_visit": round(avg_med, 4),
            "diag_voc_size": len(diag_voc.idx2word),
            "pro_voc_size": len(pro_voc.idx2word),
            "med_voc_size": len(med_voc.idx2word),
            "ddi_adj_shape": list(ddi_adj.shape),
            "ddi_known_pairs": int(ddi_adj.sum() / 2),
        },
    }
    with open(ROOT / "data/mimic-iv/records_final3.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("wrote records_final3.pkl, records_final3_hadm_ids.pkl, voc_final3.pkl, ddi_A_final3.pkl, records_final3.meta.json")


if __name__ == "__main__":
    main()
