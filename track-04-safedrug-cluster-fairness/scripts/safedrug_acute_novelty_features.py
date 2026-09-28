"""SafeDrug acute-attention novelty key features (R2).

Novelty is computed over TRUE chronology (ADMITTIME, from ADMISSIONS.csv,
joined by HADM_ID; ties broken by records order), not over records_final's
own storage order. SafeDrug's own preprocessing orders each patient's visits
by ascending HADM_ID, not ADMITTIME (verified 2026-09-07: this disagrees
with true chronology for a large share of adjacent visit pairs -- see the
`chronology` block in novelty_meta.json and design doc S2.4). The model only
ever sees records_final's own order, so the OUTPUT arrays stay indexed and
positionally aligned by that records order -- but which visits count as
"earlier" for accumulating a patient's code history is always resolved by
ADMITTIME, never by records order.

See docs/superpowers/specs/2026-09-07-safedrug-acute-attention-design.md
("R2") for the full contract.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import dill
import numpy as np
import pandas as pd

DEFAULT_ADMISSIONS_CSV = r"C:\Users\Administrator\Desktop\unfair\ADMISSIONS.csv"


def compute_novelty_for_patient(diag_lists: list[list[int]], proc_lists: list[list[int]], admittimes=None):
    """diag_lists / proc_lists: one list of code-indices per visit, indexed
    by records_final's own storage position (position 0 = however that
    visit is stored, not necessarily chronologically first). admittimes:
    optional list of the same length giving each visit's ADMITTIME (or any
    orderable timestamp); when omitted, records order is treated as
    chronological order (used only where there is no ADMITTIME disagreement
    to model -- callers with real data always pass admittimes).

    Visits are processed in ascending (ADMITTIME, records-position) order --
    i.e. true chronology, ties broken by records order. Each visit
    accumulates the union of all *strictly chronologically earlier* visits'
    own codes (same modality only: diag history never contaminates proc,
    and vice versa).

    Returns (diag_novelty, proc_novelty), each a list INDEXED BY RECORDS
    ORDER position (matching the input lists' own indexing) and positionally
    aligned with that visit's own code list. The chronologically-first
    visit's codes are all 0 ("no history"); a later visit's code is 1 ("seen
    before") if it appears in the union of its strictly-chronologically-
    earlier visits, else 2 ("new")."""
    n = len(diag_lists)
    if admittimes is None:
        chrono_order = list(range(n))
    else:
        chrono_order = sorted(range(n), key=lambda i: (admittimes[i], i))

    diag_out: list[list[int] | None] = [None] * n
    proc_out: list[list[int] | None] = [None] * n
    seen_diag: set[int] = set()
    seen_proc: set[int] = set()
    for rank, i in enumerate(chrono_order):
        dcodes = diag_lists[i]
        pcodes = proc_lists[i]
        if rank == 0:
            diag_out[i] = [0] * len(dcodes)
            proc_out[i] = [0] * len(pcodes)
        else:
            diag_out[i] = [1 if c in seen_diag else 2 for c in dcodes]
            proc_out[i] = [1 if c in seen_proc else 2 for c in pcodes]
        seen_diag.update(dcodes)
        seen_proc.update(pcodes)
    return diag_out, proc_out


def split_boundaries(n_patients: int) -> tuple[int, int]:
    """SafeDrug.py's own train/test/eval patient-index split boundaries
    (scripts/safedrug_train_dump.py's split_data, duplicated here rather
    than imported -- see design doc R9). Returns (split_point, eval_len):
    patients [0, split_point) are train, [split_point, split_point+eval_len)
    are test, [split_point+eval_len, n_patients) are eval."""
    split_point = int(n_patients * 2 / 3)
    eval_len = int((n_patients - split_point) / 2)
    return split_point, eval_len


def split_of_patient(patient_index: int, split_point: int, eval_len: int) -> str:
    if patient_index < split_point:
        return "train"
    if patient_index < split_point + eval_len:
        return "test"
    return "eval"


def admittime_or_raise(admittime_by_hadm: dict, hadm_id) -> "pd.Timestamp":
    """`admittime_by_hadm[hadm_id]`, but with an error message that names the
    missing HADM_ID and its source, instead of the bare `KeyError(<hadm_id>)`
    a raw dict subscript would raise."""
    try:
        return admittime_by_hadm[hadm_id]
    except KeyError as exc:
        raise KeyError(
            f"HADM_ID {hadm_id} has no ADMITTIME entry (admittime_by_hadm lookup "
            "failed) -- check that this HADM_ID is present in the ADMISSIONS.csv "
            "used to build the lookup"
        ) from exc


def count_chronology_violations(records_final_hadm_ids, master_visits_df: pd.DataFrame) -> dict:
    """For each patient (records_final's own list index), compares
    consecutive visits' ADMITTIME (looked up via master_visits_df, keyed by
    HADM_ID) against records order. Returns counts only -- purely
    diagnostic; the actual novelty computation (compute_novelty_for_patient,
    called from main() with ADMITTIME-resolved chronology) never uses this
    function's output to reorder anything."""
    admittime_by_hadm = dict(zip(master_visits_df["HADM_ID"], pd.to_datetime(master_visits_df["ADMITTIME"])))
    total_pairs = 0
    violations = 0
    violating_patients = 0
    for patient_hadms in records_final_hadm_ids:
        pair_violation = False
        for i in range(len(patient_hadms) - 1):
            t0 = admittime_or_raise(admittime_by_hadm, patient_hadms[i])
            t1 = admittime_or_raise(admittime_by_hadm, patient_hadms[i + 1])
            total_pairs += 1
            if t0 > t1:
                violations += 1
                pair_violation = True
        if pair_violation:
            violating_patients += 1
    return {
        "total_adjacent_pairs": total_pairs,
        "violations": violations,
        "violating_patients": violating_patients,
        "n_patients": len(records_final_hadm_ids),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug acute-attention novelty features (R2)")
    parser.add_argument(
        "--records-final", type=str,
        default=r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\records_final.pkl",
    )
    parser.add_argument(
        "--hadm-ids-pkl", type=str,
        default="out/acute_driver_audit/safedrug_mimic3_cohort/records_final_hadm_ids.pkl",
    )
    parser.add_argument(
        "--master-visits-csv", type=str,
        default="out/acute_driver_audit/safedrug_mimic3_cohort/master_visits.csv",
    )
    parser.add_argument(
        "--admissions-csv", type=str,
        default=DEFAULT_ADMISSIONS_CSV,
        help="ADMISSIONS.csv used to resolve TRUE ADMITTIME chronology for novelty "
        "(joined by HADM_ID; ties broken by records order). Distinct from "
        "--master-visits-csv, which is used only for the diagnostic chronology "
        "violation counts.",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_final = dill.load(open(args.records_final, "rb"))
    with open(args.hadm_ids_pkl, "rb") as fh:
        hadm_ids = dill.load(fh, encoding="latin1")
    if len(records_final) != len(hadm_ids):
        raise ValueError(f"records_final has {len(records_final)} patients but hadm_ids has {len(hadm_ids)}")

    master = pd.read_csv(args.master_visits_csv)
    chrono = count_chronology_violations(hadm_ids, master)
    split_point, eval_len = split_boundaries(len(records_final))

    admissions = pd.read_csv(args.admissions_csv, usecols=["HADM_ID", "ADMITTIME"])
    admittime_by_hadm = dict(zip(admissions["HADM_ID"], pd.to_datetime(admissions["ADMITTIME"])))

    novelty = {}
    split_counts = {"train": {0: 0, 1: 0, 2: 0}, "test": {0: 0, 1: 0, 2: 0}, "eval": {0: 0, 1: 0, 2: 0}}
    for patient_index, visits in enumerate(records_final):
        diag_lists = [v[0] for v in visits]
        proc_lists = [v[1] for v in visits]
        patient_hadms = hadm_ids[patient_index]
        admittimes = [admittime_or_raise(admittime_by_hadm, h) for h in patient_hadms]
        diag_novelty, proc_novelty = compute_novelty_for_patient(diag_lists, proc_lists, admittimes=admittimes)
        split_name = split_of_patient(patient_index, split_point, eval_len)
        for visit_index in range(len(visits)):
            hadm_id = patient_hadms[visit_index]
            d_arr = np.array(diag_novelty[visit_index], dtype=np.int8)
            p_arr = np.array(proc_novelty[visit_index], dtype=np.int8)
            novelty[hadm_id] = {"diag": d_arr, "proc": p_arr}
            for state in d_arr.tolist() + p_arr.tolist():
                split_counts[split_name][state] += 1

    with open(out_dir / "novelty.pkl", "wb") as fh:
        pickle.dump(novelty, fh)

    def share(counts):
        total = sum(counts.values())
        return {
            "no_history": counts[0] / total if total else 0.0,
            "seen_before": counts[1] / total if total else 0.0,
            "new": counts[2] / total if total else 0.0,
            "n_codes": total,
        }

    meta = {
        "n_patients": len(records_final),
        "n_visits": sum(len(v) for v in records_final),
        "novelty_order": "ADMITTIME",
        "chronology": chrono,
        "code_state_share_by_split": {k: share(v) for k, v in split_counts.items()},
    }
    (out_dir / "novelty_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] wrote novelty.pkl for {len(novelty)} visits to {out_dir}")


if __name__ == "__main__":
    main()
