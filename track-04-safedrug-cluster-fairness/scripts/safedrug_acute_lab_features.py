"""SafeDrug acute-attention first-24h lab query features (R3).

Gated at the caller level (Task E), not inside this script: run only when
--labevents resolves to an existing file. LABEVENTS.csv.gz and
D_LABITEMS.csv are present on this machine at
C:\\Users\\Administrator\\Desktop\\MIMIC-III_v1.4\\ (27,854,055 rows,
gzip-compressed) -- read here in chunks via pandas (chunksize ~2,000,000,
usecols restricted to the columns this script needs, compression inferred
from the .gz extension) so peak memory stays well under 8 GB. Rows are
filtered down to the fixed 30-item lab panel and this cohort's HADM_IDs
before any window / bin logic runs.

See docs/superpowers/specs/2026-09-07-safedrug-acute-attention-design.md
("R3") for the full contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import numpy as np
import pandas as pd

LAB_PANEL = {
    50912: "creatinine", 51006: "bun", 50983: "sodium", 50971: "potassium",
    50882: "bicarbonate", 50902: "chloride", 50931: "glucose", 50868: "anion_gap",
    50893: "calcium", 50960: "magnesium", 50970: "phosphate", 51301: "wbc",
    51222: "hemoglobin", 51221: "hematocrit", 51265: "platelets", 51237: "inr",
    51274: "pt", 51275: "ptt", 50813: "lactate", 51003: "troponin_t",
    50910: "ck", 50911: "ck_mb", 50885: "bilirubin_total", 50861: "alt",
    50878: "ast", 50863: "alkaline_phosphatase", 50862: "albumin",
    50820: "ph", 50818: "pco2", 50821: "po2",
}


def split_boundaries(n_patients: int) -> tuple[int, int]:
    """Duplicated from safedrug_acute_novelty_features.py (see design doc
    R9) -- SafeDrug.py's own train/test/eval patient-index split."""
    split_point = int(n_patients * 2 / 3)
    eval_len = int((n_patients - split_point) / 2)
    return split_point, eval_len


def first_value_in_window(rows: pd.DataFrame, admittime, window_before_hours: float = 6, window_after_hours: float = 24):
    """rows: LABEVENTS rows already filtered to one (HADM_ID, ITEMID) pair,
    with CHARTTIME (datetime) and VALUENUM columns. Returns the VALUENUM of
    the earliest CHARTTIME within [admittime-window_before_hours,
    admittime+window_after_hours], ignoring null VALUENUM rows. None if no
    row qualifies -- never a falsy 0.0 confused with "missing"."""
    lo = admittime - pd.Timedelta(hours=window_before_hours)
    hi = admittime + pd.Timedelta(hours=window_after_hours)
    window = rows[(rows["CHARTTIME"] >= lo) & (rows["CHARTTIME"] <= hi) & rows["VALUENUM"].notna()]
    if window.empty:
        return None
    earliest = window.sort_values("CHARTTIME").iloc[0]
    return float(earliest["VALUENUM"])


def compute_quantile_edges(values, n_bins: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("cannot compute quantile edges from zero values")
    quantiles = [i / n_bins for i in range(1, n_bins)]
    return np.quantile(values, quantiles).astype(np.float32)


def bin_value(value, edges, missing_bin: int) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing_bin
    return int(np.searchsorted(edges, value, side="right"))


def load_labevents_for_cohort(labevents_path, itemids: set[int], hadm_ids: set[int], chunksize: int = 2_000_000) -> pd.DataFrame:
    """Streams LABEVENTS.csv(.gz) in chunks (compression inferred from the
    file extension), keeping only rows for the fixed 30-item panel and this
    cohort's HADM_IDs -- the real table is ~27.8M rows; this keeps peak
    memory bounded to one chunk plus the (small) filtered accumulator."""
    usecols = ["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
    kept = []
    reader = pd.read_csv(labevents_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk = chunk[chunk["ITEMID"].isin(itemids) & chunk["HADM_ID"].isin(hadm_ids)]
        if not chunk.empty:
            kept.append(chunk)
    if not kept:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(kept, ignore_index=True)
    out["CHARTTIME"] = pd.to_datetime(out["CHARTTIME"])
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug acute-attention lab query features (R3)")
    parser.add_argument("--labevents", type=str, required=True)
    parser.add_argument("--d-labitems", type=str, required=True)
    parser.add_argument("--admissions", type=str, default="ADMISSIONS.csv")
    parser.add_argument(
        "--hadm-ids-pkl", type=str,
        default="out/acute_driver_audit/safedrug_mimic3_cohort/records_final_hadm_ids.pkl",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.labevents).exists():
        raise FileNotFoundError(
            f"{args.labevents} does not exist -- R3 is gated: only invoke this "
            "script when LABEVENTS.csv(.gz) is actually present on this machine."
        )

    d_labitems = pd.read_csv(args.d_labitems)
    missing_items = sorted(set(LAB_PANEL) - set(d_labitems["ITEMID"]))
    if missing_items:
        print(f"WARNING: {len(missing_items)} panel ITEMIDs not found in {args.d_labitems}: {missing_items}")

    with open(args.hadm_ids_pkl, "rb") as fh:
        hadm_ids_nested = dill.load(fh, encoding="latin1")
    cohort_hadm_ids = [h for patient in hadm_ids_nested for h in patient]
    split_point, eval_len = split_boundaries(len(hadm_ids_nested))

    admissions = pd.read_csv(args.admissions, usecols=["HADM_ID", "ADMITTIME"])
    admittime_by_hadm = dict(zip(admissions["HADM_ID"], pd.to_datetime(admissions["ADMITTIME"])))

    labevents = load_labevents_for_cohort(
        args.labevents, set(LAB_PANEL), set(cohort_hadm_ids), chunksize=args.chunksize
    )
    grouped = {(int(h), int(i)): g for (h, i), g in labevents.groupby(["HADM_ID", "ITEMID"])}

    items = sorted(LAB_PANEL)
    n_items = len(items)
    values = np.full((len(cohort_hadm_ids), n_items), np.nan, dtype=np.float64)
    for row, hadm_id in enumerate(cohort_hadm_ids):
        admittime = admittime_by_hadm.get(hadm_id)
        if admittime is None:
            continue
        for col, itemid in enumerate(items):
            rows = grouped.get((hadm_id, itemid))
            if rows is None:
                continue
            val = first_value_in_window(rows, admittime)
            values[row, col] = val if val is not None else np.nan

    train_patient_of_row = []
    for p, patient in enumerate(hadm_ids_nested):
        train_patient_of_row.extend([p] * len(patient))
    is_train_row = np.array(train_patient_of_row) < split_point

    edges = np.zeros((n_items, 4), dtype=np.float32)
    for col in range(n_items):
        train_values = values[is_train_row, col]
        train_values = train_values[~np.isnan(train_values)]
        edges[col] = compute_quantile_edges(train_values, n_bins=5) if len(train_values) else np.zeros(4, dtype=np.float32)

    missing_bin = 5
    bins = np.full((len(cohort_hadm_ids), n_items), missing_bin, dtype=np.int8)
    for row in range(len(cohort_hadm_ids)):
        for col in range(n_items):
            v = values[row, col]
            bins[row, col] = bin_value(None if np.isnan(v) else v, edges[col], missing_bin)

    has_any_lab = (bins != missing_bin).any(axis=1).astype(np.int8)

    np.savez(
        out_dir / "lab_query.npz",
        HADM_ID=np.array(cohort_hadm_ids, dtype=np.int64),
        bins=bins,
        edges=edges,
        has_any_lab=has_any_lab,
    )

    coverage = {
        LAB_PANEL[itemid]: {
            "itemid": itemid,
            "n_resolved": int((bins[:, col] != missing_bin).sum()),
            "coverage_frac": float((bins[:, col] != missing_bin).mean()),
        }
        for col, itemid in enumerate(items)
    }
    meta = {"n_visits": len(cohort_hadm_ids), "n_items": n_items, "missing_bin": missing_bin, "coverage": coverage}
    (out_dir / "lab_query_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] wrote lab_query.npz for {len(cohort_hadm_ids)} visits to {out_dir}")


if __name__ == "__main__":
    main()
