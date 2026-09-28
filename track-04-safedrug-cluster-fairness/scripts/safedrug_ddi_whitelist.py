"""SafeDrug condition-aware DDI pair whitelist builder (Intervention W, R1).

From TRAINING ground-truth medication regimens only, computes, per CCS
admitting-diagnosis category with enough training visits, which of the
337 TWOSIDES-flagged ATC-3 pairs (ddi_A_final.pkl) are prescribed together
often enough to look condition-appropriate rather than incidental --
"whitelisted" for that category. scripts/safedrug_train_dump.py (R2)
consumes this at training time to relax the DDI penalty for exactly these
pairs, on exactly these categories' visits; scripts/safedrug_ddi_pair_analysis.py
(R3) consumes it to measure whether that relaxation actually recovers the
targeted pairs.

Ground truth is read from --safedrug-data-dir's records_final.pkl (the
exact object train_one_epoch iterates), not the cohort audit's
records_reconstructed.pkl -- verified byte-identical for every patient
during planning, but this design still sources from the former so the
whitelist is guaranteed to reflect what training actually sees, not a
parallel reconstruction that merely happens to currently match it.

See docs/superpowers/specs/2026-09-08-safedrug-ddi-whitelist-design.md
("R1") for the full contract and rationale.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAFEDRUG_DATA_DEFAULT = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
COHORT_DIR_DEFAULT = ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort"
CCS_CSV_DEFAULT = ROOT / "out" / "ccs_assignments.csv"


def count_flagged_pairs(med_sets, ddi_adj: np.ndarray) -> Counter:
    """Over every visit's ground-truth medication-index set, for every
    unordered pair (a<b) that is a known DDI pair (ddi_adj[a,b]==1 or
    ddi_adj[b,a]==1), counts how many visits contain it. A specific pair
    contributes at most 1 per visit (a set has no duplicate medications)."""
    counts: Counter = Counter()
    for meds in med_sets:
        meds = sorted(meds)
        n = len(meds)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = meds[i], meds[j]
                if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                    counts[(a, b)] += 1
    return counts


def select_whitelist(counts: dict, n_visits: int, threshold: float) -> pd.DataFrame:
    """counts: {(idx_a, idx_b): n_visits_with_that_pair}, as returned by
    count_flagged_pairs. n_visits: the category's TOTAL training-visit
    count (the share denominator -- not sum(counts.values())). share = 0.0
    when n_visits == 0 (this function never divides by zero even though
    its caller always gates on min_visits first). Rows with share >=
    threshold only, sorted by (idx_a, idx_b)."""
    rows = []
    for (a, b), c in counts.items():
        share = c / n_visits if n_visits else 0.0
        if share >= threshold:
            rows.append({"idx_a": a, "idx_b": b, "n_pair_visits": c, "share": share})
    df = pd.DataFrame(rows, columns=["idx_a", "idx_b", "n_pair_visits", "share"])
    return df.sort_values(["idx_a", "idx_b"]).reset_index(drop=True)


def visit_mask(pairs, n_med: int) -> np.ndarray:
    """(n_med, n_med) uint8 symmetric mask: 1 at (a,b) and (b,a) for every
    pair in `pairs`, 0 elsewhere. All-zero for None or an empty list -- "no
    whitelist" and "empty whitelist" are the same code path on purpose
    (see design doc)."""
    mask = np.zeros((n_med, n_med), dtype=np.uint8)
    if not pairs:
        return mask
    for a, b in pairs:
        mask[a, b] = 1
        mask[b, a] = 1
    return mask


def load_training_visits_with_ccs(cohort_dir, safedrug_data_dir, ccs_csv) -> pd.DataFrame:
    """One row per TRAINING visit: HADM_ID, ccs_group (NaN if unlabeled),
    med_set (set[int], the ground-truth medication indices). Ground truth
    comes from --safedrug-data-dir's records_final.pkl; the
    (patient_index, visit_index) -> HADM_ID join comes from
    --cohort-dir's master_visits.csv (mirroring
    scripts/safedrug_train_dump.load_hadm_lookup's own join, duplicated
    here rather than imported -- this script has no dependency on the
    heavy models.py/util.py-importing safedrug_train_dump module)."""
    cohort_dir = Path(cohort_dir)
    safedrug_data_dir = Path(safedrug_data_dir)
    with (safedrug_data_dir / "records_final.pkl").open("rb") as fh:
        records = dill.load(fh)
    split_point = int(len(records) * 2 / 3)
    data_train = records[:split_point]

    master = pd.read_csv(cohort_dir / "master_visits.csv")
    hadm_lookup = {
        (int(r.safedrug_patient_index), int(r.safedrug_visit_index)): int(r.HADM_ID)
        for r in master.itertuples(index=False)
    }
    ccs = pd.read_csv(ccs_csv, encoding="utf-8-sig")[["HADM_ID", "group"]].rename(
        columns={"group": "ccs_group"}
    )
    ccs_lookup = dict(zip(ccs["HADM_ID"], ccs["ccs_group"]))

    rows = []
    for p, input_ in enumerate(data_train):
        for v, adm in enumerate(input_):
            hadm_id = hadm_lookup[(p, v)]
            rows.append({
                "HADM_ID": hadm_id,
                "ccs_group": ccs_lookup.get(hadm_id, float("nan")),
                "med_set": set(adm[2]),
            })
    return pd.DataFrame(rows)


def build_whitelist_pairs(train_df: pd.DataFrame, ddi_adj: np.ndarray, threshold: float,
                           min_visits: int) -> pd.DataFrame:
    """Per CCS category with >= min_visits training visits (unlabeled
    visits, NaN ccs_group, are dropped before grouping -- they can never
    form a category), count_flagged_pairs + select_whitelist, concatenated.
    A category with zero pairs clearing threshold contributes no rows (not
    an all-NaN row)."""
    columns = ["ccs_group", "idx_a", "idx_b", "n_pair_visits", "share", "n_visits"]
    frames = []
    for group, sub in train_df.dropna(subset=["ccs_group"]).groupby("ccs_group"):
        n_visits = len(sub)
        if n_visits < min_visits:
            continue
        counts = count_flagged_pairs(list(sub["med_set"]), ddi_adj)
        wl = select_whitelist(counts, n_visits, threshold)
        if wl.empty:
            continue
        wl.insert(0, "ccs_group", group)
        wl["n_visits"] = n_visits
        frames.append(wl)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def whitelist_coverage(train_df: pd.DataFrame, whitelist_pairs: pd.DataFrame,
                        ddi_adj: np.ndarray) -> dict:
    """Fraction of TRAINING flagged-pair occurrences (pooled across every
    training visit's ground-truth medication set) that fall inside that
    visit's own ccs_group's whitelist. A visit whose category has no
    whitelist entry (unlabeled, below min_visits, or zero pairs cleared
    threshold) contributes only to n_total_pairs, never n_covered_pairs."""
    whitelist_by_group: dict = {}
    for group, sub in whitelist_pairs.groupby("ccs_group"):
        whitelist_by_group[group] = set(zip(sub["idx_a"], sub["idx_b"]))

    total = 0
    covered = 0
    for r in train_df.itertuples(index=False):
        meds = sorted(r.med_set)
        wl = whitelist_by_group.get(r.ccs_group, set())
        n = len(meds)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = meds[i], meds[j]
                if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                    total += 1
                    if (a, b) in wl:
                        covered += 1
    coverage = covered / total if total else 0.0
    return {"n_total_pairs": total, "n_covered_pairs": covered, "coverage": coverage}


def build_whitelist_visits(all_hadm_ids, ccs_lookup: dict, category_pair_counts: dict) -> pd.DataFrame:
    """One row per HADM_ID in all_hadm_ids (the FULL cohort -- train, test,
    and eval alike, so scripts/safedrug_ddi_pair_analysis.py can look up a
    TEST visit's own category's whitelist size too). n_whitelisted_pairs
    is 0 for an unlabeled visit or one whose category has no entry in
    category_pair_counts."""
    rows = []
    for h in all_hadm_ids:
        h = int(h)
        group = ccs_lookup.get(h, float("nan"))
        n = int(category_pair_counts.get(group, 0)) if pd.notna(group) else 0
        rows.append({"HADM_ID": h, "ccs_group": group, "n_whitelisted_pairs": n})
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SafeDrug condition-aware DDI pair whitelist builder (Intervention W, R1)"
    )
    parser.add_argument("--cohort-dir", type=str, default=str(COHORT_DIR_DEFAULT))
    parser.add_argument("--safedrug-data-dir", type=str, default=str(SAFEDRUG_DATA_DEFAULT))
    parser.add_argument("--ccs-csv", type=str, default=str(CCS_CSV_DEFAULT))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-visits", type=int, default=30)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cohort_dir = Path(args.cohort_dir)
    safedrug_data_dir = Path(args.safedrug_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ddi_adj = np.asarray(dill.load(open(safedrug_data_dir / "ddi_A_final.pkl", "rb")))
    voc = dill.load(open(safedrug_data_dir / "voc_final.pkl", "rb"))
    idx2atc = voc["med_voc"].idx2word

    train_df = load_training_visits_with_ccs(cohort_dir, safedrug_data_dir, args.ccs_csv)
    whitelist_pairs = build_whitelist_pairs(train_df, ddi_adj, args.threshold, args.min_visits)
    whitelist_pairs["atc_a"] = whitelist_pairs["idx_a"].map(idx2atc)
    whitelist_pairs["atc_b"] = whitelist_pairs["idx_b"].map(idx2atc)
    whitelist_pairs = whitelist_pairs[
        ["ccs_group", "idx_a", "idx_b", "atc_a", "atc_b", "share", "n_visits"]
    ]
    whitelist_pairs.to_csv(out_dir / "whitelist_pairs.csv", index=False)

    master = pd.read_csv(cohort_dir / "master_visits.csv")
    ccs = pd.read_csv(args.ccs_csv, encoding="utf-8-sig")[["HADM_ID", "group"]].rename(
        columns={"group": "ccs_group"}
    )
    ccs_lookup = dict(zip(ccs["HADM_ID"], ccs["ccs_group"]))
    category_pair_counts = (
        whitelist_pairs.groupby("ccs_group").size().to_dict() if len(whitelist_pairs) else {}
    )
    whitelist_visits = build_whitelist_visits(master["HADM_ID"], ccs_lookup, category_pair_counts)
    whitelist_visits.to_csv(out_dir / "whitelist_visits.csv", index=False)

    coverage = whitelist_coverage(train_df, whitelist_pairs, ddi_adj)
    n_categories_considered = int(
        train_df.dropna(subset=["ccs_group"]).groupby("ccs_group").size().ge(args.min_visits).sum()
    )
    meta = {
        "threshold": args.threshold,
        "min_visits": args.min_visits,
        "n_categories_considered": n_categories_considered,
        "n_categories_with_whitelist": int(whitelist_pairs["ccs_group"].nunique()),
        "n_pairs_total": int(len(whitelist_pairs)),
        "n_pairs_by_category": {
            str(k): int(v) for k, v in whitelist_pairs.groupby("ccs_group").size().items()
        },
        "training_coverage": coverage,
    }
    (out_dir / "whitelist_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[+] wrote {len(whitelist_pairs)} whitelisted pairs across "
        f"{meta['n_categories_with_whitelist']} categories to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
