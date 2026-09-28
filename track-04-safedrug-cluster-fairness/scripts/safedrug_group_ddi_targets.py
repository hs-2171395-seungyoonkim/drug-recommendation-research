"""SafeDrug Intervention C (R3) helper: builds the per-visit DDI-target CSV.

Per training group (long_k10 by default, or ccs_group -- see --partition),
computes the observed DDI rate of the GROUND-TRUTH prescriptions of that
group's training visits, optionally shrinks it toward the global labeled
training rate via empirical-Bayes shrinkage (--shrink-k), clips the
(possibly shrunk) rate to [0.03, 0.10], and expands it to one row per
training HADM_ID (labeled visits get their group's clipped rate; unlabeled
visits get the flat default). Consumed by scripts/safedrug_train_dump.py's
--ddi-target-csv flag.

See docs/superpowers/specs/2026-09-05-safedrug-mitigation-design.md ("D-C1")
for the original (long_k10-only, unshrunk) contract. The --partition and
--shrink-k extensions default to today's exact behaviour (long_k10, no
shrinkage) so every existing call site stays byte-identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import DDI_ADJ_PATH
from safedrug_mechanism import load_training_visits

COHORT_DIR_DEFAULT = ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort"


def pairwise_ddi_rate(med_sets, ddi_adj: np.ndarray) -> float:
    """Attributed reimplementation of SOTA/SafeDrug/src/util.ddi_rate_score's
    core loop (lines 266-283), operating on an in-memory ddi_adj array
    instead of forcing a pickle path (so this pure function needs no file
    I/O and is directly unit-testable): over every unordered pair of
    medication indices within each set in med_sets, pooled across every
    set, the fraction that are a known DDI pair. Returns 0.0 when no set
    has >= 2 medications -- matching ddi_rate_score's own
    `if all_cnt == 0: return 0`."""
    all_cnt = 0
    dd_cnt = 0
    for meds in med_sets:
        meds = sorted(meds)
        n = len(meds)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = meds[i], meds[j]
                all_cnt += 1
                if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                    dd_cnt += 1
    return dd_cnt / all_cnt if all_cnt else 0.0


def load_training_gt_med_sets(cohort_dir):
    """HADM_ID -> ground-truth medication index set, for every training
    visit (patient_index < split_point). Mirrors
    scripts/safedrug_mechanism.py's load_training_med_sets's own
    split_point computation (int(len(records) * 2/3)) exactly, joined by
    (safedrug_patient_index, safedrug_visit_index) from master_visits.csv
    instead of returning a bare positional list."""
    cohort_dir = Path(cohort_dir)
    with (cohort_dir / "records_reconstructed.pkl").open("rb") as fh:
        records = dill.load(fh)
    split_point = int(len(records) * 2 / 3)
    master = pd.read_csv(cohort_dir / "master_visits.csv")
    train_master = master[master["safedrug_patient_index"] < split_point]
    out = {}
    for r in train_master.itertuples(index=False):
        p, v = int(r.safedrug_patient_index), int(r.safedrug_visit_index)
        out[int(r.HADM_ID)] = set(records[p][v][2])
    return out, split_point


def group_ground_truth_ddi_rates(train_labeled: pd.DataFrame, gt_med_sets: dict,
                                  ddi_adj: np.ndarray, clip_low: float = 0.03,
                                  clip_high: float = 0.10, partition: str = "long_k10",
                                  shrink_k: float = 0.0) -> pd.DataFrame:
    """R3's per-group target: pairwise_ddi_rate over each group's training
    visits' GROUND-TRUTH medication sets, optionally shrunk toward the
    global labeled-training rate, then clipped to [clip_low, clip_high].

    train_labeled must have HADM_ID and `partition` columns (as returned by
    safedrug_mechanism.load_training_visits, whose attach_labels call already
    joins in both long_k10 and ccs_group by HADM_ID); rows with a NaN
    `partition` value are excluded (they get the flat fallback in
    build_visit_target_csv, not a group of their own).

    shrink_k == 0 (the default) reproduces the original unshrunk rate
    exactly (raw_rate is used directly, not run through the shrinkage
    formula, so there is no floating-point rounding difference from before
    this parameter existed). shrink_k > 0 applies empirical-Bayes shrinkage:
    shrunk_rate = (n_g * raw_rate + shrink_k * r_global) / (n_g + shrink_k),
    where n_g is the group's training-visit count and r_global is
    pairwise_ddi_rate over every labeled training visit's ground-truth med
    set (not just this group's). Clipping is applied to the shrunk rate
    (identical to the raw rate when shrink_k == 0), never to the pre-clip
    raw_ddi_rate/shrunk_ddi_rate columns themselves.

    group ids are cast to int for partition="long_k10" (unchanged from
    before), left as-is (e.g. the ccs_group string) otherwise.
    """
    labeled = train_labeled.dropna(subset=[partition])
    r_global = None
    if shrink_k > 0:
        all_med_sets = [gt_med_sets[int(h)] for h in labeled["HADM_ID"]]
        r_global = pairwise_ddi_rate(all_med_sets, ddi_adj)
    rows = []
    for g, sub in labeled.groupby(partition):
        med_sets = [gt_med_sets[int(h)] for h in sub["HADM_ID"]]
        raw_rate = pairwise_ddi_rate(med_sets, ddi_adj)
        n_g = len(sub)
        if shrink_k > 0:
            shrunk_rate = (n_g * raw_rate + shrink_k * r_global) / (n_g + shrink_k)
        else:
            shrunk_rate = raw_rate
        clipped = min(clip_high, max(clip_low, shrunk_rate))
        group_id = int(g) if partition == "long_k10" else g
        rows.append({"group": group_id, "n_train_visits": n_g,
                      "raw_ddi_rate": raw_rate, "shrunk_ddi_rate": shrunk_rate,
                      "target_ddi": clipped})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def build_visit_target_csv(train_labeled: pd.DataFrame, group_table: pd.DataFrame,
                            default_target: float, partition: str = "long_k10") -> pd.DataFrame:
    """Expands group_table's per-group clipped target_ddi to one row per
    training HADM_ID: labeled visits get their `partition` group's
    target_ddi; visits with no `partition` label get default_target
    directly (not the wrapper's own .get(..., default) fallback -- every
    training HADM_ID has an explicit row here, see design doc D-C1)."""
    group_target = dict(zip(group_table["group"], group_table["target_ddi"]))
    rows = []
    for r in train_labeled.itertuples(index=False):
        value = getattr(r, partition)
        if pd.isna(value):
            target = default_target
        else:
            key = int(value) if partition == "long_k10" else value
            target = group_target[key]
        rows.append({"HADM_ID": int(r.HADM_ID), "target_ddi": float(target)})
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug group-aware DDI-target CSV builder (Intervention C)")
    parser.add_argument("--cohort-dir", type=str, default=str(COHORT_DIR_DEFAULT))
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--out-group-csv", type=str, default=None)
    parser.add_argument("--target-ddi-default", type=float, default=0.06)
    parser.add_argument("--clip-low", type=float, default=0.03)
    parser.add_argument("--clip-high", type=float, default=0.10)
    parser.add_argument(
        "--partition", choices=["long_k10", "ccs_group"], default="long_k10",
        help="training-visit grouping used for the per-group DDI target. "
        "ccs_group is joined from out/ccs_assignments.csv's group column by "
        "HADM_ID (via safedrug_mechanism.load_training_visits' attach_labels "
        "call) -- unlabeled visits still fall back to --target-ddi-default. "
        "Default long_k10 is byte-identical to before this flag existed.",
    )
    parser.add_argument(
        "--shrink-k", type=float, default=0.0,
        help="empirical-Bayes shrinkage strength K: each group's rate is "
        "shrunk toward the global labeled-training rate as "
        "(n_g*r_g + K*r_global)/(n_g+K) before clipping. 0 (default) "
        "disables shrinkage entirely -- byte-identical to before this flag "
        "existed.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cohort_dir = Path(args.cohort_dir)

    ddi_adj = dill.load(open(DDI_ADJ_PATH, "rb"))
    gt_med_sets, split_point = load_training_gt_med_sets(cohort_dir)
    train_labeled = load_training_visits(cohort_dir, split_point)

    group_table = group_ground_truth_ddi_rates(
        train_labeled, gt_med_sets, ddi_adj, args.clip_low, args.clip_high,
        partition=args.partition, shrink_k=args.shrink_k,
    )
    visit_csv = build_visit_target_csv(
        train_labeled, group_table, args.target_ddi_default, partition=args.partition
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    visit_csv.to_csv(out_csv, index=False)
    print(f"[+] wrote {len(visit_csv)} training-visit DDI targets to {out_csv}", flush=True)

    if args.out_group_csv:
        out_group_csv = Path(args.out_group_csv)
        out_group_csv.parent.mkdir(parents=True, exist_ok=True)
        group_table.to_csv(out_group_csv, index=False)
        print(f"[+] wrote {len(group_table)} group targets to {out_group_csv}", flush=True)


if __name__ == "__main__":
    main()
