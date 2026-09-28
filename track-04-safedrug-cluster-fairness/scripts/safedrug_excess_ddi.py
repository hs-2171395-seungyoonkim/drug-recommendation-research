"""SafeDrug excess-DDI group metric (Task H, part 2).

For every test-split visit, computes the pairwise DDI rate of the model's
PREDICTED medication set (ddi_pred) and of the GROUND-TRUTH medication set
(ddi_true), and their difference (excess_ddi = ddi_pred - ddi_true) --
whether the model's recommendations carry more (or fewer) DDI pairs than
the clinician's own prescriptions did, per visit. Aggregates per group
(long_k10 and/or ccs_group) with patient-bootstrap 95% CIs, and runs the
same patient-level permutation gap test scripts/safedrug_cluster_gap.py
uses (imported, not reimplemented) on excess_ddi's group range and
weighted SD.

Reuses scripts/safedrug_percluster/metrics.py's private _row_ddi_rate (the
same per-visit "DDI pairs / all pairs, NaN if < 2 meds" definition
scripts/safedrug_percluster/metrics.py:visit_metrics already applies to
predicted meds as its own ddi_rate_visit column -- this script additionally
applies it to the ground-truth set and takes the per-visit difference) and
scripts/safedrug_cluster_gap.py's gap_statistics/permutation_p (via
permutation_p, which already calls gap_statistics internally) -- no
statistic is redefined here.

See docs/superpowers/specs/2026-09-05-safedrug-mitigation-design.md ("R3")
for the group-aware-DDI-target context this metric evaluates against; this
script itself is a Task H addition, not part of that design's own R1-R7
task list.
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

from safedrug_cluster_gap import CCS_CSV, DDI_ADJ_PATH, DXTEXT_CSV, LABELS_NPZ, MIN_VISITS, N_PERM, PERM_SEED, permutation_p
from safedrug_percluster.metrics import _row_ddi_rate, attach_labels, patient_bootstrap_ci

DEFAULT_PARTITIONS = ["long_k10", "ccs_group"]
GROUP_METRICS = ["ddi_pred", "ddi_true", "excess_ddi"]


def per_visit_excess_ddi(y_gt: np.ndarray, y_pred: np.ndarray, ddi_adj: np.ndarray) -> pd.DataFrame:
    """Per-visit ddi_pred/ddi_true/excess_ddi for a batch of already-decided
    prediction rows.

    y_gt, y_pred: (n, n_med) 0/1 arrays. ddi_adj: (n_med, n_med) 0/1 DDI
    adjacency matrix. ddi_pred/ddi_true reuse
    safedrug_percluster.metrics._row_ddi_rate verbatim (NaN when a row has
    fewer than 2 meds in the respective set); excess_ddi = ddi_pred -
    ddi_true, NaN whenever either input is NaN.

    Returns a DataFrame, one row per input row, columns
    [ddi_pred, ddi_true, excess_ddi].
    """
    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)
    if y_gt.shape != y_pred.shape:
        raise ValueError("y_gt and y_pred must have identical shape")
    if y_gt.ndim != 2:
        raise ValueError("y_gt/y_pred must be 2-D (n_visits, n_med)")
    n_visits, n_med = y_gt.shape
    ddi_adj = np.asarray(ddi_adj)
    if ddi_adj.shape != (n_med, n_med):
        raise ValueError(f"ddi_adj shape {ddi_adj.shape} does not match n_med {n_med}")

    rows = []
    for i in range(n_visits):
        pred_idx = set(np.flatnonzero(y_pred[i] == 1).tolist())
        gt_idx = set(np.flatnonzero(y_gt[i] == 1).tolist())
        ddi_pred = _row_ddi_rate(pred_idx, ddi_adj)
        ddi_true = _row_ddi_rate(gt_idx, ddi_adj)
        excess = float("nan") if (np.isnan(ddi_pred) or np.isnan(ddi_true)) else ddi_pred - ddi_true
        rows.append({"ddi_pred": ddi_pred, "ddi_true": ddi_true, "excess_ddi": excess})
    return pd.DataFrame(rows)


def excess_ddi_group_table(df: pd.DataFrame, partition: str) -> pd.DataFrame:
    """One row per group in `partition`: n_visits (all rows with a non-NaN
    `partition` value, regardless of any individual metric's own NaNs),
    n_patients (from the first GROUP_METRICS entry's own patient-bootstrap
    frame -- the same convention scripts/safedrug_cluster_gap.py's
    group_summary_table uses for its own multi-metric table), and mean/95%
    CI (patient_bootstrap_ci, n_boot=2000, seed=0) for each of
    ddi_pred/ddi_true/excess_ddi. df must have `partition`, SUBJECT_ID, and
    the three GROUP_METRICS columns; NaN values in a given metric are
    dropped by patient_bootstrap_ci before that metric's own mean/CI (a
    group with too few non-NaN metric rows to bootstrap gets NaN mean/CI
    for that metric, not an excluded row)."""
    work = df.dropna(subset=[partition])
    groups = sorted(work[partition].unique())
    n_visits_per_group = work.groupby(partition).size()

    metric_frames = {
        metric: patient_bootstrap_ci(work, partition, metric, n_boot=2000, seed=0).set_index(partition)
        for metric in GROUP_METRICS
    }

    rows = []
    for g in groups:
        row = {
            "partition": partition,
            "group": g,
            "n_visits": int(n_visits_per_group.loc[g]),
            "n_patients": int(metric_frames[GROUP_METRICS[0]].loc[g, "n_patients"])
            if g in metric_frames[GROUP_METRICS[0]].index else 0,
        }
        for metric in GROUP_METRICS:
            mf = metric_frames[metric]
            if g in mf.index:
                row[f"mean_{metric}"] = mf.loc[g, "mean"]
                row[f"ci_low_{metric}"] = mf.loc[g, "ci_low"]
                row[f"ci_high_{metric}"] = mf.loc[g, "ci_high"]
            else:
                row[f"mean_{metric}"] = float("nan")
                row[f"ci_low_{metric}"] = float("nan")
                row[f"ci_high_{metric}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def excess_ddi_permutation_table(df: pd.DataFrame, partition: str, n_perm: int = N_PERM,
                                  seed: int = PERM_SEED) -> pd.DataFrame:
    """Patient-level permutation test (safedrug_cluster_gap.permutation_p,
    which itself calls safedrug_cluster_gap.gap_statistics for the observed
    range/weighted-SD -- neither is reimplemented here) on excess_ddi's
    group range and weighted SD, groups with < MIN_VISITS (30) excluded
    from both statistics. df must have `partition`, SUBJECT_ID, and
    excess_ddi columns; rows with a NaN `partition` or excess_ddi are
    dropped first. Returns columns [partition, statistic, observed,
    null_mean, null_p95, z, p_raw]."""
    work = df.dropna(subset=[partition, "excess_ddi"])
    groups = sorted(work[partition].unique())
    group_code_map = {g: i for i, g in enumerate(groups)}
    group_codes = work[partition].map(group_code_map).to_numpy()
    values = work["excess_ddi"].to_numpy(dtype=float)
    subject = work["SUBJECT_ID"].to_numpy()

    perm = permutation_p(group_codes, subject, values, len(groups), MIN_VISITS, n_perm, seed)

    rows = []
    for stat_key, stat_label in [("range", "range"), ("wsd", "weighted_sd")]:
        rows.append({
            "partition": partition,
            "statistic": stat_label,
            "observed": perm[f"observed_{stat_key}"],
            "null_mean": perm[f"null_mean_{stat_key}"],
            "null_p95": perm[f"null_p95_{stat_key}"],
            "z": perm[f"z_{stat_key}"],
            "p_raw": perm[f"p_raw_{stat_key}"],
        })
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SafeDrug excess-DDI group metric (predicted DDI rate minus ground-truth DDI rate)"
    )
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--partitions", nargs="+", default=list(DEFAULT_PARTITIONS))
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=PERM_SEED)
    parser.add_argument(
        "--ddi-adj", type=str, default=None,
        help="override the DDI adjacency pickle path (default: "
        "safedrug_cluster_gap.DDI_ADJ_PATH, an absolute Desktop path outside "
        "this repo).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = ROOT / args.eval_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ddi_adj_path = Path(args.ddi_adj) if args.ddi_adj else DDI_ADJ_PATH
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))

    npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    per_visit = per_visit_excess_ddi(npz["y_gt"], npz["y_pred"], ddi_adj)
    join_cols = pd.DataFrame({
        "HADM_ID": npz["HADM_ID"],
        "SUBJECT_ID": npz["SUBJECT_ID"],
        "split": npz["split"],
    })
    df = pd.concat([join_cols, per_visit], axis=1)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df = attach_labels(df, DXTEXT_CSV, LABELS_NPZ, CCS_CSV)

    df.to_csv(out_dir / "per_visit_excess_ddi.csv", index=False)

    group_frames = [excess_ddi_group_table(df, partition) for partition in args.partitions]
    perm_frames = [
        excess_ddi_permutation_table(df, partition, args.n_perm, args.seed)
        for partition in args.partitions
    ]

    pd.concat(group_frames, ignore_index=True).to_csv(out_dir / "table_excess_ddi_groups.csv", index=False)
    pd.concat(perm_frames, ignore_index=True).to_csv(out_dir / "table_excess_ddi_permutation.csv", index=False)

    print(f"[+] wrote excess-DDI tables for {len(df)} test visits to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
