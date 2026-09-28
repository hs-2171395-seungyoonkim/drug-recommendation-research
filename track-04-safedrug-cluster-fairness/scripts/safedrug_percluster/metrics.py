"""Pure per-visit metric functions for SafeDrug per-cluster evaluation.

Replicates SafeDrug's own per-visit metric definitions
(C:\\Users\\Administrator\\Desktop\\SOTA\\SafeDrug\\src\\util.py lines 166-263, the
inner helper functions of multi_label_metric()) as row-wise functions over a batch
of already-decided (thresholded) prediction rows, plus the label-attachment and
patient-level bootstrap-CI helpers scripts/safedrug_cluster_gap.py builds on.

See docs/superpowers/specs/2026-09-04-safedrug-per-cluster-eval-design.md
("D3. Per-visit metrics") for the full contract, including the footnote on why
this module's jaccard() does not reproduce util.py's unreachable
`0 if union == 0` dead-code guard (a set is never == 0, so the original raises
ZeroDivisionError on an empty/empty row instead of returning 0 -- this never
happens on real MIMIC-III visits, so there is no behavior to preserve there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _row_jaccard(gt_idx: set[int], pred_idx: set[int]) -> float:
    union = gt_idx | pred_idx
    if not union:
        return 0.0
    return len(gt_idx & pred_idx) / len(union)


def _row_precision(gt_idx: set[int], pred_idx: set[int]) -> float:
    if not pred_idx:
        return 0.0
    return len(gt_idx & pred_idx) / len(pred_idx)


def _row_recall(gt_idx: set[int], pred_idx: set[int]) -> float:
    if not gt_idx:
        return 0.0
    return len(gt_idx & pred_idx) / len(gt_idx)


def _row_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _row_ddi_rate(pred_idx: set[int], ddi_adj: np.ndarray) -> float:
    meds = sorted(pred_idx)
    n = len(meds)
    if n < 2:
        return float("nan")
    all_cnt = 0
    dd_cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = meds[i], meds[j]
            all_cnt += 1
            if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                dd_cnt += 1
    return dd_cnt / all_cnt


def visit_metrics(
    y_gt: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, ddi_adj: np.ndarray
) -> pd.DataFrame:
    """Per-visit metrics for a batch of already-decided prediction rows.

    y_gt, y_pred: (n, n_med) 0/1 arrays (any numeric dtype).
    y_prob: (n, n_med) float array of predicted probabilities.
    ddi_adj: (n_med, n_med) 0/1 DDI adjacency matrix (SafeDrug's ddi_A_final).

    Returns a DataFrame, one row per input row, columns
    [jaccard, precision, recall, f1, prauc, ddi_rate_visit, n_med_pred, n_med_gt].
    prauc is NaN when a row has no ground-truth positives; ddi_rate_visit is NaN
    when a row has fewer than 2 predicted medications.
    """
    y_gt = np.asarray(y_gt)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    if y_gt.shape != y_pred.shape or y_gt.shape != y_prob.shape:
        raise ValueError("y_gt, y_pred, y_prob must have identical shape")
    if y_gt.ndim != 2:
        raise ValueError("y_gt/y_pred/y_prob must be 2-D (n_visits, n_med)")
    n_visits, n_med = y_gt.shape
    ddi_adj = np.asarray(ddi_adj)
    if ddi_adj.shape != (n_med, n_med):
        raise ValueError(f"ddi_adj shape {ddi_adj.shape} does not match n_med {n_med}")

    rows = []
    for i in range(n_visits):
        gt_idx = set(np.flatnonzero(y_gt[i] == 1).tolist())
        pred_idx = set(np.flatnonzero(y_pred[i] == 1).tolist())
        precision = _row_precision(gt_idx, pred_idx)
        recall = _row_recall(gt_idx, pred_idx)
        prauc = (
            float("nan")
            if not gt_idx
            else float(average_precision_score(y_gt[i], y_prob[i]))
        )
        rows.append(
            {
                "jaccard": _row_jaccard(gt_idx, pred_idx),
                "precision": precision,
                "recall": recall,
                "f1": _row_f1(precision, recall),
                "prauc": prauc,
                "ddi_rate_visit": _row_ddi_rate(pred_idx, ddi_adj),
                "n_med_pred": len(pred_idx),
                "n_med_gt": len(gt_idx),
            }
        )
    return pd.DataFrame(rows)


_REQUIRED_LABEL_KEYS = ("long_k10", "long_k25", "short_k10", "concise_k10")


def attach_labels(
    df: pd.DataFrame,
    dxtext_csv: str | Path,
    labels_npz: str | Path,
    ccs_csv: str | Path,
) -> pd.DataFrame:
    """Attach the five pre-fixed partition labels to per-visit rows by HADM_ID.

    df must have an "HADM_ID" column. dxtext_csv is
    out/dxtext_cluster_assignments.csv; labels_npz is out/42_dxtext_labels.npz,
    whose long_k10/long_k25/short_k10/concise_k10 arrays are row-aligned
    POSITIONALLY with dxtext_csv (not by HADM_ID) -- the two must have identical
    length. ccs_csv is out/ccs_assignments.csv, merged onto df by HADM_ID
    directly (its "group" column, renamed ccs_group).

    Rows in df whose HADM_ID has no dxtext label get NaN in the four dxtext
    columns and False in has_label; rows with no ccs label get NaN in ccs_group.
    Raises ValueError if dxtext_csv's row count does not match a required
    labels_npz array's length (the positional-alignment invariant), or if either
    source file has a duplicate HADM_ID.
    """
    if "HADM_ID" not in df.columns:
        raise ValueError("df must have an HADM_ID column")

    dxtext = pd.read_csv(dxtext_csv, encoding="utf-8-sig")
    labels = np.load(labels_npz)
    for key in _REQUIRED_LABEL_KEYS:
        if key not in labels.files:
            raise ValueError(f"{labels_npz} missing required key {key!r}")
        if len(labels[key]) != len(dxtext):
            raise ValueError(
                f"{labels_npz}[{key!r}] has {len(labels[key])} rows but "
                f"{dxtext_csv} has {len(dxtext)} rows -- positional alignment broken"
            )

    label_table = pd.DataFrame(
        {
            "HADM_ID": dxtext["HADM_ID"].to_numpy(),
            **{key: labels[key] for key in _REQUIRED_LABEL_KEYS},
        }
    )
    if label_table["HADM_ID"].duplicated().any():
        raise ValueError(f"{dxtext_csv} has duplicate HADM_ID values")

    ccs = pd.read_csv(ccs_csv, encoding="utf-8-sig")[["HADM_ID", "group"]].rename(
        columns={"group": "ccs_group"}
    )
    if ccs["HADM_ID"].duplicated().any():
        raise ValueError(f"{ccs_csv} has duplicate HADM_ID values")

    out = df.merge(label_table, on="HADM_ID", how="left")
    out = out.merge(ccs, on="HADM_ID", how="left")
    out["has_label"] = out["long_k10"].notna()
    return out


def patient_bootstrap_ci(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """95% CI for the per-group mean of value_col by patient-level bootstrap.

    Resamples SUBJECT_IDs with replacement (not individual visit rows), n_boot
    draws, numpy.random.default_rng(seed); a resampled patient's draw
    multiplicity applies to every one of its visits, preserving within-patient
    correlation. df must have SUBJECT_ID, group_col, and value_col columns; rows
    with NaN group_col or value_col are dropped before resampling. Returns one
    row per remaining group value, columns
    [group_col, n_visits, n_patients, mean, ci_low, ci_high].
    """
    for col in ("SUBJECT_ID", group_col, value_col):
        if col not in df.columns:
            raise ValueError(f"df must have a {col!r} column")

    work = df[["SUBJECT_ID", group_col, value_col]].dropna(subset=[group_col, value_col])
    patients = np.sort(work["SUBJECT_ID"].unique())
    n_patients = len(patients)
    patient_pos = pd.Series(np.arange(n_patients), index=patients)

    groups = sorted(work[group_col].unique())
    raw_mean = work.groupby(group_col)[value_col].mean()
    n_visits = work.groupby(group_col)[value_col].size()
    n_pat_per_group = work.groupby(group_col)["SUBJECT_ID"].nunique()

    work_pat_idx = work["SUBJECT_ID"].map(patient_pos).to_numpy()
    values = work[value_col].to_numpy(dtype=float)
    group_arr = work[group_col].to_numpy()

    rng = np.random.default_rng(seed)
    boot_means = {g: np.full(n_boot, np.nan) for g in groups}
    for b in range(n_boot):
        draw = rng.integers(0, n_patients, size=n_patients)
        counts = np.bincount(draw, minlength=n_patients)
        weights = counts[work_pat_idx]
        mask = weights > 0
        if not mask.any():
            continue
        w = weights[mask]
        v = values[mask]
        g = group_arr[mask]
        for grp in groups:
            gm = g == grp
            if gm.any():
                boot_means[grp][b] = np.average(v[gm], weights=w[gm])

    rows = []
    for grp in groups:
        arr = boot_means[grp][~np.isnan(boot_means[grp])]
        if len(arr) == 0:
            ci_low, ci_high = float("nan"), float("nan")
        else:
            ci_low, ci_high = (float(x) for x in np.percentile(arr, [2.5, 97.5]))
        rows.append(
            {
                group_col: grp,
                "n_visits": int(n_visits.loc[grp]),
                "n_patients": int(n_pat_per_group.loc[grp]),
                "mean": float(raw_mean.loc[grp]),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)
