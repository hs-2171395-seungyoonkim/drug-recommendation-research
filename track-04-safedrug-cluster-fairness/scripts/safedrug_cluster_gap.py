"""SafeDrug per-cluster gap analysis: per-group tables, cluster-robust
adjustment, permutation significance, figures, and the Korean report.

Reuses scripts/53_gap_ksweep.py's gap-statistic definitions (range and
visit-weighted SD of per-group means, groups with >=30 visits only) and its
patient-level permutation null (largest_remainder-allocated patient
reassignment, 10,000 draws, seed 0) by faithful, attributed port -- that
script is a standalone entry point, not an importable module.

statsmodels is not installed under py -3.12 (verified) so the cluster-robust
(CR1, Cameron-Gelbach-Miller small-sample corrected) OLS sandwich by
SUBJECT_ID is implemented here manually with numpy, tested against a
hand-computed case.

See docs/specs/2026-09-04-safedrug-per-cluster-eval-design.md
("D5-D9") for the full contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_percluster.metrics import attach_labels, patient_bootstrap_ci, visit_metrics

DXTEXT_CSV = ROOT / "out" / "dxtext_cluster_assignments.csv"
LABELS_NPZ = ROOT / "out" / "42_dxtext_labels.npz"
CCS_CSV = ROOT / "out" / "ccs_assignments.csv"
TOPDX_CSV = ROOT / "out" / "dxtext_topdx_k5_k10.csv"
DDI_ADJ_PATH = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\ddi_A_final.pkl")

PARTITIONS = ["long_k10", "short_k10", "concise_k10", "long_k25", "ccs_group"]
OUTCOMES = ["jaccard", "f1", "prauc", "ddi_rate_visit", "n_med_pred", "n_med_gt", "n_dx"]
ADJUSTED_OUTCOMES = ["jaccard", "f1", "ddi_rate_visit"]
MIN_VISITS = 30
N_PERM = 10000
PERM_SEED = 0


# ============================================================= D6: manual cluster-robust OLS
def cluster_robust_ols(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS coefficients and cluster-robust (CR1, Stata-style small-sample
    corrected) standard errors, clustered by `clusters`.

    X: (n, k) design matrix (include an intercept / full dummy set yourself --
    this function does not add one). y: (n,) response. clusters: (n,) cluster
    id per row (e.g. SUBJECT_ID).

    Returns (beta, se, vcov): beta is (k,), se is (k,) (sqrt of vcov's
    diagonal), vcov is (k, k).

    V = c * (X'X)^-1 [sum_g (X_g' u_g)(X_g' u_g)'] (X'X)^-1
    c = (G / (G - 1)) * ((N - 1) / (N - K))
    (Cameron, Gelbach & Miller 2011; Stata's default vce(cluster); R's
    sandwich::vcovCL(type="CR1")).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    n, k = X.shape
    if y.shape != (n,):
        raise ValueError("y must have shape (n,)")
    if clusters.shape != (n,):
        raise ValueError("clusters must have shape (n,)")

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    if G < 2:
        raise ValueError("cluster_robust_ols needs at least 2 clusters")
    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = clusters == g
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov = correction * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(vcov))
    return beta, se, vcov


def _cell_means_design(group_codes, n_groups, log_n_dx, n_med_gt, visit_index):
    n = len(group_codes)
    X = np.zeros((n, n_groups + 3))
    X[np.arange(n), group_codes] = 1.0
    X[:, n_groups] = log_n_dx
    X[:, n_groups + 1] = n_med_gt
    X[:, n_groups + 2] = visit_index
    return X


def adjusted_group_means(df: pd.DataFrame, group_col: str, outcome_col: str, groups: list):
    """D6's cell-means model: one dummy per group (no separate intercept) +
    log(n_dx) + n_med_gt + visit_index, cluster-robust by SUBJECT_ID. Returns
    (adjusted: dict[group -> predicted mean at sample covariate means],
    coef_table: DataFrame[term, estimate, cluster_robust_se])."""
    work = df.dropna(subset=[group_col, outcome_col, "n_dx", "n_med_gt", "visit_index"])
    group_index = {g: i for i, g in enumerate(groups)}
    group_codes = work[group_col].map(group_index).to_numpy()
    log_n_dx = np.log(work["n_dx"].to_numpy(dtype=float))
    n_med_gt = work["n_med_gt"].to_numpy(dtype=float)
    visit_index = work["visit_index"].to_numpy(dtype=float)
    y = work[outcome_col].to_numpy(dtype=float)
    clusters = work["SUBJECT_ID"].to_numpy()

    X = _cell_means_design(group_codes, len(groups), log_n_dx, n_med_gt, visit_index)
    beta, se, _ = cluster_robust_ols(X, y, clusters)

    covariate_effect = (
        beta[len(groups)] * log_n_dx.mean()
        + beta[len(groups) + 1] * n_med_gt.mean()
        + beta[len(groups) + 2] * visit_index.mean()
    )
    adjusted = {g: float(beta[i] + covariate_effect) for i, g in enumerate(groups)}

    terms = list(groups) + ["log_n_dx", "n_med_gt", "visit_index"]
    coef_table = pd.DataFrame({"term": terms, "estimate": beta, "cluster_robust_se": se})
    return adjusted, coef_table


def covariate_only_residuals(df: pd.DataFrame, outcome_col: str):
    """D6/D7's covariate-only model: intercept + log(n_dx) + n_med_gt +
    visit_index, no group terms. Returns (residual Series aligned to the
    filtered index, that index)."""
    work = df.dropna(subset=[outcome_col, "n_dx", "n_med_gt", "visit_index"])
    n = len(work)
    X = np.column_stack(
        [
            np.ones(n),
            np.log(work["n_dx"].to_numpy(dtype=float)),
            work["n_med_gt"].to_numpy(dtype=float),
            work["visit_index"].to_numpy(dtype=float),
        ]
    )
    y = work[outcome_col].to_numpy(dtype=float)
    clusters = work["SUBJECT_ID"].to_numpy()
    beta, _, _ = cluster_robust_ols(X, y, clusters)
    resid = y - X @ beta
    return pd.Series(resid, index=work.index), work.index


# ============================================================= D7: gap statistic + permutation,
# ============================================================= ported from scripts/53_gap_ksweep.py
def largest_remainder(props: np.ndarray, total: int) -> np.ndarray:
    """Proportional integer allocation summing exactly to `total`, ties broken
    by largest remainder. Ported verbatim from scripts/53_gap_ksweep.py
    (lines 139-145)."""
    raw = props * total
    b = np.floor(raw).astype(int)
    rem = total - b.sum()
    if rem > 0:
        b[np.argsort(-(raw - b))[:rem]] += 1
    return b


def gap_statistics(group_codes: np.ndarray, values: np.ndarray, n_groups: int,
                    min_visits: int = MIN_VISITS):
    """Range and visit-weighted SD of per-group means, restricted to groups
    with >= min_visits observations. Adapted from scripts/53_gap_ksweep.py's
    _rng_wsd/gap_stats (lines 116-136), generalized from a fixed K and two
    parallel predictor columns to an arbitrary group count and a single value
    column (called separately per outcome here).

    group_codes: (n,) int array in [0, n_groups). values: (n,) float array.
    Returns (range_, weighted_sd, n_groups_used); (NaN, NaN, count) if fewer
    than 2 groups meet min_visits.
    """
    cnt = np.bincount(group_codes, minlength=n_groups).astype(float)
    tot = np.bincount(group_codes, weights=values, minlength=n_groups)
    ok = cnt >= min_visits
    if ok.sum() < 2:
        return float("nan"), float("nan"), int(ok.sum())
    m = tot[ok] / cnt[ok]
    w = cnt[ok]
    mbar = np.sum(w * m) / np.sum(w)
    wsd = np.sqrt(np.sum(w * (m - mbar) ** 2) / np.sum(w))
    return float(m.max() - m.min()), float(wsd), int(ok.sum())


def _build_permutation_target(group_of_visit: np.ndarray, subject_of_visit: np.ndarray,
                               n_groups: int):
    """Patient quota per group for the null, ported from
    scripts/53_gap_ksweep.py (`pat_per_cl` + `largest_remainder`, lines
    150-171): the number of DISTINCT patients observed with >=1 visit in each
    group (a patient with visits in multiple groups is counted in each one --
    this mirrors script 53 exactly), proportionally allocated across the
    n_patients total patients via largest_remainder.

    Returns (target, patients, patient_of_visit).
    """
    patients = np.sort(np.unique(subject_of_visit))
    n_patients = len(patients)
    patient_pos = {p: i for i, p in enumerate(patients)}
    patient_of_visit = np.array([patient_pos[s] for s in subject_of_visit])

    pat_per_group = np.zeros(n_groups)
    for g in range(n_groups):
        pat_per_group[g] = len(np.unique(subject_of_visit[group_of_visit == g]))
    target = largest_remainder(pat_per_group / pat_per_group.sum(), n_patients)
    return target, patients, patient_of_visit


def permutation_p(group_of_visit: np.ndarray, subject_of_visit: np.ndarray,
                   values: np.ndarray, n_groups: int, min_visits: int = MIN_VISITS,
                   n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Patient-level permutation test for the D7 gap statistics, ported from
    scripts/53_gap_ksweep.py's permutation loop (lines 191-234): patients are
    reassigned to groups preserving each group's observed *patient* count, so
    all of a permuted patient's visits move to its new group together."""
    target, patients, patient_of_visit = _build_permutation_target(
        group_of_visit, subject_of_visit, n_groups
    )
    n_patients = len(patients)
    obs_range, obs_wsd, n_used = gap_statistics(group_of_visit, values, n_groups, min_visits)

    rng = np.random.default_rng(seed)
    null_range = np.full(n_perm, np.nan)
    null_wsd = np.full(n_perm, np.nan)
    for b in range(n_perm):
        perm = rng.permutation(n_patients)
        cl_of_pat = np.empty(n_patients, dtype=int)
        s = 0
        for c, sz in enumerate(target):
            cl_of_pat[perm[s : s + sz]] = c
            s += sz
        permuted_group = cl_of_pat[patient_of_visit]
        null_range[b], null_wsd[b], _ = gap_statistics(permuted_group, values, n_groups, min_visits)

    def _summarize(observed, null):
        null = null[~np.isnan(null)]
        if len(null) == 0 or np.isnan(observed):
            return {"null_mean": float("nan"), "null_p95": float("nan"),
                    "z": float("nan"), "p_raw": float("nan")}
        p_raw = (1 + int((null >= observed).sum())) / (1 + len(null))
        sd = null.std(ddof=1)
        z = (observed - null.mean()) / sd if sd > 0 else float("nan")
        return {"null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                "z": float(z), "p_raw": float(p_raw)}

    range_summary = _summarize(obs_range, null_range)
    wsd_summary = _summarize(obs_wsd, null_wsd)
    return {
        "observed_range": obs_range,
        "observed_wsd": obs_wsd,
        "n_groups_used": n_used,
        **{f"{k}_range": v for k, v in range_summary.items()},
        **{f"{k}_wsd": v for k, v in wsd_summary.items()},
    }


# ============================================================= D3 orchestration + D5: per-group table
def load_per_visit_metrics(eval_dir: Path) -> pd.DataFrame:
    npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    ddi_adj = dill.load(open(DDI_ADJ_PATH, "rb"))
    metrics = visit_metrics(npz["y_gt"], npz["y_pred"], npz["y_prob"], ddi_adj)
    join_cols = pd.DataFrame(
        {
            "patient_index": npz["patient_index"],
            "visit_index": npz["visit_index"],
            "HADM_ID": npz["HADM_ID"],
            "SUBJECT_ID": npz["SUBJECT_ID"],
            "split": npz["split"],
            "n_dx": npz["n_diag"],  # design D3: n_dx is the records-derived n_diag, not dxtext's own n_dx
        }
    )
    df = pd.concat([join_cols, metrics], axis=1)
    df = attach_labels(df, DXTEXT_CSV, LABELS_NPZ, CCS_CSV)
    return df


def group_summary_table(df: pd.DataFrame, partition: str, scope: str) -> pd.DataFrame:
    """D5's per-group table for one (partition, scope) pair. df must already
    be restricted to has_label rows and the requested scope's split(s), and
    have no NaN partition values."""
    groups = sorted(df[partition].unique())
    n_visits_per_group = df.groupby(partition).size()

    metric_frames = {
        metric: patient_bootstrap_ci(df, partition, metric, n_boot=2000, seed=0).set_index(partition)
        for metric in OUTCOMES
    }

    rows = []
    for g in groups:
        row = {
            "partition": partition,
            "scope": scope,
            "group": g,
            "n_visits": int(n_visits_per_group.loc[g]),
            "n_patients": int(metric_frames[OUTCOMES[0]].loc[g, "n_patients"])
            if g in metric_frames[OUTCOMES[0]].index else 0,
            "small": bool(n_visits_per_group.loc[g] < MIN_VISITS),
        }
        for metric in OUTCOMES:
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


# ============================================================= D8: figures
def _normalize_group_id(g):
    """Fix-round-2: normalize a group id read back from a table_*.csv round
    trip to a stable, comparable key.

    table_group_summary.csv/table_adjusted_means.csv's "group" column mixes
    integer cluster ids (long_k10/short_k10/concise_k10/long_k25) with CCS
    group-name strings (ccs_group) in the same column. Upstream,
    attach_labels' left join puts NaN into the numeric label columns for the
    ~590 visits with no dxtext label, which forces pandas to upcast those
    otherwise-integer columns to float64 (there is no missing-value-safe
    plain int dtype) -- so numeric group ids are float (e.g. 6.0) by the
    time they reach these tables, and to_csv writes them as "6.0". On
    read-back, because the column as a whole also holds non-numeric CCS
    strings, pandas cannot infer it as numeric and keeps every entry as a
    literal object/string -- so a numeric group id round-trips as the
    *string* '6.0', not the float 6.0. A bare int(g) raises ValueError on
    that string.

    Anything that looks like a whole number -- a Python/numpy int, a
    Python/numpy float, or a numeric string including a float-formatted one
    like '6.0' -- normalizes to a plain int. Anything else (a CCS group
    name, or any other non-numeric string) passes through unchanged as a
    string. NaN passes through unchanged (there is no integer it could be).
    """
    if isinstance(g, (int, np.integer)):
        return int(g)
    if isinstance(g, (float, np.floating)):
        return g if pd.isna(g) else int(g)
    s = str(g)
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    if pd.isna(f):
        return s
    return int(f) if f.is_integer() else s


def _load_long_k10_theme_labels() -> dict:
    """빈도1 + first lift_top5 entry, for the long_k10 partition's figure
    labels only (design D8 -- not extended to short_k10/concise_k10 even
    though out/dxtext_topdx_k5_k10.csv has k=10 rows for them too)."""
    topdx = pd.read_csv(TOPDX_CSV, encoding="utf-8-sig")
    topdx = topdx[(topdx["변형"] == "long") & (topdx["k"] == 10)]
    labels = {}
    for row in topdx.itertuples(index=False):
        top_lift = str(row.lift_top5).split("|")[0].strip()
        labels[_normalize_group_id(row.cluster)] = f"{row.빈도1} / {top_lift}"
    return labels


def _figure_group_labels(partition: str, groups: list) -> list[str]:
    if partition == "long_k10":
        theme = _load_long_k10_theme_labels()
        return [theme.get(_normalize_group_id(g), str(g)) for g in groups]
    return [str(g) for g in groups]


def make_figures(eval_dir: Path, figs_dir: Path) -> None:
    # Imported locally, not at module scope: matplotlib is not installed under
    # py -3.12 (verified) -- this keeps every other function in this module,
    # and this whole module's own unit tests, independent of that dependency.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

    summary = pd.read_csv(eval_dir / "table_group_summary.csv")
    adjusted = pd.read_csv(eval_dir / "table_adjusted_means.csv")

    for partition in PARTITIONS:
        part_summary = summary[(summary["partition"] == partition) & (summary["scope"] == "test")]
        part_adjusted = adjusted[
            (adjusted["partition"] == partition)
            & (adjusted["scope"] == "test")
            & (adjusted["outcome"] == "jaccard")
        ]
        if part_summary.empty:
            continue
        merged = part_summary.merge(part_adjusted, on="group", how="left").sort_values("mean_jaccard")
        labels = _figure_group_labels(partition, merged["group"].tolist())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        y_pos = np.arange(len(merged))
        axes[0].errorbar(
            merged["mean_jaccard"], y_pos,
            xerr=[
                merged["mean_jaccard"] - merged["ci_low_jaccard"],
                merged["ci_high_jaccard"] - merged["mean_jaccard"],
            ],
            fmt="o", color="#2471a3",
        )
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_title(f"{partition} - raw mean Jaccard (95% CI)")

        axes[1].plot(merged["adjusted_mean"], y_pos, "o", color="#c0392b")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].set_title(f"{partition} - adjusted mean Jaccard")

        fig.tight_layout()
        fig.savefig(figs_dir / f"fig_{partition}_raw_vs_adjusted.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    primary = summary[(summary["partition"] == "long_k10") & (summary["scope"] == "test")].sort_values("group")
    if not primary.empty:
        labels = _figure_group_labels("long_k10", primary["group"].tolist())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        x_pos = np.arange(len(primary))
        axes[0].bar(x_pos, primary["mean_ddi_rate_visit"], color="#7d3c98")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        axes[0].set_title("long_k10 - mean DDI rate per group")
        axes[1].bar(x_pos, primary["mean_n_med_pred"], color="#2471a3")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        axes[1].set_title("long_k10 - mean predicted #meds per group")
        fig.tight_layout()
        fig.savefig(figs_dir / "fig_long_k10_ddi_nmed.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    train_log = pd.read_csv(eval_dir / "train_log.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()
    ax.plot(train_log["epoch"], train_log["eval_ja"], "o-", color="#2471a3", label="eval Jaccard")
    ax2.plot(train_log["epoch"], train_log["eval_ddi_rate"], "o-", color="#c0392b", label="eval DDI rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("eval Jaccard", color="#2471a3")
    ax2.set_ylabel("eval DDI rate", color="#c0392b")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_train_log.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================= D9: report
SAFEDRUG_MASTER_BRANCH_REFERENCE = {
    "ddi_rate": 0.0632, "ddi_rate_sd": 0.0003,
    "ja": 0.5114, "ja_sd": 0.0026,
    "avg_f1": 0.6676, "avg_f1_sd": 0.0023,
    "prauc": 0.7649, "prauc_sd": 0.0028,
}  # SOTA/SafeDrug/README.md lines 12-13 ("master branch", lr=5e-4)
SAFEDRUG_PAPER_REFERENCE = {
    "ddi_rate": 0.0589, "ja": 0.5213, "avg_f1": 0.6768, "prauc": 0.7647,
}  # SOTA/SafeDrug/README.md lines 4-6 ("archived branch" -- the IJCAI'21 paper numbers)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table formatter -- avoids the optional `tabulate`
    dependency pandas.DataFrame.to_markdown() requires, which is not
    installed under py -3.12."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_COVARIATE_TERMS = ("log_n_dx", "n_med_gt", "visit_index")


def _adjusted_means_section(eval_dir: Path, summary: pd.DataFrame) -> list[str]:
    """D6 fix-round addition: render raw-vs-adjusted group means (from
    table_adjusted_means.csv) and the covariate coefficients (from
    table_ols_coefficients.csv) that write_report previously never read --
    D9 requires this content ("D6 raw-vs-adjusted") and it was missing.

    One subsection per (partition, scope) pair present in
    table_adjusted_means.csv, primary (long_k10, test) first, then the rest
    in PARTITIONS order (test scope before test+eval); within each, one
    small table per ADJUSTED_OUTCOMES outcome (group, n_visits, raw_mean,
    adjusted_mean -- n_visits joined in from the group-summary table, since
    table_adjusted_means.csv itself does not carry it), plus one combined
    covariate-coefficient table (outcome, term, estimate, cluster_robust_se)
    restricted to the three covariate terms (the group-dummy terms are not
    shown here -- they are already folded into adjusted_mean).
    """
    adjusted = pd.read_csv(eval_dir / "table_adjusted_means.csv")
    coefficients = pd.read_csv(eval_dir / "table_ols_coefficients.csv")

    combos = sorted(
        {(p, s) for p, s in adjusted[["partition", "scope"]].itertuples(index=False)},
        key=lambda ps: (
            0 if ps == ("long_k10", "test") else 1,
            PARTITIONS.index(ps[0]) if ps[0] in PARTITIONS else len(PARTITIONS),
            0 if ps[1] == "test" else 1,
        ),
    )

    lines = ["\n## 4. 조정 전/후 군집별 평균 (D6: raw vs adjusted)\n"]
    for partition, scope in combos:
        lines.append(f"### {partition} / {scope}\n")
        part_scope_adj = adjusted[(adjusted["partition"] == partition) & (adjusted["scope"] == scope)]
        group_n_visits = summary[
            (summary["partition"] == partition) & (summary["scope"] == scope)
        ][["group", "n_visits"]]

        for outcome in ADJUSTED_OUTCOMES:
            out_adj = part_scope_adj[part_scope_adj["outcome"] == outcome]
            if out_adj.empty:
                continue
            table = out_adj.merge(group_n_visits, on="group", how="left")[
                ["group", "n_visits", "raw_mean", "adjusted_mean"]
            ]
            lines.append(f"**{outcome}**\n")
            lines.append(_df_to_markdown(table))
            lines.append("")

        coef = coefficients[
            (coefficients["partition"] == partition)
            & (coefficients["scope"] == scope)
            & (coefficients["term"].isin(_COVARIATE_TERMS))
        ][["outcome", "term", "estimate", "cluster_robust_se"]]
        if not coef.empty:
            lines.append("**공변량 계수 (cluster-robust SE)**\n")
            lines.append(_df_to_markdown(coef))
            lines.append("")
    return lines


def write_report(eval_dir: Path) -> None:
    manifest = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(eval_dir / "table_group_summary.csv")
    permutation = pd.read_csv(eval_dir / "table_permutation.csv")
    per_visit = pd.read_csv(eval_dir / "per_visit_metrics.csv")

    n_total = len(per_visit)
    n_labeled = int(per_visit["has_label"].sum())
    test_df = per_visit[per_visit["split"] == "test"]
    n_test_labeled = int(test_df["has_label"].sum())

    off_test = manifest["official_metrics"]["test"]
    ref = SAFEDRUG_MASTER_BRANCH_REFERENCE
    paper = SAFEDRUG_PAPER_REFERENCE

    lines = [
        "# SafeDrug 군집별 평가 리포트\n",
        "## 1. 학습 요약\n",
        f"- best epoch: {manifest['best_epoch']} (eval Jaccard {manifest['best_eval_jaccard']:.4f})",
        f"- test 공식 지표: Jaccard {off_test['ja']:.4f}, PRAUC {off_test['prauc']:.4f}, "
        f"F1 {off_test['avg_f1']:.4f}, DDI {off_test['ddi_rate']:.4f}, 평균 처방수 {off_test['avg_med']:.2f}",
        f"- 참고(SafeDrug README, master branch lr=5e-4): Jaccard {ref['ja']}±{ref['ja_sd']}, "
        f"DDI {ref['ddi_rate']}±{ref['ddi_rate_sd']}, F1 {ref['avg_f1']}±{ref['avg_f1_sd']}, "
        f"PRAUC {ref['prauc']}±{ref['prauc_sd']}",
        f"- 참고(논문, archived branch): Jaccard {paper['ja']}, DDI {paper['ddi_rate']}, "
        f"F1 {paper['avg_f1']}, PRAUC {paper['prauc']}",
        "\n## 2. 라벨 커버리지\n",
        f"- 전체 {n_total}방문 중 라벨 있음 {n_labeled}개",
        f"- test 분할 {len(test_df)}방문 중 라벨 있음 {n_test_labeled}개\n",
        "## 3. 군집별 표 (raw, test)\n",
    ]
    for partition in PARTITIONS:
        part = summary[(summary["partition"] == partition) & (summary["scope"] == "test")]
        lines.append(f"### {partition}\n")
        lines.append(_df_to_markdown(part))
        lines.append("")

    lines.extend(_adjusted_means_section(eval_dir, summary))

    lines.append("## 5. 순열 검정 (p_비보정 / p_bonferroni)\n")
    lines.append(_df_to_markdown(permutation))

    lines.append("\n## 6. 해석\n")
    lines.append(
        "- 위 표는 각 파티션·지표별 관측 격차와 순열 귀무분포 대비 p값을 보여준다. "
        "조정(adjusted) 열은 log(진단수)·처방수·방문순서를 통제한 뒤에도 격차가 남는지를 "
        "보여주며, 임상적 해석은 이 표가 보여주는 범위를 넘지 않는다."
    )

    (eval_dir / "REPORT_SAFEDRUG_CLUSTER_KO.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================= orchestration
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug per-cluster gap analysis")
    parser.add_argument("--eval-dir", type=str, default="out/safedrug_eval")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = ROOT / args.eval_dir
    figs_dir = eval_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = load_per_visit_metrics(eval_dir)
    df.to_csv(eval_dir / "per_visit_metrics.csv", index=False)

    labeled = df[df["has_label"]].copy()
    scopes = {"test": labeled[labeled["split"] == "test"], "test+eval": labeled}

    summary_frames, permutation_rows, adjusted_rows, coef_frames = [], [], [], []

    for partition in PARTITIONS:
        for scope_name, scope_df_all in scopes.items():
            scope_df = scope_df_all.dropna(subset=[partition]).copy()
            summary_frames.append(group_summary_table(scope_df, partition, scope_name))

            groups_all = sorted(scope_df[partition].unique())
            group_code_map = {g: i for i, g in enumerate(groups_all)}
            n_visits_per_group = scope_df.groupby(partition).size()

            for outcome in ADJUSTED_OUTCOMES:
                work = scope_df.dropna(subset=[outcome, "n_dx", "n_med_gt", "visit_index"])
                group_codes = work[partition].map(group_code_map).to_numpy()
                values = work[outcome].to_numpy(dtype=float)
                subject = work["SUBJECT_ID"].to_numpy()

                perm_raw = permutation_p(group_codes, subject, values, len(groups_all), MIN_VISITS, N_PERM, PERM_SEED)

                adjusted, coef_table = adjusted_group_means(scope_df, partition, outcome, groups_all)
                for g in groups_all:
                    raw_mean = work.loc[work[partition] == g, outcome].mean()
                    adjusted_rows.append(
                        {
                            "partition": partition, "scope": scope_name, "outcome": outcome,
                            "group": g, "raw_mean": float(raw_mean) if pd.notna(raw_mean) else float("nan"),
                            "adjusted_mean": adjusted[g],
                        }
                    )
                coef_table.insert(0, "outcome", outcome)
                coef_table.insert(0, "scope", scope_name)
                coef_table.insert(0, "partition", partition)
                coef_frames.append(coef_table)

                resid, resid_index = covariate_only_residuals(scope_df, outcome)
                resid_df = scope_df.loc[resid_index]
                resid_group_codes = resid_df[partition].map(group_code_map).to_numpy()
                resid_subject = resid_df["SUBJECT_ID"].to_numpy()
                perm_resid = permutation_p(
                    resid_group_codes, resid_subject, resid.to_numpy(), len(groups_all),
                    MIN_VISITS, N_PERM, PERM_SEED,
                )

                for source_name, perm in [("raw", perm_raw), ("residual", perm_resid)]:
                    for stat_key, stat_label in [("range", "range"), ("wsd", "weighted_sd")]:
                        p_raw = perm[f"p_raw_{stat_key}"]
                        p_bonf = float("nan") if np.isnan(p_raw) else min(1.0, p_raw * len(PARTITIONS))
                        permutation_rows.append(
                            {
                                "partition": partition, "scope": scope_name, "outcome": outcome,
                                "source": source_name, "statistic": stat_label,
                                "observed": perm[f"observed_{stat_key}"],
                                "null_mean": perm[f"null_mean_{stat_key}"],
                                "null_p95": perm[f"null_p95_{stat_key}"],
                                "z": perm[f"z_{stat_key}"],
                                "p_raw": p_raw, "p_bonferroni": p_bonf,
                            }
                        )

    pd.concat(summary_frames, ignore_index=True).to_csv(eval_dir / "table_group_summary.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(eval_dir / "table_permutation.csv", index=False)
    pd.DataFrame(adjusted_rows).to_csv(eval_dir / "table_adjusted_means.csv", index=False)
    pd.concat(coef_frames, ignore_index=True).to_csv(eval_dir / "table_ols_coefficients.csv", index=False)

    make_figures(eval_dir, figs_dir)
    write_report(eval_dir)


if __name__ == "__main__":
    main()
