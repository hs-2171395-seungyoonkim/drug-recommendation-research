"""SafeDrug mechanism decomposition: precision/recall gap, training
representation, drug rarity, and a widened (v2) adjustment.

See docs/specs/2026-09-04-safedrug-seeds-mechanism-ksweep-design.md
("D-C") for the full contract.
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

from safedrug_cluster_gap import (
    CCS_CSV,
    DXTEXT_CSV,
    LABELS_NPZ,
    MIN_VISITS,
    adjusted_group_means,
    cluster_robust_ols,
    covariate_only_residuals,
    gap_statistics,
    permutation_p,
)
from safedrug_percluster.metrics import attach_labels, patient_bootstrap_ci

COHORT_DIR_DEFAULT = ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort"
MECH_PARTITIONS = ["long_k10", "ccs_group"]
N_MED = 112


def load_training_med_sets(cohort_dir):
    cohort_dir = Path(cohort_dir)
    with (cohort_dir / "records_reconstructed.pkl").open("rb") as fh:
        records = dill.load(fh)
    split_point = int(len(records) * 2 / 3)
    med_sets = []
    for patient in records[:split_point]:
        for visit in patient:
            med_sets.append(set(visit[2]))
    return med_sets, split_point


def drug_training_frequency(train_med_sets, n_med: int = N_MED) -> np.ndarray:
    counts = np.zeros(n_med, dtype=float)
    n = len(train_med_sets)
    for meds in train_med_sets:
        for m in meds:
            counts[m] += 1
    return counts / n if n else counts


def rare_drug_threshold(drug_freq: np.ndarray) -> float:
    return float(np.percentile(drug_freq, 10))


def rarity_features(gt_lookup: dict, hadm_ids, drug_freq: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for h in hadm_ids:
        y_gt = gt_lookup[h]
        idx = np.flatnonzero(np.asarray(y_gt) == 1)
        if len(idx) == 0:
            mean_freq, min_freq, n_rare = float("nan"), float("nan"), 0
        else:
            freqs = drug_freq[idx]
            mean_freq = float(freqs.mean())
            min_freq = float(freqs.min())
            n_rare = int((freqs < threshold).sum())
        rows.append({"HADM_ID": h, "mean_gt_freq": mean_freq, "min_gt_freq": min_freq,
                     "n_rare_gt": n_rare})
    return pd.DataFrame(rows)


def load_training_visits(cohort_dir, split_point: int) -> pd.DataFrame:
    master = pd.read_csv(Path(cohort_dir) / "master_visits.csv")
    train = master[master["safedrug_patient_index"] < split_point][["HADM_ID"]].copy()
    return attach_labels(train, DXTEXT_CSV, LABELS_NPZ, CCS_CSV)


def training_representation(train_labeled: pd.DataFrame, partition: str,
                             test_group_counts: pd.Series) -> pd.DataFrame:
    work = train_labeled.dropna(subset=[partition])
    total_train = len(work)
    train_counts = work.groupby(partition).size()
    total_test = test_group_counts.sum()
    groups = sorted(set(train_counts.index) | set(test_group_counts.index))
    rows = []
    for g in groups:
        train_n = int(train_counts.get(g, 0))
        test_n = int(test_group_counts.get(g, 0))
        train_share = train_n / total_train if total_train else float("nan")
        test_share = test_n / total_test if total_test else float("nan")
        ratio = train_share / test_share if test_share else float("nan")
        rows.append({"partition": partition, "group": g, "train_n_visits": train_n,
                     "train_share": train_share, "test_n_visits": test_n,
                     "test_share": test_share, "share_ratio": ratio})
    return pd.DataFrame(rows)


def precision_recall_gap_table(df: pd.DataFrame, partition: str) -> pd.DataFrame:
    groups = sorted(df[partition].unique())
    n_visits = df.groupby(partition).size()
    ci_p = patient_bootstrap_ci(df, partition, "precision", n_boot=2000, seed=0).set_index(partition)
    ci_r = patient_bootstrap_ci(df, partition, "recall", n_boot=2000, seed=0).set_index(partition)
    ci_f = patient_bootstrap_ci(df, partition, "f1", n_boot=2000, seed=0).set_index(partition)
    diff = df["n_med_pred"] - df["n_med_gt"]
    over_under = diff.groupby(df[partition]).mean()

    rows = []
    for g in groups:
        rows.append({
            "partition": partition, "group": g, "n_visits": int(n_visits.loc[g]),
            "n_patients": int(ci_p.loc[g, "n_patients"]) if g in ci_p.index else 0,
            "small": bool(n_visits.loc[g] < MIN_VISITS),
            "mean_precision": ci_p.loc[g, "mean"] if g in ci_p.index else float("nan"),
            "ci_low_precision": ci_p.loc[g, "ci_low"] if g in ci_p.index else float("nan"),
            "ci_high_precision": ci_p.loc[g, "ci_high"] if g in ci_p.index else float("nan"),
            "mean_recall": ci_r.loc[g, "mean"] if g in ci_r.index else float("nan"),
            "ci_low_recall": ci_r.loc[g, "ci_low"] if g in ci_r.index else float("nan"),
            "ci_high_recall": ci_r.loc[g, "ci_high"] if g in ci_r.index else float("nan"),
            "mean_f1": ci_f.loc[g, "mean"] if g in ci_f.index else float("nan"),
            "ci_low_f1": ci_f.loc[g, "ci_low"] if g in ci_f.index else float("nan"),
            "ci_high_f1": ci_f.loc[g, "ci_high"] if g in ci_f.index else float("nan"),
            "mean_over_under": float(over_under.loc[g]),
        })
    return pd.DataFrame(rows)


def precision_recall_permutation(df: pd.DataFrame, partition: str) -> pd.DataFrame:
    groups_all = sorted(df[partition].unique())
    group_code_map = {g: i for i, g in enumerate(groups_all)}
    rows = []
    for source_metric in ("precision", "recall"):
        work = df.dropna(subset=[source_metric])
        group_codes = work[partition].map(group_code_map).to_numpy()
        values = work[source_metric].to_numpy(dtype=float)
        subject = work["SUBJECT_ID"].to_numpy()
        perm = permutation_p(group_codes, subject, values, len(groups_all))
        for stat_key, stat_label in [("range", "range"), ("wsd", "weighted_sd")]:
            p_raw = perm[f"p_raw_{stat_key}"]
            p_bonf = float("nan") if np.isnan(p_raw) else min(1.0, p_raw * len(MECH_PARTITIONS))
            rows.append({
                "partition": partition, "scope": "test", "source_metric": source_metric,
                "statistic": stat_label, "observed": perm[f"observed_{stat_key}"],
                "null_mean": perm[f"null_mean_{stat_key}"], "null_p95": perm[f"null_p95_{stat_key}"],
                "z": perm[f"z_{stat_key}"], "p_raw": p_raw, "p_bonferroni": p_bonf,
            })
    return pd.DataFrame(rows)


def mechanism_correlations(group_table: pd.DataFrame, partition: str, source: str) -> pd.DataFrame:
    big = group_table[~group_table["small"]]
    rows = []
    for predictor in ("train_share", "mean_gt_freq", "mean_n_med_gt"):
        sub = big.dropna(subset=["adjusted_mean_jaccard", predictor])
        if len(sub) < 3 or sub[predictor].nunique() < 2:
            rows.append({"source": source, "partition": partition, "predictor": predictor,
                         "n_groups": len(sub), "spearman_r": float("nan")})
            continue
        rho = float(sub["adjusted_mean_jaccard"].corr(sub[predictor], method="spearman"))
        rows.append({"source": source, "partition": partition, "predictor": predictor,
                     "n_groups": len(sub), "spearman_r": rho})
    return pd.DataFrame(rows)


def _cell_means_design_v2(group_codes, n_groups, log_n_dx, n_med_gt, visit_index,
                           mean_gt_freq, n_rare_gt):
    n = len(group_codes)
    X = np.zeros((n, n_groups + 5))
    X[np.arange(n), group_codes] = 1.0
    X[:, n_groups] = log_n_dx
    X[:, n_groups + 1] = n_med_gt
    X[:, n_groups + 2] = visit_index
    X[:, n_groups + 3] = mean_gt_freq
    X[:, n_groups + 4] = n_rare_gt
    return X


def adjusted_group_means_v2(df: pd.DataFrame, partition: str, outcome: str, groups: list):
    work = df.dropna(subset=[partition, outcome, "n_dx", "n_med_gt", "visit_index",
                              "mean_gt_freq", "n_rare_gt"])
    group_index = {g: i for i, g in enumerate(groups)}
    group_codes = work[partition].map(group_index).to_numpy()
    log_n_dx = np.log(work["n_dx"].to_numpy(dtype=float))
    n_med_gt = work["n_med_gt"].to_numpy(dtype=float)
    visit_index = work["visit_index"].to_numpy(dtype=float)
    mean_gt_freq = work["mean_gt_freq"].to_numpy(dtype=float)
    n_rare_gt = work["n_rare_gt"].to_numpy(dtype=float)
    y = work[outcome].to_numpy(dtype=float)
    clusters = work["SUBJECT_ID"].to_numpy()

    X = _cell_means_design_v2(group_codes, len(groups), log_n_dx, n_med_gt, visit_index,
                               mean_gt_freq, n_rare_gt)
    beta, se, _ = cluster_robust_ols(X, y, clusters)

    covariate_effect = (
        beta[len(groups)] * log_n_dx.mean()
        + beta[len(groups) + 1] * n_med_gt.mean()
        + beta[len(groups) + 2] * visit_index.mean()
        + beta[len(groups) + 3] * mean_gt_freq.mean()
        + beta[len(groups) + 4] * n_rare_gt.mean()
    )
    adjusted = {g: float(beta[i] + covariate_effect) for i, g in enumerate(groups)}
    terms = list(groups) + ["log_n_dx", "n_med_gt", "visit_index", "mean_gt_freq", "n_rare_gt"]
    coef_table = pd.DataFrame({"term": terms, "estimate": beta, "cluster_robust_se": se})
    return adjusted, coef_table


def covariate_only_residuals_v2(df: pd.DataFrame, outcome: str):
    work = df.dropna(subset=[outcome, "n_dx", "n_med_gt", "visit_index",
                              "mean_gt_freq", "n_rare_gt"])
    n = len(work)
    X = np.column_stack([
        np.ones(n),
        np.log(work["n_dx"].to_numpy(dtype=float)),
        work["n_med_gt"].to_numpy(dtype=float),
        work["visit_index"].to_numpy(dtype=float),
        work["mean_gt_freq"].to_numpy(dtype=float),
        work["n_rare_gt"].to_numpy(dtype=float),
    ])
    y = work[outcome].to_numpy(dtype=float)
    clusters = work["SUBJECT_ID"].to_numpy()
    beta, _, _ = cluster_robust_ols(X, y, clusters)
    resid = y - X @ beta
    return pd.Series(resid, index=work.index), work.index


def _load_source_frames(eval_dir: Path, pooled_csv: Path) -> dict:
    sources = {}
    seed0 = pd.read_csv(eval_dir / "per_visit_metrics.csv")
    seed0 = seed0[(seed0["split"] == "test") & (seed0["has_label"])].copy()
    sources["seed0"] = seed0
    if pooled_csv.exists():
        sources["pooled"] = pd.read_csv(pooled_csv)
    else:
        print(f"[!] {pooled_csv} not found -- running mechanism decomposition on "
              "seed0 only", flush=True)
    return sources


def _gt_lookup(eval_dir: Path) -> dict:
    npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    return {int(h): npz["y_gt"][i] for i, h in enumerate(npz["HADM_ID"])}


def _group_over_under_and_rarity(scope_df, partition):
    rarity_group = scope_df.groupby(partition)[["mean_gt_freq", "min_gt_freq", "n_rare_gt"]].mean()
    rarity_group = rarity_group.reset_index().rename(columns={partition: "group"})
    mean_n_med_gt = scope_df.groupby(partition)["n_med_gt"].mean().reset_index()
    mean_n_med_gt = mean_n_med_gt.rename(columns={partition: "group", "n_med_gt": "mean_n_med_gt"})
    return rarity_group, mean_n_med_gt


def make_mechanism_figure(groups_df: pd.DataFrame, figs_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

    g = groups_df
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(g["mean_precision"], g["mean_recall"], s=np.clip(g["n_visits"] / 5, 8, 80))
    axes[0].set_xlabel("precision"); axes[0].set_ylabel("recall")
    axes[0].set_title("long_k10 - recall vs precision per group")

    axes[1].scatter(g["train_share"], g["adjusted_mean_jaccard"], s=30)
    axes[1].set_xlabel("train share"); axes[1].set_ylabel("adjusted mean Jaccard")
    axes[1].set_title("train representation vs performance")

    axes[2].scatter(g["mean_gt_freq"], g["adjusted_mean_jaccard"], s=30)
    axes[2].set_xlabel("mean ground-truth drug training frequency")
    axes[2].set_ylabel("adjusted mean Jaccard")
    axes[2].set_title("drug rarity vs performance")

    fig.tight_layout()
    fig.savefig(figs_dir / "fig_mechanism_long_k10.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug mechanism decomposition")
    parser.add_argument("--eval-dir", type=str, default="out/safedrug_eval")
    parser.add_argument("--pooled-csv", type=str,
                         default="out/safedrug_eval/seeds/per_visit_pooled.csv")
    parser.add_argument("--cohort-dir", type=str, default=str(COHORT_DIR_DEFAULT))
    parser.add_argument("--out-dir", type=str, default="out/safedrug_eval/mechanism")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dir = ROOT / args.eval_dir
    pooled_csv = ROOT / args.pooled_csv
    cohort_dir = Path(args.cohort_dir)
    out_dir = ROOT / args.out_dir
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_source_frames(eval_dir, pooled_csv)
    gt_lookup = _gt_lookup(eval_dir)

    train_med_sets, split_point = load_training_med_sets(cohort_dir)
    drug_freq = drug_training_frequency(train_med_sets)
    threshold = rare_drug_threshold(drug_freq)
    train_labeled = load_training_visits(cohort_dir, split_point)

    groups_frames, gap_frames, corr_frames, v2_frames = [], [], [], []

    for source_name, raw_df in sources.items():
        rarity = rarity_features(gt_lookup, raw_df["HADM_ID"].tolist(), drug_freq, threshold)
        df = raw_df.merge(rarity, on="HADM_ID", how="left")

        for partition in MECH_PARTITIONS:
            scope_df = df.dropna(subset=[partition]).copy()
            groups_all = sorted(scope_df[partition].unique())
            group_code_map = {g: i for i, g in enumerate(groups_all)}

            pr_table = precision_recall_gap_table(scope_df, partition)
            gap_table = precision_recall_permutation(scope_df, partition)
            gap_table.insert(0, "source", source_name)
            gap_frames.append(gap_table)

            test_group_counts = scope_df.groupby(partition).size()
            rep_table = training_representation(train_labeled, partition, test_group_counts)
            rarity_group, mean_n_med_gt = _group_over_under_and_rarity(scope_df, partition)

            adjusted_v1, _coef_v1 = adjusted_group_means(scope_df, partition, "jaccard", groups_all)
            adj_v1_df = pd.DataFrame({"group": groups_all,
                                       "adjusted_mean_jaccard": [adjusted_v1[g] for g in groups_all]})

            combined = pr_table.merge(
                rep_table[["group", "train_n_visits", "train_share", "test_share", "share_ratio"]],
                on="group", how="left",
            )
            combined = combined.merge(rarity_group, on="group", how="left")
            combined = combined.merge(mean_n_med_gt, on="group", how="left")
            combined = combined.merge(adj_v1_df, on="group", how="left")
            combined.insert(0, "source", source_name)
            groups_frames.append(combined)

            corr_frames.append(mechanism_correlations(combined, partition, source_name))

            adjusted_v2, _coef_v2 = adjusted_group_means_v2(scope_df, partition, "jaccard", groups_all)

            resid_v1, resid_idx_v1 = covariate_only_residuals(scope_df, "jaccard")
            gc_v1 = scope_df.loc[resid_idx_v1, partition].map(group_code_map).to_numpy()
            range_v1, wsd_v1, n_used_v1 = gap_statistics(gc_v1, resid_v1.to_numpy(), len(groups_all))

            resid_v2, resid_idx_v2 = covariate_only_residuals_v2(scope_df, "jaccard")
            gc_v2 = scope_df.loc[resid_idx_v2, partition].map(group_code_map).to_numpy()
            range_v2, wsd_v2, n_used_v2 = gap_statistics(gc_v2, resid_v2.to_numpy(), len(groups_all))

            for g in groups_all:
                v2_frames.append(pd.DataFrame([{
                    "source": source_name, "partition": partition, "group": g,
                    "adjusted_mean_v1": adjusted_v1[g], "adjusted_mean_v2": adjusted_v2[g],
                    "gap_range_v1": range_v1, "gap_wsd_v1": wsd_v1,
                    "gap_range_v2": range_v2, "gap_wsd_v2": wsd_v2,
                    "n_groups_used_v1": n_used_v1, "n_groups_used_v2": n_used_v2,
                }]))

    table_groups = pd.concat(groups_frames, ignore_index=True)
    table_gap = pd.concat(gap_frames, ignore_index=True)
    table_corr = pd.concat(corr_frames, ignore_index=True)
    table_v2 = pd.concat(v2_frames, ignore_index=True)

    table_groups.to_csv(out_dir / "table_mechanism_groups.csv", index=False)
    table_gap.to_csv(out_dir / "table_mechanism_precision_recall_gap.csv", index=False)
    table_corr.to_csv(out_dir / "table_mechanism_correlations.csv", index=False)
    table_v2.to_csv(out_dir / "table_mechanism_adjusted_v2.csv", index=False)

    fig_source = "pooled" if "pooled" in sources else "seed0"
    fig_df = table_groups[(table_groups["source"] == fig_source)
                           & (table_groups["partition"] == "long_k10")]
    if not fig_df.empty:
        make_mechanism_figure(fig_df, figs_dir)

    print(f"[+] wrote mechanism tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
