"""SafeDrug k sweep: does any (variant, k) diagnosis-clustering configuration
separate SafeDrug's performance more sharply than the pre-fixed long_k10,
under a max-statistic selection-corrected permutation test.

The permutation loop is a faithful, attributed port of
scripts/53_gap_ksweep.py lines 191-234 (max-statistic over 42 configs),
generalized from that script's fixed const/copy-prev predictor pair to this
task's raw/adjusted range/weighted-SD statistic pair -- reusing
scripts/safedrug_cluster_gap.py's gap_statistics/largest_remainder (imported)
throughout rather than redefining what "gap" or "null" mean.

See docs/specs/2026-09-04-safedrug-seeds-mechanism-ksweep-design.md
("D-D") for the full contract.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import (
    DXTEXT_CSV,
    LABELS_NPZ,
    MIN_VISITS,
    PARTITIONS as REFERENCE_PARTITIONS,
    covariate_only_residuals,
    gap_statistics,
    largest_remainder,
)
from safedrug_percluster import labels as percluster_labels

VARIANTS = ["short", "long", "concise"]
K_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
N_PERM_DEFAULT = 2000
SWEEP_SEED = 0
ARI_MIN = 0.85
REFERENCE_QUALITY_KEY = {
    "long_k10": ("long", 10), "short_k10": ("short", 10),
    "concise_k10": ("concise", 10), "long_k25": ("long", 25),
}


def attach_variant_label(df: pd.DataFrame, dxtext_csv, labels_npz, key: str) -> pd.Series:
    dxtext = pd.read_csv(dxtext_csv, encoding="utf-8-sig")
    labels = np.load(labels_npz)
    if len(labels[key]) != len(dxtext):
        raise ValueError(
            f"{labels_npz}[{key!r}] has {len(labels[key])} rows but {dxtext_csv} has "
            f"{len(dxtext)} rows -- positional alignment broken"
        )
    lookup = dict(zip(dxtext["HADM_ID"], labels[key]))
    return df["HADM_ID"].map(lookup)


def config_gap_stats(df: pd.DataFrame, group_col: str) -> dict:
    groups_all = sorted(df[group_col].dropna().unique())
    group_code_map = {g: i for i, g in enumerate(groups_all)}
    n_groups = len(groups_all)

    group_codes = df[group_col].map(group_code_map).to_numpy()
    values = df["jaccard"].to_numpy(dtype=float)
    subject = df["SUBJECT_ID"].to_numpy()

    raw_range, raw_wsd, n_used_raw = gap_statistics(group_codes, values, n_groups, MIN_VISITS)
    min_group_size = int(np.bincount(group_codes, minlength=n_groups).min()) if n_groups else 0

    resid, resid_index = covariate_only_residuals(df, "jaccard")
    resid_df = df.loc[resid_index]
    resid_group_codes = resid_df[group_col].map(group_code_map).to_numpy()
    resid_subject = resid_df["SUBJECT_ID"].to_numpy()
    resid_values = resid.to_numpy()
    adj_range, adj_wsd, n_used_adj = gap_statistics(resid_group_codes, resid_values, n_groups, MIN_VISITS)

    return {
        "n_groups_total": n_groups, "min_group_size": min_group_size,
        "raw_range": raw_range, "raw_wsd": raw_wsd, "n_groups_used_raw": n_used_raw,
        "adj_range": adj_range, "adj_wsd": adj_wsd, "n_groups_used_adj": n_used_adj,
        "group_codes": group_codes, "subject": subject,
        "resid_group_codes": resid_group_codes, "resid_subject": resid_subject,
        "resid_values": resid_values, "raw_values": values, "n_groups": n_groups,
    }


def _build_permutation_target(group_of_visit, subject_of_visit, n_groups):
    """Faithful, attributed port of
    safedrug_cluster_gap._build_permutation_target / scripts/53_gap_ksweep.py's
    pat_per_cl + largest_remainder (lines 150-171)."""
    patients = np.sort(np.unique(subject_of_visit))
    n_patients = len(patients)
    patient_pos = {p: i for i, p in enumerate(patients)}
    patient_of_visit = np.array([patient_pos[s] for s in subject_of_visit])

    pat_per_group = np.zeros(n_groups)
    for g in range(n_groups):
        pat_per_group[g] = len(np.unique(subject_of_visit[group_of_visit == g]))
    total = pat_per_group.sum()
    props = pat_per_group / total if total else np.zeros(n_groups)
    target = largest_remainder(props, n_patients)
    return target, patients, patient_of_visit


def _permuted_group(rng, target, pat_of_visit):
    """Reassign patients to groups per `target` (a per-group patient quota
    summing to the total patient count, from _build_permutation_target), then
    broadcast each visit's permuted group from its own patient's assignment."""
    perm = rng.permutation(target.sum())
    cl_of_pat = np.empty(target.sum(), dtype=int)
    s = 0
    for c, sz in enumerate(target):
        cl_of_pat[perm[s : s + sz]] = c
        s += sz
    return cl_of_pat[pat_of_visit]


def _prep_config(name_variant, name_k, df, group_col):
    stats = config_gap_stats(df, group_col)
    raw_target, raw_patients, raw_pat_of_visit = _build_permutation_target(
        stats["group_codes"], stats["subject"], stats["n_groups"]
    )
    resid_target, resid_patients, resid_pat_of_visit = _build_permutation_target(
        stats["resid_group_codes"], stats["resid_subject"], stats["n_groups"]
    )
    return {
        "variant": name_variant, "k": name_k, "n_groups": stats["n_groups"],
        "raw_target": raw_target, "raw_pat_of_visit": raw_pat_of_visit,
        "raw_values": stats["raw_values"],
        "resid_target": resid_target, "resid_pat_of_visit": resid_pat_of_visit,
        "resid_values": stats["resid_values"],
        "obs": (stats["raw_range"], stats["raw_wsd"], stats["adj_range"], stats["adj_wsd"]),
        "stats": stats,
    }


def _run_permutation_pool(preps, n_perm, seed):
    rng = np.random.default_rng(seed)
    null = np.full((n_perm, len(preps), 4), np.nan)
    t0 = time.time()
    for b in range(n_perm):
        for i, p in enumerate(preps):
            permuted_raw = _permuted_group(rng, p["raw_target"], p["raw_pat_of_visit"])
            r_range, r_wsd, _ = gap_statistics(permuted_raw, p["raw_values"], p["n_groups"], MIN_VISITS)

            permuted_resid = _permuted_group(rng, p["resid_target"], p["resid_pat_of_visit"])
            a_range, a_wsd, _ = gap_statistics(permuted_resid, p["resid_values"], p["n_groups"], MIN_VISITS)

            null[b, i] = (r_range, r_wsd, a_range, a_wsd)
        if (b + 1) % 500 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (b + 1) * (n_perm - b - 1)
            print(f"[.] permutation {b + 1}/{n_perm} {elapsed:.0f}s (remaining ~{remaining:.0f}s)",
                  flush=True)
    return null


_STAT_NAMES = ["range_raw", "wsd_raw", "range_adj", "wsd_adj"]


def _stat_p_values(obs, null_col, null_max_col=None):
    nd = null_col[~np.isnan(null_col)]
    p_raw = (1 + int((nd >= obs).sum())) / (1 + len(nd)) if len(nd) else float("nan")
    if null_max_col is None:
        return p_raw, float("nan")
    nm = null_max_col[~np.isnan(null_max_col)]
    p_sel = (1 + int((nm >= obs).sum())) / (1 + len(nm)) if len(nm) else float("nan")
    return p_raw, p_sel


def _null_zscore_max(null: np.ndarray):
    """Controller ruling (Task D fix round 1): the raw max-statistic
    correction (`null_max` / `p_selcorr_*`) is dominated by the large-k
    configs, since `gap_statistics`'s range/weighted-SD grow with the number
    of groups even under the null -- so a small-k config's real effect can
    never look extreme relative to a null pool full of large-k noise, and
    `p_selcorr_*` is ~1.0 almost everywhere.

    This standardizes each config's null distribution to *its own* mean/sd
    (over all n_perm permutations of that config) before taking the
    cross-config max per round, removing the scale dependence on k while
    still selecting over all swept configs each round -- same purpose as
    `null_max`/`_stat_p_values`'s `p_sel` branch, on a standardized scale.

    null: (n_perm, n_configs, 4) array from `_run_permutation_pool`.
    Returns (null_mean, null_sd, null_z, null_z_max): null_mean/null_sd have
    shape (n_configs, 4) (each config's own null moments, per stat); null_z is
    `null` standardized per-config/per-stat; null_z_max has shape (n_perm, 4)
    (max standardized null value across configs, per stat, per round). A
    config with zero null variance for a stat gets NaN sd (and NaN z) for
    that stat -- standardization is undefined, not zero.
    """
    n_perm, n_configs, n_stats = null.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        null_mean = np.nanmean(null, axis=0)
        null_sd = np.nanstd(null, axis=0, ddof=0)
        null_sd = np.where(null_sd == 0, np.nan, null_sd)
        with np.errstate(invalid="ignore", divide="ignore"):
            null_z = (null - null_mean[None, :, :]) / null_sd[None, :, :]
        if n_configs:
            null_z_max = np.nanmax(null_z, axis=1)
        else:
            null_z_max = np.full((n_perm, n_stats), np.nan)
    return null_mean, null_sd, null_z, null_z_max


def run_sweep(source_dfs: dict, configs: list, reference_partitions: list,
              quality: pd.DataFrame, n_perm: int = N_PERM_DEFAULT, seed: int = SWEEP_SEED) -> pd.DataFrame:
    all_rows = []

    for source_name, df in source_dfs.items():
        preps = []
        for variant, k, key in configs:
            work = df.copy()
            work["_cfg_group"] = attach_variant_label(work, DXTEXT_CSV, LABELS_NPZ, key)
            work = work.dropna(subset=["_cfg_group"]).copy()
            preps.append(_prep_config(variant, k, work, "_cfg_group"))

        null = _run_permutation_pool(preps, n_perm, seed) if preps else np.zeros((n_perm, 0, 4))
        null_max = np.nanmax(null, axis=1) if preps else np.full((n_perm, 4), np.nan)
        null_mean, null_sd, null_z, null_z_max = _null_zscore_max(null)

        for i, p in enumerate(preps):
            row = {
                "source": source_name, "variant": p["variant"], "k": p["k"], "is_reference": False,
                "n_groups_total": p["stats"]["n_groups_total"],
                "min_group_size": p["stats"]["min_group_size"],
                "raw_range": p["stats"]["raw_range"], "raw_wsd": p["stats"]["raw_wsd"],
                "n_groups_used_raw": p["stats"]["n_groups_used_raw"],
                "adj_range": p["stats"]["adj_range"], "adj_wsd": p["stats"]["adj_wsd"],
                "n_groups_used_adj": p["stats"]["n_groups_used_adj"],
            }
            for j, name in enumerate(_STAT_NAMES):
                p_raw, p_sel = _stat_p_values(p["obs"][j], null[:, i, j], null_max[:, j])
                row[f"p_raw_{name}"] = p_raw
                row[f"p_selcorr_{name}"] = p_sel

                obs_z = (p["obs"][j] - null_mean[i, j]) / null_sd[i, j]
                _, p_sel_z = _stat_p_values(obs_z, null_z[:, i, j], null_z_max[:, j])
                row[f"p_selcorr_z_{name}"] = p_sel_z
            all_rows.append(row)

        for partition in reference_partitions:
            work = df.dropna(subset=[partition]).copy()
            if work.empty:
                continue
            ref_prep = _prep_config(partition, float("nan"), work, partition)
            ref_null = _run_permutation_pool([ref_prep], n_perm, seed)[:, 0, :]

            # D-D5: k = the partition's own k (long_k10/short_k10/concise_k10 -> 10,
            # long_k25 -> 25); NaN only for ccs_group (no k to report).
            partition_k = REFERENCE_QUALITY_KEY.get(partition, (None, float("nan")))[1]
            row = {
                "source": source_name, "variant": partition, "k": partition_k, "is_reference": True,
                "n_groups_total": ref_prep["stats"]["n_groups_total"],
                "min_group_size": ref_prep["stats"]["min_group_size"],
                "raw_range": ref_prep["stats"]["raw_range"], "raw_wsd": ref_prep["stats"]["raw_wsd"],
                "n_groups_used_raw": ref_prep["stats"]["n_groups_used_raw"],
                "adj_range": ref_prep["stats"]["adj_range"], "adj_wsd": ref_prep["stats"]["adj_wsd"],
                "n_groups_used_adj": ref_prep["stats"]["n_groups_used_adj"],
            }
            for j, name in enumerate(_STAT_NAMES):
                p_raw, _ = _stat_p_values(ref_prep["obs"][j], ref_null[:, j], None)
                row[f"p_raw_{name}"] = p_raw
                row[f"p_selcorr_{name}"] = float("nan")
                row[f"p_selcorr_z_{name}"] = float("nan")
            all_rows.append(row)

    table = pd.DataFrame(all_rows)
    if table.empty:
        return table

    q = quality[["변형", "k", "실루엣", "ARI_시드간", "최소군집", "평균순위"]].rename(
        columns={"변형": "variant"}
    )
    table = table.merge(q, on=["variant", "k"], how="left")

    for idx, row in table[table["is_reference"]].iterrows():
        key = REFERENCE_QUALITY_KEY.get(row["variant"])
        if key is None:
            continue
        variant_q, k_q = key
        match = quality[(quality["변형"] == variant_q) & (quality["k"] == k_q)]
        if not match.empty:
            for col in ["실루엣", "ARI_시드간", "최소군집", "평균순위"]:
                table.loc[idx, col] = match.iloc[0][col]

    table["meets_min_group"] = table["n_groups_used_raw"] == table["n_groups_total"]
    table["meets_ari"] = table["ARI_시드간"] >= ARI_MIN
    table["meets_significance"] = (
        (table["p_selcorr_range_adj"] < 0.05) | (table["p_selcorr_wsd_adj"] < 0.05)
    )
    table["qualifies"] = (
        (~table["is_reference"]) & table["meets_min_group"].fillna(False)
        & table["meets_ari"].fillna(False) & table["meets_significance"].fillna(False)
    )

    # Controller ruling (Task D fix round 1): standardized-scale counterpart of
    # `meets_significance`/`qualifies`, using the z-standardized selection
    # correction (p_selcorr_z_*) instead of the raw one, since the raw
    # max-statistic pool is dominated by large-k configs (see
    # `_null_zscore_max`). `qualifies` itself is left untouched.
    table["meets_significance_z"] = (
        (table["p_selcorr_z_range_adj"] < 0.05) | (table["p_selcorr_z_wsd_adj"] < 0.05)
    )
    table["qualifies_z"] = (
        (~table["is_reference"]) & table["meets_min_group"].fillna(False)
        & table["meets_ari"].fillna(False) & table["meets_significance_z"].fillna(False)
    )
    return table


def _load_source_dfs(eval_dir: Path, pooled_csv: Path) -> dict:
    sources = {}
    seed0 = pd.read_csv(eval_dir / "per_visit_metrics.csv")
    seed0 = seed0[(seed0["split"] == "test") & (seed0["has_label"])].copy()
    sources["seed0"] = seed0
    if pooled_csv.exists():
        sources["pooled"] = pd.read_csv(pooled_csv)
    else:
        print(f"[!] {pooled_csv} not found -- running k sweep on seed0 only", flush=True)
    return sources


def make_ksweep_figure(table: pd.DataFrame, figs_dir: Path, lang: str = "en") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percluster_labels.apply_korean_font()
    ft = percluster_labels.figure_text
    dx = table[~table["is_reference"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = {"seed0": "-", "pooled": "--"}
    for source_name, sub_s in dx.groupby("source"):
        ls = styles.get(source_name, "-")
        for variant, sub in sub_s.groupby("variant"):
            sub = sub.sort_values("k")
            axes[0].plot(sub["k"], sub["adj_range"], ls, marker="o", ms=4,
                         label=f"{variant} ({source_name})")
            sig = sub[sub["meets_significance"] == True]  # noqa: E712
            if not sig.empty:
                axes[0].plot(sig["k"], sig["adj_range"], "o", ms=7, mfc="none", mec="black")
            axes[1].plot(sub["k"], sub["ARI_시드간"], ls, marker="s", ms=4,
                         label=f"{variant} ({source_name})")
    axes[1].axhline(ARI_MIN, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("k"); axes[0].set_ylabel(ft("ksweep_ylabel_range", lang))
    axes[0].set_title(ft("ksweep_panel0_title", lang))
    axes[1].set_xlabel("k"); axes[1].set_ylabel("ARI (시드간)")
    axes[1].set_title(ft("ksweep_panel1_title", lang))
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(
        figs_dir / f"fig_k_sweep{percluster_labels.fig_suffix(lang)}.png",
        dpi=160, bbox_inches="tight",
    )
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug k sweep")
    parser.add_argument("--eval-dir", type=str, default="out/safedrug_eval")
    parser.add_argument("--pooled-csv", type=str,
                         default="out/safedrug_eval/seeds/per_visit_pooled.csv")
    parser.add_argument("--partition-quality", type=str, default="out/table77_partition_quality.csv")
    parser.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    parser.add_argument("--seed", type=int, default=SWEEP_SEED)
    parser.add_argument("--out-dir", type=str, default="out/safedrug_eval/ksweep")
    parser.add_argument(
        "--lang", choices=["en", "ko"], default="en",
        help="figure language. ko implies --figures-only.",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="skip the sweep computation; regenerate the k-sweep figure only, "
        "from table_k_sweep.csv already written to --out-dir.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = ROOT / args.out_dir
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    figures_only = args.figures_only or args.lang != "en"

    if figures_only:
        table = pd.read_csv(out_dir / "table_k_sweep.csv")
        make_ksweep_figure(table, figs_dir, lang=args.lang)
        print(f"[+] wrote k-sweep figures ({args.lang}) to {figs_dir}", flush=True)
        return

    eval_dir = ROOT / args.eval_dir
    pooled_csv = ROOT / args.pooled_csv
    quality = pd.read_csv(ROOT / args.partition_quality, encoding="utf-8-sig")

    sources = _load_source_dfs(eval_dir, pooled_csv)
    configs = [(v, k, f"{v}_k{k}") for v in VARIANTS for k in K_GRID]

    table = run_sweep(sources, configs, REFERENCE_PARTITIONS, quality,
                       n_perm=args.n_perm, seed=args.seed)
    table.to_csv(out_dir / "table_k_sweep.csv", index=False)

    dx_seed0 = table[(~table["is_reference"]) & (table["source"] == "seed0")]
    if len(dx_seed0) >= 3:
        rho = dx_seed0["평균순위"].corr(dx_seed0["adj_range"], method="spearman")
        print(f"[=] Spearman(quality rank, adjusted gap) over 42 dxtext configs (seed0) = "
              f"{rho:.3f}", flush=True)

    make_ksweep_figure(table, figs_dir, lang=args.lang)
    print(f"[+] wrote k-sweep table to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
