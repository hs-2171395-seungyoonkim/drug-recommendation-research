"""Seed robustness for the SafeDrug per-cluster evaluation.

Aggregates the four seed-0..3 gap-analysis outputs (each already produced by
running scripts/safedrug_cluster_gap.py --eval-dir <dir>) into per-seed and
cross-seed tables, and builds a seed-pooled per-visit metric (mean over seeds,
same visits) that is re-run through safedrug_cluster_gap.py's own group-table /
adjustment / permutation machinery (imported, not copied).

See docs/specs/2026-09-04-safedrug-seeds-mechanism-ksweep-design.md
("D-B") for the full contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import safedrug_cluster_gap as scg
from safedrug_cluster_gap import ADJUSTED_OUTCOMES, PARTITIONS
from safedrug_percluster import labels as percluster_labels
from safedrug_percluster.labels import short_group_label

DEFAULT_EVAL_DIRS = [
    "out/safedrug_eval",
    "out/safedrug_eval/seed_1",
    "out/safedrug_eval/seed_2",
    "out/safedrug_eval/seed_3",
]

POOLED_METRIC_COLUMNS = [
    "jaccard", "precision", "recall", "f1", "prauc", "ddi_rate_visit",
    "n_med_pred", "n_med_gt", "n_dx",
]
STATIC_COLUMNS = [
    "patient_index", "visit_index", "HADM_ID", "SUBJECT_ID", "split",
    "long_k10", "long_k25", "short_k10", "concise_k10", "ccs_group", "has_label",
]

_SEED_DIR_RE = re.compile(r"seed_(\d+)$")


def seed_label(eval_dir) -> int:
    """out/safedrug_eval -> 0 (the base directory is the original seed);
    .../seed_N -> N, parsed from the directory's own basename."""
    m = _SEED_DIR_RE.search(Path(eval_dir).name)
    return int(m.group(1)) if m else 0


def load_seed_official(eval_dirs) -> pd.DataFrame:
    rows = []
    for d in eval_dirs:
        d = Path(d)
        manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        seed = seed_label(d)
        for split, metrics in manifest["official_metrics"].items():
            rows.append({"seed": seed, "split": split, **metrics})
    return pd.DataFrame(rows)


def load_seed_groups(eval_dirs) -> pd.DataFrame:
    frames = []
    for d in eval_dirs:
        d = Path(d)
        seed = seed_label(d)
        adjusted = pd.read_csv(d / "table_adjusted_means.csv")
        adjusted = adjusted[adjusted["scope"] == "test"].copy()
        summary = pd.read_csv(d / "table_group_summary.csv")
        summary = summary[summary["scope"] == "test"][["partition", "group", "n_visits"]]
        merged = adjusted.merge(summary, on=["partition", "group"], how="left")
        merged.insert(0, "seed", seed)
        frames.append(merged[["seed", "partition", "outcome", "group", "n_visits",
                               "raw_mean", "adjusted_mean"]])
    return pd.concat(frames, ignore_index=True)


def load_seed_gap(eval_dirs) -> pd.DataFrame:
    frames = []
    for d in eval_dirs:
        d = Path(d)
        seed = seed_label(d)
        perm = pd.read_csv(d / "table_permutation.csv")
        perm = perm[perm["scope"] == "test"].copy()
        perm.insert(0, "seed", seed)
        frames.append(perm)
    return pd.concat(frames, ignore_index=True)


def aggregate_seed_groups(seed_groups: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_groups.groupby(["partition", "outcome", "group"])
    return grouped.agg(
        n_seeds=("adjusted_mean", "size"),
        mean_adjusted=("adjusted_mean", "mean"),
        sd_adjusted=("adjusted_mean", lambda s: s.std(ddof=1)),
        mean_raw=("raw_mean", "mean"),
        sd_raw=("raw_mean", lambda s: s.std(ddof=1)),
    ).reset_index()


def seed_rank_correlations(seed_groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (partition, outcome), sub in seed_groups.groupby(["partition", "outcome"]):
        wide = sub.pivot(index="group", columns="seed", values="adjusted_mean")
        seeds = sorted(wide.columns)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = seeds[i], seeds[j]
                pair = wide[[a, b]].dropna()
                n = len(pair)
                rho = float(pair[a].corr(pair[b], method="spearman")) if n >= 3 else float("nan")
                rows.append({"partition": partition, "outcome": outcome,
                             "seed_a": a, "seed_b": b, "n_groups": n, "spearman_r": rho})
    return pd.DataFrame(rows)


def count_significant_seeds(seed_gap: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    grouped = seed_gap.groupby(["partition", "outcome", "source", "statistic"])
    return grouped.agg(
        n_seeds=("p_raw", "size"),
        n_significant=("p_raw", lambda s: int((s < alpha).sum())),
    ).reset_index()


def pool_per_visit_metrics(eval_dirs) -> pd.DataFrame:
    eval_dirs = [Path(d) for d in eval_dirs]
    frames = []
    for d in eval_dirs:
        df = pd.read_csv(d / "per_visit_metrics.csv")
        df = df[(df["split"] == "test") & (df["has_label"])].copy()
        df = df.sort_values("HADM_ID").reset_index(drop=True)
        frames.append(df)

    base_ids = set(frames[0]["HADM_ID"])
    for i in range(1, len(frames)):
        ids = set(frames[i]["HADM_ID"])
        if ids != base_ids:
            raise ValueError(
                f"HADM_ID mismatch between {eval_dirs[0]} and {eval_dirs[i]}: "
                f"{len(base_ids ^ ids)} visits differ -- seed-pooled visit sets "
                "must be identical (test-split visits do not depend on the seed)"
            )

    pooled = frames[0][STATIC_COLUMNS].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for col in POOLED_METRIC_COLUMNS:
            mat = np.stack([f[col].to_numpy(dtype=float) for f in frames], axis=0)
            pooled[col] = np.nanmean(mat, axis=0)
    return pooled


def pooled_group_tables(pooled_df: pd.DataFrame, n_perm: int = None):
    """Runs safedrug_cluster_gap's own group_summary_table / adjusted_group_means /
    covariate_only_residuals / permutation_p (all imported, unmodified) over
    pooled_df, scope fixed to "test" (pooled_df is already test-scope only).

    n_perm=None (the production default) uses safedrug_cluster_gap.N_PERM
    (10,000). Tests pass a small n_perm explicitly to stay fast -- threaded
    through to every permutation_p call as an explicit keyword argument, NOT
    via monkeypatching the safedrug_cluster_gap.N_PERM module attribute:
    permutation_p's own `n_perm=N_PERM` default is evaluated once, at
    permutation_p's *definition* time (standard Python default-argument
    binding), so reassigning the module attribute afterwards would silently
    have no effect on calls that rely on that default.
    """
    n_perm = scg.N_PERM if n_perm is None else n_perm
    groups_frames = []
    perm_rows = []

    for partition in PARTITIONS:
        scope_df = pooled_df.dropna(subset=[partition]).copy()
        summary = scg.group_summary_table(scope_df, partition, "test")
        groups_all = sorted(scope_df[partition].unique())
        group_code_map = {g: i for i, g in enumerate(groups_all)}

        adjusted_by_group = {g: {} for g in groups_all}
        for outcome in ADJUSTED_OUTCOMES:
            work = scope_df.dropna(subset=[outcome, "n_dx", "n_med_gt", "visit_index"])
            group_codes = work[partition].map(group_code_map).to_numpy()
            values = work[outcome].to_numpy(dtype=float)
            subject = work["SUBJECT_ID"].to_numpy()

            perm_raw = scg.permutation_p(group_codes, subject, values, len(groups_all),
                                          n_perm=n_perm)

            adjusted, _coef = scg.adjusted_group_means(scope_df, partition, outcome, groups_all)
            for g in groups_all:
                adjusted_by_group[g][f"adjusted_mean_{outcome}"] = adjusted[g]

            resid, resid_index = scg.covariate_only_residuals(scope_df, outcome)
            resid_df = scope_df.loc[resid_index]
            resid_group_codes = resid_df[partition].map(group_code_map).to_numpy()
            resid_subject = resid_df["SUBJECT_ID"].to_numpy()
            perm_resid = scg.permutation_p(resid_group_codes, resid_subject,
                                            resid.to_numpy(), len(groups_all), n_perm=n_perm)

            for source_name, perm in [("raw", perm_raw), ("residual", perm_resid)]:
                for stat_key, stat_label in [("range", "range"), ("wsd", "weighted_sd")]:
                    p_raw = perm[f"p_raw_{stat_key}"]
                    p_bonf = float("nan") if np.isnan(p_raw) else min(1.0, p_raw * len(PARTITIONS))
                    perm_rows.append({
                        "partition": partition, "scope": "test", "outcome": outcome,
                        "source": source_name, "statistic": stat_label,
                        "observed": perm[f"observed_{stat_key}"],
                        "null_mean": perm[f"null_mean_{stat_key}"],
                        "null_p95": perm[f"null_p95_{stat_key}"],
                        "z": perm[f"z_{stat_key}"], "p_raw": p_raw, "p_bonferroni": p_bonf,
                    })

        adj_df = pd.DataFrame(
            [{"group": g, **vals} for g, vals in adjusted_by_group.items()]
        )
        summary = summary.merge(adj_df, on="group", how="left")
        groups_frames.append(summary)

    table_pooled_groups = pd.concat(groups_frames, ignore_index=True)
    table_pooled_permutation = pd.DataFrame(perm_rows)
    return table_pooled_groups, table_pooled_permutation


def make_seed_figure(pooled_groups: pd.DataFrame, seed_groups: pd.DataFrame, figs_dir: Path,
                      lang: str = "en"):
    """long_k10/jaccard by seed: per-seed points, a mean +/- SD error bar
    across seeds, and the pooled (per-visit-mean) marker, groups ordered by
    the pooled adjusted mean, short curated y-axis labels.

    Fix-round note: pooled_groups' "group" column is computed fresh in this
    process (pooled_group_tables never round-trips through a CSV), so a
    numeric long_k10 id stays a plain float (e.g. 0.0). seed_groups' "group"
    column instead comes from load_seed_groups() -> pd.read_csv() over each
    seed's table_adjusted_means.csv, whose "group" column mixes numeric ids
    with ccs_group name strings across partitions and so round-trips as an
    object column of literal strings ("0.0", not 0.0). A bare `.isin()` join
    between the two, without normalizing both sides first, silently matches
    nothing -- every seed row got dropped and only the pooled diamonds ever
    rendered. Both sides are normalized via safedrug_cluster_gap's own
    _normalize_group_id (imported as `scg` at this module's top) before the
    join, exactly like safedrug_cluster_gap.make_figures does for its own
    group ids.

    Returns the closed Figure (for tests to inspect its axes/artists) --
    main()'s caller ignores the return value.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percluster_labels.apply_korean_font()

    pooled_long = pooled_groups[pooled_groups["partition"] == "long_k10"].copy()
    pooled_long["group"] = pooled_long["group"].map(scg._normalize_group_id)
    pooled_long = pooled_long.sort_values("adjusted_mean_jaccard")
    order = pooled_long["group"].tolist()
    group_pos = {g: i for i, g in enumerate(order)}

    seed_long = seed_groups[(seed_groups["partition"] == "long_k10")
                             & (seed_groups["outcome"] == "jaccard")].copy()
    seed_long["group"] = seed_long["group"].map(scg._normalize_group_id)
    seed_long = seed_long[seed_long["group"].isin(group_pos)]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    seed_label = percluster_labels.figure_text("seed_legend_seed", lang)
    for seed, sub in seed_long.groupby("seed"):
        y = [group_pos[g] for g in sub["group"]]
        ax.scatter(sub["adjusted_mean"], y, s=22, alpha=0.7, zorder=2,
                   label=seed_label.format(seed=seed))

    agg = aggregate_seed_groups(seed_long)
    agg = agg[agg["group"].isin(group_pos)]
    agg_y = [group_pos[g] for g in agg["group"]]
    ax.errorbar(
        agg["mean_adjusted"], agg_y, xerr=agg["sd_adjusted"],
        fmt="s", color="#e67e22", ecolor="#e67e22", elinewidth=1.5, capsize=4,
        markersize=6, linestyle="none", zorder=3,
        label=percluster_labels.figure_text("seed_legend_mean_sd", lang),
    )

    ax.scatter(pooled_long["adjusted_mean_jaccard"], [group_pos[g] for g in order],
               marker="D", color="#0b0b0b", s=40, zorder=4,
               label=percluster_labels.figure_text("seed_legend_pooled", lang))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short_group_label("long_k10", g, lang=lang) for g in order], fontsize=8)
    ax.set_xlabel(percluster_labels.figure_text("seed_figure_xlabel", lang))
    ax.set_title(percluster_labels.figure_text("seed_figure_title", lang))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(
        figs_dir / f"fig_seed_long_k10{percluster_labels.fig_suffix(lang)}.png",
        dpi=160, bbox_inches="tight",
    )
    plt.close(fig)
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug seed robustness")
    parser.add_argument("--eval-dirs", nargs="+", default=DEFAULT_EVAL_DIRS)
    parser.add_argument("--out-dir", type=str, default="out/safedrug_eval/seeds")
    parser.add_argument(
        "--lang", choices=["en", "ko"], default="en",
        help="figure language. ko implies --figures-only.",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="skip every table computation; regenerate the seed figure only, "
        "from table_pooled_groups.csv / table_seed_groups.csv already written "
        "to --out-dir.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    eval_dirs = [ROOT / d for d in args.eval_dirs]
    out_dir = ROOT / args.out_dir
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    figures_only = args.figures_only or args.lang != "en"

    if figures_only:
        pooled_groups = pd.read_csv(out_dir / "table_pooled_groups.csv")
        seed_groups = pd.read_csv(out_dir / "table_seed_groups.csv")
        make_seed_figure(pooled_groups, seed_groups, figs_dir, lang=args.lang)
        print(f"[+] wrote seed robustness figures ({args.lang}) to {figs_dir}", flush=True)
        return

    load_seed_official(eval_dirs).to_csv(out_dir / "table_seed_official.csv", index=False)

    seed_groups = load_seed_groups(eval_dirs)
    seed_groups.to_csv(out_dir / "table_seed_groups.csv", index=False)

    seed_rank_correlations(seed_groups).to_csv(out_dir / "table_seed_rank_correlations.csv", index=False)

    load_seed_gap(eval_dirs).to_csv(out_dir / "table_seed_gap.csv", index=False)

    pooled = pool_per_visit_metrics(eval_dirs)
    pooled.to_csv(out_dir / "per_visit_pooled.csv", index=False)

    pooled_groups, pooled_perm = pooled_group_tables(pooled)
    pooled_groups.to_csv(out_dir / "table_pooled_groups.csv", index=False)
    pooled_perm.to_csv(out_dir / "table_pooled_permutation.csv", index=False)

    make_seed_figure(pooled_groups, seed_groups, figs_dir, lang=args.lang)
    print(f"[+] wrote seed robustness tables to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
