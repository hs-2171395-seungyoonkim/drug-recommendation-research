"""SafeDrug Intervention E seed report (R6).

Aggregates the 4-seed x 3-variant (ccs/k10/global) decode-budget dumps --
each already processed by scripts/safedrug_cluster_gap.py and
scripts/safedrug_excess_ddi.py (Task C's job, not this script's) -- into
overall/excess-gap/Jaccard-gap/group tables, a two-panel excess-DDI figure,
and a Korean report. No new statistics: every number here is read from files
those two scripts already know how to write.

See docs/superpowers/specs/2026-09-06-safedrug-decode-ddi-budget-design.md
("R6") for the full contract, including why table_adjusted_means.csv's and
table_excess_ddi_groups.csv's own "group" columns need
safedrug_cluster_gap._normalize_group_id before they can be joined.
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

from safedrug_cluster_gap import DDI_ADJ_PATH, _normalize_group_id
from safedrug_mitigation_compare import flat_test_metrics
from safedrug_percluster import labels as percluster_labels
from safedrug_percluster.labels import short_group_label

VARIANTS = ("baseline", "ccs", "k10", "global")
AXES = ("long_k10", "ccs_group")


def _variant_dir(seed_baseline_dir: Path, variant_root: Path, seed_idx: int, variant: str) -> Path:
    if variant == "baseline":
        return seed_baseline_dir
    return variant_root / f"seed{seed_idx}" / variant


def build_table_e_overall(baseline_dirs: list, variant_root: Path, ddi_adj: np.ndarray) -> pd.DataFrame:
    """Per (seed, variant) row: flat_test_metrics (imported unmodified from
    safedrug_mitigation_compare -- SafeDrug's own mean-of-per-visit
    convention, plus the aggregate DDI rate) and the variant's own chosen
    p_floor (NaN for baseline, which has no decode_policy block)."""
    rows = []
    for seed_idx, baseline_dir in enumerate(baseline_dirs):
        for variant in VARIANTS:
            d = _variant_dir(baseline_dir, variant_root, seed_idx, variant)
            metrics = flat_test_metrics(d, ddi_adj)
            p_floor = float("nan")
            if variant != "baseline":
                manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
                p_floor = manifest["decode_policy"]["p_floor"]
            rows.append({
                "seed": seed_idx, "variant": variant, "p_floor": p_floor,
                "jaccard_mean": metrics["jaccard_mean"], "f1_mean": metrics["f1_mean"],
                "prauc_mean": metrics["prauc_mean"], "ddi_rate": metrics["ddi_rate"],
                "avg_n_med_pred": metrics["avg_n_med_pred"], "n_test_visits": metrics["n_test_visits"],
            })
    return pd.DataFrame(rows)


def build_table_e_excess_gap(baseline_dirs: list, variant_root: Path) -> pd.DataFrame:
    """Long format: one row per (seed, variant, axis, statistic) read from
    that dir's excess_ddi/table_excess_ddi_permutation.csv, plus one
    additional seed="mean" row per (variant, axis, statistic) -- the plain
    arithmetic mean of observed/p_raw across the numeric seed rows (a
    descriptive summary only, not a meta-analytic p-value combination)."""
    rows = []
    for seed_idx, baseline_dir in enumerate(baseline_dirs):
        for variant in VARIANTS:
            d = _variant_dir(baseline_dir, variant_root, seed_idx, variant)
            perm = pd.read_csv(d / "excess_ddi" / "table_excess_ddi_permutation.csv")
            for axis in AXES:
                sub = perm[perm["partition"] == axis]
                for _, r in sub.iterrows():
                    rows.append({
                        "seed": seed_idx, "variant": variant, "axis": axis,
                        "statistic": r["statistic"], "observed": r["observed"], "p_raw": r["p_raw"],
                    })
    table = pd.DataFrame(rows)

    mean_rows = []
    for (variant, axis, statistic), sub in table.groupby(["variant", "axis", "statistic"]):
        mean_rows.append({
            "seed": "mean", "variant": variant, "axis": axis, "statistic": statistic,
            "observed": float(sub["observed"].mean()), "p_raw": float(sub["p_raw"].mean()),
        })
    return pd.concat([table, pd.DataFrame(mean_rows)], ignore_index=True)


def build_table_e_jaccard_gap(baseline_dirs: list, variant_root: Path) -> pd.DataFrame:
    """Long format: one row per (seed, variant, source in {raw, residual})
    -- the raw/covariate-adjusted Jaccard range and its Bonferroni p, read
    from table_permutation.csv restricted to partition="long_k10",
    scope="test", outcome="jaccard", statistic="range" (the same filter
    safedrug_mitigation_compare.variant_gap_table already applies for "the"
    Jaccard gap). Plus a seed="mean" row per (variant, source)."""
    rows = []
    for seed_idx, baseline_dir in enumerate(baseline_dirs):
        for variant in VARIANTS:
            d = _variant_dir(baseline_dir, variant_root, seed_idx, variant)
            perm = pd.read_csv(d / "table_permutation.csv")
            sub = perm[
                (perm["partition"] == "long_k10") & (perm["scope"] == "test")
                & (perm["outcome"] == "jaccard") & (perm["statistic"] == "range")
            ]
            for _, r in sub.iterrows():
                rows.append({
                    "seed": seed_idx, "variant": variant, "source": r["source"],
                    "observed": r["observed"], "p_bonferroni": r["p_bonferroni"],
                })
    table = pd.DataFrame(rows)

    mean_rows = []
    for (variant, source), sub in table.groupby(["variant", "source"]):
        mean_rows.append({
            "seed": "mean", "variant": variant, "source": source,
            "observed": float(sub["observed"].mean()), "p_bonferroni": float(sub["p_bonferroni"].mean()),
        })
    return pd.concat([table, pd.DataFrame(mean_rows)], ignore_index=True)


def build_table_e_groups(baseline_dirs: list, variant_root: Path) -> pd.DataFrame:
    """Per (variant, axis, group): seed-mean of mean_ddi_pred/mean_ddi_true/
    mean_excess_ddi (from excess_ddi/table_excess_ddi_groups.csv) joined to
    seed-mean adjusted-mean Jaccard (from table_adjusted_means.csv, filtered
    to outcome="jaccard", scope="test"), plus min/max of mean_excess_ddi
    across seeds (the figure's whiskers).

    Both source tables mix numeric long_k10 ids and string ccs_group names
    in one shared "group" column (each built by pd.concat-ing per-partition
    frames), which round-trips a long_k10 id like "6.0" (a string), not 6 --
    the same problem safedrug_cluster_gap._normalize_group_id was written to
    fix for that module's own tables. Both sides of the join are normalized
    with it here.
    """
    rows = []
    for seed_idx, baseline_dir in enumerate(baseline_dirs):
        for variant in VARIANTS:
            d = _variant_dir(baseline_dir, variant_root, seed_idx, variant)
            excess_groups = pd.read_csv(d / "excess_ddi" / "table_excess_ddi_groups.csv")
            adjusted = pd.read_csv(d / "table_adjusted_means.csv")
            adj_jac = adjusted[(adjusted["outcome"] == "jaccard") & (adjusted["scope"] == "test")]

            for axis in AXES:
                axis_excess = excess_groups[excess_groups["partition"] == axis]
                axis_adj = adj_jac[adj_jac["partition"] == axis].copy()
                axis_adj["_norm_group"] = axis_adj["group"].apply(_normalize_group_id)

                for _, r in axis_excess.iterrows():
                    norm_g = _normalize_group_id(r["group"])
                    match = axis_adj[axis_adj["_norm_group"] == norm_g]
                    adj_val = float(match["adjusted_mean"].iloc[0]) if len(match) else float("nan")
                    rows.append({
                        "seed": seed_idx, "variant": variant, "axis": axis, "group": norm_g,
                        "mean_ddi_pred": r["mean_ddi_pred"], "mean_ddi_true": r["mean_ddi_true"],
                        "mean_excess_ddi": r["mean_excess_ddi"], "adj_jaccard": adj_val,
                    })
    per_seed = pd.DataFrame(rows)

    out = per_seed.groupby(["variant", "axis", "group"], as_index=False).agg(
        n_seeds=("seed", "count"),
        mean_ddi_pred=("mean_ddi_pred", "mean"),
        mean_ddi_true=("mean_ddi_true", "mean"),
        mean_excess_ddi=("mean_excess_ddi", "mean"),
        min_excess_ddi=("mean_excess_ddi", "min"),
        max_excess_ddi=("mean_excess_ddi", "max"),
        adj_jaccard=("adj_jaccard", "mean"),
    )
    return out


def make_fig_e_excess_ddi(table_e_groups: pd.DataFrame, figs_dir: Path, lang: str = "en") -> None:
    """Two panels (long_k10, ccs_group): per-group mean excess DDI for
    baseline vs ccs vs global (not k10 -- the figure's job is "does the
    acute-axis-targeted budget beat a same-size-but-undifferentiated
    control", a 3-way comparison; k10's own numbers are in the tables),
    seed-mean point with min-max whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percluster_labels.apply_korean_font()
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_variants = ["baseline", "ccs", "global"]
    colors = {"baseline": "#888888", "ccs": "#c0392b", "global": "#2471a3"}
    markers = {"baseline": "o", "ccs": "s", "global": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, axis in zip(axes, AXES):
        axis_df = table_e_groups[table_e_groups["axis"] == axis]
        groups = sorted(axis_df["group"].unique(), key=lambda g: str(g))
        y_pos = np.arange(len(groups))
        labels = [short_group_label(axis, g, lang=lang) for g in groups]

        for i, variant in enumerate(fig_variants):
            sub = axis_df[axis_df["variant"] == variant].set_index("group")
            means, err_low, err_high = [], [], []
            for g in groups:
                if g not in sub.index:
                    means.append(np.nan)
                    err_low.append(0)
                    err_high.append(0)
                    continue
                means.append(sub.loc[g, "mean_excess_ddi"])
                err_low.append(sub.loc[g, "mean_excess_ddi"] - sub.loc[g, "min_excess_ddi"])
                err_high.append(sub.loc[g, "max_excess_ddi"] - sub.loc[g, "mean_excess_ddi"])
            offset = (i - 1) * 0.15
            ax.errorbar(
                means, y_pos + offset, xerr=[err_low, err_high], fmt=markers[variant],
                color=colors[variant], label=variant, markersize=5, capsize=2,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="#888888", linewidth=1)
        ax.set_title(percluster_labels.partition_label(axis, lang))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_E_excess_ddi.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table formatter (tabulate not installed) -- same
    helper duplicated in safedrug_cluster_gap.py and
    safedrug_mitigation_compare.py; kept local here for the same reason
    those two don't share it (no common report-formatting module exists)."""
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


def write_report_ko(out_dir: Path, overall: pd.DataFrame, excess_gap: pd.DataFrame,
                     jaccard_gap: pd.DataFrame, groups: pd.DataFrame) -> None:
    lines = [
        "# SafeDrug 개입 E: 디코딩 단계 방문별 DDI 예산 리포트\n",
        "## 1. 개입 설명\n",
        "- 개입 E는 재학습 없이, 평가된 확률 벡터에 대해 방문별 DDI 예산(b_v)을 "
        "디코딩 규칙(REMOVE→ADD)으로 적용한다. b_v는 훈련 데이터의 그룹별 실제 DDI율에서 "
        "유도되며(축: ccs_group=주요, long_k10=민감도), 방문 하나에만 적용되므로 "
        "다른 그룹으로 새어나갈 수 없다.",
        "- **ccs**: CCS 주진단 축(급성 카테고리)의 예산. **k10**: long_k10 축의 예산(민감도 확인). "
        "**global**: 모든 방문에 동일한 r_global 예산을 적용하는 대조군.\n",
        "## 2. 전체 지표 (test, seed x variant)\n",
        _df_to_markdown(overall),
        "\n## 3. 격차(gap) 통계: excess DDI (축별)\n",
        _df_to_markdown(excess_gap),
        "\n## 4. 격차(gap) 통계: Jaccard 범위 (long_k10, raw/조정)\n",
        _df_to_markdown(jaccard_gap),
        "\n## 5. 군집별 표 (시드 평균)\n",
        _df_to_markdown(groups),
        "\n## 6. 해석\n",
    ]

    interp = []
    for axis in AXES:
        for variant in ("ccs", "k10"):
            base_rows = excess_gap[
                (excess_gap["variant"] == "baseline") & (excess_gap["axis"] == axis)
                & (excess_gap["statistic"] == "range") & (excess_gap["seed"] != "mean")
            ]
            var_rows = excess_gap[
                (excess_gap["variant"] == variant) & (excess_gap["axis"] == axis)
                & (excess_gap["statistic"] == "range") & (excess_gap["seed"] != "mean")
            ]
            merged = base_rows[["seed", "observed"]].merge(
                var_rows[["seed", "observed"]], on="seed", suffixes=("_base", "_var")
            )
            n_closed = int((merged["observed_var"] < merged["observed_base"]).sum())
            interp.append(
                f"- {axis} 축, {variant} 변형: excess-DDI 범위가 baseline 대비 좁아진 시드 수 "
                f"{n_closed}/{len(merged)}."
            )
    lines.extend(interp)
    lines.append(
        "\n- global 대조군과 ccs/k10의 격차 축소 정도를 비교하면, 예산의 크기 자체(하나의 숫자)가 "
        "아니라 그룹별로 예산을 다르게 준 것이 격차 축소에 기여했는지를 판단할 수 있다."
    )
    lines.append(
        "- 위 표는 test 지표, excess-DDI 및 Jaccard 격차 통계, 군집별 시드 평균을 보여준다. "
        "해석은 이 표들이 보여주는 범위를 넘지 않는다."
    )
    (out_dir / "REPORT_DECODE_BUDGET_KO.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug decode-time DDI budget seed report (Intervention E)")
    parser.add_argument("--baseline-dirs", nargs="+", required=True)
    parser.add_argument("--variant-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--lang", choices=["en", "ko"], default="en")
    parser.add_argument("--ddi-adj", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    baseline_dirs = [ROOT / d for d in args.baseline_dirs]
    variant_root = ROOT / args.variant_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ddi_adj_path = Path(args.ddi_adj) if args.ddi_adj else DDI_ADJ_PATH
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))

    overall = build_table_e_overall(baseline_dirs, variant_root, ddi_adj)
    overall.to_csv(out_dir / "table_E_overall.csv", index=False)

    excess_gap = build_table_e_excess_gap(baseline_dirs, variant_root)
    excess_gap.to_csv(out_dir / "table_E_excess_gap.csv", index=False)

    jaccard_gap = build_table_e_jaccard_gap(baseline_dirs, variant_root)
    jaccard_gap.to_csv(out_dir / "table_E_jaccard_gap.csv", index=False)

    groups = build_table_e_groups(baseline_dirs, variant_root)
    groups.to_csv(out_dir / "table_E_groups.csv", index=False)

    figs_dir = out_dir / "figs"
    make_fig_e_excess_ddi(groups, figs_dir, lang=args.lang)

    write_report_ko(out_dir, overall, excess_gap, jaccard_gap, groups)

    print(f"[+] wrote Intervention E seed report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
