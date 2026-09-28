"""SafeDrug mitigation comparison (R4): overall, per-group, and gap tables
across the baseline and every intervention variant already processed by
scripts/safedrug_cluster_gap.py, plus a paired patient-bootstrap CI on the
raw-Jaccard-range change and a Korean report.

See docs/superpowers/specs/2026-09-05-safedrug-mitigation-design.md ("D-D")
for the full contract.
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

from safedrug_cluster_gap import DDI_ADJ_PATH, MIN_VISITS, adjusted_group_means
from safedrug_mechanism import precision_recall_permutation
from safedrug_percluster import labels as percluster_labels
from safedrug_percluster.labels import short_group_label

GAP_OUTCOMES = ["jaccard", "precision", "recall", "ddi_rate_visit"]


# ============================================================= D-D1: overall table
def global_ddi_rate(y_pred: np.ndarray, ddi_adj: np.ndarray) -> float:
    """Attributed reimplementation of SOTA/SafeDrug/src/util.ddi_rate_score's
    core loop (patient/visit nesting never affects the aggregate rate --
    verified by reading its source, see design doc). Over every predicted-
    medication PAIR within each visit, pooled across every visit in
    y_pred, the fraction that are a known DDI pair. Returns 0.0 when no
    visit predicts >= 2 medications."""
    all_cnt = 0
    dd_cnt = 0
    for row in y_pred:
        idx = np.flatnonzero(row == 1)
        n = len(idx)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = idx[i], idx[j]
                all_cnt += 1
                if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                    dd_cnt += 1
    return dd_cnt / all_cnt if all_cnt else 0.0


def _is_posthoc(manifest: dict) -> bool:
    """True when `manifest` was written by a post-hoc (non-retraining)
    variant script -- Intervention B's threshold_policy block
    (safedrug_group_threshold.py) or Intervention E's decode_policy block
    (scripts/safedrug_decode_budget.py). Both leave official_metrics as a
    verbatim copy of the source dump's own numbers, so
    official_metrics_reflect_variant must be False for either. This is the
    ONLY call site in the repo that inspects threshold_policy (verified via
    `git grep threshold_policy` before this edit) -- widening it here is a
    complete, minimal fix."""
    return "threshold_policy" in manifest or "decode_policy" in manifest


def flat_test_metrics(eval_dir: Path, ddi_adj: np.ndarray) -> dict:
    """R4's uniform (non-'official') per-variant summary: flat mean over
    test-split rows of per_visit_metrics.csv's jaccard/f1/prauc/n_med_pred,
    plus the TRUE aggregate DDI rate (global_ddi_rate, not a per-visit
    mean). Applied identically to every variant including the baseline and
    the two retrained interventions, so comparisons stay apples-to-apples
    (design doc D-D1) -- alongside the dir's own official_metrics.test for
    reference."""
    per_visit = pd.read_csv(eval_dir / "per_visit_metrics.csv")
    test = per_visit[per_visit["split"] == "test"]
    npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    test_mask = npz["split"] == "test"
    manifest = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))
    official = manifest["official_metrics"]["test"]
    return {
        "n_test_visits": int(len(test)),
        "jaccard_mean": float(test["jaccard"].mean()),
        "f1_mean": float(test["f1"].mean()),
        "prauc_mean": float(test["prauc"].mean(skipna=True)),
        "avg_n_med_pred": float(test["n_med_pred"].mean()),
        "ddi_rate": global_ddi_rate(npz["y_pred"][test_mask], ddi_adj),
        "official_jaccard": official["ja"],
        "official_f1": official["avg_f1"],
        "official_prauc": official["prauc"],
        "official_ddi_rate": official["ddi_rate"],
        "official_avg_med": official["avg_med"],
        "official_metrics_reflect_variant": not _is_posthoc(manifest),
    }


# ============================================================= D-D2: group table
def group_metric_table(per_visit_metrics_csv: Path, outcomes=None) -> pd.DataFrame:
    """Raw and D6-adjusted per-long_k10-group means for each outcome in
    `outcomes` (default GAP_OUTCOMES), from one dir's per_visit_metrics.csv
    (test split, has_label rows only). Reuses adjusted_group_means
    (imported, unmodified from safedrug_cluster_gap) for precision/recall
    too -- cluster_gap.py's own ADJUSTED_OUTCOMES never includes those."""
    outcomes = GAP_OUTCOMES if outcomes is None else outcomes
    df = pd.read_csv(per_visit_metrics_csv)
    scope = df[(df["split"] == "test") & (df["has_label"])].dropna(subset=["long_k10"]).copy()
    groups = sorted(scope["long_k10"].unique())
    n_visits = scope.groupby("long_k10").size()

    out = pd.DataFrame({"group": groups, "n_visits": [int(n_visits.loc[g]) for g in groups]})
    for outcome in outcomes:
        work = scope.dropna(subset=[outcome, "n_dx", "n_med_gt", "visit_index"])
        raw = work.groupby("long_k10")[outcome].mean()
        adjusted, _coef = adjusted_group_means(scope, "long_k10", outcome, groups)
        out[f"raw_{outcome}"] = [raw.get(g, float("nan")) for g in groups]
        out[f"adj_{outcome}"] = [adjusted[g] for g in groups]
    return out


def mitigation_group_deltas(baseline_groups: pd.DataFrame, variant_groups: pd.DataFrame,
                             metric_cols: list) -> pd.DataFrame:
    """Returns a copy of variant_groups (its own raw_*/adj_* columns
    unchanged, same shape as baseline_groups) with one new delta_<col> =
    variant's value - baseline's value column added per entry in
    metric_cols (baseline_groups looked up by 'group'). Keeping the same
    column set for baseline and variant rows -- rather than an outer merge
    that would duplicate every baseline column under a second name -- is
    what lets build_mitigation_tables stack every variant's rows (baseline
    included, delta_* left NaN there) into one single-shaped
    table_mitigation_groups.csv."""
    baseline_by_group = baseline_groups.set_index("group")
    out = variant_groups.copy()
    for col in metric_cols:
        base_col = out["group"].map(baseline_by_group[col])
        out[f"delta_{col}"] = out[col] - base_col
    return out


# ============================================================= D-D3: gap table + paired bootstrap
def variant_gap_table(eval_dir: Path) -> pd.DataFrame:
    """jaccard/ddi_rate_visit: read directly from the dir's own
    table_permutation.csv (already computed by safedrug_cluster_gap.py --
    no new statistics). precision/recall: computed fresh via
    safedrug_mechanism.precision_recall_permutation (imported, unmodified)
    -- source="raw" only, that function has no residual variant."""
    perm = pd.read_csv(eval_dir / "table_permutation.csv")
    jac_ddi = perm[
        (perm["partition"] == "long_k10") & (perm["scope"] == "test")
        & (perm["outcome"].isin(["jaccard", "ddi_rate_visit"]))
    ][["outcome", "source", "statistic", "observed", "null_mean", "null_p95", "z", "p_raw", "p_bonferroni"]].copy()

    per_visit = pd.read_csv(eval_dir / "per_visit_metrics.csv")
    scope = per_visit[(per_visit["split"] == "test") & (per_visit["has_label"])].dropna(subset=["long_k10"])
    pr = precision_recall_permutation(scope, "long_k10")
    pr = pr[pr["partition"] == "long_k10"].rename(columns={"source_metric": "outcome"}).copy()
    pr["source"] = "raw"
    pr = pr[["outcome", "source", "statistic", "observed", "null_mean", "null_p95", "z", "p_raw", "p_bonferroni"]]

    return pd.concat([jac_ddi, pr], ignore_index=True)


def weighted_gap_range(group_codes: np.ndarray, values: np.ndarray, weights: np.ndarray,
                        n_groups: int, min_visits: int) -> float:
    """Range statistic (max group mean - min group mean), weighted by each
    row's patient-draw multiplicity -- the weighted analogue of
    safedrug_cluster_gap.gap_statistics's own cnt/tot/ok/m/mbar bincount
    logic, generalized with an explicit weights array instead of an
    implicit weights-of-1 (gap_statistics itself is not modified)."""
    cnt = np.bincount(group_codes, weights=weights, minlength=n_groups)
    tot = np.bincount(group_codes, weights=values * weights, minlength=n_groups)
    ok = cnt >= min_visits
    if ok.sum() < 2:
        return float("nan")
    m = tot[ok] / cnt[ok]
    return float(m.max() - m.min())


def joint_patient_bootstrap_delta_range(baseline_df: pd.DataFrame, variant_df: pd.DataFrame,
                                         outcome: str, n_groups: int, group_code_map: dict,
                                         min_visits: int = MIN_VISITS, n_boot: int = 2000,
                                         seed: int = 0):
    """R4's paired patient bootstrap for the raw range-statistic change
    (Delta-range = variant_range - baseline_range). baseline_df/variant_df
    must already be restricted to the SAME test-split, has_label,
    outcome-non-null visit set, in IDENTICAL HADM_ID order (data_test's
    visit membership is a pure function of the fixed input data list, so
    every variant's test split covers the same HADM_IDs -- see design doc).
    Draws the SAME resampled SUBJECT_ID multiset for both frames on every
    draw, generalizing safedrug_percluster.metrics.patient_bootstrap_ci's
    own draw/bincount/weight pattern to a group-range statistic via
    weighted_gap_range. Returns (observed_delta, ci_low, ci_high)."""
    if not (baseline_df["HADM_ID"].to_numpy() == variant_df["HADM_ID"].to_numpy()).all():
        raise ValueError("baseline_df and variant_df must have identical, identically-ordered HADM_ID")

    group_codes = baseline_df["long_k10"].map(group_code_map).to_numpy()
    base_values = baseline_df[outcome].to_numpy(dtype=float)
    var_values = variant_df[outcome].to_numpy(dtype=float)
    subject = baseline_df["SUBJECT_ID"].to_numpy()

    patients = np.sort(np.unique(subject))
    n_patients = len(patients)
    patient_pos = {p: i for i, p in enumerate(patients)}
    patient_of_row = np.array([patient_pos[s] for s in subject])

    ones = np.ones(len(group_codes))
    base_range_obs = weighted_gap_range(group_codes, base_values, ones, n_groups, min_visits)
    var_range_obs = weighted_gap_range(group_codes, var_values, ones, n_groups, min_visits)
    observed_delta = var_range_obs - base_range_obs

    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = rng.integers(0, n_patients, size=n_patients)
        counts = np.bincount(draw, minlength=n_patients)
        weights = counts[patient_of_row]
        base_range = weighted_gap_range(group_codes, base_values, weights, n_groups, min_visits)
        var_range = weighted_gap_range(group_codes, var_values, weights, n_groups, min_visits)
        deltas[b] = var_range - base_range

    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        return observed_delta, float("nan"), float("nan")
    ci_low, ci_high = (float(x) for x in np.percentile(valid, [2.5, 97.5]))
    return observed_delta, ci_low, ci_high


# ============================================================= D-D4: figures + report
def make_mitigation_figures(table_gap: pd.DataFrame, group_table_all: pd.DataFrame,
                             figs_dir: Path, lang: str = "ko") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percluster_labels.apply_korean_font()

    jac_range = table_gap[
        (table_gap["outcome"] == "jaccard") & (table_gap["source"] == "raw")
        & (table_gap["statistic"] == "range")
    ].copy()
    variants = ["baseline"] + [v for v in jac_range["variant"].unique() if v != "baseline"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(variants))
    deltas, err_low, err_high = [], [], []
    for v in variants:
        if v == "baseline":
            deltas.append(0.0)
            err_low.append(0.0)
            err_high.append(0.0)
            continue
        row = jac_range[jac_range["variant"] == v].iloc[0]
        deltas.append(row["delta_vs_baseline"])
        err_low.append(row["delta_vs_baseline"] - row["delta_ci_low"])
        err_high.append(row["delta_ci_high"] - row["delta_vs_baseline"])
    ax.errorbar(deltas, y_pos, xerr=[err_low, err_high], fmt="o", color="#c0392b")
    ax.axvline(0, color="#888888", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants)
    ax.set_xlabel("Δ raw Jaccard range vs baseline")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_mitigation_gap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    jac_groups = group_table_all[["variant", "group", "adj_jaccard"]].copy()
    baseline_order = jac_groups[jac_groups["variant"] == "baseline"].sort_values("adj_jaccard")
    order = baseline_order["group"].tolist()
    group_pos = {g: i for i, g in enumerate(order)}
    variants_g = ["baseline"] + [v for v in jac_groups["variant"].unique() if v != "baseline"]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.4 * len(order) + 1)))
    markers = ["o", "s", "^", "D", "P", "X"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    for i, v in enumerate(variants_g):
        sub = jac_groups[jac_groups["variant"] == v]
        y = [group_pos[g] for g in sub["group"]]
        ax.scatter(sub["adj_jaccard"], y, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], s=70, label=v, edgecolors="white", linewidths=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short_group_label("long_k10", g, lang=lang) for g in order])
    ax.set_xlabel("adjusted mean Jaccard")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_mitigation_groups.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table formatter (tabulate is not installed) --
    duplicated from safedrug_cluster_gap.write_report's own helper."""
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


VARIANT_DESCRIPTIONS_KO = {
    "intervention_a": "개입 A: 학습 손실에서 희귀 약물에 더 큰 가중치를 부여 (약물 라벨 불균형 가설 검증)",
    "global_0.5": "개입 B 대조군: 기존과 동일한 전역 0.5 임계값 (재현 확인용 복사본)",
    "global_tuned": "개입 B-1: eval 분할에서 튜닝한 단일 전역 임계값",
    "group_tuned": "개입 B-2: eval 분할에서 군집별로 튜닝한 임계값",
    "intervention_c": "개입 C: 군집별 실제 DDI율에서 유도한 DDI 목표치로 재학습 (전역 DDI 제약 가설 검증)",
}


def write_mitigation_report(out_dir: Path, overall_rows: list, table_gap: pd.DataFrame,
                             group_table_all: pd.DataFrame, lang: str = "ko") -> None:
    lines = ["# SafeDrug 개입(보강) 실험 비교 리포트\n", "## 1. 개입 설명\n"]
    for row in overall_rows:
        v = row["variant"]
        desc = VARIANT_DESCRIPTIONS_KO.get(v, v)
        lines.append(f"- **{v}**: {desc}")

    lines.append("\n## 2. 전체 지표 (test, variant별)\n")
    lines.append(_df_to_markdown(pd.DataFrame(overall_rows)))

    lines.append("\n## 3. 군집별 raw/adjusted 지표\n")
    lines.append(_df_to_markdown(group_table_all))

    lines.append("\n## 4. 격차(gap) 통계 및 변화\n")
    lines.append(_df_to_markdown(table_gap))

    lines.append("\n## 5. 해석\n")
    lines.append(
        "- 위 표는 각 개입의 test 지표, 군집별 raw/adjusted 성능, 그리고 raw Jaccard 범위의 "
        "변화(Δrange)와 그 부트스트랩 신뢰구간을 보여준다. Δrange의 신뢰구간이 0을 포함하지 않고 "
        "음수이면 해당 개입이 군집 간 격차를 줄였다는 근거이며, 이는 표에 나타난 범위를 넘어서지 "
        "않는 해석이다."
    )
    (out_dir / "REPORT_MITIGATION_KO.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================= orchestration
def build_mitigation_tables(baseline_dir: Path, variants: dict, out_dir: Path,
                             n_boot: int = 2000, seed: int = 0, lang: str = "ko") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ddi_adj = dill.load(open(DDI_ADJ_PATH, "rb"))

    overall_rows = [{"variant": "baseline", **flat_test_metrics(baseline_dir, ddi_adj)}]
    for name, d in variants.items():
        overall_rows.append({"variant": name, **flat_test_metrics(d, ddi_adj)})
    pd.DataFrame(overall_rows).to_csv(out_dir / "table_mitigation_overall.csv", index=False)

    baseline_groups = group_metric_table(baseline_dir / "per_visit_metrics.csv")
    metric_cols = [c for c in baseline_groups.columns if c not in ("group", "n_visits")]

    baseline_row = baseline_groups.copy()
    baseline_row.insert(0, "variant", "baseline")
    for col in metric_cols:
        baseline_row[f"delta_{col}"] = float("nan")  # nothing to diff baseline against

    group_rows = [baseline_row]
    for name, d in variants.items():
        vt = group_metric_table(d / "per_visit_metrics.csv")
        merged = mitigation_group_deltas(baseline_groups, vt, metric_cols)
        merged.insert(0, "variant", name)
        group_rows.append(merged)

    group_table_all = pd.concat(group_rows, ignore_index=True)
    group_table_all.to_csv(out_dir / "table_mitigation_groups.csv", index=False)

    baseline_gap = variant_gap_table(baseline_dir)
    baseline_gap.insert(0, "variant", "baseline")

    baseline_per_visit = pd.read_csv(baseline_dir / "per_visit_metrics.csv")
    baseline_scope = baseline_per_visit[
        (baseline_per_visit["split"] == "test") & (baseline_per_visit["has_label"])
    ].dropna(subset=["long_k10"]).sort_values("HADM_ID").reset_index(drop=True)
    groups_all = sorted(baseline_scope["long_k10"].unique())
    group_code_map = {g: i for i, g in enumerate(groups_all)}

    for col in ("delta_vs_baseline", "delta_ci_low", "delta_ci_high", "baseline_observed"):
        baseline_gap[col] = float("nan")
    gap_rows = [baseline_gap]

    for name, d in variants.items():
        vgap = variant_gap_table(d)
        vgap.insert(0, "variant", name)
        for col in ("delta_vs_baseline", "delta_ci_low", "delta_ci_high", "baseline_observed"):
            vgap[col] = float("nan")

        variant_per_visit = pd.read_csv(d / "per_visit_metrics.csv")
        variant_scope = variant_per_visit[
            (variant_per_visit["split"] == "test") & (variant_per_visit["has_label"])
        ].dropna(subset=["long_k10"])

        for outcome in GAP_OUTCOMES:
            base_o = baseline_scope.dropna(subset=[outcome])
            var_o = variant_scope[variant_scope["HADM_ID"].isin(base_o["HADM_ID"])].dropna(subset=[outcome])
            common = sorted(set(base_o["HADM_ID"]) & set(var_o["HADM_ID"]))
            base_o = base_o[base_o["HADM_ID"].isin(common)].sort_values("HADM_ID").reset_index(drop=True)
            var_o = var_o[var_o["HADM_ID"].isin(common)].sort_values("HADM_ID").reset_index(drop=True)

            obs_delta, ci_low, ci_high = joint_patient_bootstrap_delta_range(
                base_o, var_o, outcome, len(groups_all), group_code_map, MIN_VISITS, n_boot, seed,
            )
            base_row_obs = baseline_gap[
                (baseline_gap["outcome"] == outcome) & (baseline_gap["source"] == "raw")
                & (baseline_gap["statistic"] == "range")
            ]["observed"]
            baseline_observed_range = float(base_row_obs.iloc[0]) if len(base_row_obs) else float("nan")

            mask = (vgap["outcome"] == outcome) & (vgap["source"] == "raw") & (vgap["statistic"] == "range")
            vgap.loc[mask, "delta_vs_baseline"] = obs_delta
            vgap.loc[mask, "delta_ci_low"] = ci_low
            vgap.loc[mask, "delta_ci_high"] = ci_high
            vgap.loc[mask, "baseline_observed"] = baseline_observed_range

        gap_rows.append(vgap)

    table_gap = pd.concat(gap_rows, ignore_index=True)
    table_gap.to_csv(out_dir / "table_mitigation_gap.csv", index=False)

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    make_mitigation_figures(table_gap, group_table_all, figs_dir, lang=lang)
    write_mitigation_report(out_dir, overall_rows, table_gap, group_table_all, lang=lang)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug mitigation comparison (R4)")
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--variant", action="append", nargs=2, metavar=("NAME", "DIR"),
                         required=True, dest="variants")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--lang", choices=["en", "ko"], default="ko")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    baseline_dir = ROOT / args.baseline_dir
    variants = {name: ROOT / d for name, d in args.variants}
    out_dir = ROOT / args.out_dir
    build_mitigation_tables(baseline_dir, variants, out_dir, n_boot=args.n_boot, seed=args.seed, lang=args.lang)
    print(f"[+] wrote mitigation comparison to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
