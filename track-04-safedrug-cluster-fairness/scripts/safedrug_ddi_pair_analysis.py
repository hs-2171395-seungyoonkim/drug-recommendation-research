"""SafeDrug DDI pair-level analysis (Intervention W, R3).

Reports, for a baseline dump plus any number of named variant dumps (e.g.
the no-DDI-penalty ablation and Intervention W itself), whether easing the
DDI penalty for condition-appropriate pairs actually recovers those
specific pairs without materially increasing OTHER (non-whitelisted)
pairs' excess DDI -- the central empirical question Intervention W exists
to answer.

One flat "pair events" table (build_pair_events) replaces the parallel
per-axis Counters the precedent scratch script used: every requested
cross-tab (miss rate by long_k10, by ccs_group, inside vs outside
whitelist, the guideline-pair table) is a groupby over this one table,
computed once, not once per cross-tab.

See docs/superpowers/specs/2026-09-08-safedrug-ddi-whitelist-design.md
("R3") for the full contract, including the excess-DDI split's exact
formula and why the guideline-pair table pools across categories.
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

from safedrug_cluster_gap import CCS_CSV, DDI_ADJ_PATH, DXTEXT_CSV, LABELS_NPZ
from safedrug_percluster.metrics import _row_ddi_rate, attach_labels

SAFEDRUG_DATA_DEFAULT = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")

# The scratch script's (noddi_compare.py) original 10 guideline pairs, verified
# against it during planning.
GUIDELINE_PAIRS = [
    ("N02A", "B01A"), ("C07A", "A12B"), ("B01A", "C10A"), ("N02B", "C10A"),
    ("C07A", "C10A"), ("C07A", "C09A"), ("B01A", "C09A"), ("C10A", "C01D"),
    ("A06A", "N06A"), ("N06A", "B01A"),
]
GUIDELINE_SHARE_THRESHOLD = 0.10


# ================================================================= per-visit metrics
def compute_per_visit_table(y_gt: np.ndarray, y_pred: np.ndarray, ddi_adj: np.ndarray) -> pd.DataFrame:
    """One row per input row: jaccard, precision, recall, n_pred, n_true,
    ddi_pred, ddi_true, excess (= ddi_pred - ddi_true). ddi_pred/ddi_true
    reuse safedrug_percluster.metrics._row_ddi_rate verbatim (imported,
    not reimplemented) -- NaN when fewer than 2 meds in the respective
    set; excess is NaN whenever either side is NaN."""
    n = y_gt.shape[0]
    rows = []
    for i in range(n):
        gt = set(np.flatnonzero(y_gt[i]).tolist())
        pr = set(np.flatnonzero(y_pred[i]).tolist())
        union = gt | pr
        jaccard = len(gt & pr) / len(union) if union else 0.0
        precision = len(gt & pr) / len(pr) if pr else 0.0
        recall = len(gt & pr) / len(gt) if gt else 0.0
        ddi_pred = _row_ddi_rate(pr, ddi_adj)
        ddi_true = _row_ddi_rate(gt, ddi_adj)
        excess = float("nan") if (np.isnan(ddi_pred) or np.isnan(ddi_true)) else ddi_pred - ddi_true
        rows.append({
            "jaccard": jaccard, "precision": precision, "recall": recall,
            "n_pred": len(pr), "n_true": len(gt),
            "ddi_pred": ddi_pred, "ddi_true": ddi_true, "excess": excess,
        })
    return pd.DataFrame(rows)


def overall_table(per_visit: pd.DataFrame) -> pd.DataFrame:
    """per_visit must have a 'variant' column plus compute_per_visit_table's
    own columns. One row per variant: n_visits and the per-visit mean of
    each metric (pandas' own NaN-skipping .mean())."""
    return per_visit.groupby("variant", as_index=False).agg(
        n_visits=("jaccard", "size"),
        jaccard_mean=("jaccard", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        n_pred_mean=("n_pred", "mean"),
        ddi_pred_mean=("ddi_pred", "mean"),
        ddi_true_mean=("ddi_true", "mean"),
        excess_mean=("excess", "mean"),
    )


# ================================================================= pair events
def load_whitelist_by_group(whitelist_pairs: pd.DataFrame) -> dict:
    out: dict = {}
    for group, sub in whitelist_pairs.groupby("ccs_group"):
        out[group] = set(zip(sub["idx_a"], sub["idx_b"]))
    return out


def build_pair_events(df: pd.DataFrame, y_gt: np.ndarray, y_pred: np.ndarray,
                       ddi_adj: np.ndarray, idx2atc: dict, whitelist_by_group: dict) -> pd.DataFrame:
    """One row per (test visit, ground-truth DDI-flagged pair) occurrence.

    df must be row-aligned with y_gt/y_pred and have HADM_ID, long_k10,
    ccs_group columns (attach_labels' own output shape).
    whitelist_by_group: ccs_group -> set of (idx_a, idx_b) pairs, idx_a <
    idx_b (load_whitelist_by_group's output) -- a visit's ccs_group absent
    here (NaN, or no whitelist for that category) contributes an empty
    set, so every one of that visit's flagged pairs is "outside"
    (whitelisted=False) by construction.

    missed: True iff at least one drug of the pair is absent from that
    visit's predicted set (matches noddi_compare.py's own
    `mf = (a not in ps) or (b not in ps)` rule, verified against it
    during planning).
    """
    rows = []
    n = y_gt.shape[0]
    for i in range(n):
        gt_sorted = sorted(np.flatnonzero(y_gt[i]).tolist())
        pred_set = set(np.flatnonzero(y_pred[i]).tolist())
        group_ccs = df["ccs_group"].iloc[i]
        wl = whitelist_by_group.get(group_ccs, set())
        n_gt = len(gt_sorted)
        for x in range(n_gt):
            for y in range(x + 1, n_gt):
                a, b = gt_sorted[x], gt_sorted[y]
                if not (ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1):
                    continue
                rows.append({
                    "HADM_ID": df["HADM_ID"].iloc[i],
                    "long_k10": df["long_k10"].iloc[i],
                    "ccs_group": group_ccs,
                    "idx_a": a, "idx_b": b,
                    "atc_a": idx2atc[a], "atc_b": idx2atc[b],
                    "missed": (a not in pred_set) or (b not in pred_set),
                    "whitelisted": (a, b) in wl,
                })
    return pd.DataFrame(rows, columns=[
        "HADM_ID", "long_k10", "ccs_group", "idx_a", "idx_b", "atc_a", "atc_b", "missed", "whitelisted",
    ])


def miss_rate_by_group(pair_events: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """pair_events must have a 'variant' column. Rows with a NaN
    group_col value are dropped first."""
    work = pair_events.dropna(subset=[group_col])
    out = work.groupby(["variant", group_col], as_index=False).agg(
        n_pairs=("missed", "size"), n_missed=("missed", "sum"),
    )
    out["miss_rate"] = out["n_missed"] / out["n_pairs"]
    return out


def miss_rate_inside_outside_whitelist(pair_events: pd.DataFrame) -> pd.DataFrame:
    out = pair_events.groupby(["variant", "whitelisted"], as_index=False).agg(
        n_pairs=("missed", "size"), n_missed=("missed", "sum"),
    )
    out["miss_rate"] = out["n_missed"] / out["n_pairs"]
    return out


def guideline_pair_table(pair_events: pd.DataFrame, whitelist_pairs: pd.DataFrame,
                          fixed_pairs=GUIDELINE_PAIRS,
                          share_threshold: float = GUIDELINE_SHARE_THRESHOLD) -> pd.DataFrame:
    """Miss rate for each named pair, POOLED over every test visit
    containing it regardless of the visit's own category (groupby
    (variant, pair), not (variant, ccs_group, pair) -- matches
    noddi_compare.py's own flat Counter, which never restricted by
    category). Named pairs = fixed_pairs union every whitelist_pairs.csv
    row with share >= share_threshold (deduplicated by (atc_a, atc_b))."""
    if len(whitelist_pairs):
        extended = set(zip(
            whitelist_pairs.loc[whitelist_pairs["share"] >= share_threshold, "atc_a"],
            whitelist_pairs.loc[whitelist_pairs["share"] >= share_threshold, "atc_b"],
        ))
    else:
        extended = set()
    wanted = set(fixed_pairs) | extended

    work = pair_events.copy()
    work["pair"] = list(zip(work["atc_a"], work["atc_b"]))
    work = work[work["pair"].isin(wanted)]
    out = work.groupby(["variant", "pair"], as_index=False).agg(
        n_pairs=("missed", "size"), n_missed=("missed", "sum"),
    )
    out["miss_rate"] = out["n_missed"] / out["n_pairs"]
    atc = pd.DataFrame(out["pair"].tolist(), columns=["atc_a", "atc_b"], index=out.index)
    return pd.concat([out.drop(columns=["pair"]), atc], axis=1)


# ================================================================= excess-DDI split
def per_visit_excess_split(df: pd.DataFrame, y_gt: np.ndarray, y_pred: np.ndarray,
                            ddi_adj: np.ndarray, whitelist_by_group: dict) -> pd.DataFrame:
    """Per test visit, excess DDI split into a whitelisted-pair component
    and a non-whitelisted ('other') component -- design doc R3's exact
    formula: for X in {predicted, true}, rate_X_wl = (# DDI pairs in X
    that are whitelisted)/C(|X|,2), rate_X_other = (# DDI pairs in X that
    are not)/C(|X|,2) (both NaN if |X|<2); excess_wl = rate_pred_wl -
    rate_true_wl, excess_other = rate_pred_other - rate_true_other (NaN if
    either side is NaN). excess_wl + excess_other equals the ordinary
    excess_ddi (ddi_pred - ddi_true) whenever both are non-NaN."""
    rows = []
    n = y_gt.shape[0]
    for i in range(n):
        pred = sorted(np.flatnonzero(y_pred[i]).tolist())
        true = sorted(np.flatnonzero(y_gt[i]).tolist())
        group_ccs = df["ccs_group"].iloc[i]
        wl = whitelist_by_group.get(group_ccs, set())

        def split_rate(idx_list):
            m = len(idx_list)
            all_cnt = m * (m - 1) // 2
            if all_cnt == 0:
                return float("nan"), float("nan")
            dd_wl = dd_other = 0
            for x in range(m):
                for y in range(x + 1, m):
                    a, b = idx_list[x], idx_list[y]
                    if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                        if (a, b) in wl:
                            dd_wl += 1
                        else:
                            dd_other += 1
            return dd_wl / all_cnt, dd_other / all_cnt

        rate_pred_wl, rate_pred_other = split_rate(pred)
        rate_true_wl, rate_true_other = split_rate(true)
        excess_wl = (float("nan") if (np.isnan(rate_pred_wl) or np.isnan(rate_true_wl))
                     else rate_pred_wl - rate_true_wl)
        excess_other = (float("nan") if (np.isnan(rate_pred_other) or np.isnan(rate_true_other))
                        else rate_pred_other - rate_true_other)
        rows.append({
            "HADM_ID": df["HADM_ID"].iloc[i], "long_k10": df["long_k10"].iloc[i], "ccs_group": group_ccs,
            "rate_pred_wl": rate_pred_wl, "rate_pred_other": rate_pred_other,
            "rate_true_wl": rate_true_wl, "rate_true_other": rate_true_other,
            "excess_wl": excess_wl, "excess_other": excess_other,
        })
    return pd.DataFrame(rows)


def excess_split_by_cluster(per_visit_excess: pd.DataFrame, group_col: str = "long_k10") -> pd.DataFrame:
    """per_visit_excess must have a 'variant' column. Rows with NaN
    group_col are dropped first."""
    work = per_visit_excess.dropna(subset=[group_col])
    return work.groupby(["variant", group_col], as_index=False).agg(
        n_visits=("excess_wl", "size"),
        mean_excess_wl=("excess_wl", "mean"),
        mean_excess_other=("excess_other", "mean"),
    )


# ================================================================= figure + report
def make_fig_pair_missrate_by_cluster(missrate_long_k10: pd.DataFrame, figs_dir: Path, lang: str = "en") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from safedrug_percluster import labels as percluster_labels
    from safedrug_percluster.labels import short_group_label

    percluster_labels.apply_korean_font()
    figs_dir.mkdir(parents=True, exist_ok=True)

    variants = list(dict.fromkeys(missrate_long_k10["variant"]))  # order-preserving, baseline first
    groups = sorted(missrate_long_k10["long_k10"].unique())
    y_pos = np.arange(len(groups))
    tick_labels = [short_group_label("long_k10", g, lang=lang) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 6))
    n_variants = len(variants)
    for i, variant in enumerate(variants):
        sub = missrate_long_k10[missrate_long_k10["variant"] == variant].set_index("long_k10")
        vals = [sub.loc[g, "miss_rate"] if g in sub.index else np.nan for g in groups]
        offset = (i - (n_variants - 1) / 2) * 0.15
        ax.scatter(vals, y_pos + offset, label=variant, s=30)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("miss rate" if lang != "ko" else "누락률")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n_variants, 4), frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_pair_missrate_by_cluster.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table formatter (tabulate not installed) -- same
    helper duplicated in safedrug_cluster_gap.py / safedrug_mitigation_compare.py
    / safedrug_decode_budget_report.py; kept local here for the same
    reason those don't share it (no common report-formatting module
    exists in this repo)."""
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


def write_report_ko(out_dir: Path, overall: pd.DataFrame, missrate_long_k10: pd.DataFrame,
                     missrate_ccs: pd.DataFrame, missrate_wl: pd.DataFrame,
                     guideline: pd.DataFrame, excess_split: pd.DataFrame) -> None:
    lines = [
        "# SafeDrug 개입 W: 조건부 DDI 페어 화이트리스트 - 페어 수준 분석\n",
        "## 1. 전체 지표 (test)\n", _df_to_markdown(overall),
        "\n## 2. long_k10 군집별 상호작용 페어 누락률\n", _df_to_markdown(missrate_long_k10),
        "\n## 3. CCS 범주별 상호작용 페어 누락률\n", _df_to_markdown(missrate_ccs),
        "\n## 4. 화이트리스트 내부 vs 외부 누락률\n", _df_to_markdown(missrate_wl),
        "\n## 5. 가이드라인 페어 누락률\n", _df_to_markdown(guideline),
        "\n## 6. 군집별 초과 DDI (화이트리스트 vs 기타)\n", _df_to_markdown(excess_split),
    ]
    (out_dir / "REPORT_PAIR_ANALYSIS_KO.md").write_text("\n".join(lines), encoding="utf-8")


def load_dir(eval_dir: Path) -> dict:
    npz = np.load(eval_dir / "per_visit_predictions.npz", allow_pickle=False)
    mask = npz["split"] == "test"
    df = pd.DataFrame({"HADM_ID": npz["HADM_ID"][mask], "SUBJECT_ID": npz["SUBJECT_ID"][mask]})
    df = attach_labels(df, DXTEXT_CSV, LABELS_NPZ, CCS_CSV)
    return {"df": df.reset_index(drop=True), "y_gt": npz["y_gt"][mask], "y_pred": npz["y_pred"][mask]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug DDI pair-level analysis (Intervention W, R3)")
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--variant", action="append", nargs=2, metavar=("NAME", "DIR"), default=[])
    parser.add_argument("--whitelist-pairs", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--lang", choices=["en", "ko"], default="en")
    parser.add_argument("--ddi-adj", type=str, default=None)
    parser.add_argument("--safedrug-data-dir", type=str, default=str(SAFEDRUG_DATA_DEFAULT))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    ddi_adj_path = Path(args.ddi_adj) if args.ddi_adj else DDI_ADJ_PATH
    ddi_adj = np.asarray(dill.load(open(ddi_adj_path, "rb")))
    voc = dill.load(open(Path(args.safedrug_data_dir) / "voc_final.pkl", "rb"))
    idx2atc = voc["med_voc"].idx2word

    whitelist_pairs = pd.read_csv(args.whitelist_pairs)
    whitelist_by_group = load_whitelist_by_group(whitelist_pairs)

    dirs = {"baseline": Path(args.baseline_dir)}
    for name, d in args.variant:
        dirs[name] = Path(d)

    per_visit_frames, pair_event_frames, excess_frames = [], [], []
    for variant_name, eval_dir in dirs.items():
        loaded = load_dir(eval_dir)
        df, y_gt, y_pred = loaded["df"], loaded["y_gt"], loaded["y_pred"]

        pv = compute_per_visit_table(y_gt, y_pred, ddi_adj)
        pv.insert(0, "variant", variant_name)
        per_visit_frames.append(pv)

        pe = build_pair_events(df, y_gt, y_pred, ddi_adj, idx2atc, whitelist_by_group)
        pe.insert(0, "variant", variant_name)
        pair_event_frames.append(pe)

        ex = per_visit_excess_split(df, y_gt, y_pred, ddi_adj, whitelist_by_group)
        ex.insert(0, "variant", variant_name)
        excess_frames.append(ex)

    per_visit = pd.concat(per_visit_frames, ignore_index=True)
    pair_events = pd.concat(pair_event_frames, ignore_index=True)
    excess = pd.concat(excess_frames, ignore_index=True)

    overall = overall_table(per_visit)
    overall.to_csv(out_dir / "table_overall.csv", index=False)

    missrate_long_k10 = miss_rate_by_group(pair_events, "long_k10")
    missrate_long_k10.to_csv(out_dir / "table_missrate_by_long_k10.csv", index=False)

    missrate_ccs = miss_rate_by_group(pair_events, "ccs_group")
    missrate_ccs.to_csv(out_dir / "table_missrate_by_ccs.csv", index=False)

    missrate_wl = miss_rate_inside_outside_whitelist(pair_events)
    missrate_wl.to_csv(out_dir / "table_missrate_whitelist.csv", index=False)

    guideline = guideline_pair_table(pair_events, whitelist_pairs)
    guideline.to_csv(out_dir / "table_guideline_pairs.csv", index=False)

    excess_split = excess_split_by_cluster(excess, "long_k10")
    excess_split.to_csv(out_dir / "table_excess_split_by_cluster.csv", index=False)

    make_fig_pair_missrate_by_cluster(missrate_long_k10, figs_dir, lang=args.lang)

    write_report_ko(out_dir, overall, missrate_long_k10, missrate_ccs, missrate_wl, guideline, excess_split)

    print(f"[+] wrote DDI pair-level analysis for {len(dirs)} variant(s) to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
