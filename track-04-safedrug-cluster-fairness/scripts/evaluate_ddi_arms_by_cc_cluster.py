"""Stage 1: does the DDI loss cost accuracy where ground-truth DDI is high?

Arms (same wrapper, records, seeds 0..3, 50 epochs, threshold 0.5):
  ON  = track-04 baseline, target_ddi 0.06 / kp 0.05
        SafeDrug_군집별평가_20260904/개입실험/B_thresholds/seed{k}/global_0.5/
  OFF = run_safedrug_ddi_off.py, target_ddi 1.0 (rate gate never fires)
        out/safedrug_ddi_off/seed{k}/

Unit of analysis: a visit of the primary split (test), scored by both arms
under every seed; per-visit metrics are averaged over the 4 seeds per arm,
then delta = ON - OFF. Clusters: the 24 CC clusters (clusterCC_bert) plus
NO_CC, attached by HADM_ID. Cluster premise: stage-0 ground-truth DDI rate
per cluster (data/manifests/cluster-ddi-rate-stage0.json, all splits).

Decision rule (fixed before any OFF result was seen, 2026-09-18):
  H1 cost      : mean delta Jaccard over all primary-split visits < 0 and its
                 patient-clustered bootstrap 95% CI excludes 0.
  H2 spread    : patient-level permutation test (safedrug_cluster_gap.permutation_p,
                 10,000 draws, seed 0, min 30 visits/cluster) on per-visit
                 delta Jaccard across the 24 CC clusters: p_range < 0.05.
  H3 direction : Spearman rho over the 24 clusters between the stage-0
                 ground-truth DDI rate (fixed, observed) and cluster-mean delta
                 Jaccard is negative, with patient-level permutation p < 0.05
                 (clusters re-drawn by patient quota, premise vector fixed).
  Hypothesis SUPPORTED iff H1 and H2 and H3 hold. Otherwise NOT SUPPORTED,
  with the failing component named. Sensitivity (test+eval split) and the
  delta-DDI / delta-Jaccard tracking are reported but do not decide.
  Any later change must be labelled post hoc.

Outputs
  data/manifests/ddi-arms-cc-cluster-stage1.json
  out/ddi_arms_cc_cluster_stage1.md

Run:  py -3.12 scripts/evaluate_ddi_arms_by_cc_cluster.py [--splits test] [--n-perm 10000]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from measure_cluster_ddi_rate import (  # noqa: E402
    DDI_A, NO_CC, K, load_cc_labels, load_cluster_names, visit_ddi_counts, visit_rate,
)
from safedrug_cluster_gap import (  # noqa: E402
    MIN_VISITS, N_PERM, PERM_SEED, _build_permutation_target, permutation_p,
)

ON_ROOT = Path(r"C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds")
OFF_ROOT = ROOT / "out" / "safedrug_ddi_off"
STAGE0 = ROOT / "data" / "manifests" / "cluster-ddi-rate-stage0.json"
MANIFEST = ROOT / "data" / "manifests" / "ddi-arms-cc-cluster-stage1.json"
REPORT = ROOT / "out" / "ddi_arms_cc_cluster_stage1.md"
SEEDS = [0, 1, 2, 3]
ALPHA = 0.05
N_BOOT = 2000
RULE = ("SUPPORTED iff H1 (mean dJaccard < 0, patient-bootstrap 95% CI excludes 0) AND "
        "H2 (permutation p_range of dJaccard across 24 CC clusters < 0.05) AND "
        "H3 (Spearman rho[stage-0 GT DDI rate, cluster-mean dJaccard] < 0 with patient-level "
        "permutation p < 0.05). Primary split = test, seeds 0..3 averaged per visit. Fixed before results.")


# ------------------------------------------------------------------ per-visit metrics
def jaccard(pred: set, true: set) -> float:
    union = pred | true
    return len(pred & true) / len(union) if union else 1.0


def f1(pred: set, true: set) -> float:
    inter = len(pred & true)
    if inter == 0:
        return 0.0
    p, r = inter / len(pred), inter / len(true)
    return 2 * p * r / (p + r)


def visit_metrics(y_gt_row: np.ndarray, y_pred_row: np.ndarray, ddi_A: np.ndarray) -> dict:
    true = set(int(i) for i in np.flatnonzero(y_gt_row))
    pred = sorted(int(i) for i in np.flatnonzero(y_pred_row))
    dd_p, pairs_p = visit_ddi_counts(pred, ddi_A)
    dd_t, pairs_t = visit_ddi_counts(sorted(true), ddi_A)
    return {"jaccard": jaccard(set(pred), true), "f1": f1(set(pred), true),
            "n_med_pred": len(pred), "n_med_gt": len(true),
            "ddi_pred": visit_rate(dd_p, pairs_p), "dd_cnt_pred": dd_p,
            "ddi_true": visit_rate(dd_t, pairs_t), "dd_cnt_true": dd_t}


def arm_dirs(arm: str) -> dict[int, Path]:
    if arm == "ON":
        return {k: ON_ROOT / f"seed{k}" / "global_0.5" for k in SEEDS}
    return {k: OFF_ROOT / f"seed{k}" for k in SEEDS}


def load_arm(arm: str, splits: list[str], ddi_A: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per-visit metrics for every seed of one arm, plus official test metrics."""
    frames, official = [], {}
    for k, d in arm_dirs(arm).items():
        z = np.load(d / "per_visit_predictions.npz")
        keep = np.isin(z["split"], splits)
        rows = []
        for i in np.flatnonzero(keep):
            m = visit_metrics(z["y_gt"][i], z["y_pred"][i], ddi_A)
            m.update({"HADM_ID": int(z["HADM_ID"][i]), "SUBJECT_ID": int(z["SUBJECT_ID"][i]),
                      "split": str(z["split"][i]), "seed": k, "arm": arm})
            rows.append(m)
        frames.append(pd.DataFrame(rows))
        man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        off = man["official_metrics"]
        off = ast.literal_eval(off) if isinstance(off, str) else off
        official[k] = {"best_epoch": man["best_epoch"], "target_ddi": man["hyperparameters"]["target_ddi"],
                       "test": off["test"], "eval": off["eval"]}
    return pd.concat(frames, ignore_index=True), official


def check_alignment(on: pd.DataFrame, off: pd.DataFrame) -> dict:
    """Both arms must score the same visits with the same ground truth."""
    a = on.groupby("HADM_ID")[["n_med_gt", "dd_cnt_true"]].first()
    b = off.groupby("HADM_ID")[["n_med_gt", "dd_cnt_true"]].first()
    if set(a.index) != set(b.index):
        raise AssertionError("ON and OFF dumps cover different HADM_IDs")
    if not a.sort_index().equals(b.sort_index()):
        raise AssertionError("ground truth differs between ON and OFF dumps")
    per_seed = {arm: df.groupby("seed").size().to_dict() for arm, df in [("ON", on), ("OFF", off)]}
    return {"n_visits": int(len(a)), "seeds_per_arm": per_seed}


def seed_average(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["HADM_ID", "SUBJECT_ID"], as_index=False)
    out = g[cols].mean()
    out["n_seeds"] = g.size()["size"].to_numpy()
    return out


def patient_bootstrap_mean(values: np.ndarray, patients: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    groups = pd.Series(np.arange(len(values))).groupby(patients).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        draws[b] = values[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(values)), "n_patients": int(len(keys)), "n_boot": n_boot}


def spearman(x, y) -> float:
    x, y = pd.Series(x, dtype=float).rank(), pd.Series(y, dtype=float).rank()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_permutation(group_of_visit: np.ndarray, subject_of_visit: np.ndarray, values: np.ndarray,
                         premise: np.ndarray, n_groups: int, n_perm: int, seed: int) -> dict:
    """Observed Spearman between the fixed premise vector (one value per group)
    and the group means of `values`; null from patient-level reassignment with
    the observed per-group patient quota (same construction as permutation_p).
    One-sided p for rho <= observed (the hypothesis predicts a negative rho)."""
    target, patients, patient_of_visit = _build_permutation_target(group_of_visit, subject_of_visit, n_groups)

    def group_means(g):
        cnt = np.bincount(g, minlength=n_groups).astype(float)
        tot = np.bincount(g, weights=values, minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            return tot / cnt

    obs = spearman(premise, group_means(group_of_visit))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    n_pat = len(patients)
    for b in range(n_perm):
        perm = rng.permutation(n_pat)
        cl = np.empty(n_pat, dtype=int)
        s = 0
        for c, sz in enumerate(target):
            cl[perm[s:s + sz]] = c
            s += sz
        null[b] = spearman(premise, group_means(cl[patient_of_visit]))
    null = null[~np.isnan(null)]
    p_neg = (1 + int((null <= obs).sum())) / (1 + len(null))
    p_two = (1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + len(null))
    return {"rho": float(obs), "p_one_sided_negative": float(p_neg), "p_two_sided": float(p_two),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "n_perm": n_perm, "seed": seed}


# ------------------------------------------------------------------ report
def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render(m: dict) -> str:
    names = m["cluster_names"]
    v = m["verdict"]
    L = ["# DDI loss ON vs OFF — CC 군집별 정확도 손실 (1단계)", "",
         f"생성 {m['generated_at_utc']} · 주 분할 {'+'.join(m['splits'])} · 시드 {m['seeds']} (방문별 시드 평균) · 순열 {m['n_perm']:,}회 · 부트스트랩 {N_BOOT:,}회", "",
         f"**판정 기준(결과 확인 전 고정):** {m['rule']}", "",
         f"## 판정: **가설 {'지지' if v['supported'] else '미지지'}** — H1 {'✓' if v['H1'] else '✗'} · H2 {'✓' if v['H2'] else '✗'} · H3 {'✓' if v['H3'] else '✗'}", "",
         f"- H1 전체 ΔJaccard(ON − OFF) = {fmt(v['h1']['mean'])} [{fmt(v['h1']['ci_low'])}, {fmt(v['h1']['ci_high'])}] (환자 {v['h1']['n_patients']:,}, 방문 {v['h1']['n']:,})",
         f"- H2 군집 간 ΔJaccard range = {fmt(v['h2']['observed_range'])} (null 평균 {fmt(v['h2']['null_mean_range'])}, p95 {fmt(v['h2']['null_p95_range'])}), p = {fmt(v['h2']['p_raw_range'])}; 사용 군집 {v['h2']['n_groups_used']}",
         f"- H3 Spearman ρ(정답 DDI율, ΔJaccard) = {fmt(v['h3']['rho'], 3)}, 단측 p = {fmt(v['h3']['p_one_sided_negative'])} (양측 {fmt(v['h3']['p_two_sided'])}; null 평균 {fmt(v['h3']['null_mean'], 3)} ± {fmt(v['h3']['null_sd'], 3)})", "",
         "## 1. Arm 공식 지표 (test split, run_manifest official_metrics, 4 seed mean ± sd)", "",
         "| 지표 | ON (target 0.06) | OFF (target 1.0) | Δ(ON − OFF) |", "|---|---|---|---|"]
    for met in ["ja", "prauc", "avg_f1", "ddi_rate", "avg_med"]:
        o, f = m["official"]["ON"][met], m["official"]["OFF"][met]
        L.append(f"| {met} | {fmt(o['mean'])} ± {fmt(o['sd'])} | {fmt(f['mean'])} ± {fmt(f['sd'])} | {fmt(o['mean'] - f['mean'])} |")
    L += [f"| best_epoch (seed 순) | {m['official']['ON']['best_epochs']} | {m['official']['OFF']['best_epochs']} | |",
          "", f"덤프 정렬: {m['alignment']['n_visits']:,} 방문, 시드별 행 수 {m['alignment']['seeds_per_arm']}", "",
          "## 2. 군집별 ON − OFF (정답 DDI율 내림차순)", "",
          "| 군집 | 방문 | 환자 | 정답 DDI율 (0단계) | Jaccard ON | OFF | ΔJaccard [95% CI] | DDI율 ON | OFF | ΔDDI | 약물 수 ON | OFF |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(m["cluster_table"], key=lambda r: (r["cluster_label"] == NO_CC, -(r["gt_ddi_stage0"] if r["gt_ddi_stage0"] == r["gt_ddi_stage0"] else -1)))
    for r in rows:
        L.append(f"| {names[r['cluster_label']]} | {r['n_visits']:,} | {r['n_patients']:,} | {fmt(r['gt_ddi_stage0'])} | {fmt(r['jaccard_on'])} | {fmt(r['jaccard_off'])} "
                 f"| {fmt(r['d_jaccard'])} [{fmt(r['d_jaccard_ci_low'])}, {fmt(r['d_jaccard_ci_high'])}] | {fmt(r['ddi_pred_on'])} | {fmt(r['ddi_pred_off'])} | {fmt(r['d_ddi_pred'])} "
                 f"| {fmt(r['n_med_pred_on'], 2)} | {fmt(r['n_med_pred_off'], 2)} |")
    t = m["tracking"]
    L += ["", "## 3. 추적 통계 (24개 CC 군집)", "",
          "| 항목 | 값 |", "|---|---|",
          f"| Spearman ρ(정답 DDI율 0단계, ΔJaccard) | {fmt(t['rho_gt0_djaccard'], 3)} |",
          f"| Spearman ρ(정답 DDI율 test, ΔJaccard) | {fmt(t['rho_gt_test_djaccard'], 3)} |",
          f"| Spearman ρ(ΔDDI 예측, ΔJaccard) | {fmt(t['rho_dddi_djaccard'], 3)} |",
          f"| Spearman ρ(정답 DDI율 0단계, ΔDDI 예측) | {fmt(t['rho_gt0_dddi'], 3)} |",
          f"| 방문 수준 Spearman ρ(정답 DDI율, ΔJaccard) | {fmt(t['rho_visit_ddi_true_djaccard'], 3)} |",
          f"| 상위 사분위(정답 DDI율 상위 6 군집) ΔJaccard | {fmt(t['top_quartile']['mean'])} [{fmt(t['top_quartile']['ci_low'])}, {fmt(t['top_quartile']['ci_high'])}] (n {t['top_quartile']['n']:,}) |",
          f"| 나머지 18 군집 ΔJaccard | {fmt(t['rest']['mean'])} [{fmt(t['rest']['ci_low'])}, {fmt(t['rest']['ci_high'])}] (n {t['rest']['n']:,}) |",
          f"| 차이 (상위 − 나머지) | {fmt(t['top_minus_rest']['mean'])} [{fmt(t['top_minus_rest']['ci_low'])}, {fmt(t['top_minus_rest']['ci_high'])}] |",
          f"| ΔDDI 예측 range 순열 p / ΔF1 range 순열 p | {fmt(m['permutation']['d_ddi_pred']['p_raw_range'])} / {fmt(m['permutation']['d_f1']['p_raw_range'])} |", ""]
    if m.get("sensitivity"):
        s = m["sensitivity"]
        L += ["## 4. 민감도 (test + eval split)", "",
              f"- H1 ΔJaccard {fmt(s['h1']['mean'])} [{fmt(s['h1']['ci_low'])}, {fmt(s['h1']['ci_high'])}]; H2 p_range {fmt(s['h2']['p_raw_range'])}; "
              f"H3 ρ {fmt(s['h3']['rho'], 3)}, p {fmt(s['h3']['p_one_sided_negative'])}", ""]
    L += ["## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    v, t, o = m["verdict"], m["tracking"], m["official"]
    names = m["cluster_names"]
    rows = [r for r in m["cluster_table"] if r["cluster_label"] != NO_CC]
    worst = min(rows, key=lambda r: r["d_jaccard"]); best = max(rows, key=lambda r: r["d_jaccard"])
    out = [f"- **가설 {'지지' if v['supported'] else '미지지'}.** DDI loss를 끄면 test Jaccard {fmt(o['OFF']['ja']['mean'])} vs 켜면 {fmt(o['ON']['ja']['mean'])}, "
           f"DDI율 {fmt(o['OFF']['ddi_rate']['mean'])} vs {fmt(o['ON']['ddi_rate']['mean'])}. 방문별 ΔJaccard 평균 {fmt(v['h1']['mean'])} [{fmt(v['h1']['ci_low'])}, {fmt(v['h1']['ci_high'])}] → H1 {'성립' if v['H1'] else '불성립'}."]
    out.append(f"- 군집별 ΔJaccard는 {fmt(worst['d_jaccard'])}({names[worst['cluster_label']]})에서 {fmt(best['d_jaccard'])}({names[best['cluster_label']]}) 사이, range 순열 p = {fmt(v['h2']['p_raw_range'])} → H2 {'성립' if v['H2'] else '불성립'}.")
    out.append(f"- 0단계 정답 DDI율과 ΔJaccard의 군집 상관 ρ = {fmt(v['h3']['rho'], 2)}(단측 p {fmt(v['h3']['p_one_sided_negative'])}) → H3 {'성립' if v['H3'] else '불성립'}. "
               f"정답 DDI율 상위 6 군집의 ΔJaccard {fmt(t['top_quartile']['mean'])} vs 나머지 {fmt(t['rest']['mean'])}, 차이 {fmt(t['top_minus_rest']['mean'])} [{fmt(t['top_minus_rest']['ci_low'])}, {fmt(t['top_minus_rest']['ci_high'])}].")
    out.append(f"- DDI 감소량(ΔDDI 예측)과 정확도 손실의 군집 상관 ρ = {fmt(t['rho_dddi_djaccard'], 2)}; 정답 DDI율과 ΔDDI 예측의 상관 ρ = {fmt(t['rho_gt0_dddi'], 2)}. "
               f"방문 수준에서는 정답 DDI율과 ΔJaccard의 ρ = {fmt(t['rho_visit_ddi_true_djaccard'], 2)}.")
    return out


# ------------------------------------------------------------------ main
def run_analysis(on: pd.DataFrame, off: pd.DataFrame, cc: pd.DataFrame, stage0: dict, n_perm: int, seed: int) -> dict:
    cols = ["jaccard", "f1", "n_med_pred", "ddi_pred", "dd_cnt_pred", "ddi_true", "n_med_gt"]
    a, b = seed_average(on, cols), seed_average(off, cols)
    df = a.merge(b, on=["HADM_ID", "SUBJECT_ID"], suffixes=("_on", "_off"))
    for c in ["jaccard", "f1", "ddi_pred", "dd_cnt_pred", "n_med_pred"]:
        df[f"d_{c}"] = df[f"{c}_on"] - df[f"{c}_off"]
    df = df.merge(cc[["HADM_ID", "cluster_label"]], on="HADM_ID", how="left")
    df["cluster_label"] = df["cluster_label"].fillna("UNLABELLED")
    gt0 = {r["cluster_label"]: r["ddi_rate_mean"] for r in stage0["cluster_table"]}

    # cluster table
    g = df.groupby("cluster_label", sort=True)
    table = []
    for lab, sub in g:
        boot = patient_bootstrap_mean(sub["d_jaccard"].to_numpy(), sub["SUBJECT_ID"].to_numpy(), seed=seed)
        table.append({"cluster_label": lab, "n_visits": int(len(sub)), "n_patients": int(sub["SUBJECT_ID"].nunique()),
                      "gt_ddi_stage0": gt0.get(lab, float("nan")), "gt_ddi_test_mean": float(sub["ddi_true_on"].mean()),
                      "jaccard_on": float(sub["jaccard_on"].mean()), "jaccard_off": float(sub["jaccard_off"].mean()),
                      "d_jaccard": boot["mean"], "d_jaccard_ci_low": boot["ci_low"], "d_jaccard_ci_high": boot["ci_high"],
                      "f1_on": float(sub["f1_on"].mean()), "f1_off": float(sub["f1_off"].mean()),
                      "ddi_pred_on": float(sub["ddi_pred_on"].mean()), "ddi_pred_off": float(sub["ddi_pred_off"].mean()),
                      "d_ddi_pred": float(sub["d_ddi_pred"].mean()),
                      "n_med_pred_on": float(sub["n_med_pred_on"].mean()), "n_med_pred_off": float(sub["n_med_pred_off"].mean())})

    ccdf = df[df["cluster_label"].str.match(r"^\d\d$")].copy()
    group = ccdf["cluster_label"].astype(int).to_numpy()
    subj = ccdf["SUBJECT_ID"].to_numpy()
    h1 = patient_bootstrap_mean(df["d_jaccard"].to_numpy(), df["SUBJECT_ID"].to_numpy(), seed=seed)
    perm = {}
    for c in ["d_jaccard", "d_f1", "d_ddi_pred"]:
        vals = ccdf[c].to_numpy(dtype=float)
        ok = ~np.isnan(vals)
        perm[c] = permutation_p(group[ok], subj[ok], vals[ok], n_groups=K, min_visits=MIN_VISITS, n_perm=n_perm, seed=seed)
        perm[c].update({"n_visits": int(ok.sum()), "n_perm": n_perm, "seed": seed})
    premise = np.array([gt0[f"{k:02d}"] for k in range(K)])
    h3 = spearman_permutation(group, subj, ccdf["d_jaccard"].to_numpy(dtype=float), premise, K, n_perm, seed)

    ct = pd.DataFrame([r for r in table if r["cluster_label"] != NO_CC and r["cluster_label"] != "UNLABELLED"])
    top_labels = ct.sort_values("gt_ddi_stage0", ascending=False)["cluster_label"].head(6).tolist()
    top = ccdf[ccdf["cluster_label"].isin(top_labels)]; rest = ccdf[~ccdf["cluster_label"].isin(top_labels)]
    tq = patient_bootstrap_mean(top["d_jaccard"].to_numpy(), top["SUBJECT_ID"].to_numpy(), seed=seed)
    rq = patient_bootstrap_mean(rest["d_jaccard"].to_numpy(), rest["SUBJECT_ID"].to_numpy(), seed=seed)
    # difference of two independent patient-bootstrap means
    rng = np.random.default_rng(seed)
    def _boot(sub):
        groups = pd.Series(np.arange(len(sub))).groupby(sub["SUBJECT_ID"].to_numpy()).indices; keys = list(groups); vals = sub["d_jaccard"].to_numpy()
        return np.array([vals[np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), size=len(keys))])].mean() for _ in range(N_BOOT)])
    diff = _boot(top) - _boot(rest)
    tracking = {
        "rho_gt0_djaccard": spearman(ct["gt_ddi_stage0"], ct["d_jaccard"]),
        "rho_gt_test_djaccard": spearman(ct["gt_ddi_test_mean"], ct["d_jaccard"]),
        "rho_dddi_djaccard": spearman(ct["d_ddi_pred"], ct["d_jaccard"]),
        "rho_gt0_dddi": spearman(ct["gt_ddi_stage0"], ct["d_ddi_pred"]),
        "rho_visit_ddi_true_djaccard": spearman(ccdf["ddi_true_on"].fillna(0), ccdf["d_jaccard"]),
        "top_quartile_clusters": top_labels, "top_quartile": tq, "rest": rq,
        "top_minus_rest": {"mean": float(tq["mean"] - rq["mean"]), "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5))},
    }
    verdict = {"H1": bool(h1["mean"] < 0 and h1["ci_high"] < 0), "H2": bool(perm["d_jaccard"]["p_raw_range"] < ALPHA),
               "H3": bool(h3["rho"] < 0 and h3["p_one_sided_negative"] < ALPHA), "h1": h1, "h2": perm["d_jaccard"], "h3": h3}
    verdict["supported"] = bool(verdict["H1"] and verdict["H2"] and verdict["H3"])
    return {"cluster_table": table, "permutation": perm, "tracking": tracking, "verdict": verdict,
            "n_visits_primary": int(len(df)), "n_unlabelled": int((df["cluster_label"] == "UNLABELLED").sum())}


def official_summary(official: dict) -> dict:
    out = {}
    for met in ["ja", "prauc", "avg_f1", "ddi_rate", "avg_med"]:
        vals = np.array([official[k]["test"][met] for k in SEEDS])
        out[met] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "per_seed": vals.tolist()}
    out["best_epochs"] = [official[k]["best_epoch"] for k in SEEDS]
    out["target_ddi"] = sorted({official[k]["target_ddi"] for k in SEEDS})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["test"], choices=["test", "eval"])
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=PERM_SEED)
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    ddi_A = np.asarray(dill.load(open(DDI_A, "rb")))
    cc = load_cc_labels(); names = load_cluster_names() | {"UNLABELLED": "UNLABELLED"}
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))

    all_splits = ["test", "eval"] if not args.no_sensitivity else args.splits
    on_all, on_off = load_arm("ON", all_splits, ddi_A)
    off_all, off_off = load_arm("OFF", all_splits, ddi_A)
    if on_off[0]["target_ddi"] != 0.06 or off_off[0]["target_ddi"] != 1.0:
        raise AssertionError(f"arm definitions: ON target {on_off[0]['target_ddi']}, OFF target {off_off[0]['target_ddi']}")
    on, off = on_all[on_all["split"].isin(args.splits)], off_all[off_all["split"].isin(args.splits)]
    alignment = check_alignment(on, off)
    print(f"aligned {alignment['n_visits']:,} visits; running primary analysis on {args.splits}", flush=True)
    primary = run_analysis(on, off, cc, stage0, args.n_perm, args.seed)
    sens = None
    if not args.no_sensitivity and set(all_splits) != set(args.splits):
        s = run_analysis(on_all, off_all, cc, stage0, args.n_perm, args.seed)
        sens = {"splits": all_splits, "h1": s["verdict"]["h1"], "h2": s["verdict"]["h2"], "h3": s["verdict"]["h3"], "supported": s["verdict"]["supported"]}

    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
         "rule": RULE, "alpha": ALPHA, "splits": args.splits, "seeds": SEEDS, "n_perm": args.n_perm, "perm_seed": args.seed,
         "arms": {"ON": {"root": str(ON_ROOT), "target_ddi": 0.06}, "OFF": {"root": str(OFF_ROOT), "target_ddi": 1.0}},
         "inputs": {"ddi_A_final.pkl": hashlib.sha256(DDI_A.read_bytes()).hexdigest(), "stage0_manifest": str(STAGE0),
                    "dumps": {arm: {k: hashlib.sha256((d / "per_visit_predictions.npz").read_bytes()).hexdigest()[:16] for k, d in arm_dirs(arm).items()} for arm in ["ON", "OFF"]}},
         "cluster_names": names, "alignment": alignment,
         "official": {"ON": official_summary(on_off), "OFF": official_summary(off_off)},
         **primary, "sensitivity": sens, "post_hoc_changes": []}
    m["interpretation"] = interpret(m)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    REPORT.write_text(render(m), encoding="utf-8")
    v = m["verdict"]
    print(f"verdict: {'SUPPORTED' if v['supported'] else 'NOT SUPPORTED'} (H1 {v['H1']}, H2 {v['H2']}, H3 {v['H3']}); "
          f"dJaccard {v['h1']['mean']:+.4f}, p_range {v['h2']['p_raw_range']:.4f}, rho {v['h3']['rho']:+.3f} p {v['h3']['p_one_sided_negative']:.4f}; "
          f"wrote {REPORT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
