"""Two pre-checks for placing an acute/chronic axis on the CHANGE decision (MIMIC-IV).

Q1  Resolved acute diagnoses -> stop failures?
    resolved_acute = previous-visit acute diagnoses (HCUP chronic=0, V/E/Z status
    codes excluded) absent in the current visit. Outcome = per-transition stale
    keep rate (stopped drugs the model still predicts) of SafeDrug (5 seeds) and
    GAMENet (3 seeds), seed-averaged.
    Rule (fixed before results): (a) beta_std(resolved_acute) on SafeDrug stale keep
    rate > 0 with patient-bootstrap 95% CI excluding 0, after [log n_stopped,
    n_prev_meds, n_dx, log gap, emergency] + resolved_chronic + resolved_status;
    (b) beta(resolved_acute) - beta(resolved_chronic) > 0 with CI excluding 0.
    "Stop gap exists" iff (a); "chronicity needed" iff (a) and (b).

Q2  Drug persistence prior.
    prior[d] = P(d in visit t | d in visit t-1) over consecutive visit pairs of
    TRAIN patients. Stop-decision AUC (previous drugs, continued=1) of the prior
    alone, of the model alone, and of a logistic combination of the two fitted on
    half the test patients and scored on the other half (2-fold by patient
    parity, both directions). Gate counterfactual: previous drugs re-decided by
    the cross-fitted combination (keep iff p >= 0.5); non-previous predictions
    unchanged. Rule: "simple gate worth building" iff gate dJaccard > 0 with
    patient-bootstrap 95% CI excluding 0 for SafeDrug.

Outputs: results/resolved_acute_stop_check_mimic4.{md,json}
Run:  py -3.12 scripts/resolved_acute_stop_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:\Users\Administrator\Desktop\ServerityMed\src")
import acute_lib as A  # noqa: E402
from acute_label_corr import load_mimic4  # noqa: E402
from change_visit_split import MIMIC4_DUMP, MIMIC4_RECORDS, split_patients  # noqa: E402
from cvsplit_lib import multihot_rows_to_sets  # noqa: E402
from stop_decision_mimic4 import auc  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
GAMENET_DIR = ROOT / "out" / "mimic4_gamenet"
BASE = ["log_n_stopped", "n_prev_meds", "n_dx", "log_gap", "emergency"]
RES = ["n_resolved_acute", "n_resolved_chronic", "n_resolved_status"]
RULE_Q1 = ("(a) beta_std(n_resolved_acute) on SafeDrug per-transition stale keep rate > 0, patient-bootstrap 95% CI excludes 0, "
           "after [log n_stopped, n_prev_meds, n_dx, log gap, emergency] + n_resolved_chronic + n_resolved_status; "
           "(b) beta(resolved_acute) - beta(resolved_chronic) > 0, CI excludes 0. Stop gap exists iff (a); chronicity needed iff (a) and (b). Fixed before results.")
RULE_Q2 = ("Gate = logistic(logit model prob, logit persistence prior) cross-fitted by patient parity on test; previous drugs kept iff p >= 0.5, "
           "other predictions unchanged. Simple gate worth building iff SafeDrug dJaccard > 0 with patient-bootstrap 95% CI excluding 0. Fixed before results.")


# ------------------------------------------------------------------ diagnosis-side features
def dx_change_features(patients: list, test_idx: list, labeler) -> pd.DataFrame:
    rows = []
    for pi in test_idx:
        visits = patients[pi]
        for i in range(1, len(visits)):
            prev, cur = visits[i - 1], visits[i]
            def cls(c):
                if A.is_status_code(*c):
                    return "status"
                f = labeler.chronic(*c)
                return "acute" if f == 0 else ("chronic" if f == 1 else "unknown")
            resolved = prev["dx"] - cur["dx"]; new = cur["dx"] - prev["dx"]; persist = prev["dx"] & cur["dx"]
            r = {"patient": pi, "visit_pos": i, "n_dx": len(cur["dx"]), "n_prev_dx": len(prev["dx"]),
                 "n_prev_meds": len(prev["meds"]), "n_stopped": len(prev["meds"] - cur["meds"]), "n_added": len(cur["meds"] - prev["meds"]),
                 "gap_days": (cur["admit"] - prev["admit"]).total_seconds() / 86400 if cur.get("admit") is not None and prev.get("admit") is not None else np.nan,
                 "emergency": float(bool(cur.get("emergency")))}
            for name, s in [("resolved", resolved), ("new", new), ("persist", persist)]:
                for k in ("acute", "chronic", "status", "unknown"):
                    r[f"n_{name}_{k}"] = sum(1 for c in s if cls(c) == k)
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ medication-side: dumps and persistence prior
def persistence_prior(records: list, train_idx: list, n_med: int) -> tuple[np.ndarray, np.ndarray]:
    kept = np.zeros(n_med); seen = np.zeros(n_med)
    for pi in train_idx:
        p = records[pi]
        for a, b in zip(p[:-1], p[1:]):
            sa, sb = set(a[2]), set(b[2])
            for m in sa:
                seen[m] += 1; kept[m] += m in sb
    prior = np.where(seen > 0, kept / np.maximum(seen, 1), kept.sum() / max(seen.sum(), 1))
    return prior, seen


def load_arm(paths: list[Path], prior: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-transition (seed-averaged) stale counts + drug-level rows for previous drugs (all seeds)."""
    trans, drug = [], []
    for path in paths:
        with np.load(path) as z:
            z = {k: z[k] for k in z.files}
        gt, pr = multihot_rows_to_sets(z["y_gt"]), multihot_rows_to_sets(z["y_pred"])
        order = np.lexsort((z["visit_index"], z["patient_index"]))
        by_p: dict = defaultdict(list)
        for i in order:
            by_p[int(z["patient_index"][i])].append(i)
        seed = path.parent.name if path.name == "per_visit_predictions.npz" else path.stem
        for p, idxs in by_p.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                if z["split"][b] != "test":
                    continue
                prev, cur, pred, prob = gt[a], gt[b], pr[b], z["y_prob"][b]
                stopped = prev - cur
                trans.append({"patient": p, "visit_pos": int(z["visit_index"][b]), "seed": seed, "row": b,
                              "fp_stale": len(pred & stopped), "n_stopped_dump": len(stopped),
                              "stale_keep_rate": (len(pred & stopped) / len(stopped)) if stopped else np.nan,
                              "jaccard": len(pred & cur) / len(pred | cur) if (pred | cur) else 1.0})
                for m in prev:
                    drug.append((seed, p, int(z["visit_index"][b]), b, m, float(prob[m]), 1 if m in cur else 0, float(prior[m])))
    t = pd.DataFrame(trans)
    num = ["fp_stale", "n_stopped_dump", "stale_keep_rate", "jaccard"]
    t_avg = t.groupby(["patient", "visit_pos"], as_index=False)[num].mean()
    d = pd.DataFrame(drug, columns=["seed", "patient", "visit_pos", "row", "med", "prob", "continued", "prior"])
    return t_avg, d


# ------------------------------------------------------------------ Q1
def q1_regressions(df: pd.DataFrame, arm_col: str, n_boot: int, seed: int) -> dict:
    d = df[df["n_stopped"] > 0].dropna(subset=[arm_col]).copy()
    d["log_n_stopped"] = np.log1p(d["n_stopped"]); d["log_gap"] = np.log1p(d["gap_days"].fillna(d["gap_days"].median()).clip(lower=0))
    cols = BASE + RES
    y = d[arm_col].to_numpy(float)
    _, r2_b = A.ols_standardized(d[BASE].to_numpy(float), y)
    beta, r2 = A.ols_standardized(d[cols].to_numpy(float), y)
    def _stat(s):
        b, _ = A.ols_standardized(s[cols].to_numpy(float), s[arm_col].to_numpy(float))
        return np.append(b, b[cols.index("n_resolved_acute")] - b[cols.index("n_resolved_chronic")])
    bb = A.cluster_bootstrap(_stat, d, "patient", n_boot=n_boot, seed=seed)
    names = cols + ["acute_minus_chronic"]
    return {"n": int(len(d)), "r2_base": r2_b, "r2_full": r2,
            "betas": [{"feature": c, "beta_std": pt, "ci_low": lo, "ci_high": hi} for c, pt, lo, hi in zip(names, bb["point"], bb["ci_low"], bb["ci_high"])]}


def bins(df: pd.DataFrame, col: str, arms: list[str]) -> list[dict]:
    out = []
    for label, lo, hi in [("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6+", 6, 10**9)]:
        s = df[(df[col] >= lo) & (df[col] <= hi)]
        if not len(s):
            continue
        r = {"bin": label, "n": int(len(s)), "mean_n_stopped": float(s["n_stopped"].mean()), "mean_n_added": float(s["n_added"].mean())}
        for a in arms:
            ss = s[s["n_stopped"] > 0]
            r[f"stale_keep_{a}"] = float((ss[f"fp_stale_{a}"].sum() / ss["n_stopped"].sum())) if ss["n_stopped"].sum() else np.nan
        out.append(r)
    return out


# ------------------------------------------------------------------ Q2
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def crossfit_keep_prob(d: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([logit(d["prob"].to_numpy()), logit(d["prior"].to_numpy())])
    y = d["continued"].to_numpy()
    fold = (d["patient"].to_numpy() % 2)
    out = np.empty(len(d))
    coefs = {}
    for f in (0, 1):
        fit, score = fold != f, fold == f
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X[fit], y[fit])
        out[score] = lr.predict_proba(X[score])[:, 1]
        coefs[f"fit_on_fold_{1 - f}"] = {"w_model": float(lr.coef_[0][0]), "w_prior": float(lr.coef_[0][1]), "b": float(lr.intercept_[0])}
    return out, coefs


def q2_for_arm(d: pd.DataFrame, paths: list[Path], n_boot: int, seed: int) -> dict:
    d = d.copy()
    d["keep_combo"], coefs = crossfit_keep_prob(d)
    res = {"n_rows": int(len(d)), "auc_model": auc(d["prob"].to_numpy(), d["continued"].to_numpy()),
           "auc_prior": auc(d["prior"].to_numpy(), d["continued"].to_numpy()),
           "auc_combo_crossfit": auc(d["keep_combo"].to_numpy(), d["continued"].to_numpy()), "logistic": coefs}
    # keep rate of STOPPED drugs by prior tertile (does the model already use persistence?)
    edges = np.quantile(d["prior"], [1 / 3, 2 / 3]); d["_t"] = np.digitize(d["prior"], edges)
    res["prior_edges"] = edges.tolist()
    res["by_prior_tertile"] = [{"tertile": ["low-persistence", "mid", "high-persistence"][int(t)], "n": int(len(s)), "share_continued": float(s["continued"].mean()),
                                "keep_rate_stopped_model": float((s.loc[s["continued"] == 0, "prob"] >= 0.5).mean()),
                                "keep_rate_stopped_gate": float((s.loc[s["continued"] == 0, "keep_combo"] >= 0.5).mean()),
                                "keep_rate_cont_model": float((s.loc[s["continued"] == 1, "prob"] >= 0.5).mean()),
                                "keep_rate_cont_gate": float((s.loc[s["continued"] == 1, "keep_combo"] >= 0.5).mean())} for t, s in d.groupby("_t")]
    # gate counterfactual per seed-row: rebuild predicted sets for previous drugs
    keep = d.assign(keep=(d["keep_combo"] >= 0.5)).groupby(["seed", "row"])
    keep_sets = {k: set(g.loc[g["keep"], "med"].astype(int)) for k, g in keep}
    prev_sets = {k: set(g["med"].astype(int)) for k, g in keep}
    rows = []
    for path in paths:
        with np.load(path) as z:
            z = {k: z[k] for k in z.files}
        seed_name = path.parent.name if path.name == "per_visit_predictions.npz" else path.stem
        gt, pr = multihot_rows_to_sets(z["y_gt"]), multihot_rows_to_sets(z["y_pred"])
        for (s, b), prev in prev_sets.items():
            if s != seed_name:
                continue
            cur, pred = gt[b], pr[b]
            new_pred = (pred - prev) | keep_sets[(s, b)]
            stopped, cont = prev - cur, prev & cur
            rows.append({"seed": s, "patient": int(z["patient_index"][b]), "row": b,
                         "j_model": len(pred & cur) / len(pred | cur) if (pred | cur) else 1.0, "j_gate": len(new_pred & cur) / len(new_pred | cur) if (new_pred | cur) else 1.0,
                         "stale_model": len(pred & stopped), "stale_gate": len(new_pred & stopped), "fn_cont_model": len(cont - pred), "fn_cont_gate": len(cont - new_pred),
                         "n_pred_model": len(pred), "n_pred_gate": len(new_pred)})
    g = pd.DataFrame(rows)
    ga = g.groupby(["patient", "row"], as_index=False)[["j_model", "j_gate", "stale_model", "stale_gate", "fn_cont_model", "fn_cont_gate", "n_pred_model", "n_pred_gate"]].mean()
    ga["d"] = ga["j_gate"] - ga["j_model"]
    bb = A.cluster_bootstrap(lambda s: s["d"].mean(), ga, "patient", n_boot=n_boot, seed=seed)
    res["gate"] = {"jaccard_model": float(ga["j_model"].mean()), "jaccard_gate": float(ga["j_gate"].mean()), "d_jaccard": float(ga["d"].mean()),
                   "ci_low": bb["ci_low"][0], "ci_high": bb["ci_high"][0], "n_transitions": int(len(ga)),
                   "stale_per_transition": [float(ga["stale_model"].mean()), float(ga["stale_gate"].mean())],
                   "fn_cont_per_transition": [float(ga["fn_cont_model"].mean()), float(ga["fn_cont_gate"].mean())],
                   "n_pred": [float(ga["n_pred_model"].mean()), float(ga["n_pred_gate"].mean())]}
    return res


# ------------------------------------------------------------------ report
def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render(m: dict) -> str:
    v = m["verdict"]
    L = ["# 해소된 급성 진단 → 중단 실패? / 약물 지속률 사전확률 — MIMIC-IV", "",
         f"생성 {m['generated_at_utc']} · test 전이 {m['n_transitions']:,} · SafeDrug {m['n_seeds_sd']} seed, GAMENet {m['n_seeds_gn']} seed", "",
         f"**Q1 기준(고정):** {RULE_Q1}", "", f"**Q2 기준(고정):** {RULE_Q2}", "",
         f"## 판정: Q1 중단 오류 {'존재' if v['q1_stop_gap'] else '미확인'} · 급성/만성 구분 {'필요' if v['q1_chronicity'] else '미확인'} · Q2 단순 게이트 {'가치 있음' if v['q2_gate'] else '가치 미확인'}", "",
         "## Q1-1. 진단 변화와 실제 처방 변화 (전이 수준 Spearman)", "",
         "| 진단 변화 | ρ(n_stopped) | ρ(n_added) |", "|---|---|---|"]
    for k, r in m["q1_corr"].items():
        L.append(f"| {k} | {fmt(r['rho_stopped'])} | {fmt(r['rho_added'])} |")
    L += ["", "## Q1-2. 해소된 급성 진단 수별 stale 유지율(중단됐어야 할 약 중 계속 예측, pooled)", "",
          "| 해소된 급성 수 | 전이 | 평균 중단 수 | 평균 added | SafeDrug | GAMENet |", "|---|---|---|---|---|---|"]
    for r in m["q1_bins_acute"]:
        L.append(f"| {r['bin']} | {r['n']:,} | {fmt(r['mean_n_stopped'], 2)} | {fmt(r['mean_n_added'], 2)} | {fmt(r['stale_keep_sd'])} | {fmt(r['stale_keep_gn'])} |")
    L += ["", "해소된 **만성** 진단 수별:", "", "| 해소된 만성 수 | 전이 | 평균 중단 수 | SafeDrug | GAMENet |", "|---|---|---|---|---|"]
    for r in m["q1_bins_chronic"]:
        L.append(f"| {r['bin']} | {r['n']:,} | {fmt(r['mean_n_stopped'], 2)} | {fmt(r['stale_keep_sd'])} | {fmt(r['stale_keep_gn'])} |")
    for arm, key in [("SafeDrug (판정)", "q1_reg_sd"), ("GAMENet (참고)", "q1_reg_gn")]:
        r = m[key]
        L += ["", f"## Q1-3. 회귀 — y = stale 유지율, {arm} (R² 기본 {fmt(r['r2_base'])} → +해소 진단 {fmt(r['r2_full'])}, n {r['n']:,})", "", "| 특성 | β_std | 95% CI |", "|---|---|---|"]
        for b in r["betas"]:
            L.append(f"| {b['feature']} | {fmt(b['beta_std'])} | [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}] |")
    L += ["", "## Q2-1. ATC3 지속률 사전확률 (train, 연속 방문 쌍)", "",
          f"전체 평균 지속률 {fmt(m['prior_overall'])} · 가장 낮은 10 / 가장 높은 10:", "",
          "| 낮은 지속률 (급성용) | P(유지) | n | 높은 지속률 (유지용) | P(유지) | n |", "|---|---|---|---|---|---|"]
    for lo, hi in zip(m["prior_lowest"], m["prior_highest"]):
        L.append(f"| {lo['atc3']} | {fmt(lo['prior'])} | {lo['n']:,} | {hi['atc3']} | {fmt(hi['prior'])} | {hi['n']:,} |")
    L += ["", "## Q2-2. 중단 판단 AUC와 게이트", "",
          "| 항목 | SafeDrug | GAMENet |", "|---|---|---|"]
    a, g = m["q2_sd"], m["q2_gn"]
    L += [f"| AUC 모델 확률 단독 | {fmt(a['auc_model'])} | {fmt(g['auc_model'])} |",
          f"| AUC 지속률 사전확률 단독 | {fmt(a['auc_prior'])} | {fmt(g['auc_prior'])} |",
          f"| AUC 로지스틱(모델 + 사전확률, 교차 적합) | {fmt(a['auc_combo_crossfit'])} | {fmt(g['auc_combo_crossfit'])} |",
          f"| 로지스틱 계수 (w_model, w_prior) | {fmt(a['logistic']['fit_on_fold_1']['w_model'], 2)}, {fmt(a['logistic']['fit_on_fold_1']['w_prior'], 2)} | {fmt(g['logistic']['fit_on_fold_1']['w_model'], 2)}, {fmt(g['logistic']['fit_on_fold_1']['w_prior'], 2)} |",
          f"| 게이트 Jaccard: 모델 → 게이트 | {fmt(a['gate']['jaccard_model'], 4)} → {fmt(a['gate']['jaccard_gate'], 4)} | {fmt(g['gate']['jaccard_model'], 4)} → {fmt(g['gate']['jaccard_gate'], 4)} |",
          f"| ΔJaccard [95% CI] | {fmt(a['gate']['d_jaccard'], 4)} [{fmt(a['gate']['ci_low'], 4)}, {fmt(a['gate']['ci_high'], 4)}] | {fmt(g['gate']['d_jaccard'], 4)} [{fmt(g['gate']['ci_low'], 4)}, {fmt(g['gate']['ci_high'], 4)}] |",
          f"| stale FP (전이당) 모델 → 게이트 | {fmt(a['gate']['stale_per_transition'][0], 2)} → {fmt(a['gate']['stale_per_transition'][1], 2)} | {fmt(g['gate']['stale_per_transition'][0], 2)} → {fmt(g['gate']['stale_per_transition'][1], 2)} |",
          f"| 유지 누락 (전이당) 모델 → 게이트 | {fmt(a['gate']['fn_cont_per_transition'][0], 2)} → {fmt(a['gate']['fn_cont_per_transition'][1], 2)} | {fmt(g['gate']['fn_cont_per_transition'][0], 2)} → {fmt(g['gate']['fn_cont_per_transition'][1], 2)} |",
          f"| 예측 크기 모델 → 게이트 | {fmt(a['gate']['n_pred'][0], 2)} → {fmt(a['gate']['n_pred'][1], 2)} | {fmt(g['gate']['n_pred'][0], 2)} → {fmt(g['gate']['n_pred'][1], 2)} |", "",
          "지속률 3분위별 (직전 약물; 모델 vs 게이트, 임계 0.5):", "",
          "| 지속률 | 행 | 유지 비율 | 중단 약물 유지율 모델 → 게이트 (SD) | 유지 약물 유지율 모델 → 게이트 (SD) | 중단 약물 유지율 (GN) |", "|---|---|---|---|---|---|"]
    for rs, rg in zip(a["by_prior_tertile"], g["by_prior_tertile"]):
        L.append(f"| {rs['tertile']} | {rs['n']:,} | {fmt(rs['share_continued'])} | {fmt(rs['keep_rate_stopped_model'])} → {fmt(rs['keep_rate_stopped_gate'])} | {fmt(rs['keep_rate_cont_model'])} → {fmt(rs['keep_rate_cont_gate'])} | {fmt(rg['keep_rate_stopped_model'])} → {fmt(rg['keep_rate_stopped_gate'])} |")
    L += ["", "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    v = m["verdict"]; r = m["q1_reg_sd"]; g = lambda f: next(b for b in r["betas"] if b["feature"] == f)
    ba, bc, bd = g("n_resolved_acute"), g("n_resolved_chronic"), g("acute_minus_chronic")
    c = m["q1_corr"]
    out = [f"- 진단 변화 ↔ 처방 변화: 해소된 급성 진단 수와 중단 약물 수의 ρ = {fmt(c['n_resolved_acute']['rho_stopped'])}, 해소된 만성 {fmt(c['n_resolved_chronic']['rho_stopped'])}; (참고) 신규 급성 ↔ added ρ = {fmt(c['n_new_acute']['rho_added'])}."]
    out.append(f"- Q1: SafeDrug stale 유지율에 대한 해소된 급성 계수 {fmt(ba['beta_std'])} [{fmt(ba['ci_low'])}, {fmt(ba['ci_high'])}], 해소된 만성 {fmt(bc['beta_std'])}, 차이 {fmt(bd['beta_std'])} [{fmt(bd['ci_low'])}, {fmt(bd['ci_high'])}] → 중단 오류 {'존재' if v['q1_stop_gap'] else '미확인'}, 급성/만성 구분 {'필요' if v['q1_chronicity'] else '미확인'}.")
    a = m["q2_sd"]
    out.append(f"- Q2: 중단 판단 AUC 모델 {fmt(a['auc_model'])}, 지속률 사전확률 단독 {fmt(a['auc_prior'])}, 결합 {fmt(a['auc_combo_crossfit'])}. 게이트 ΔJaccard {fmt(a['gate']['d_jaccard'], 4)} [{fmt(a['gate']['ci_low'], 4)}, {fmt(a['gate']['ci_high'], 4)}], stale FP {fmt(a['gate']['stale_per_transition'][0], 2)} → {fmt(a['gate']['stale_per_transition'][1], 2)}, 유지 누락 {fmt(a['gate']['fn_cont_per_transition'][0], 2)} → {fmt(a['gate']['fn_cont_per_transition'][1], 2)} → 단순 게이트 {'가치 있음' if v['q2_gate'] else '가치 미확인'}.")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(MIMIC4_RECORDS, "rb") as f:
        records = dill.load(f)
    train_idx, test_idx, _ = split_patients(records)
    n_med = max(m for p in records for adm in p for m in adm[2]) + 1
    prior, seen = persistence_prior(records, train_idx, n_med)
    labeler = A.CodeLabeler()
    patients, medw, _ = load_mimic4()
    feats = dx_change_features(patients, test_idx, labeler)
    print(f"features: {len(feats):,} transitions", flush=True)
    sd_paths = sorted(MIMIC4_DUMP.glob("baseline-seed-*.npz"))
    gn_paths = sorted(p for p in GAMENET_DIR.glob("seed-*/per_visit_predictions.npz") if "smoke" not in p.parent.name)
    t_sd, d_sd = load_arm(sd_paths, prior); t_gn, d_gn = load_arm(gn_paths, prior)
    df = feats.merge(t_sd.add_suffix("_sd").rename(columns={"patient_sd": "patient", "visit_pos_sd": "visit_pos"}), on=["patient", "visit_pos"], how="inner")
    df = df.merge(t_gn.add_suffix("_gn").rename(columns={"patient_gn": "patient", "visit_pos_gn": "visit_pos"}), on=["patient", "visit_pos"], how="inner")
    if len(df) != len(feats) or not np.allclose(df["n_stopped"], df["n_stopped_dump_sd"]):
        raise AssertionError(f"join/ground-truth mismatch: feats {len(feats)}, joined {len(df)}")
    print(f"joined {len(df):,}; drug rows sd {len(d_sd):,} gn {len(d_gn):,}", flush=True)

    q1_corr = {k: {"rho_stopped": A.spearman(df[k], df["n_stopped"]), "rho_added": A.spearman(df[k], df["n_added"])}
               for k in ["n_resolved_acute", "n_resolved_chronic", "n_resolved_status", "n_new_acute", "n_new_chronic", "n_persist_chronic", "n_persist_acute"]}
    q1_sd = q1_regressions(df, "stale_keep_rate_sd", args.n_boot, args.seed)
    q1_gn = q1_regressions(df, "stale_keep_rate_gn", args.n_boot, args.seed)
    q2_sd = q2_for_arm(d_sd, sd_paths, args.n_boot, args.seed)
    q2_gn = q2_for_arm(d_gn, gn_paths, args.n_boot, args.seed)
    ba = next(b for b in q1_sd["betas"] if b["feature"] == "n_resolved_acute"); bd = next(b for b in q1_sd["betas"] if b["feature"] == "acute_minus_chronic")
    verdict = {"q1_stop_gap": bool(ba["beta_std"] > 0 and ba["ci_low"] > 0), "q1_chronicity": bool(ba["beta_std"] > 0 and ba["ci_low"] > 0 and bd["beta_std"] > 0 and bd["ci_low"] > 0),
               "q2_gate": bool(q2_sd["gate"]["d_jaccard"] > 0 and q2_sd["gate"]["ci_low"] > 0)}
    order = np.argsort(prior)
    valid = [i for i in order if seen[i] >= 50]
    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "n_transitions": int(len(df)), "n_seeds_sd": len(sd_paths), "n_seeds_gn": len(gn_paths),
         "rule_q1": RULE_Q1, "rule_q2": RULE_Q2, "verdict": verdict, "q1_corr": q1_corr,
         "q1_bins_acute": bins(df, "n_resolved_acute", ["sd", "gn"]), "q1_bins_chronic": bins(df, "n_resolved_chronic", ["sd", "gn"]),
         "q1_reg_sd": q1_sd, "q1_reg_gn": q1_gn, "q2_sd": q2_sd, "q2_gn": q2_gn,
         "prior_overall": float(np.average(prior, weights=np.maximum(seen, 1))),
         "prior_lowest": [{"atc3": medw[int(i)], "prior": float(prior[i]), "n": int(seen[i])} for i in valid[:10]],
         "prior_highest": [{"atc3": medw[int(i)], "prior": float(prior[i]), "n": int(seen[i])} for i in valid[::-1][:10]],
         "coverage": {"share_transitions_any_resolved_acute": float((df["n_resolved_acute"] > 0).mean()), "mean_resolved_acute": float(df["n_resolved_acute"].mean()),
                      "mean_resolved_chronic": float(df["n_resolved_chronic"].mean()), "mean_n_stopped": float(df["n_stopped"].mean())}}
    m["interpretation"] = interpret(m)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "resolved_acute_stop_check_mimic4.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = render(m); (RESULTS / "resolved_acute_stop_check_mimic4.md").write_text(md, encoding="utf-8"); print(md)


if __name__ == "__main__":
    main()
