"""(B) Acute-label generation and correlation with medication change.

    python scripts/acute_label_corr.py --dataset mimic3 [--order chronological|safedrug]
    python scripts/acute_label_corr.py --dataset mimic4

Pure statistics, no model. Uses the full cohort (all splits): this is a
label-vs-outcome description, not model selection.

MIMIC-III: SafeDrug records order visits by HADM_ID, which is not
chronological (50% of transitions go backwards in time). The primary run
re-orders each patient's visits by ADMITTIME (master_visits.csv sidecar);
--order safedrug reproduces the benchmark's order as a sensitivity check.
MIMIC-IV final5 is chronological already.

Outputs (aggregate only): results/acute_label_corr_<dataset>[_safedrug-order].{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import acute_lib as A  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
SERVERITYMED = Path(r"C:\Users\Administrator\Desktop\ServerityMed")
sys.path.insert(0, str(SERVERITYMED / "src"))  # records_final5.pkl pickles a class from mimic_iv_rebuild

M3_RECORDS = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\records_final.pkl")
M3_VOC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\voc_final.pkl")
M3_MASTER = Path(r"C:\Users\Administrator\Desktop\unfair\out\acute_driver_audit\safedrug_mimic3_cohort\master_visits.csv")
M3_DXICD = Path(r"C:\Users\Administrator\Desktop\unfair\DIAGNOSES_ICD.csv")
M4_RECORDS = SERVERITYMED / "data/mimic-iv/records_final5.pkl"
M4_VOC = SERVERITYMED / "data/mimic-iv/voc_final5.pkl"
M4_HADM = SERVERITYMED / "data/mimic-iv/records_final5_hadm_ids.pkl"
M4_ADM = SERVERITYMED / "data/raw_mimic_iv/admissions.csv.gz"
M4_DX = SERVERITYMED / "data/raw_mimic_iv/diagnoses_icd.csv"
M4_ORGAN = SERVERITYMED / "data/mimic-iv/organ_function_features_final5.pkl"
LAB_MARKERS = ["creatinine", "bun", "alt", "ast", "bilirubin_total", "albumin"]

CORR_FEATURES = [
    "n_new_acute_prev", "n_new_acute_excl_status_prev", "n_new_status_prev", "n_new_chronic_prev", "n_new_unknown_prev", "n_new_prev",
    "n_new_acute_hist", "n_new_acute_excl_status_hist", "n_new_chronic_hist", "n_new_hist",
    "n_acute", "n_chronic", "n_dx", "n_dropped_prev", "n_prev_meds", "gap_days", "emergency",
]


def _load(path):
    with open(path, "rb") as f:
        return dill.load(f)


def load_mimic3(order: str):
    records, voc = _load(M3_RECORDS), _load(M3_VOC)
    dxw, medw = voc["diag_voc"].idx2word, voc["med_voc"].idx2word
    mv = pd.read_csv(M3_MASTER, parse_dates=["ADMITTIME"])
    n_visits = sum(len(p) for p in records)
    if len(mv) != n_visits or mv["safedrug_patient_index"].nunique() != len(records):
        raise AssertionError("master_visits.csv does not align with records_final.pkl")
    key = {(int(p), int(v)): (int(h), t, a) for p, v, h, t, a in zip(
        mv["safedrug_patient_index"], mv["safedrug_visit_index"], mv["HADM_ID"], mv["ADMITTIME"], mv["ADMISSION_TYPE"])}
    dx = pd.read_csv(M3_DXICD, usecols=["HADM_ID", "SEQ_NUM", "ICD9_CODE"], dtype={"ICD9_CODE": str})
    dx1 = dx[dx["SEQ_NUM"] == 1].dropna(subset=["ICD9_CODE"]).drop_duplicates("HADM_ID")
    principal = {int(h): (str(c), 9) for h, c in zip(dx1["HADM_ID"], dx1["ICD9_CODE"])}
    patients = []
    for pi, p in enumerate(records):
        visits = []
        for vi, visit in enumerate(p):
            h, t, a = key[(pi, vi)]
            visits.append({"hadm": h, "dx": {A.split_code(dxw[i]) for i in visit[0]}, "meds": set(int(m) for m in visit[2]),
                           "principal": principal.get(h), "admit": t, "emergency": a in ("EMERGENCY", "URGENT"),
                           "safedrug_pos": vi})
        if order == "chronological":
            visits.sort(key=lambda v: (v["admit"], v["hadm"]))
        patients.append(visits)
    meta = {"records": str(M3_RECORDS), "visit_order": order, "n_patients": len(records), "n_visits": n_visits,
            "principal_source": str(M3_DXICD), "principal_coverage": len(principal) / n_visits if n_visits else None}
    return patients, medw, meta


def load_mimic4():
    records, voc, hadm = _load(M4_RECORDS), _load(M4_VOC), _load(M4_HADM)
    dxw, medw = voc["diag_voc"].idx2word, voc["med_voc"].idx2word
    adm = pd.read_csv(M4_ADM, usecols=["hadm_id", "admittime", "admission_type"], parse_dates=["admittime"]).set_index("hadm_id")
    dx = pd.read_csv(M4_DX, usecols=["hadm_id", "seq_num", "icd_code", "icd_version"], dtype={"icd_code": str})
    dx1 = dx[dx["seq_num"] == 1].dropna(subset=["icd_code"]).drop_duplicates("hadm_id")
    principal = {int(h): (str(c).strip(), int(v)) for h, c, v in zip(dx1["hadm_id"], dx1["icd_code"], dx1["icd_version"])}
    organ = _load(M4_ORGAN)
    patients = []
    for pi, p in enumerate(records):
        visits = []
        for vi, visit in enumerate(p):
            h = int(hadm[pi][vi])
            a = str(adm.at[h, "admission_type"])
            labs = {m: organ[pi][vi].get(f"{m}_log_ratio_delta") for m in LAB_MARKERS}
            visits.append({"hadm": h, "dx": {A.split_code(dxw[i]) for i in visit[0]}, "meds": set(int(m) for m in visit[2]),
                           "principal": principal.get(h), "admit": adm.at[h, "admittime"],
                           "emergency": ("EMER" in a) or ("URGENT" in a), "labs": labs, "safedrug_pos": vi})
        patients.append(visits)
    n_visits = sum(len(p) for p in records)
    meta = {"records": str(M4_RECORDS), "visit_order": "chronological (records order, verified)", "n_patients": len(records),
            "n_visits": n_visits, "principal_source": str(M4_DX), "principal_coverage": len(principal) / n_visits}
    return patients, medw, meta


def build_frame(patients, labeler) -> pd.DataFrame:
    rows = []
    for pi, visits in enumerate(patients):
        for r in A.transition_features(visits, labeler):
            r["patient"] = pi
            v = visits[r["visit_pos"]]
            if "labs" in v:
                for m, val in v["labs"].items():
                    r[f"lab_absdelta_{m}"] = abs(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df["change_mag"] = 1.0 - df["prev_cur_jaccard"]
    df["emergency"] = df["emergency"].astype(float)
    df["gap_days"] = df["gap_days"].astype(float)
    return df


def _design(df, cols):
    X = df[cols].copy()
    if "gap_days" in cols:
        X["gap_days"] = np.log1p(X["gap_days"].fillna(X["gap_days"].median()).clip(lower=0))
    return X.fillna(0).to_numpy(dtype=float)


def analyses(df: pd.DataFrame, medw, labeler, n_boot: int) -> dict:
    out = {}
    n = len(df)
    out["coverage"] = {
        "n_transitions": n, "n_patients": int(df["patient"].nunique()),
        "mean_n_dx": float(df["n_dx"].mean()),
        "dx_code_share_labelled": float((df["n_acute"].sum() + df["n_chronic"].sum()) / df["n_dx"].sum()),
        "dx_code_share_acute_among_labelled": float(df["n_acute"].sum() / (df["n_acute"].sum() + df["n_chronic"].sum())),
        "status_code_share_of_acute": float(df["n_acute_status"].sum() / df["n_acute"].sum()),
        "share_visits_any_new_acute_excl_status_prev": float((df["n_new_acute_excl_status_prev"] > 0).mean()),
        "mean_n_new_acute_excl_status_prev": float(df["n_new_acute_excl_status_prev"].mean()),
        "share_visits_any_new_acute_prev": float((df["n_new_acute_prev"] > 0).mean()),
        "share_visits_any_new_chronic_prev": float((df["n_new_chronic_prev"] > 0).mean()),
        "share_visits_any_new_acute_hist": float((df["n_new_acute_hist"] > 0).mean()),
        "mean_n_new_acute_prev": float(df["n_new_acute_prev"].mean()),
        "mean_n_new_chronic_prev": float(df["n_new_chronic_prev"].mean()),
        "mean_n_new_prev": float(df["n_new_prev"].mean()),
        "mean_n_added": float(df["n_added"].mean()), "median_n_added": float(df["n_added"].median()),
        "mean_n_stopped": float(df["n_stopped"].mean()),
        "mean_change_mag": float(df["change_mag"].mean()),
        "share_emergency": float(df["emergency"].mean()),
        "median_gap_days": float(df["gap_days"].median()),
        "principal_labelled_share": float(df["principal_chronic"].notna().mean()),
    }
    # --- correlations
    corr = []
    for f in CORR_FEATURES:
        x = df[f].fillna(df[f].median()) if f == "gap_days" else df[f]
        row = {"feature": f,
               "rho_n_added": A.spearman(x, df["n_added"]),
               "rho_n_stopped": A.spearman(x, df["n_stopped"]),
               "rho_change_mag": A.spearman(x, df["change_mag"])}
        corr.append(row)
    key_feats = ["n_new_acute_prev", "n_new_acute_excl_status_prev", "n_new_status_prev", "n_new_chronic_prev", "n_new_prev", "n_new_acute_hist", "n_dx", "n_prev_meds"]
    def _rhos(d):
        return [A.spearman(d[f], d["n_added"]) for f in key_feats]
    boot = A.cluster_bootstrap(_rhos, df, "patient", n_boot=n_boot)
    ci = {f: (lo, hi) for f, lo, hi in zip(key_feats, boot["ci_low"], boot["ci_high"])}
    for row in corr:
        if row["feature"] in ci:
            row["rho_n_added_ci"] = ci[row["feature"]]
    out["correlations"] = corr
    # --- 2x2 group means
    g = df.assign(any_new_acute=df["n_new_acute_prev"] > 0, any_new_chronic=df["n_new_chronic_prev"] > 0)
    grp = g.groupby(["any_new_acute", "any_new_chronic"]).agg(
        n=("n_added", "size"), mean_n_added=("n_added", "mean"), mean_n_stopped=("n_stopped", "mean"),
        mean_change_mag=("change_mag", "mean"), mean_n_new_prev=("n_new_prev", "mean")).reset_index()
    out["group_2x2"] = grp.to_dict("records")
    # --- principal diagnosis
    p = df.assign(principal_chronic=df["principal_chronic"].map({0: "acute", 1: "chronic"}).fillna("unknown"),
                  principal_new=df["principal_new_prev"].map({True: "new", False: "seen"}).fillna("n/a"))
    pg = p.groupby(["principal_chronic", "principal_new"]).agg(
        n=("n_added", "size"), mean_n_added=("n_added", "mean"), mean_n_stopped=("n_stopped", "mean"),
        mean_change_mag=("change_mag", "mean")).reset_index()
    out["principal"] = pg.to_dict("records")
    # --- nested OLS
    base = ["n_dx", "n_prev_meds", "gap_days", "emergency"]
    models = {
        "M0 base (n_dx, n_prev_meds, log gap, emergency)": base,
        "M1 = M0 + n_new_prev": base + ["n_new_prev"],
        "M2 = M0 + new acute/chronic/unknown split": base + ["n_new_acute_prev", "n_new_chronic_prev", "n_new_unknown_prev"],
        "M3 = M2 + n_dropped_prev": base + ["n_new_acute_prev", "n_new_chronic_prev", "n_new_unknown_prev", "n_dropped_prev"],
        "M2h = M0 + new-vs-history acute/chronic split": base + ["n_new_acute_hist", "n_new_chronic_hist"],
        "M4 = M0 + acute(excl. status)/status/chronic/unknown split": base + ["n_new_acute_excl_status_prev", "n_new_status_prev", "n_new_chronic_prev", "n_new_unknown_prev"],
    }
    nested = {}
    for name, cols in models.items():
        X = _design(df, cols)
        for target in ["n_added", "n_stopped", "change_mag"]:
            _, r2 = A.ols_standardized(X, df[target].to_numpy(dtype=float))
            nested.setdefault(name, {})[target] = r2
    out["nested_r2"] = nested
    cols = models["M4 = M0 + acute(excl. status)/status/chronic/unknown split"] + ["n_dropped_prev"]
    def _betas(d):
        b, _ = A.ols_standardized(_design(d, cols), d["n_added"].to_numpy(dtype=float))
        return b
    bb = A.cluster_bootstrap(_betas, df, "patient", n_boot=n_boot)
    out["ols_M3_n_added"] = [{"feature": c, "beta_std": pt, "ci_low": lo, "ci_high": hi}
                             for c, pt, lo, hi in zip(cols, bb["point"], bb["ci_low"], bb["ci_high"])]
    # --- acute type analysis
    base_rate = Counter()
    for meds in df["added_meds"]:
        base_rate.update(meds)
    base_rate = {m: c / n for m, c in base_rate.items()}
    cat_rows = defaultdict(list)
    code_rows = defaultdict(list)
    for idx, (codes, meds, n_added) in enumerate(zip(df["new_acute_codes"], df["added_meds"], df["n_added"])):
        for c in codes:
            cat_rows[labeler.category(*c)].append((idx, n_added, meds))
            code_rows[c].append((idx, n_added, meds))
    def _summ(rows, label, min_support=30):
        idxs = {r[0] for r in rows}
        added = Counter()
        for _, _, meds in rows:
            added.update(meds)
        n_t = len(idxs)
        # lift = P(ATC3 added | new acute dx in this group) / P(ATC3 added | any transition);
        # require the class to be added in >= 5% of the group's occurrences (and >= 10 times)
        # so a 3x lift on a 0.3% base rate cannot top the list.
        lifts = sorted(((m, (cnt / len(rows)) / base_rate[m], cnt / len(rows)) for m, cnt in added.items()
                        if base_rate.get(m, 0) > 0 and cnt >= 10 and cnt / len(rows) >= 0.05),
                       key=lambda t: -t[1])[:3]
        return {"label": label, "n_transitions": n_t, "n_occurrences": len(rows),
                "mean_n_added": float(np.mean([r[1] for r in rows])),
                "top_added_by_lift": [{"atc3": medw[m], "lift": round(l, 2), "share": round(s, 3)} for m, l, s in lifts]}
    cats = sorted(cat_rows.items(), key=lambda kv: -len(kv[1]))[:25]
    out["acute_types_by_category"] = [_summ(rows, label) for label, rows in cats]
    codes = sorted(code_rows.items(), key=lambda kv: -len(kv[1]))[:20]
    out["acute_types_by_code"] = [dict(_summ(rows, f"{c[0]} (v{c[1]}) {labeler.describe(*c)}"), code=c[0]) for c, rows in codes]
    out["overall_mean_n_added"] = float(df["n_added"].mean())
    # --- labs (MIMIC-IV only)
    lab_cols = [c for c in df.columns if c.startswith("lab_absdelta_")]
    if lab_cols:
        labs = {}
        for c in lab_cols:
            sub = df[df[c].notna()]
            labs[c] = {"n": len(sub), "rho_with_n_new_acute_prev": A.spearman(sub[c], sub["n_new_acute_prev"]),
                       "rho_with_n_added": A.spearman(sub[c], sub["n_added"]),
                       "rho_with_change_mag": A.spearman(sub[c], sub["change_mag"])}
        X2 = _design(df, cols)
        lab_X = np.column_stack([df[c].fillna(0).to_numpy(float) for c in lab_cols] +
                                [df[c].isna().astype(float).to_numpy() for c in lab_cols])
        y = df["n_added"].to_numpy(float)
        _, r2_m3 = A.ols_standardized(X2, y)
        _, r2_lab = A.ols_standardized(np.column_stack([X2, lab_X]), y)
        _, r2_labonly = A.ols_standardized(np.column_stack([_design(df, base), lab_X]), y)
        labs["nested_r2_n_added"] = {"M3": r2_m3, "M3 + |lab log-ratio delta| (+missing flags)": r2_lab,
                                     "M0 + labs only": r2_labonly}
        out["labs"] = labs
    return out


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_md(dataset, meta, res) -> str:
    L = []
    cv = res["coverage"]
    L.append(f"# 급성 라벨 × 처방 변화 상관 — {dataset.upper()} ({meta['visit_order']})")
    L.append("")
    L.append(f"생성 {meta['generated_at_utc']} · 전이 {cv['n_transitions']:,}건 / 환자 {cv['n_patients']:,}명 (전체 코호트, 모델 없음)")
    L.append("")
    L.append("급성 = HCUP CCI(ICD-9) / CCIR v2023.1(ICD-10)에서 chronic=0. 신규(prev) = 직전 방문에 없던 진단, 신규(hist) = 환자의 이전 모든 방문에 없던 진단. "
             "added/stopped = 직전 방문 정답 처방 대비 추가/중단된 ATC3 수. change_mag = 1 − Jaccard(직전, 현재 정답).")
    L.append("")
    L.append("## 1. 라벨 커버리지와 분포")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    for k, lab in [("dx_code_share_labelled", "진단 코드 중 CCI/CCIR 판정 가능 비율"),
                   ("dx_code_share_acute_among_labelled", "판정된 코드 중 급성(chronic=0) 비율"),
                   ("status_code_share_of_acute", "급성(chronic=0)으로 판정된 코드 중 상태/이력 V·E·Z 코드 비율"),
                   ("share_visits_any_new_acute_excl_status_prev", "신규(prev) 급성(상태 코드 제외) ≥1 인 전이 비율"),
                   ("mean_n_new_acute_excl_status_prev", "전이당 신규(prev) 급성(상태 코드 제외) 수"),
                   ("mean_n_dx", "방문당 진단 수"),
                   ("share_visits_any_new_acute_prev", "신규(prev) 급성 진단 ≥1 인 전이 비율"),
                   ("share_visits_any_new_chronic_prev", "신규(prev) 만성 진단 ≥1 인 전이 비율"),
                   ("share_visits_any_new_acute_hist", "신규(hist) 급성 진단 ≥1 인 전이 비율"),
                   ("mean_n_new_acute_prev", "전이당 신규(prev) 급성 수"), ("mean_n_new_chronic_prev", "전이당 신규(prev) 만성 수"),
                   ("mean_n_added", "전이당 added 약물 수"), ("median_n_added", "added 중앙값"), ("mean_n_stopped", "전이당 stopped 약물 수"),
                   ("mean_change_mag", "평균 change_mag"), ("share_emergency", "응급/긴급 입원 비율"), ("median_gap_days", "방문 간격 중앙값(일)"),
                   ("principal_labelled_share", "주진단(SEQ_NUM=1) CCI 판정 가능 비율")]:
        L.append(f"| {lab} | {fmt(cv[k])} |")
    L.append("")
    L.append("## 2. Spearman 상관")
    L.append("")
    L.append("| 특성 | ρ(n_added) [95% CI, 환자 군집 부트스트랩] | ρ(n_stopped) | ρ(change_mag) |")
    L.append("|---|---|---|---|")
    for r in res["correlations"]:
        ci = r.get("rho_n_added_ci")
        citxt = f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else ""
        L.append(f"| {r['feature']} | {fmt(r['rho_n_added'])}{citxt} | {fmt(r['rho_n_stopped'])} | {fmt(r['rho_change_mag'])} |")
    L.append("")
    L.append("## 3. 신규 급성 × 신규 만성 (직전 방문 대비) 2×2")
    L.append("")
    L.append("| 신규 급성 ≥1 | 신규 만성 ≥1 | n | mean added | mean stopped | mean change_mag | mean 신규 진단 수 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["group_2x2"]:
        L.append(f"| {r['any_new_acute']} | {r['any_new_chronic']} | {r['n']:,} | {fmt(r['mean_n_added'], 2)} | {fmt(r['mean_n_stopped'], 2)} | {fmt(r['mean_change_mag'])} | {fmt(r['mean_n_new_prev'], 2)} |")
    L.append("")
    L.append("## 4. 주진단(SEQ_NUM=1) 급성/만성 × 신규 여부")
    L.append("")
    L.append("| 주진단 CCI | 직전 방문에 있었나 | n | mean added | mean stopped | mean change_mag |")
    L.append("|---|---|---|---|---|---|")
    for r in res["principal"]:
        L.append(f"| {r['principal_chronic']} | {r['principal_new']} | {r['n']:,} | {fmt(r['mean_n_added'], 2)} | {fmt(r['mean_n_stopped'], 2)} | {fmt(r['mean_change_mag'])} |")
    L.append("")
    L.append("## 5. 중첩 OLS — R² (표준화 없음, 절편 포함)")
    L.append("")
    L.append("| 모델 | R²(n_added) | R²(n_stopped) | R²(change_mag) |")
    L.append("|---|---|---|---|")
    for name, d in res["nested_r2"].items():
        L.append(f"| {name} | {fmt(d['n_added'])} | {fmt(d['n_stopped'])} | {fmt(d['change_mag'])} |")
    L.append("")
    L.append("M4 + n_dropped_prev 표준화 계수 (y = n_added), 환자 군집 부트스트랩 95% CI:")
    L.append("")
    L.append("| 특성 | β_std | 95% CI |")
    L.append("|---|---|---|")
    for r in res["ols_M3_n_added"]:
        L.append(f"| {r['feature']} | {fmt(r['beta_std'])} | [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |")
    L.append("")
    L.append(f"## 6. 신규(prev) 급성 진단(상태 코드 제외) 유형별 added 약물 (전체 평균 added {fmt(res['overall_mean_n_added'], 2)})")
    L.append("")
    L.append("| 범주 | 전이 수 | mean added | lift 상위 added ATC3 (lift, 해당 전이 중 비율) |")
    L.append("|---|---|---|---|")
    for r in res["acute_types_by_category"]:
        tops = "; ".join(f"{t['atc3']} ({t['lift']}×, {t['share']:.2f})" for t in r["top_added_by_lift"])
        L.append(f"| {r['label']} | {r['n_transitions']:,} | {fmt(r['mean_n_added'], 2)} | {tops} |")
    L.append("")
    L.append("코드 수준 상위 20:")
    L.append("")
    L.append("| 코드 | 전이 수 | mean added | lift 상위 added ATC3 |")
    L.append("|---|---|---|---|")
    for r in res["acute_types_by_code"]:
        tops = "; ".join(f"{t['atc3']} ({t['lift']}×)" for t in r["top_added_by_lift"])
        L.append(f"| {r['label'][:70]} | {r['n_transitions']:,} | {fmt(r['mean_n_added'], 2)} | {tops} |")
    L.append("")
    if "labs" in res:
        L.append("## 7. (부가) 검사 수치 궤적과의 상관 — MIMIC-IV")
        L.append("")
        L.append("| |log-ratio Δ| 마커 | n(측정) | ρ(신규 급성 수) | ρ(n_added) | ρ(change_mag) |")
        L.append("|---|---|---|---|---|")
        for c, d in res["labs"].items():
            if c.startswith("lab_absdelta_"):
                L.append(f"| {c.replace('lab_absdelta_', '')} | {d['n']:,} | {fmt(d['rho_with_n_new_acute_prev'])} | {fmt(d['rho_with_n_added'])} | {fmt(d['rho_with_change_mag'])} |")
        L.append("")
        L.append("| 모델 (y = n_added) | R² |")
        L.append("|---|---|")
        for k, v in res["labs"]["nested_r2_n_added"].items():
            L.append(f"| {k} | {fmt(v, 4)} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic3", "mimic4"])
    ap.add_argument("--order", default="chronological", choices=["chronological", "safedrug"], help="mimic3 only")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    labeler = A.CodeLabeler()
    if args.dataset == "mimic3":
        patients, medw, meta = load_mimic3(args.order)
    else:
        if args.order != "chronological":
            raise SystemExit("mimic4 records are already chronological; --order applies to mimic3 only")
        patients, medw, meta = load_mimic4()
    df = build_frame(patients, labeler)
    print(f"[{args.dataset}] transitions {len(df):,}, patients {df['patient'].nunique():,}")
    res = analyses(df, medw, labeler, args.n_boot)
    meta.update({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset, "n_boot": args.n_boot,
                 "label_sources": {"cci9": str(A.CCI9_PATH), "ccir10": str(A.CCIR10_PATH), "ccs9": str(A.CCS9_REF), "ccsr10": str(A.CCSR10_PATH)}})
    tag = args.dataset + ("_safedrug-order" if args.order == "safedrug" else "")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"acute_label_corr_{tag}.json").write_text(json.dumps({"meta": meta, "results": res}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md = render_md(args.dataset, meta, res)
    (RESULTS / f"acute_label_corr_{tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
