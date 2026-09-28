"""Does SafeDrug's error on ADDED medications concentrate where new acute diagnoses are many?

The prestudy so far showed the signal exists in the data (new acute diagnoses
correlate with added drugs). This asks the model-side question the proposed
module actually targets: are the added drugs the baseline model MISSES
concentrated in visits with many new acute diagnoses, over and above what the
plain count of new diagnoses explains?

Data (MIMIC-IV final5, chronological): baseline SafeDrug test-split dumps for
5 seeds (out/mimic4/baseline-seed-*.npz, written by dump_mimic4_predictions.py)
joined on (patient_index, visit_index) with the acute-label transition
features of acute_label_corr.py (HCUP CCI/CCIR chronic=0 = acute; V/E/Z status
codes split out).

Per transition (visit i >= 1 of a test patient), per seed, then averaged over seeds:
  true_added   = cur_true - prev_true
  hit_added    = |pred & true_added|,  miss_rate = 1 - hit_added / |true_added|  (NaN if none added)
  pred_added   = pred - prev_true,     added_precision = |pred_added & true_added| / |pred_added|
  cont_recall  = |pred & (cur_true & prev_true)| / |cur_true & prev_true|      (how well continued drugs are kept)

Decision rule (fixed 2026-09-18 before results were seen):
  (a) NOVELTY GAP: OLS of miss_rate on [log n_added, n_prev_meds, n_dx, log gap, emergency]
      + [n_new_acute(excl. status), n_new_status, n_new_chronic, n_new_unknown];
      standardised beta of n_new_acute > 0 with patient-cluster bootstrap 95% CI excluding 0.
  (b) CHRONICITY GAP: beta(n_new_acute) - beta(n_new_chronic) > 0 with bootstrap 95% CI excluding 0.
  The module's rationale is COMPLETE iff (a) and (b). (a) without (b) means the
  error tracks novelty in general (a MICRON-style residual would do), not
  chronicity. Descriptive tables, n_added-stratified tables and per-class miss
  rates are reported but do not decide.

Outputs (aggregate only): results/model_error_vs_acute_mimic4.{json,md}
Run:  py -3.12 scripts/model_error_vs_acute.py [--n-boot 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import acute_lib as A  # noqa: E402
from acute_label_corr import load_mimic4  # noqa: E402
from change_visit_split import MIMIC4_DUMP, split_patients  # noqa: E402
from cvsplit_lib import multihot_rows_to_sets  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
ACUTE_BINS = [("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6+", 6, 10**9)]
BASE = ["log_n_added", "n_prev_meds", "n_dx", "log_gap", "emergency"]
NOVELTY = ["n_new_acute_excl_status_prev", "n_new_status_prev", "n_new_chronic_prev", "n_new_unknown_prev"]
RULE = ("(a) beta_std(n_new_acute excl. status) on per-transition added miss_rate > 0 with patient-bootstrap 95% CI "
        "excluding 0, after [log n_added, n_prev_meds, n_dx, log gap, emergency] + status/chronic/unknown counts; "
        "(b) beta(acute) - beta(chronic) > 0 with CI excluding 0. Rationale COMPLETE iff (a) and (b). Fixed before results.")


# ------------------------------------------------------------------ per-transition model error
def added_metrics(prev_true: set, cur_true: set, pred: set) -> dict:
    true_added = cur_true - prev_true
    continued = cur_true & prev_true
    pred_added = pred - prev_true
    hit = len(pred & true_added)
    return {
        "n_added": len(true_added), "n_hit_added": hit, "n_missed_added": len(true_added) - hit,
        "miss_rate": (1 - hit / len(true_added)) if true_added else float("nan"),
        "n_pred_added": len(pred_added),
        "added_precision": (len(pred_added & true_added) / len(pred_added)) if pred_added else float("nan"),
        "cont_recall": (len(pred & continued) / len(continued)) if continued else float("nan"),
        "jaccard": len(pred & cur_true) / len(pred | cur_true) if (pred | cur_true) else 1.0,
        "n_pred": len(pred),
        "missed_added_codes": sorted(true_added - pred),
    }


def load_test_errors(ddi_unused=None) -> tuple[pd.DataFrame, dict]:
    """Seed-averaged added-drug error per (patient_index, visit_index) transition of the test split."""
    dumps = sorted(MIMIC4_DUMP.glob("baseline-seed-*.npz"))
    if not dumps:
        raise FileNotFoundError(MIMIC4_DUMP)
    per_seed = []
    missed_counter: dict = defaultdict(Counter)   # seed -> Counter of missed added codes (for per-class rates)
    added_counter: dict = defaultdict(Counter)
    acute_of_visit = None
    for path in dumps:
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files}   # materialise once: NpzFile re-decompresses on every z[key]
        seed = path.stem.replace("baseline-", "")
        gt, pr = multihot_rows_to_sets(z["y_gt"]), multihot_rows_to_sets(z["y_pred"])
        order = np.lexsort((z["visit_index"], z["patient_index"]))
        by_p: dict = defaultdict(list)
        for i in order:
            by_p[int(z["patient_index"][i])].append(i)
        rows = []
        for p, idxs in by_p.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                if int(z["visit_index"][b]) != int(z["visit_index"][a]) + 1:
                    raise ValueError(f"patient {p}: visit gap in dump")
                if z["split"][b] != "test":
                    continue
                m = added_metrics(gt[a], gt[b], pr[b])
                m.update({"patient": p, "visit_pos": int(z["visit_index"][b]), "seed": seed})
                rows.append(m)
        per_seed.append(pd.DataFrame(rows))
    df = pd.concat(per_seed, ignore_index=True)
    num = ["n_added", "n_hit_added", "n_missed_added", "miss_rate", "n_pred_added", "added_precision", "cont_recall", "jaccard", "n_pred"]
    avg = df.groupby(["patient", "visit_pos"], as_index=False)[num].mean()
    avg["n_seeds"] = df.groupby(["patient", "visit_pos"]).size().to_numpy()
    if not (avg["n_seeds"] == len(dumps)).all():
        raise AssertionError("some transitions are missing under some seeds")
    # per-code miss counts pooled over seeds (each seed contributes its own misses)
    codes_missed, codes_added = Counter(), Counter()
    for r in df.itertuples(index=False):
        codes_missed.update(r.missed_added_codes)
    added_sets = df.drop_duplicates(["patient", "visit_pos"])   # true added is seed-invariant
    return avg, {"n_seeds": len(dumps), "seeds": [p.stem for p in dumps], "per_seed_rows": df, "codes_missed": codes_missed}


def acute_features_for(patients: list, test_idx: list, labeler) -> pd.DataFrame:
    rows = []
    for pi in test_idx:
        for r in A.transition_features(patients[pi], labeler):
            r["patient"] = pi
            rows.append(r)
    out = pd.DataFrame(rows)
    out["emergency"] = out["emergency"].astype(float)
    return out


# ------------------------------------------------------------------ statistics
def design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].copy()
    return X.fillna(0).to_numpy(dtype=float)


def nested_betas(df: pd.DataFrame, y_col: str, n_boot: int, seed: int) -> dict:
    cols = BASE + NOVELTY
    y = df[y_col].to_numpy(dtype=float)
    _, r2_base = A.ols_standardized(design(df, BASE), y)
    _, r2_nov = A.ols_standardized(design(df, BASE + ["n_new_prev"]), y)
    beta, r2_full = A.ols_standardized(design(df, cols), y)

    def _stat(d):
        b, _ = A.ols_standardized(design(d, cols), d[y_col].to_numpy(dtype=float))
        ia, ic = cols.index("n_new_acute_excl_status_prev"), cols.index("n_new_chronic_prev")
        return np.append(b, b[ia] - b[ic])
    bb = A.cluster_bootstrap(_stat, df, "patient", n_boot=n_boot, seed=seed)
    names = cols + ["acute_minus_chronic"]
    return {"r2_base": r2_base, "r2_base_plus_n_new_prev": r2_nov, "r2_full": r2_full,
            "betas": [{"feature": n, "beta_std": pt, "ci_low": lo, "ci_high": hi} for n, pt, lo, hi in zip(names, bb["point"], bb["ci_low"], bb["ci_high"])],
            "n": int(len(df)), "n_patients": int(df["patient"].nunique())}


def bin_table(df: pd.DataFrame, count_col: str) -> list[dict]:
    out = []
    tot_missed, tot_added = df["n_missed_added"].sum(), df["n_added"].sum()
    for label, lo, hi in ACUTE_BINS:
        sub = df[(df[count_col] >= lo) & (df[count_col] <= hi)]
        if len(sub) == 0:
            continue
        with_added = sub[sub["n_added"] > 0]
        pooled = sub["n_missed_added"].sum() / sub["n_added"].sum() if sub["n_added"].sum() else float("nan")
        out.append({"bin": label, "n_transitions": int(len(sub)), "share_transitions": len(sub) / len(df),
                    "mean_n_added": float(sub["n_added"].mean()), "mean_n_missed": float(sub["n_missed_added"].mean()),
                    "pooled_miss_rate": float(pooled), "mean_miss_rate": float(with_added["miss_rate"].mean()),
                    "mean_added_precision": float(with_added["added_precision"].mean()),
                    "mean_cont_recall": float(sub["cont_recall"].mean()), "mean_jaccard": float(sub["jaccard"].mean()),
                    "share_of_all_missed": float(sub["n_missed_added"].sum() / tot_missed), "share_of_all_added": float(sub["n_added"].sum() / tot_added)})
    return out


def stratified_table(df: pd.DataFrame) -> list[dict]:
    d = df[df["n_added"] > 0].copy()
    d["q_added"] = pd.qcut(d["n_added"], 4, labels=False, duplicates="drop")
    out = []
    for q, sq in d.groupby("q_added"):
        row = {"n_added_quartile": int(q) + 1, "n_added_min": int(sq["n_added"].min()), "n_added_max": int(sq["n_added"].max())}
        for label, lo, hi in ACUTE_BINS:
            sub = sq[(sq["n_new_acute_excl_status_prev"] >= lo) & (sq["n_new_acute_excl_status_prev"] <= hi)]
            row[f"pooled_miss_{label}"] = float(sub["n_missed_added"].sum() / sub["n_added"].sum()) if sub["n_added"].sum() else float("nan")
            row[f"n_{label}"] = int(len(sub))
        out.append(row)
    return out


def per_class_table(per_seed_rows: pd.DataFrame, feats: pd.DataFrame, medw: dict, n_seeds: int, top: int = 15) -> list[dict]:
    """For each ATC3, miss rate among its true additions overall vs in acute-heavy
    (>=3 new acute dx) transitions. True additions are seed-invariant, so one
    seed's rows give the denominators; misses are averaged over seeds."""
    key = feats.set_index(["patient", "visit_pos"])
    first_seed = per_seed_rows["seed"].iloc[0]
    rows0 = per_seed_rows[per_seed_rows["seed"] == first_seed]
    added_all, added_heavy = Counter(), Counter()
    for r in rows0.itertuples(index=False):
        codes = key.loc[(r.patient, r.visit_pos), "added_meds"]
        heavy = key.loc[(r.patient, r.visit_pos), "n_new_acute_excl_status_prev"] >= 3
        added_all.update(codes)
        if heavy:
            added_heavy.update(codes)
    missed_all, missed_heavy = Counter(), Counter()
    for r in per_seed_rows.itertuples(index=False):
        heavy = key.loc[(r.patient, r.visit_pos), "n_new_acute_excl_status_prev"] >= 3
        missed_all.update(r.missed_added_codes)
        if heavy:
            missed_heavy.update(r.missed_added_codes)
    out = []
    for code, n in added_all.most_common(top):
        out.append({"atc3": medw[code], "n_true_added": int(n),
                    "miss_rate_all": missed_all[code] / n_seeds / n,
                    "n_true_added_acute_heavy": int(added_heavy[code]),
                    "miss_rate_acute_heavy": (missed_heavy[code] / n_seeds / added_heavy[code]) if added_heavy[code] else float("nan")})
    return out


# ------------------------------------------------------------------ report
def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render(m: dict) -> str:
    v = m["verdict"]; reg = m["regression_miss_rate"]
    L = ["# 모델이 놓친 added 약물은 신규 급성 진단이 많은 방문에 몰리는가 — MIMIC-IV", "",
         f"생성 {m['generated_at_utc']} · test split 전이 {m['n_transitions']:,}건 / 환자 {m['n_patients']:,}명 · baseline SafeDrug {m['n_seeds']} seed 평균 · 부트스트랩 {m['n_boot']}회", "",
         f"**판정 기준(결과 확인 전 고정):** {m['rule']}", "",
         f"## 판정: (a) 신규성 오류 {'존재' if v['a_novelty_gap'] else '미확인'} · (b) 급성/만성 분리 오류 {'존재' if v['b_chronicity_gap'] else '미확인'} → 모듈 근거 **{'완성' if v['rationale_complete'] else '미완성'}**", "",
         f"- 전체: added 약물 {m['overall']['n_added_total']:,}건 중 모델이 놓친 비율(pooled) {fmt(m['overall']['pooled_miss_rate'])}, 유지 약물 recall {fmt(m['overall']['mean_cont_recall'])}, Jaccard {fmt(m['overall']['mean_jaccard'], 4)}",
         f"- (a) β_std(신규 급성, 상태 코드 제외) = {fmt(v['beta_acute'])} [{fmt(v['beta_acute_ci'][0])}, {fmt(v['beta_acute_ci'][1])}]",
         f"- (b) β(급성) − β(만성) = {fmt(v['beta_diff'])} [{fmt(v['beta_diff_ci'][0])}, {fmt(v['beta_diff_ci'][1])}]; β_std(신규 만성) = {fmt(v['beta_chronic'])}", "",
         "## 1. 신규 급성 진단 수 구간별 (상태 코드 제외)", "",
         "| 신규 급성 수 | 전이 | 비율 | 평균 added | 평균 누락 | pooled 누락률 | 전이 평균 누락률 | added 정밀도 | 유지 recall | Jaccard | 전체 누락 중 비율 / 전체 added 중 비율 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in m["bins_by_acute"]:
        L.append(f"| {r['bin']} | {r['n_transitions']:,} | {fmt(r['share_transitions'])} | {fmt(r['mean_n_added'], 2)} | {fmt(r['mean_n_missed'], 2)} | {fmt(r['pooled_miss_rate'])} | {fmt(r['mean_miss_rate'])} "
                 f"| {fmt(r['mean_added_precision'])} | {fmt(r['mean_cont_recall'])} | {fmt(r['mean_jaccard'])} | {fmt(r['share_of_all_missed'])} / {fmt(r['share_of_all_added'])} |")
    L += ["", "같은 표를 신규 **만성** 진단 수로:", "", "| 신규 만성 수 | 전이 | 평균 added | pooled 누락률 | 전이 평균 누락률 | 유지 recall | Jaccard |", "|---|---|---|---|---|---|---|"]
    for r in m["bins_by_chronic"]:
        L.append(f"| {r['bin']} | {r['n_transitions']:,} | {fmt(r['mean_n_added'], 2)} | {fmt(r['pooled_miss_rate'])} | {fmt(r['mean_miss_rate'])} | {fmt(r['mean_cont_recall'])} | {fmt(r['mean_jaccard'])} |")
    L += ["", "## 2. added 수 사분위 × 신규 급성 구간 — pooled 누락률 (added 수 고정)", "",
          "| added 수 사분위 | 급성 0 | 1-2 | 3-5 | 6+ |", "|---|---|---|---|---|"]
    for r in m["stratified"]:
        cells = " | ".join(f"{fmt(r[f'pooled_miss_{b}'])} (n {r[f'n_{b}']:,})" for b, _, _ in ACUTE_BINS)
        L.append(f"| Q{r['n_added_quartile']} [{r['n_added_min']}–{r['n_added_max']}] | {cells} |")
    L += ["", "## 3. 회귀 — y = 전이별 added 누락률 (added ≥ 1인 전이), 표준화 계수, 환자 군집 부트스트랩 95% CI", "",
          f"R²: 기본 공변량 {fmt(reg['r2_base'])} → +신규 진단 수 {fmt(reg['r2_base_plus_n_new_prev'])} → +급성/상태/만성/미상 분리 {fmt(reg['r2_full'])}", "",
          "| 특성 | β_std | 95% CI |", "|---|---|---|"]
    for b in reg["betas"]:
        L.append(f"| {b['feature']} | {fmt(b['beta_std'])} | [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}] |")
    reg2 = m["regression_missed_count"]
    L += ["", f"보조: y = 누락 약물 **개수** (모든 전이). R² {fmt(reg2['r2_base'])} → {fmt(reg2['r2_base_plus_n_new_prev'])} → {fmt(reg2['r2_full'])}; "
          + "; ".join(f"{b['feature']} {fmt(b['beta_std'])} [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}]" for b in reg2["betas"] if b["feature"] in ("n_new_acute_excl_status_prev", "n_new_chronic_prev", "acute_minus_chronic")), "",
          "## 4. 약물 종류별 누락률 (실제 added 상위 15 ATC3): 전체 vs 신규 급성 ≥3 전이", "",
          "| ATC3 | added 건수 | 누락률 전체 | added 건수 (급성≥3) | 누락률 (급성≥3) |", "|---|---|---|---|---|"]
    for r in m["per_class"]:
        L.append(f"| {r['atc3']} | {r['n_true_added']:,} | {fmt(r['miss_rate_all'])} | {r['n_true_added_acute_heavy']:,} | {fmt(r['miss_rate_acute_heavy'])} |")
    L += ["", "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    v = m["verdict"]; b = m["bins_by_acute"]; c = m["bins_by_chronic"]
    lo, hi = b[0], b[-1]
    out = [f"- 신규 급성 진단이 0개인 전이의 added 누락률(pooled)은 {fmt(lo['pooled_miss_rate'])}, 6개 이상이면 {fmt(hi['pooled_miss_rate'])}. "
           f"신규 만성으로 나누면 {fmt(c[0]['pooled_miss_rate'])} → {fmt(c[-1]['pooled_miss_rate'])}. 유지 약물 recall은 구간에 걸쳐 {fmt(min(r['mean_cont_recall'] for r in b))}~{fmt(max(r['mean_cont_recall'] for r in b))}로 평평하다."]
    out.append(f"- added 수를 고정해도(사분위 층화) 급성 구간에 따른 누락률 차이가 {'남는다' if v['a_novelty_gap'] else '뚜렷하지 않다'}; 회귀에서 β(급성) {fmt(v['beta_acute'])} [{fmt(v['beta_acute_ci'][0])}, {fmt(v['beta_acute_ci'][1])}] → (a) {'성립' if v['a_novelty_gap'] else '불성립'}.")
    out.append(f"- β(급성) − β(만성) = {fmt(v['beta_diff'])} [{fmt(v['beta_diff_ci'][0])}, {fmt(v['beta_diff_ci'][1])}] → (b) {'성립' if v['b_chronicity_gap'] else '불성립'}. "
               + ("급성 진단이 끌어온 added 약물을 모델이 만성 진단이 끌어온 것보다 더 자주 놓친다: 모듈이 겨냥하는 오류가 존재한다." if v["rationale_complete"] else
                  ("오류는 신규 진단 수 전반을 따라가고 급성/만성 구분은 추가 설명력이 없다: MICRON류 신규성 신호로 충분할 가능성." if v["a_novelty_gap"] else "신규 진단 수가 누락률을 설명하지 못한다.")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    err, meta = load_test_errors()
    print(f"model error rows: {len(err):,} transitions, {meta['n_seeds']} seeds", flush=True)
    labeler = A.CodeLabeler()
    patients, medw, meta4 = load_mimic4()
    import dill
    from change_visit_split import MIMIC4_RECORDS
    with open(MIMIC4_RECORDS, "rb") as f:
        n_patients = len(dill.load(f))
    _, test_idx, _ = split_patients(list(range(n_patients)))
    feats = acute_features_for(patients, test_idx, labeler)
    df = err.merge(feats, on=["patient", "visit_pos"], how="inner", suffixes=("", "_lab"))
    if len(df) != len(err) or len(df) != len(feats):
        raise AssertionError(f"join mismatch: errors {len(err)}, labels {len(feats)}, joined {len(df)}")
    if not np.allclose(df["n_added"], df["n_added_lab"]):
        raise AssertionError("n_added from dumps disagrees with n_added from records")
    df["log_n_added"] = np.log1p(df["n_added"])
    df["log_gap"] = np.log1p(df["gap_days"].fillna(df["gap_days"].median()).clip(lower=0))
    with_added = df[df["n_added"] > 0]
    reg = nested_betas(with_added, "miss_rate", args.n_boot, args.seed)
    reg2 = nested_betas(df, "n_missed_added", args.n_boot, args.seed)
    ba = next(b for b in reg["betas"] if b["feature"] == "n_new_acute_excl_status_prev")
    bc = next(b for b in reg["betas"] if b["feature"] == "n_new_chronic_prev")
    bd = next(b for b in reg["betas"] if b["feature"] == "acute_minus_chronic")
    verdict = {"a_novelty_gap": bool(ba["beta_std"] > 0 and ba["ci_low"] > 0), "b_chronicity_gap": bool(bd["beta_std"] > 0 and bd["ci_low"] > 0),
               "beta_acute": ba["beta_std"], "beta_acute_ci": [ba["ci_low"], ba["ci_high"]], "beta_chronic": bc["beta_std"],
               "beta_diff": bd["beta_std"], "beta_diff_ci": [bd["ci_low"], bd["ci_high"]]}
    verdict["rationale_complete"] = bool(verdict["a_novelty_gap"] and verdict["b_chronicity_gap"])
    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "rule": RULE, "n_boot": args.n_boot, "seed": args.seed,
         "n_seeds": meta["n_seeds"], "seeds": meta["seeds"], "n_transitions": int(len(df)), "n_patients": int(df["patient"].nunique()),
         "overall": {"n_added_total": int(df["n_added"].sum()), "pooled_miss_rate": float(df["n_missed_added"].sum() / df["n_added"].sum()),
                     "mean_miss_rate": float(with_added["miss_rate"].mean()), "mean_cont_recall": float(df["cont_recall"].mean()),
                     "mean_jaccard": float(df["jaccard"].mean()), "mean_n_added": float(df["n_added"].mean()),
                     "rho_miss_rate_vs_n_new_acute": A.spearman(with_added["n_new_acute_excl_status_prev"], with_added["miss_rate"]),
                     "rho_miss_rate_vs_n_new_chronic": A.spearman(with_added["n_new_chronic_prev"], with_added["miss_rate"]),
                     "rho_miss_rate_vs_n_added": A.spearman(with_added["n_added"], with_added["miss_rate"])},
         "bins_by_acute": bin_table(df, "n_new_acute_excl_status_prev"), "bins_by_chronic": bin_table(df, "n_new_chronic_prev"),
         "stratified": stratified_table(df), "regression_miss_rate": reg, "regression_missed_count": reg2,
         "per_class": per_class_table(meta["per_seed_rows"], feats, medw, meta["n_seeds"]), "verdict": verdict}
    m["interpretation"] = interpret(m)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_error_vs_acute_mimic4.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = render(m)
    (RESULTS / "model_error_vs_acute_mimic4.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
