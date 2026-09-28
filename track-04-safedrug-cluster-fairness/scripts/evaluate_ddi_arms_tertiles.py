"""Stage 1c (POST HOC to stages 1/1b): DDI-loss cost by ground-truth-DDI tertile.

Stage 1b showed the cost of the DDI loss (ON - OFF Jaccard) points the same
way in every analysis -- larger where ground-truth DDI is high -- but the
24-cluster range statistic lacked power once set sizes were matched. This
collapses the 24 CC clusters into three pre-defined tertiles and tests one
contrast, trading resolution for power.

Tertiles. The 24 CC clusters ranked by their stage-0 ground-truth DDI rate
(mean per-visit rate over ALL splits, data/manifests/cluster-ddi-rate-stage0.json,
fixed before any model result): top 8 = HIGH, next 8 = MID, bottom 8 = LOW.
A visit inherits its cluster's tertile. NO_CC and unlabelled visits are
tabulated but excluded from the tests.

Outcome. Per-visit dJaccard = ON - OFF, seeds 0..3 averaged per visit.
PRIMARY decode = per-visit size matching (stage 1b primary: OFF emits top-k_i
with k_i = ON's predicted size), test split.

Decision rule (fixed 2026-09-18 before this analysis's results were seen):
  H1 : mean dJaccard over all primary-split visits < 0, patient-bootstrap
       95% CI excludes 0 (carried over from stage 1/1b).
  HC : contrast C = mean dJaccard(HIGH) - mean dJaccard(LOW) < 0 with
       patient-level permutation one-sided p < 0.05 (patients reassigned to
       tertiles preserving each tertile's observed patient count, as in
       scripts/safedrug_cluster_gap.py; 10,000 draws, seed 0).
  SUPPORTED iff H1 and HC. Reported but not deciding: 3-group range
  permutation p, monotonicity HIGH <= MID <= LOW, the @0.5 and global-threshold
  decodes, and test+eval. This analysis is post hoc relative to the stage-1
  plan; its own rule was not changed after seeing its results.

Outputs
  data/manifests/ddi-arms-tertiles-stage1c.json
  out/ddi_arms_tertiles_stage1c.md

Run:  py -3.12 scripts/evaluate_ddi_arms_tertiles.py [--n-perm 10000]
"""
from __future__ import annotations

import argparse
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

from evaluate_ddi_arms_by_cc_cluster import (  # noqa: E402
    ALPHA, DDI_A, N_BOOT, NO_CC, SEEDS, STAGE0, arm_dirs, fmt, load_cc_labels, load_cluster_names,
    patient_bootstrap_mean, seed_average,
)
from evaluate_ddi_arms_size_matched import decode_off, frame_from, load_dumps  # noqa: E402
from measure_cluster_ddi_rate import K  # noqa: E402
from safedrug_cluster_gap import MIN_VISITS, N_PERM, PERM_SEED, _build_permutation_target, permutation_p  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "ddi-arms-tertiles-stage1c.json"
REPORT = ROOT / "out" / "ddi_arms_tertiles_stage1c.md"
TERTILES = ["HIGH", "MID", "LOW"]
RULE = ("POST HOC to stages 1/1b. 24 CC clusters -> tertiles by stage-0 GT DDI rate (8/8/8, fixed). PRIMARY = per-visit "
        "size-matched decode, test split, seeds averaged per visit. SUPPORTED iff H1 (mean dJaccard < 0, patient-bootstrap "
        "95% CI excludes 0) AND HC (mean dJaccard[HIGH] - mean dJaccard[LOW] < 0, one-sided patient-level permutation "
        "p < 0.05, 10,000 draws, seed 0). 3-group range p, monotonicity, other decodes and test+eval are reported only. "
        "Rule fixed before this analysis's results were seen.")


# ------------------------------------------------------------------ tertiles
def tertile_map(stage0: dict) -> tuple[dict[str, str], pd.DataFrame]:
    rows = [r for r in stage0["cluster_table"] if r["cluster_label"] != NO_CC]
    if len(rows) != K:
        raise AssertionError(f"expected {K} CC clusters in stage-0 table, found {len(rows)}")
    ranked = sorted(rows, key=lambda r: -r["ddi_rate_mean"])
    mapping, table = {}, []
    for i, r in enumerate(ranked):
        t = TERTILES[i // (K // 3)]
        mapping[r["cluster_label"]] = t
        table.append({"rank": i + 1, "cluster_label": r["cluster_label"], "tertile": t,
                      "gt_ddi_rate_mean": r["ddi_rate_mean"], "n_visits_all_splits": r["n_visits"]})
    return mapping, pd.DataFrame(table)


def visit_deltas(on: pd.DataFrame, off: pd.DataFrame, cc: pd.DataFrame, tmap: dict[str, str]) -> pd.DataFrame:
    cols = ["jaccard", "f1", "n_med_pred", "ddi_pred", "ddi_true", "n_med_gt"]
    a, b = seed_average(on, cols), seed_average(off, cols)
    df = a.merge(b, on=["HADM_ID", "SUBJECT_ID"], suffixes=("_on", "_off"))
    if len(df) != len(a) or len(df) != len(b):
        raise AssertionError("ON/OFF visit sets differ")
    for c in ["jaccard", "f1", "ddi_pred", "n_med_pred"]:
        df[f"d_{c}"] = df[f"{c}_on"] - df[f"{c}_off"]
    df = df.merge(cc[["HADM_ID", "cluster_label"]], on="HADM_ID", how="left")
    df["cluster_label"] = df["cluster_label"].fillna("UNLABELLED")
    df["tertile"] = df["cluster_label"].map(tmap).fillna(df["cluster_label"])   # NO_CC / UNLABELLED keep their label
    return df


# ------------------------------------------------------------------ tests
def contrast_permutation(group: np.ndarray, subject: np.ndarray, values: np.ndarray,
                         n_groups: int, hi: int, lo: int, n_perm: int, seed: int) -> dict:
    """C = mean(values | group==hi) - mean(values | group==lo); null by patient-level
    reassignment preserving each group's patient quota. One-sided p for C <= observed."""
    target, patients, patient_of_visit = _build_permutation_target(group, subject, n_groups)

    def contrast(g):
        return values[g == hi].mean() - values[g == lo].mean()

    obs = contrast(group)
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
        null[b] = contrast(cl[patient_of_visit])
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p05": float(np.percentile(null, 5)),
            "p_one_sided_negative": float((1 + int((null <= obs).sum())) / (1 + n_perm)),
            "p_two_sided": float((1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + n_perm)),
            "z": float((obs - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else float("nan"),
            "n_perm": n_perm, "seed": seed}


def contrast_bootstrap(df: pd.DataFrame, value_col: str, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Joint patient bootstrap of mean(HIGH) - mean(LOW): patients resampled once,
    both tertile means recomputed from the same draw (a patient may sit in both)."""
    groups = df.groupby("SUBJECT_ID").indices
    keys = list(groups)
    vals = df[value_col].to_numpy(dtype=float)
    is_hi = (df["tertile"] == "HIGH").to_numpy()
    is_lo = (df["tertile"] == "LOW").to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), size=len(keys))])
        h, l = is_hi[idx], is_lo[idx]
        draws[b] = vals[idx][h].mean() - vals[idx][l].mean()
    obs = vals[is_hi].mean() - vals[is_lo].mean()
    return {"mean": float(obs), "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)), "n_boot": n_boot}


def analyse(df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    tdf = df[df["tertile"].isin(TERTILES)]
    code = tdf["tertile"].map({t: i for i, t in enumerate(TERTILES)}).to_numpy()
    subj = tdf["SUBJECT_ID"].to_numpy()
    table = []
    for t in TERTILES + [NO_CC, "UNLABELLED"]:
        sub = df[df["tertile"] == t]
        if len(sub) == 0:
            continue
        b = patient_bootstrap_mean(sub["d_jaccard"].to_numpy(), sub["SUBJECT_ID"].to_numpy(), seed=seed)
        table.append({"tertile": t, "n_visits": int(len(sub)), "n_patients": int(sub["SUBJECT_ID"].nunique()),
                      "gt_ddi_true_mean": float(sub["ddi_true_on"].mean()),
                      "jaccard_on": float(sub["jaccard_on"].mean()), "jaccard_off": float(sub["jaccard_off"].mean()),
                      "d_jaccard": b["mean"], "d_jaccard_ci_low": b["ci_low"], "d_jaccard_ci_high": b["ci_high"],
                      "f1_on": float(sub["f1_on"].mean()), "f1_off": float(sub["f1_off"].mean()), "d_f1": float(sub["d_f1"].mean()),
                      "ddi_pred_on": float(sub["ddi_pred_on"].mean()), "ddi_pred_off": float(sub["ddi_pred_off"].mean()),
                      "d_ddi_pred": float(sub["d_ddi_pred"].mean()),
                      "n_med_pred_on": float(sub["n_med_pred_on"].mean()), "n_med_pred_off": float(sub["n_med_pred_off"].mean())})
    h1 = patient_bootstrap_mean(df["d_jaccard"].to_numpy(), df["SUBJECT_ID"].to_numpy(), seed=seed)
    vals = tdf["d_jaccard"].to_numpy(dtype=float)
    hc = contrast_permutation(code, subj, vals, len(TERTILES), hi=0, lo=2, n_perm=n_perm, seed=seed)
    hc["bootstrap"] = contrast_bootstrap(tdf, "d_jaccard", seed=seed)
    rng_test = permutation_p(code, subj, vals, n_groups=len(TERTILES), min_visits=MIN_VISITS, n_perm=n_perm, seed=seed)
    means = {r["tertile"]: r["d_jaccard"] for r in table if r["tertile"] in TERTILES}
    monotone = bool(means["HIGH"] <= means["MID"] <= means["LOW"])
    extra = {}
    for c in ["d_f1", "d_ddi_pred"]:
        v = tdf[c].to_numpy(dtype=float); ok = ~np.isnan(v)
        extra[c] = contrast_permutation(code[ok], subj[ok], v[ok], len(TERTILES), 0, 2, n_perm, seed)
    verdict = {"H1": bool(h1["mean"] < 0 and h1["ci_high"] < 0), "HC": bool(hc["observed"] < 0 and hc["p_one_sided_negative"] < ALPHA)}
    verdict["supported"] = bool(verdict["H1"] and verdict["HC"])
    return {"table": table, "h1": h1, "contrast": hc, "range3": rng_test, "monotone_high_le_mid_le_low": monotone,
            "contrast_other": extra, "verdict": verdict, "n_visits": int(len(df)), "n_visits_tested": int(len(tdf))}


# ------------------------------------------------------------------ report
def render(m: dict) -> str:
    P = m["primary"]; v = P["verdict"]; hc = P["contrast"]
    L = ["# DDI loss 비용의 정답 DDI율 3분위 검정 (1c단계, 1/1b단계에 대한 사후 분석)", "",
         f"생성 {m['generated_at_utc']} · 주 분할 test · 방문별 크기 일치 디코딩 · 시드 {SEEDS} 방문별 평균 · 순열 {m['n_perm']:,}회 · 부트스트랩 {N_BOOT:,}회", "",
         f"**판정 기준(이 분석의 결과 확인 전 고정):** {m['rule']}", "",
         f"## 판정: **가설 {'지지' if v['supported'] else '미지지'}** — H1 {'✓' if v['H1'] else '✗'} · HC {'✓' if v['HC'] else '✗'}", "",
         f"- H1 전체 ΔJaccard = {fmt(P['h1']['mean'])} [{fmt(P['h1']['ci_low'])}, {fmt(P['h1']['ci_high'])}] (방문 {P['h1']['n']:,}, 환자 {P['h1']['n_patients']:,})",
         f"- HC 대비 ΔJ(상) − ΔJ(하) = {fmt(hc['observed'])} [부트스트랩 {fmt(hc['bootstrap']['ci_low'])}, {fmt(hc['bootstrap']['ci_high'])}]; 순열 null 평균 {fmt(hc['null_mean'])} ± {fmt(hc['null_sd'])}, z {fmt(hc['z'], 2)}, 단측 p = {fmt(hc['p_one_sided_negative'])} (양측 {fmt(hc['p_two_sided'])})",
         f"- 보조: 3군 range 순열 p = {fmt(P['range3']['p_raw_range'])}, 단조(상 ≤ 중 ≤ 하) = {P['monotone_high_le_mid_le_low']}; ΔF1 대비 p = {fmt(P['contrast_other']['d_f1']['p_one_sided_negative'])}, ΔDDI 예측 대비 p = {fmt(P['contrast_other']['d_ddi_pred']['p_one_sided_negative'])}", "",
         "## 1. 3분위 구성 (0단계 정답 DDI율, 전체 분할, 고정)", "",
         "| 분위 | 군집 (순위 순) | 정답 DDI율 범위 |", "|---|---|---|"]
    names = m["cluster_names"]
    for t in TERTILES:
        rows = [r for r in m["tertile_table"] if r["tertile"] == t]
        L.append(f"| {t} | {', '.join(names[r['cluster_label']] for r in rows)} | {fmt(min(r['gt_ddi_rate_mean'] for r in rows))} – {fmt(max(r['gt_ddi_rate_mean'] for r in rows))} |")
    L += ["", "## 2. 분위별 ON − OFF (주 분석)", "",
          "| 분위 | 방문 | 환자 | 정답 DDI율(test) | Jaccard ON | OFF_m | ΔJaccard [95% CI] | ΔF1 | DDI ON | OFF_m | ΔDDI | 약물 수 ON / OFF_m |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in P["table"]:
        L.append(f"| {r['tertile']} | {r['n_visits']:,} | {r['n_patients']:,} | {fmt(r['gt_ddi_true_mean'])} | {fmt(r['jaccard_on'])} | {fmt(r['jaccard_off'])} "
                 f"| {fmt(r['d_jaccard'])} [{fmt(r['d_jaccard_ci_low'])}, {fmt(r['d_jaccard_ci_high'])}] | {fmt(r['d_f1'])} | {fmt(r['ddi_pred_on'])} | {fmt(r['ddi_pred_off'])} | {fmt(r['d_ddi_pred'])} "
                 f"| {fmt(r['n_med_pred_on'], 2)} / {fmt(r['n_med_pred_off'], 2)} |")
    L += ["", "## 3. 민감도 (같은 3분위, 다른 디코딩·분할)", "",
          "| 분석 | ΔJ 상 / 중 / 하 | 대비 상−하 [95% CI] | 단측 p | 3군 range p | 단조 | 판정 |", "|---|---|---|---|---|---|---|"]
    for key, label in [("primary", "방문별 크기 일치, test (주)"), ("none_test", "OFF @0.5, test (1단계 디코딩)"), ("global_test", "전역 임계 일치, test"),
                       ("visit_test_eval", "방문별 크기 일치, test+eval"), ("none_test_eval", "OFF @0.5, test+eval"), ("global_test_eval", "전역 임계 일치, test+eval")]:
        r = m["primary"] if key == "primary" else m["sensitivity"][key]
        mm = {x["tertile"]: x["d_jaccard"] for x in r["table"]}
        c = r["contrast"]
        L.append(f"| {label} | {fmt(mm['HIGH'])} / {fmt(mm['MID'])} / {fmt(mm['LOW'])} | {fmt(c['observed'])} [{fmt(c['bootstrap']['ci_low'])}, {fmt(c['bootstrap']['ci_high'])}] "
                 f"| {fmt(c['p_one_sided_negative'])} | {fmt(r['range3']['p_raw_range'])} | {'Y' if r['monotone_high_le_mid_le_low'] else 'N'} | {'지지' if r['verdict']['supported'] else '미지지'} |")
    L += ["", "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    P = m["primary"]; v = P["verdict"]; hc = P["contrast"]
    mm = {x["tertile"]: x for x in P["table"]}
    out = [f"- **{'지지' if v['supported'] else '미지지'}.** 정답 DDI율 상위 8군집의 ΔJaccard {fmt(mm['HIGH']['d_jaccard'])} vs 하위 8군집 {fmt(mm['LOW']['d_jaccard'])}, "
           f"대비 {fmt(hc['observed'])} [{fmt(hc['bootstrap']['ci_low'])}, {fmt(hc['bootstrap']['ci_high'])}], 환자 단위 순열 단측 p = {fmt(hc['p_one_sided_negative'])}."]
    out.append(f"- 중위는 {fmt(mm['MID']['d_jaccard'])}로 {'단조 감소' if P['monotone_high_le_mid_le_low'] else '단조가 아님'}. "
               f"예측 DDI 감소량은 상 {fmt(mm['HIGH']['d_ddi_pred'])} / 중 {fmt(mm['MID']['d_ddi_pred'])} / 하 {fmt(mm['LOW']['d_ddi_pred'])}.")
    sens = m["sensitivity"]
    n_sup = sum(1 for r in sens.values() if r["verdict"]["supported"])
    out.append(f"- 민감도 {len(sens)}개 중 {n_sup}개가 같은 판정. 단측 p 범위 {fmt(min(r['contrast']['p_one_sided_negative'] for r in sens.values()))} ~ {fmt(max(r['contrast']['p_one_sided_negative'] for r in sens.values()))}.")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=PERM_SEED)
    args = ap.parse_args(argv)
    t0 = time.time()
    ddi_A = np.asarray(dill.load(open(DDI_A, "rb")))
    cc = load_cc_labels(); names = load_cluster_names() | {"UNLABELLED": "UNLABELLED"}
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    tmap, ttable = tertile_map(stage0)
    print("tertiles:", {t: [c for c, tt in tmap.items() if tt == t] for t in TERTILES}, flush=True)

    on_z, off_z = load_dumps("ON"), load_dumps("OFF")
    splits_all = ["test", "eval"]
    on_all = pd.concat([frame_from("ON", s, on_z[s], on_z[s]["y_pred"], splits_all, ddi_A) for s in SEEDS], ignore_index=True)
    decoded = {mode: decode_off(on_z, off_z, mode, splits_all, ddi_A)[0] for mode in ["visit", "none", "global"]}

    def sub(df, splits):
        return df[df["split"].isin(splits)]

    results = {}
    for key, mode, splits in [("primary", "visit", ["test"]), ("none_test", "none", ["test"]), ("global_test", "global", ["test"]),
                              ("visit_test_eval", "visit", splits_all), ("none_test_eval", "none", splits_all), ("global_test_eval", "global", splits_all)]:
        df = visit_deltas(sub(on_all, splits), sub(decoded[mode], splits), cc, tmap)
        results[key] = analyse(df, args.n_perm, args.seed)
        r = results[key]; c = r["contrast"]; mm = {x["tertile"]: x["d_jaccard"] for x in r["table"]}
        print(f"{key}: dJ H/M/L {mm['HIGH']:+.4f}/{mm['MID']:+.4f}/{mm['LOW']:+.4f}  contrast {c['observed']:+.4f} p {c['p_one_sided_negative']:.4f}  "
              f"range3 p {r['range3']['p_raw_range']:.4f} -> {'SUPPORTED' if r['verdict']['supported'] else 'not'}", flush=True)

    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
         "rule": RULE, "alpha": ALPHA, "n_perm": args.n_perm, "perm_seed": args.seed,
         "post_hoc_of": ["data/manifests/ddi-arms-cc-cluster-stage1.json", "data/manifests/ddi-arms-cc-cluster-stage1b-size-matched.json"],
         "inputs": {"stage0_manifest": str(STAGE0), "ddi_A_final.pkl": hashlib.sha256(DDI_A.read_bytes()).hexdigest(),
                    "dumps": {arm: {k: hashlib.sha256((d / "per_visit_predictions.npz").read_bytes()).hexdigest()[:16] for k, d in arm_dirs(arm).items()} for arm in ["ON", "OFF"]}},
         "cluster_names": names, "tertile_map": tmap, "tertile_table": ttable.to_dict("records"),
         "primary": results["primary"], "sensitivity": {k: r for k, r in results.items() if k != "primary"}, "post_hoc_changes": []}
    m["interpretation"] = interpret(m)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    REPORT.write_text(render(m), encoding="utf-8")
    print(f"wrote {REPORT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
