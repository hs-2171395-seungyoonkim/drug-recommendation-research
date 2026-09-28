"""Stage 1b (POST HOC to stage 1): ON vs OFF with the predicted set size matched.

Stage 1 found the OFF arm (no DDI loss) predicts ~1.8 more drugs per visit, so
part of its Jaccard advantage may be "predicting more" rather than "ranking
better". This re-decodes the OFF arm from its saved probabilities so that the
set size equals ON's, then repeats the stage-1 analysis. No retraining.

Two matchings (both from the same OFF dumps, same seeds 0..3):
  visit  (PRIMARY)   per visit i and seed s, k_i = |ON_s(i) at threshold 0.5|;
                     OFF_s(i) = top-k_i of OFF_s's probabilities. ON's own
                     threshold set IS its top-k_i, so both arms decode as
                     "top-k_i of own probabilities" with identical k_i. Any
                     remaining difference is ranking quality only.
  global (SENSITIVITY) per seed s, a single threshold t_s chosen on the EVAL
                     split so that OFF_s's mean predicted size equals ON_s's
                     mean predicted size there; applied unchanged to test.

Decision rule for THIS analysis (fixed 2026-09-18 before its results were seen;
identical to stage 1, applied to the visit-matched primary on the test split):
  H1 mean dJaccard(ON - OFF_matched) < 0 with patient-bootstrap 95% CI excluding 0;
  H2 permutation p_range of dJaccard across the 24 CC clusters < 0.05;
  H3 Spearman rho[stage-0 GT DDI rate, cluster-mean dJaccard] < 0, one-sided
     patient-level permutation p < 0.05.
  SUPPORTED iff H1 and H2 and H3. This whole analysis is post hoc relative to
  the stage-1 plan; its own rule was not changed after seeing its results.

Outputs
  data/manifests/ddi-arms-cc-cluster-stage1b-size-matched.json
  out/ddi_arms_cc_cluster_stage1b_size_matched.md

Run:  py -3.12 scripts/evaluate_ddi_arms_size_matched.py [--n-perm 10000]
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

import evaluate_ddi_arms_by_cc_cluster as S1  # noqa: E402
from evaluate_ddi_arms_by_cc_cluster import (  # noqa: E402
    ALPHA, DDI_A, N_BOOT, NO_CC, SEEDS, STAGE0, arm_dirs, check_alignment, fmt, load_cc_labels,
    load_cluster_names, run_analysis, visit_metrics,
)
from measure_cluster_ddi_rate import visit_ddi_counts  # noqa: E402
from safedrug_cluster_gap import N_PERM, PERM_SEED  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "ddi-arms-cc-cluster-stage1b-size-matched.json"
REPORT = ROOT / "out" / "ddi_arms_cc_cluster_stage1b_size_matched.md"
RULE = ("POST HOC follow-up to stage 1. PRIMARY = per-visit size matching (OFF decodes top-k_i with k_i = ON's "
        "predicted size at threshold 0.5). SUPPORTED iff H1 (mean dJaccard < 0, patient-bootstrap 95% CI excludes 0) "
        "AND H2 (permutation p_range across 24 CC clusters < 0.05) AND H3 (Spearman rho[stage-0 GT DDI, cluster dJaccard] "
        "< 0, one-sided permutation p < 0.05), test split, seeds 0..3 averaged per visit. Global-threshold matching and "
        "test+eval are sensitivity only. Rule fixed before this analysis's results were seen.")


# ------------------------------------------------------------------ decoding
def topk_rows(prob: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Row i -> 0/1 vector of the k[i] largest probabilities (stable on ties)."""
    out = np.zeros(prob.shape, dtype=np.uint8)
    for i in range(prob.shape[0]):
        ki = int(k[i])
        if ki <= 0:
            continue
        idx = np.argsort(-prob[i], kind="stable")[:ki]
        out[i, idx] = 1
    return out


def threshold_rows(prob: np.ndarray, t: float) -> np.ndarray:
    return (prob >= t).astype(np.uint8)


def find_threshold_for_mean_size(prob: np.ndarray, target_mean: float, tol: float = 1e-3, iters: int = 60) -> float:
    """Bisection on t so that mean_i |{prob_i >= t}| is closest to target_mean
    (the count is non-increasing in t)."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = threshold_rows(prob, mid).sum(1).mean()
        if abs(m - target_mean) < tol:
            return mid
        if m > target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_dumps(arm: str) -> dict[int, dict]:
    out = {}
    for k, d in arm_dirs(arm).items():
        z = np.load(d / "per_visit_predictions.npz")
        out[k] = {key: z[key] for key in ["HADM_ID", "SUBJECT_ID", "split", "y_gt", "y_pred", "y_prob"]}
    return out


def frame_from(arm: str, seed: int, z: dict, y_pred: np.ndarray, splits: list[str], ddi_A: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in np.flatnonzero(np.isin(z["split"], splits)):
        m = visit_metrics(z["y_gt"][i], y_pred[i], ddi_A)
        m.update({"HADM_ID": int(z["HADM_ID"][i]), "SUBJECT_ID": int(z["SUBJECT_ID"][i]),
                  "split": str(z["split"][i]), "seed": seed, "arm": arm})
        rows.append(m)
    return pd.DataFrame(rows)


def decode_off(on: dict, off: dict, mode: str, splits: list[str], ddi_A: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Returns the OFF per-visit frame under `mode` plus matching diagnostics."""
    frames, diag = [], {}
    for s in SEEDS:
        zo, zf = on[s], off[s]
        if not np.array_equal(zo["HADM_ID"], zf["HADM_ID"]) or not np.array_equal(zo["y_gt"], zf["y_gt"]):
            raise AssertionError(f"seed {s}: ON/OFF dumps are not row-aligned")
        if mode == "none":
            y = zf["y_pred"]
            diag[s] = {}
        elif mode == "visit":
            k = zo["y_pred"].sum(1)
            y = topk_rows(zf["y_prob"], k)
            if not np.array_equal(y.sum(1), k):
                raise AssertionError("top-k decode did not reproduce the requested sizes")
            # sanity: ON's own top-k equals its threshold set
            if not np.array_equal(topk_rows(zo["y_prob"], k), zo["y_pred"]):
                raise AssertionError(f"seed {s}: ON top-k differs from its threshold-0.5 set (ties?)")
            diag[s] = {"k_source": "ON per-visit predicted size", "mean_k_test": float(k[zo["split"] == "test"].mean())}
        elif mode == "global":
            ev = zo["split"] == "eval"
            target = float(zo["y_pred"][ev].sum(1).mean())
            t = find_threshold_for_mean_size(zf["y_prob"][ev], target)
            y = threshold_rows(zf["y_prob"], t)
            te = zo["split"] == "test"
            diag[s] = {"threshold": t, "eval_target_mean_size_on": target,
                       "eval_mean_size_off_matched": float(y[ev].sum(1).mean()),
                       "test_mean_size_on": float(zo["y_pred"][te].sum(1).mean()),
                       "test_mean_size_off_matched": float(y[te].sum(1).mean()),
                       "test_mean_size_off_at_0.5": float(zf["y_pred"][te].sum(1).mean())}
        else:
            raise ValueError(mode)
        frames.append(frame_from("OFF", s, zf, y, splits, ddi_A))
    return pd.concat(frames, ignore_index=True), diag


def arm_level(df: pd.DataFrame) -> dict:
    """Visit-level means per seed, then mean ± sd over seeds; DDI pooled per seed as util does."""
    per = []
    for s, sub in df.groupby("seed"):
        pooled = sub["dd_cnt_pred"].sum() / max(1, sum(v * (v - 1) // 2 for v in sub["n_med_pred"]))
        per.append({"jaccard": sub["jaccard"].mean(), "f1": sub["f1"].mean(), "ddi_pooled": pooled,
                    "ddi_visit_mean": sub["ddi_pred"].mean(), "n_med_pred": sub["n_med_pred"].mean()})
    per = pd.DataFrame(per)
    return {c: {"mean": float(per[c].mean()), "sd": float(per[c].std(ddof=1)), "per_seed": per[c].round(5).tolist()} for c in per.columns}


# ------------------------------------------------------------------ report
def render(m: dict) -> str:
    names = m["cluster_names"]
    P = m["primary"]; v = P["verdict"]
    L = ["# DDI loss ON vs OFF, 약물 수 일치 재측정 (1b단계, 1단계에 대한 사후 분석)", "",
         f"생성 {m['generated_at_utc']} · 주 분할 test · 시드 {SEEDS} (방문별 시드 평균) · 순열 {m['n_perm']:,}회 · 부트스트랩 {N_BOOT:,}회 · 재학습 없음", "",
         f"**판정 기준(이 분석의 결과 확인 전 고정):** {m['rule']}", "",
         f"## 판정 (방문별 크기 일치, 주 분석): **가설 {'지지' if v['supported'] else '미지지'}** — H1 {'✓' if v['H1'] else '✗'} · H2 {'✓' if v['H2'] else '✗'} · H3 {'✓' if v['H3'] else '✗'}", "",
         f"- H1 전체 ΔJaccard(ON − OFF_matched) = {fmt(v['h1']['mean'])} [{fmt(v['h1']['ci_low'])}, {fmt(v['h1']['ci_high'])}] (환자 {v['h1']['n_patients']:,}, 방문 {v['h1']['n']:,})",
         f"- H2 군집 간 range = {fmt(v['h2']['observed_range'])} (null 평균 {fmt(v['h2']['null_mean_range'])}, p95 {fmt(v['h2']['null_p95_range'])}), p = {fmt(v['h2']['p_raw_range'])}; 사용 군집 {v['h2']['n_groups_used']}; wSD p = {fmt(v['h2']['p_raw_wsd'])}",
         f"- H3 Spearman ρ = {fmt(v['h3']['rho'], 3)}, 단측 p = {fmt(v['h3']['p_one_sided_negative'])} (양측 {fmt(v['h3']['p_two_sided'])})", "",
         "## 1. Arm 수준 (test split, 방문 평균의 4 seed mean ± sd; DDI pooled = util 정의)", "",
         "| Arm | Jaccard | F1 | DDI pooled | 약물 수 |", "|---|---|---|---|---|"]
    for key, label in [("ON_0.5", "ON @0.5 (기준)"), ("OFF_0.5", "OFF @0.5 (1단계)"), ("OFF_visit", "OFF 방문별 크기 일치"), ("OFF_global", "OFF 전역 임계 일치")]:
        a = m["arm_level"][key]
        L.append(f"| {label} | {fmt(a['jaccard']['mean'])} ± {fmt(a['jaccard']['sd'])} | {fmt(a['f1']['mean'])} ± {fmt(a['f1']['sd'])} "
                 f"| {fmt(a['ddi_pooled']['mean'])} ± {fmt(a['ddi_pooled']['sd'])} | {fmt(a['n_med_pred']['mean'], 2)} ± {fmt(a['n_med_pred']['sd'], 2)} |")
    L += ["", "전역 임계값(시드별, eval에서 맞춤 → test 적용):", "", "| 시드 | t | eval 목표 크기(ON) | eval OFF 크기 | test ON | test OFF 일치 | test OFF @0.5 |", "|---|---|---|---|---|---|---|"]
    for s, d in m["global_matching"].items():
        L.append(f"| {s} | {fmt(d['threshold'], 4)} | {fmt(d['eval_target_mean_size_on'], 2)} | {fmt(d['eval_mean_size_off_matched'], 2)} | {fmt(d['test_mean_size_on'], 2)} | {fmt(d['test_mean_size_off_matched'], 2)} | {fmt(d['test_mean_size_off_at_0.5'], 2)} |")
    L += ["", "## 2. 군집별 ON − OFF_matched (방문별 크기 일치; 정답 DDI율 내림차순)", "",
          "| 군집 | 방문 | 정답 DDI율 (0단계) | Jaccard ON | OFF_m | ΔJaccard [95% CI] | ΔJaccard 1단계(@0.5) | DDI ON | OFF_m | ΔDDI |", "|---|---|---|---|---|---|---|---|---|---|"]
    s1 = {r["cluster_label"]: r for r in m["stage1_cluster_table"]}
    rows = sorted(P["cluster_table"], key=lambda r: (r["cluster_label"] in (NO_CC, "UNLABELLED"), -(r["gt_ddi_stage0"] if r["gt_ddi_stage0"] == r["gt_ddi_stage0"] else -1)))
    for r in rows:
        L.append(f"| {names[r['cluster_label']]} | {r['n_visits']:,} | {fmt(r['gt_ddi_stage0'])} | {fmt(r['jaccard_on'])} | {fmt(r['jaccard_off'])} "
                 f"| {fmt(r['d_jaccard'])} [{fmt(r['d_jaccard_ci_low'])}, {fmt(r['d_jaccard_ci_high'])}] | {fmt(s1.get(r['cluster_label'], {}).get('d_jaccard'))} "
                 f"| {fmt(r['ddi_pred_on'])} | {fmt(r['ddi_pred_off'])} | {fmt(r['d_ddi_pred'])} |")
    t = P["tracking"]
    L += ["", "## 3. 추적 통계 (24 CC 군집, 방문별 크기 일치)", "", "| 항목 | 값 |", "|---|---|",
          f"| Spearman ρ(정답 DDI율 0단계, ΔJaccard) | {fmt(t['rho_gt0_djaccard'], 3)} |",
          f"| Spearman ρ(ΔDDI 예측, ΔJaccard) | {fmt(t['rho_dddi_djaccard'], 3)} |",
          f"| Spearman ρ(정답 DDI율 0단계, ΔDDI 예측) | {fmt(t['rho_gt0_dddi'], 3)} |",
          f"| 상위 6 군집 ΔJaccard | {fmt(t['top_quartile']['mean'])} [{fmt(t['top_quartile']['ci_low'])}, {fmt(t['top_quartile']['ci_high'])}] |",
          f"| 나머지 18 군집 ΔJaccard | {fmt(t['rest']['mean'])} [{fmt(t['rest']['ci_low'])}, {fmt(t['rest']['ci_high'])}] |",
          f"| 차이 (상위 − 나머지) | {fmt(t['top_minus_rest']['mean'])} [{fmt(t['top_minus_rest']['ci_low'])}, {fmt(t['top_minus_rest']['ci_high'])}] |",
          f"| ΔDDI 예측 range 순열 p / ΔF1 range 순열 p | {fmt(P['permutation']['d_ddi_pred']['p_raw_range'])} / {fmt(P['permutation']['d_f1']['p_raw_range'])} |", "",
          "## 4. 민감도", "", "| 분석 | H1 ΔJaccard [95% CI] | H2 p(range) | H3 ρ (단측 p) | 판정 |", "|---|---|---|---|---|"]
    for key, label in [("stage1_none", "1단계 원안 (OFF @0.5, test)"), ("primary", "방문별 크기 일치, test (주)"),
                       ("global_test", "전역 임계 일치, test"), ("visit_test_eval", "방문별 크기 일치, test+eval"), ("global_test_eval", "전역 임계 일치, test+eval")]:
        r = m["summary_rows"][key]
        L.append(f"| {label} | {fmt(r['h1_mean'])} [{fmt(r['h1_lo'])}, {fmt(r['h1_hi'])}] | {fmt(r['h2_p'])} | {fmt(r['h3_rho'], 3)} ({fmt(r['h3_p'])}) | {'지지' if r['supported'] else '미지지'} |")
    L += ["", "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def summary_row(res: dict) -> dict:
    v = res["verdict"]
    return {"h1_mean": v["h1"]["mean"], "h1_lo": v["h1"]["ci_low"], "h1_hi": v["h1"]["ci_high"], "h2_p": v["h2"]["p_raw_range"],
            "h3_rho": v["h3"]["rho"], "h3_p": v["h3"]["p_one_sided_negative"], "supported": v["supported"]}


def interpret(m: dict) -> list[str]:
    P = m["primary"]; v = P["verdict"]; t = P["tracking"]; a = m["arm_level"]; names = m["cluster_names"]
    rows = [r for r in P["cluster_table"] if r["cluster_label"] not in (NO_CC, "UNLABELLED")]
    worst = min(rows, key=lambda r: r["d_jaccard"]); best = max(rows, key=lambda r: r["d_jaccard"])
    s1 = m["summary_rows"]["stage1_none"]
    out = [f"- 약물 수를 방문별로 똑같이 맞추면 OFF의 Jaccard 우위는 {fmt(-s1['h1_mean'])}(1단계)에서 {fmt(-v['h1']['mean'])}로 "
           f"{'줄지만 남는다' if v['H1'] else '사라진다'} (ΔJaccard {fmt(v['h1']['mean'])} [{fmt(v['h1']['ci_low'])}, {fmt(v['h1']['ci_high'])}]). "
           f"같은 크기에서 OFF의 DDI pooled {fmt(a['OFF_visit']['ddi_pooled']['mean'])} vs ON {fmt(a['ON_0.5']['ddi_pooled']['mean'])}."]
    out.append(f"- 군집별 ΔJaccard는 {fmt(worst['d_jaccard'])}({names[worst['cluster_label']]}) ~ {fmt(best['d_jaccard'])}({names[best['cluster_label']]}), range 순열 p = {fmt(v['h2']['p_raw_range'])} → H2 {'성립' if v['H2'] else '불성립'}.")
    out.append(f"- 0단계 정답 DDI율과의 ρ = {fmt(v['h3']['rho'], 2)}(단측 p {fmt(v['h3']['p_one_sided_negative'])}) → H3 {'성립' if v['H3'] else '불성립'}. "
               f"상위 6 군집 {fmt(t['top_quartile']['mean'])} vs 나머지 {fmt(t['rest']['mean'])}, 차이 {fmt(t['top_minus_rest']['mean'])} [{fmt(t['top_minus_rest']['ci_low'])}, {fmt(t['top_minus_rest']['ci_high'])}].")
    g = m["summary_rows"]["global_test"]
    out.append(f"- 전역 임계값 일치(민감도): ΔJaccard {fmt(g['h1_mean'])}, H2 p {fmt(g['h2_p'])}, H3 ρ {fmt(g['h3_rho'], 2)} (p {fmt(g['h3_p'])}) → {'지지' if g['supported'] else '미지지'}.")
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
    stage1 = json.loads(S1.MANIFEST.read_text(encoding="utf-8"))

    on_z, off_z = load_dumps("ON"), load_dumps("OFF")
    splits_all = ["test", "eval"]
    on_all = pd.concat([frame_from("ON", s, on_z[s], on_z[s]["y_pred"], splits_all, ddi_A) for s in SEEDS], ignore_index=True)
    off_none, _ = decode_off(on_z, off_z, "none", splits_all, ddi_A)
    off_visit, diag_visit = decode_off(on_z, off_z, "visit", splits_all, ddi_A)
    off_global, diag_global = decode_off(on_z, off_z, "global", splits_all, ddi_A)
    print(f"decoded; global thresholds {[round(diag_global[s]['threshold'], 4) for s in SEEDS]}", flush=True)

    def sub(df, splits):
        return df[df["split"].isin(splits)]

    on_t = sub(on_all, ["test"])
    alignment = check_alignment(on_t, sub(off_visit, ["test"]))
    results = {}
    for key, off_df, splits in [("primary", off_visit, ["test"]), ("global_test", off_global, ["test"]),
                                ("visit_test_eval", off_visit, splits_all), ("global_test_eval", off_global, splits_all),
                                ("stage1_none", off_none, ["test"])]:
        t1 = time.time()
        results[key] = run_analysis(sub(on_all, splits), sub(off_df, splits), cc, stage0, args.n_perm, args.seed)
        v = results[key]["verdict"]
        print(f"{key}: dJ {v['h1']['mean']:+.4f} [{v['h1']['ci_low']:+.4f},{v['h1']['ci_high']:+.4f}] p_range {v['h2']['p_raw_range']:.4f} "
              f"rho {v['h3']['rho']:+.3f} p {v['h3']['p_one_sided_negative']:.4f} -> {'SUPPORTED' if v['supported'] else 'not'} ({time.time() - t1:.0f}s)", flush=True)
    # stage-1 numbers must be reproduced by the 'none' decode
    s1_h1 = stage1["verdict"]["h1"]["mean"]
    if abs(results["stage1_none"]["verdict"]["h1"]["mean"] - s1_h1) > 1e-9:
        raise AssertionError("re-decoding OFF at 0.5 does not reproduce the stage-1 dJaccard")

    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.time() - t0, 1),
         "rule": RULE, "alpha": ALPHA, "n_perm": args.n_perm, "perm_seed": args.seed, "post_hoc_of": str(S1.MANIFEST),
         "inputs": {"ddi_A_final.pkl": hashlib.sha256(DDI_A.read_bytes()).hexdigest(),
                    "dumps": {arm: {k: hashlib.sha256((d / "per_visit_predictions.npz").read_bytes()).hexdigest()[:16] for k, d in arm_dirs(arm).items()} for arm in ["ON", "OFF"]}},
         "cluster_names": names, "alignment": alignment,
         "arm_level": {"ON_0.5": arm_level(on_t), "OFF_0.5": arm_level(sub(off_none, ["test"])),
                       "OFF_visit": arm_level(sub(off_visit, ["test"])), "OFF_global": arm_level(sub(off_global, ["test"]))},
         "visit_matching": diag_visit, "global_matching": diag_global,
         "primary": results["primary"],
         "sensitivity": {k: results[k] for k in ["global_test", "visit_test_eval", "global_test_eval", "stage1_none"]},
         "summary_rows": {k: summary_row(r) for k, r in results.items()},
         "stage1_cluster_table": stage1["cluster_table"], "post_hoc_changes": []}
    m["interpretation"] = interpret(m)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    REPORT.write_text(render(m), encoding="utf-8")
    print(f"wrote {REPORT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
