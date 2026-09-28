"""Stop decisions and error headroom of the MIMIC-IV baseline SafeDrug.

Two questions on the 5-seed test dumps (no retraining):

1. Can the model tell which previous drugs to STOP? Among drugs in the previous
   true regimen, how well does the predicted probability separate continued
   (label 1) from stopped (label 0)? AUC overall, by previous-regimen size and by
   drug training frequency; keep-rate of stopped drugs and drop-rate of
   continued drugs at threshold 0.5.

2. Where is the headroom? Every error of a prediction falls in exactly one of
   four bins: stale FP (predicted but stopped), novel FP (predicted, never in
   prev nor cur), missed-added FN, missed-continued FN. For each bin, the
   counterfactual Jaccard if that bin alone were fixed gives an upper bound on
   what a method targeting that error could gain.

Exploratory, descriptive; patient-cluster bootstrap CIs on the headroom.
Outputs: results/stop_decision_mimic4.{md,json}
Run:  py -3.12 scripts/stop_decision_mimic4.py
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
from change_visit_split import MIMIC4_DUMP, MIMIC4_RECORDS  # noqa: E402
from cvsplit_lib import multihot_rows_to_sets  # noqa: E402
from prev_meds_dependence import train_frequencies  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def decompose(prev: set, cur: set, pred: set) -> dict:
    continued, stopped, added = prev & cur, prev - cur, cur - prev
    stale_fp, novel_fp = pred & stopped, pred - prev - cur
    missed_added, missed_cont = added - pred, continued - pred
    base = jac(pred, cur)
    return {
        "n_prev": len(prev), "n_cur": len(cur), "n_pred": len(pred), "n_stopped": len(stopped), "n_added": len(added), "n_cont": len(continued),
        "fp_stale": len(stale_fp), "fp_novel": len(novel_fp), "fn_added": len(missed_added), "fn_cont": len(missed_cont),
        "jaccard": base,
        "j_fix_stale": jac(pred - stale_fp, cur), "j_fix_novel": jac(pred - novel_fp, cur),
        "j_fix_fn_added": jac(pred | missed_added, cur), "j_fix_fn_cont": jac(pred | missed_cont, cur),
        "j_fix_all_fp": jac(pred & cur, cur), "j_fix_all_fn": jac(pred | cur, cur),
        "stale_keep_rate": (len(stale_fp) / len(stopped)) if stopped else float("nan"),
        "cont_drop_rate": (len(missed_cont) / len(continued)) if continued else float("nan"),
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC; ties get average rank."""
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def load(records: list):
    n_med = max(m for p in records for adm in p for m in adm[2]) + 1
    train_freq = train_frequencies(records, n_med)
    dumps = sorted(MIMIC4_DUMP.glob("baseline-seed-*.npz"))
    trans_frames, drug_rows = [], []
    for path in dumps:
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files}
        gt, pr = multihot_rows_to_sets(z["y_gt"]), multihot_rows_to_sets(z["y_pred"])
        order = np.lexsort((z["visit_index"], z["patient_index"]))
        by_p: dict = defaultdict(list)
        for i in order:
            by_p[int(z["patient_index"][i])].append(i)
        rows = []
        for p, idxs in by_p.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                if z["split"][b] != "test":
                    continue
                prev, cur, pred, prob = gt[a], gt[b], pr[b], z["y_prob"][b]
                d = decompose(prev, cur, pred); d.update({"patient": p, "visit_pos": int(z["visit_index"][b]), "seed": path.stem})
                rows.append(d)
                for m in prev:
                    drug_rows.append((path.stem, p, len(prev), m, float(prob[m]), 1 if m in cur else 0, float(train_freq[m])))
        trans_frames.append(pd.DataFrame(rows))
    trans = pd.concat(trans_frames, ignore_index=True)
    num = [c for c in trans.columns if c not in ("patient", "visit_pos", "seed")]
    trans_avg = trans.groupby(["patient", "visit_pos"], as_index=False)[num].mean()
    drugs = pd.DataFrame(drug_rows, columns=["seed", "patient", "n_prev", "med", "prob", "continued", "train_freq"])
    return trans_avg, drugs, train_freq, [p.stem for p in dumps]


def headroom(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    fixes = ["j_fix_stale", "j_fix_novel", "j_fix_fn_added", "j_fix_fn_cont", "j_fix_all_fp", "j_fix_all_fn"]
    def _stat(d):
        return np.array([d[f].mean() - d["jaccard"].mean() for f in fixes])
    bb = A.cluster_bootstrap(_stat, df, "patient", n_boot=n_boot, seed=seed)
    out = {"jaccard": float(df["jaccard"].mean())}
    for f, pt, lo, hi in zip(fixes, bb["point"], bb["ci_low"], bb["ci_high"]):
        out[f] = {"jaccard": float(df[f].mean()), "gain": float(pt), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def by_bins(df: pd.DataFrame, col: str, q: int = 5) -> list[dict]:
    d = df.copy(); d["_b"] = pd.qcut(d[col], q, labels=False, duplicates="drop")
    out = []
    for b, sub in d.groupby("_b"):
        out.append({"lo": int(sub[col].min()), "hi": int(sub[col].max()), "n": int(len(sub)),
                    **{k: float(sub[k].mean()) for k in ["n_pred", "n_cur", "fp_stale", "fp_novel", "fn_added", "fn_cont", "jaccard", "j_fix_stale", "j_fix_novel", "j_fix_fn_added", "j_fix_fn_cont", "stale_keep_rate", "cont_drop_rate"]}})
    return out


def auc_tables(drugs: pd.DataFrame, train_freq: np.ndarray) -> dict:
    out = {"overall": {"auc": auc(drugs["prob"].to_numpy(), drugs["continued"].to_numpy()), "n_drug_rows": int(len(drugs)),
                       "share_continued": float(drugs["continued"].mean()),
                       "keep_rate_stopped_at_0.5": float((drugs.loc[drugs["continued"] == 0, "prob"] >= 0.5).mean()),
                       "keep_rate_continued_at_0.5": float((drugs.loc[drugs["continued"] == 1, "prob"] >= 0.5).mean()),
                       "mean_prob_continued": float(drugs.loc[drugs["continued"] == 1, "prob"].mean()),
                       "mean_prob_stopped": float(drugs.loc[drugs["continued"] == 0, "prob"].mean())}}
    d = drugs.copy(); d["_b"] = pd.qcut(d["n_prev"], 5, labels=False, duplicates="drop")
    out["by_n_prev"] = [{"lo": int(s["n_prev"].min()), "hi": int(s["n_prev"].max()), "n": int(len(s)), "auc": auc(s["prob"].to_numpy(), s["continued"].to_numpy()),
                         "share_continued": float(s["continued"].mean()), "keep_rate_stopped": float((s.loc[s["continued"] == 0, "prob"] >= 0.5).mean()),
                         "keep_rate_continued": float((s.loc[s["continued"] == 1, "prob"] >= 0.5).mean())} for _, s in d.groupby("_b")]
    edges = np.quantile(train_freq[train_freq > 0], [1 / 3, 2 / 3])
    d["_r"] = np.digitize(d["train_freq"], edges)  # 0 rare, 1 mid, 2 common (by code-level tertiles)
    out["rarity_edges"] = edges.tolist()
    out["by_rarity"] = [{"tertile": ["rare", "mid", "common"][int(r)], "n": int(len(s)), "auc": auc(s["prob"].to_numpy(), s["continued"].to_numpy()),
                         "share_continued": float(s["continued"].mean()), "keep_rate_stopped": float((s.loc[s["continued"] == 0, "prob"] >= 0.5).mean()),
                         "keep_rate_continued": float((s.loc[s["continued"] == 1, "prob"] >= 0.5).mean()),
                         "mean_prob_continued": float(s.loc[s["continued"] == 1, "prob"].mean()), "mean_prob_stopped": float(s.loc[s["continued"] == 0, "prob"].mean())}
                        for r, s in d.groupby("_r")]
    # per-seed AUC spread
    out["auc_per_seed"] = {s: auc(g["prob"].to_numpy(), g["continued"].to_numpy()) for s, g in drugs.groupby("seed")}
    # base-rate reference: a predictor that always keeps everything has keep_rate 1 for both; AUC of train_freq alone as a "prior" baseline
    out["auc_train_freq_only"] = auc(drugs["train_freq"].to_numpy(), drugs["continued"].to_numpy())
    return out


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render(m: dict) -> str:
    h = m["headroom"]; a = m["auc"]; o = m["overall"]
    L = ["# 중단 판단과 오류 여유(headroom) — MIMIC-IV baseline SafeDrug", "",
         f"생성 {m['generated_at_utc']} · test 전이 {m['n_transitions']:,}건 / 환자 {m['n_patients']:,}명 · {len(m['seeds'])} seed · 탐색적", "",
         "## 1. 예측 오류의 네 갈래 (전이당 평균 개수)", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 정답 크기 / 예측 크기 | {fmt(o['n_cur'], 2)} / {fmt(o['n_pred'], 2)} |",
         f"| FP: 중단됐는데 예측(stale) / 어디에도 없는데 예측(novel) | {fmt(o['fp_stale'], 2)} / {fmt(o['fp_novel'], 2)} (stale 비중 {fmt(o['fp_stale'] / (o['fp_stale'] + o['fp_novel']))}) |",
         f"| FN: added 누락 / 유지 약물 누락 | {fmt(o['fn_added'], 2)} / {fmt(o['fn_cont'], 2)} (added 비중 {fmt(o['fn_added'] / (o['fn_added'] + o['fn_cont']))}) |",
         f"| 중단 약물 중 계속 예측한 비율 / 유지 약물 중 빠뜨린 비율 | {fmt(o['stale_keep_rate'])} / {fmt(o['cont_drop_rate'])} |", "",
         "## 2. 각 오류 하나만 완벽히 고치면 Jaccard가 얼마나 오르나 (상한, 환자 부트스트랩 95% CI)", "",
         f"실제 Jaccard {fmt(h['jaccard'], 4)}", "",
         "| 고치는 오류 | Jaccard | 이득 [95% CI] |", "|---|---|---|"]
    for key, label in [("j_fix_stale", "stale FP 제거 (완벽한 중단 판단)"), ("j_fix_novel", "novel FP 제거"), ("j_fix_fn_added", "added 누락 복구 (완벽한 추가 판단)"),
                       ("j_fix_fn_cont", "유지 약물 누락 복구"), ("j_fix_all_fp", "모든 FP 제거"), ("j_fix_all_fn", "모든 FN 복구")]:
        r = h[key]; L.append(f"| {label} | {fmt(r['jaccard'], 4)} | +{fmt(r['gain'], 4)} [{fmt(r['ci_low'], 4)}, {fmt(r['ci_high'], 4)}] |")
    L += ["", "직전 약물 수 5분위별 오류 구성과 상한:", "",
          "| 직전 약물 수 | 전이 | 예측/정답 | stale FP | novel FP | added FN | 유지 FN | Jaccard | +stale 제거 | +added 복구 | 중단 약물 유지율 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in m["by_n_prev"]:
        L.append(f"| {r['lo']}–{r['hi']} | {r['n']:,} | {fmt(r['n_pred'], 1)}/{fmt(r['n_cur'], 1)} | {fmt(r['fp_stale'], 2)} | {fmt(r['fp_novel'], 2)} | {fmt(r['fn_added'], 2)} | {fmt(r['fn_cont'], 2)} | {fmt(r['jaccard'])} "
                 f"| +{fmt(r['j_fix_stale'] - r['jaccard'])} | +{fmt(r['j_fix_fn_added'] - r['jaccard'])} | {fmt(r['stale_keep_rate'])} |")
    L += ["", "## 3. 중단 판단: 직전 약물의 예측 확률이 유지/중단을 가르는가", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 직전 약물 행 수 (seed 합산) | {a['overall']['n_drug_rows']:,} (유지 비율 {fmt(a['overall']['share_continued'])}) |",
          f"| AUC(유지=1 vs 중단=0) | {fmt(a['overall']['auc'])} (seed별 {', '.join(fmt(v) for v in a['auc_per_seed'].values())}) |",
          f"| 참고: 약물 학습 빈도만으로의 AUC | {fmt(a['auc_train_freq_only'])} |",
          f"| 평균 확률 유지 / 중단 | {fmt(a['overall']['mean_prob_continued'])} / {fmt(a['overall']['mean_prob_stopped'])} |",
          f"| 0.5 기준: 중단 약물을 계속 예측 / 유지 약물을 계속 예측 | {fmt(a['overall']['keep_rate_stopped_at_0.5'])} / {fmt(a['overall']['keep_rate_continued_at_0.5'])} |", "",
          "| 직전 약물 수 | 행 | 유지 비율 | AUC | 중단 약물 유지율 | 유지 약물 유지율 |", "|---|---|---|---|---|---|"]
    for r in a["by_n_prev"]:
        L.append(f"| {r['lo']}–{r['hi']} | {r['n']:,} | {fmt(r['share_continued'])} | {fmt(r['auc'])} | {fmt(r['keep_rate_stopped'])} | {fmt(r['keep_rate_continued'])} |")
    L += ["", f"약물 희귀도 3분위(코드 단위, 학습 빈도 경계 {', '.join(fmt(e) for e in a['rarity_edges'])}):", "",
          "| 희귀도 | 행 | 유지 비율 | AUC | 중단 약물 유지율 | 유지 약물 유지율 | 평균 확률 유지 / 중단 |", "|---|---|---|---|---|---|---|"]
    for r in a["by_rarity"]:
        L.append(f"| {r['tertile']} | {r['n']:,} | {fmt(r['share_continued'])} | {fmt(r['auc'])} | {fmt(r['keep_rate_stopped'])} | {fmt(r['keep_rate_continued'])} | {fmt(r['mean_prob_continued'])} / {fmt(r['mean_prob_stopped'])} |")
    L += ["", "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    h, a, o = m["headroom"], m["auc"], m["overall"]
    out = [f"- 오류 구성: 전이당 FP {fmt(o['fp_stale'] + o['fp_novel'], 2)}개(그중 stale {fmt(o['fp_stale'] / (o['fp_stale'] + o['fp_novel']))}), FN {fmt(o['fn_added'] + o['fn_cont'], 2)}개(그중 added {fmt(o['fn_added'] / (o['fn_added'] + o['fn_cont']))}). "
           f"중단된 약물의 {fmt(o['stale_keep_rate'])}를 계속 예측하고, added의 {fmt(m['pooled_added_miss'])}를 놓친다."]
    gains = {k: h[k]["gain"] for k in ["j_fix_stale", "j_fix_novel", "j_fix_fn_added", "j_fix_fn_cont"]}
    best = max(gains, key=gains.get)
    names = {"j_fix_stale": "완벽한 중단 판단", "j_fix_novel": "novel FP 제거", "j_fix_fn_added": "완벽한 추가 판단", "j_fix_fn_cont": "유지 누락 복구"}
    out.append("- 상한: " + ", ".join(f"{names[k]} +{fmt(v, 3)}" for k, v in gains.items()) + f". 가장 큰 여유는 **{names[best]}**({fmt(gains[best], 3)}).")
    out.append(f"- 중단 판단 AUC {fmt(a['overall']['auc'])}: 직전 약물 중 유지될 것과 중단될 것을 {'어느 정도' if a['overall']['auc'] > 0.7 else '거의 못'} 가른다. 약물 학습 빈도만으로도 AUC {fmt(a['auc_train_freq_only'])}이므로 모델이 더하는 판별력은 {fmt(a['overall']['auc'] - a['auc_train_freq_only'])}. "
               f"희귀도별 AUC {', '.join(f'{r['tertile']} {fmt(r['auc'])}' for r in a['by_rarity'])}.")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(MIMIC4_RECORDS, "rb") as f:
        records = dill.load(f)
    trans, drugs, train_freq, seeds = load(records)
    print(f"{len(trans):,} transitions, {len(drugs):,} prev-drug rows", flush=True)
    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds, "n_transitions": int(len(trans)), "n_patients": int(trans["patient"].nunique()),
         "overall": {k: float(trans[k].mean()) for k in ["n_prev", "n_cur", "n_pred", "fp_stale", "fp_novel", "fn_added", "fn_cont", "jaccard", "stale_keep_rate", "cont_drop_rate"]},
         "pooled_added_miss": float(trans["fn_added"].sum() / trans["n_added"].sum()),
         "headroom": headroom(trans, args.n_boot, args.seed), "by_n_prev": by_bins(trans, "n_prev"), "auc": auc_tables(drugs, train_freq)}
    m["interpretation"] = interpret(m)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stop_decision_mimic4.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = render(m); (RESULTS / "stop_decision_mimic4.md").write_text(md, encoding="utf-8"); print(md)


if __name__ == "__main__":
    main()
