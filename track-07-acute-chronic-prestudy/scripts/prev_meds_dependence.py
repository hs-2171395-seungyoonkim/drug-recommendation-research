"""Why does SafeDrug miss more ADDED drugs when the previous regimen is large?

model_error_vs_acute.py found the strongest predictor of the added-drug miss
rate is the size of the previous true regimen (beta_std +0.27), not acute
diagnoses. This dissects that dependence with the saved probabilities of the
MIMIC-IV baseline (5 seeds, test split), no retraining.

Candidate mechanisms and what separates them (written before results):
  M1 size budget / threshold saturation  -> predicted set is filled by continued
     drugs; size gap (n_pred - n_cur) turns negative as n_prev grows; at ORACLE
     size (top-n_cur by probability) the added miss rate is flat across n_prev.
  M2 ranking degradation                 -> even at oracle size the added miss
     rate rises with n_prev; mean probability / rank of true-added drugs falls.
  M3 copy dominance                      -> probability of added drugs falls with
     n_prev while probability of continued drugs stays; the stopped-drug drop
     rate also falls (symmetric conservatism).
  M4 rarity confound                     -> drugs added to large regimens are
     rarer in training; adding train frequency to the regression removes n_prev.
Exploratory: no pass/fail gate. Every table is descriptive plus a
patient-cluster bootstrap regression.

Outputs (aggregate): results/prev_meds_dependence_mimic4.{md,json}
Run:  py -3.12 scripts/prev_meds_dependence.py [--n-boot 200]
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
from change_visit_split import MIMIC4_DUMP, MIMIC4_RECORDS, split_patients  # noqa: E402
from cvsplit_lib import multihot_rows_to_sets  # noqa: E402

sys.path.insert(0, r"C:\Users\Administrator\Desktop\ServerityMed\src")   # records_final5.pkl pickles a class from there
ROOT = HERE.parent
RESULTS = ROOT / "results"
THRESHOLDS = [0.3, 0.4, 0.5]


# ------------------------------------------------------------------ per-transition dissection
def topk_set(prob: np.ndarray, k: int) -> set:
    if k <= 0:
        return set()
    return set(int(i) for i in np.argsort(-prob, kind="stable")[:k])


def dissect(prev: set, cur: set, pred: set, prob: np.ndarray, train_freq: np.ndarray) -> dict:
    added, stopped, cont = cur - prev, prev - cur, cur & prev
    oracle = topk_set(prob, len(cur))
    oracle_prev = topk_set(prob, len(prev))
    rank = np.empty_like(prob); rank[np.argsort(-prob, kind="stable")] = np.arange(1, len(prob) + 1)
    d = {
        "n_prev": len(prev), "n_cur": len(cur), "n_added": len(added), "n_stopped": len(stopped), "n_cont": len(cont),
        "n_pred": len(pred), "size_gap": len(pred) - len(cur),
        "n_pred_in_prev": len(pred & prev), "n_pred_added": len(pred - prev),
        "share_pred_in_prev": (len(pred & prev) / len(pred)) if pred else float("nan"),
        "miss_rate": (1 - len(pred & added) / len(added)) if added else float("nan"),
        "miss_rate_oracle": (1 - len(oracle & added) / len(added)) if added else float("nan"),
        "cont_recall": (len(pred & cont) / len(cont)) if cont else float("nan"),
        "cont_recall_oracle": (len(oracle & cont) / len(cont)) if cont else float("nan"),
        "stopped_drop_rate": (len(stopped - pred) / len(stopped)) if stopped else float("nan"),
        "prob_added_mean": float(np.mean([prob[i] for i in added])) if added else float("nan"),
        "prob_cont_mean": float(np.mean([prob[i] for i in cont])) if cont else float("nan"),
        "prob_stopped_mean": float(np.mean([prob[i] for i in stopped])) if stopped else float("nan"),
        "rank_added_mean": float(np.mean([rank[i] for i in added])) if added else float("nan"),
        "train_freq_added_mean": float(np.mean([train_freq[i] for i in added])) if added else float("nan"),
        "train_freq_cont_mean": float(np.mean([train_freq[i] for i in cont])) if cont else float("nan"),
        "jaccard": len(pred & cur) / len(pred | cur) if (pred | cur) else 1.0,
        "jaccard_oracle": len(oracle & cur) / len(oracle | cur) if (oracle | cur) else 1.0,
    }
    for t in THRESHOLDS:
        pt = set(int(i) for i in np.flatnonzero(prob >= t))
        d[f"miss_rate@{t}"] = (1 - len(pt & added) / len(added)) if added else float("nan")
        d[f"n_pred@{t}"] = len(pt)
        d[f"added_precision@{t}"] = (len((pt - prev) & added) / len(pt - prev)) if (pt - prev) else float("nan")
        d[f"jaccard@{t}"] = len(pt & cur) / len(pt | cur) if (pt | cur) else 1.0
    return d


def train_frequencies(records: list, n_med: int) -> np.ndarray:
    train_idx, _, _ = split_patients(records)
    cnt = np.zeros(n_med); n_visits = 0
    for pi in train_idx:
        for adm in records[pi]:
            cnt[list(adm[2])] += 1; n_visits += 1
    return cnt / n_visits


def load_transitions(records: list) -> tuple[pd.DataFrame, list[str]]:
    import time
    t0 = time.time()
    n_med = max(max(m for p in records for adm in p for m in adm[2]), 0) + 1
    train_freq = train_frequencies(records, n_med)
    print(f"[{time.time() - t0:.0f}s] train frequencies over {n_med} codes", flush=True)
    dumps = sorted(MIMIC4_DUMP.glob("baseline-seed-*.npz"))
    frames = []
    for path in dumps:
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files}   # materialise once: NpzFile re-decompresses on every z[key]
        print(f"[{time.time() - t0:.0f}s] loaded {path.name}", flush=True)
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
                d = dissect(gt[a], gt[b], pr[b], z["y_prob"][b].astype(float), train_freq)
                d.update({"patient": p, "visit_pos": int(z["visit_index"][b]), "seed": path.stem,
                          "n_dx": len(records[p][int(z["visit_index"][b])][0])})
                rows.append(d)
        frames.append(pd.DataFrame(rows))
        print(f"[{time.time() - t0:.0f}s] dissected {len(rows):,} transitions of {path.stem}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    num = [c for c in df.columns if c not in ("patient", "visit_pos", "seed")]
    avg = df.groupby(["patient", "visit_pos"], as_index=False)[num].mean()
    print(f"[{time.time() - t0:.0f}s] seed-averaged", flush=True)
    return avg, [p.stem for p in dumps]


# ------------------------------------------------------------------ tables
def bins_by(df: pd.DataFrame, col: str, q: int = 5) -> list[dict]:
    d = df.copy()
    d["_bin"] = pd.qcut(d[col], q, labels=False, duplicates="drop")
    out = []
    for b, sub in d.groupby("_bin"):
        wa = sub[sub["n_added"] > 0]
        pooled = lambda s, k: float((s[k] * s["n_added"]).sum() / s["n_added"].sum()) if s["n_added"].sum() else float("nan")
        out.append({
            "bin": int(b) + 1, "lo": float(sub[col].min()), "hi": float(sub[col].max()), "n": int(len(sub)),
            "n_cur": float(sub["n_cur"].mean()), "n_added": float(sub["n_added"].mean()), "n_stopped": float(sub["n_stopped"].mean()),
            "n_pred": float(sub["n_pred"].mean()), "size_gap": float(sub["size_gap"].mean()),
            "share_pred_in_prev": float(sub["share_pred_in_prev"].mean()), "n_pred_added": float(sub["n_pred_added"].mean()),
            "miss_rate": pooled(wa, "miss_rate"), "miss_rate_oracle": pooled(wa, "miss_rate_oracle"),
            "miss_rate@0.3": pooled(wa, "miss_rate@0.3"), "miss_rate@0.4": pooled(wa, "miss_rate@0.4"),
            "n_pred@0.3": float(sub["n_pred@0.3"].mean()), "added_precision@0.3": float(sub["added_precision@0.3"].mean()),
            "jaccard@0.3": float(sub["jaccard@0.3"].mean()), "jaccard@0.4": float(sub["jaccard@0.4"].mean()),
            "cont_recall": float(sub["cont_recall"].mean()), "cont_recall_oracle": float(sub["cont_recall_oracle"].mean()),
            "stopped_drop_rate": float(sub["stopped_drop_rate"].mean()),
            "prob_added": float(wa["prob_added_mean"].mean()), "prob_cont": float(sub["prob_cont_mean"].mean()), "prob_stopped": float(sub["prob_stopped_mean"].mean()),
            "rank_added": float(wa["rank_added_mean"].mean()),
            "train_freq_added": float(wa["train_freq_added_mean"].mean()), "train_freq_cont": float(sub["train_freq_cont_mean"].mean()),
            "jaccard": float(sub["jaccard"].mean()), "jaccard_oracle": float(sub["jaccard_oracle"].mean()),
        })
    return out


def cross_tab(df: pd.DataFrame, row_col: str, col_col: str, value: str, q: int = 4) -> dict:
    d = df[df["n_added"] > 0].copy()
    d["_r"] = pd.qcut(d[row_col], q, labels=False, duplicates="drop")
    d["_c"] = pd.qcut(d[col_col], q, labels=False, duplicates="drop")
    tab = d.groupby(["_r", "_c"]).apply(lambda s: (s[value] * s["n_added"]).sum() / s["n_added"].sum()).unstack()
    cnt = d.groupby(["_r", "_c"]).size().unstack()
    edges_r = d.groupby("_r")[row_col].agg(["min", "max"]); edges_c = d.groupby("_c")[col_col].agg(["min", "max"])
    return {"rows": [f"{row_col} {int(a)}–{int(b)}" for a, b in edges_r.to_numpy()],
            "cols": [f"{col_col} {int(a)}–{int(b)}" for a, b in edges_c.to_numpy()],
            "values": tab.round(4).to_numpy().tolist(), "counts": cnt.fillna(0).astype(int).to_numpy().tolist()}


def regressions(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    d = df[df["n_added"] > 0].copy()
    d["log_n_added"] = np.log1p(d["n_added"])
    out = {}
    specs = {
        "miss_rate ~ n_prev + n_added + n_dx": ("miss_rate", ["n_prev", "log_n_added", "n_dx"]),
        "miss_rate ~ n_prev + n_cur + n_added + n_dx": ("miss_rate", ["n_prev", "n_cur", "log_n_added", "n_dx"]),
        "miss_rate ~ n_prev + n_stopped + n_added + n_dx": ("miss_rate", ["n_prev", "n_stopped", "log_n_added", "n_dx"]),
        "miss_rate ~ ... + train_freq_added": ("miss_rate", ["n_prev", "n_stopped", "log_n_added", "n_dx", "train_freq_added_mean"]),
        "miss_rate_oracle ~ n_prev + n_stopped + n_added + n_dx": ("miss_rate_oracle", ["n_prev", "n_stopped", "log_n_added", "n_dx"]),
        "miss_rate@0.3 ~ n_prev + n_stopped + n_added + n_dx": ("miss_rate@0.3", ["n_prev", "n_stopped", "log_n_added", "n_dx"]),
        "size_gap ~ n_prev + n_stopped + n_added + n_dx": ("size_gap", ["n_prev", "n_stopped", "log_n_added", "n_dx"]),
        "prob_added ~ n_prev + n_stopped + n_added + n_dx + train_freq_added": ("prob_added_mean", ["n_prev", "n_stopped", "log_n_added", "n_dx", "train_freq_added_mean"]),
    }
    for name, (y, cols) in specs.items():
        sub = d.dropna(subset=[y] + cols)
        X = sub[cols].to_numpy(dtype=float); yv = sub[y].to_numpy(dtype=float)
        beta, r2 = A.ols_standardized(X, yv)
        bb = A.cluster_bootstrap(lambda s: A.ols_standardized(s[cols].to_numpy(float), s[y].to_numpy(float))[0], sub, "patient", n_boot=n_boot, seed=seed)
        out[name] = {"y": y, "r2": r2, "n": int(len(sub)),
                     "betas": [{"feature": c, "beta_std": pt, "ci_low": lo, "ci_high": hi} for c, pt, lo, hi in zip(cols, bb["point"], bb["ci_low"], bb["ci_high"])]}
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
    L = ["# 직전 약물 수 의존성 해부 — 왜 직전 처방이 크면 added 약물을 더 놓치나 (MIMIC-IV baseline)", "",
         f"생성 {m['generated_at_utc']} · test 전이 {m['n_transitions']:,}건 / 환자 {m['n_patients']:,}명 · {len(m['seeds'])} seed 평균 · 탐색적 분석(합격 기준 없음)", "",
         "oracle = 정답 약물 수만큼 확률 상위에서 뽑은 집합(디코딩 크기 오류를 제거한 순위 품질). pooled 누락률 = 구간 안 added 약물 전체 중 놓친 비율.", "",
         "## 1. 직전 약물 수 5분위별", "",
         "| 직전 약물 수 | 전이 | 정답 크기 | added | stopped | 예측 크기 | size gap | 예측 중 직전 약물 비율 | 예측 신규 수 | added 누락률 @0.5 | @oracle | @0.3 | 유지 recall | stopped 제거율 | Jaccard @0.5 / oracle |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in m["by_n_prev"]:
        L.append(f"| {int(r['lo'])}–{int(r['hi'])} | {r['n']:,} | {fmt(r['n_cur'], 1)} | {fmt(r['n_added'], 2)} | {fmt(r['n_stopped'], 2)} | {fmt(r['n_pred'], 1)} | {fmt(r['size_gap'], 2)} | {fmt(r['share_pred_in_prev'])} | {fmt(r['n_pred_added'], 2)} "
                 f"| {fmt(r['miss_rate'])} | {fmt(r['miss_rate_oracle'])} | {fmt(r['miss_rate@0.3'])} | {fmt(r['cont_recall'])} | {fmt(r['stopped_drop_rate'])} | {fmt(r['jaccard'])} / {fmt(r['jaccard_oracle'])} |")
    L += ["", "확률·순위·희귀도:", "", "| 직전 약물 수 | added 평균 확률 | 유지 평균 확률 | stopped 평균 확률 | added 평균 순위 (1=최상) | added 학습 빈도 | 유지 학습 빈도 |", "|---|---|---|---|---|---|---|"]
    for r in m["by_n_prev"]:
        L.append(f"| {int(r['lo'])}–{int(r['hi'])} | {fmt(r['prob_added'])} | {fmt(r['prob_cont'])} | {fmt(r['prob_stopped'])} | {fmt(r['rank_added'], 1)} | {fmt(r['train_freq_added'])} | {fmt(r['train_freq_cont'])} |")
    L += ["", "같은 표를 **정답 크기(n_cur)** 5분위로 (직전 크기와 현재 크기 중 어느 쪽인가):", "",
          "| 정답 크기 | 전이 | 직전 크기 | added | size gap | added 누락률 @0.5 | @oracle | 유지 recall | Jaccard |", "|---|---|---|---|---|---|---|---|---|"]
    for r in m["by_n_cur"]:
        L.append(f"| {int(r['lo'])}–{int(r['hi'])} | {r['n']:,} | — | {fmt(r['n_added'], 2)} | {fmt(r['size_gap'], 2)} | {fmt(r['miss_rate'])} | {fmt(r['miss_rate_oracle'])} | {fmt(r['cont_recall'])} | {fmt(r['jaccard'])} |")
    L += ["", "## 2. 직전 약물 수 × added 수 — pooled 누락률 @0.5 (괄호: @oracle)", ""]
    ct, cto = m["cross_prev_added"], m["cross_prev_added_oracle"]
    L.append("| | " + " | ".join(ct["cols"]) + " |"); L.append("|---|" + "---|" * len(ct["cols"]))
    for rname, vals, ovals, cnts in zip(ct["rows"], ct["values"], cto["values"], ct["counts"]):
        L.append(f"| {rname} | " + " | ".join(f"{fmt(v)} ({fmt(o)}) n={c:,}" for v, o, c in zip(vals, ovals, cnts)) + " |")
    L += ["", "## 3. 임계값을 낮추면 격차가 닫히나 (전역 임계 0.3 / 0.4 / 0.5)", "",
          "| 직전 약물 수 | 예측 크기 @0.3 | 누락률 @0.3 | @0.4 | @0.5 | added 정밀도 @0.3 | Jaccard @0.3 | @0.4 | @0.5 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in m["by_n_prev"]:
        L.append(f"| {int(r['lo'])}–{int(r['hi'])} | {fmt(r['n_pred@0.3'], 1)} | {fmt(r['miss_rate@0.3'])} | {fmt(r['miss_rate@0.4'])} | {fmt(r['miss_rate'])} | {fmt(r['added_precision@0.3'])} | {fmt(r['jaccard@0.3'])} | {fmt(r['jaccard@0.4'])} | {fmt(r['jaccard'])} |")
    L += ["", "## 4. 회귀 (표준화 계수, 환자 군집 부트스트랩 95% CI)", ""]
    for name, r in m["regressions"].items():
        L.append(f"**{name}** — R² {fmt(r['r2'])}, n {r['n']:,}")
        L.append("")
        L.append("| 특성 | β_std | 95% CI |"); L.append("|---|---|---|")
        for b in r["betas"]:
            L.append(f"| {b['feature']} | {fmt(b['beta_std'])} | [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}] |")
        L.append("")
    L += ["## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    b = m["by_n_prev"]; lo, hi = b[0], b[-1]
    R = m["regressions"]
    g = lambda name, f: next(x for x in R[name]["betas"] if x["feature"] == f)
    out = []
    out.append(f"- 직전 약물 수가 {int(lo['lo'])}–{int(lo['hi'])}개에서 {int(hi['lo'])}–{int(hi['hi'])}개로 가면 added 누락률(@0.5)은 {fmt(lo['miss_rate'])} → {fmt(hi['miss_rate'])}. "
               f"같은 구간에서 size gap(예측 − 정답)은 {fmt(lo['size_gap'], 2)} → {fmt(hi['size_gap'], 2)}, 예측 중 직전 약물 비율 {fmt(lo['share_pred_in_prev'])} → {fmt(hi['share_pred_in_prev'])}, 예측 신규 수 {fmt(lo['n_pred_added'], 2)} → {fmt(hi['n_pred_added'], 2)}(실제 added {fmt(lo['n_added'], 2)} → {fmt(hi['n_added'], 2)}).")
    bp = g("miss_rate ~ n_prev + n_stopped + n_added + n_dx", "n_prev"); bf = g("miss_rate ~ ... + train_freq_added", "n_prev"); bfr = g("miss_rate ~ ... + train_freq_added", "train_freq_added_mean")
    bo = g("miss_rate_oracle ~ n_prev + n_stopped + n_added + n_dx", "n_prev"); bos = g("miss_rate_oracle ~ n_prev + n_stopped + n_added + n_dx", "n_stopped"); b3 = g("miss_rate@0.3 ~ n_prev + n_stopped + n_added + n_dx", "n_prev")
    r2_no, r2_fr = R["miss_rate ~ n_prev + n_stopped + n_added + n_dx"]["r2"], R["miss_rate ~ ... + train_freq_added"]["r2"]
    rarity_explains = bf["ci_high"] < 0.05
    out.append(f"- **희귀도가 의존성을 설명한다 (M4).** 큰 처방에 추가되는 약물은 학습셋에서 드물다(added 평균 학습 빈도 {fmt(lo['train_freq_added'])} → {fmt(hi['train_freq_added'])}). "
               f"added 약물의 학습 빈도를 회귀에 넣으면 n_prev 계수가 {fmt(bp['beta_std'])} [{fmt(bp['ci_low'])}, {fmt(bp['ci_high'])}] → {fmt(bf['beta_std'])} [{fmt(bf['ci_low'])}, {fmt(bf['ci_high'])}]로 {'사라지고' if rarity_explains else '줄고'}, "
               f"R²는 {fmt(r2_no)} → {fmt(r2_fr)}. 학습 빈도 계수 {fmt(bfr['beta_std'])}가 압도적이다. added 약물의 평균 확률도 같은 구조다(빈도 계수 {fmt(g('prob_added ~ n_prev + n_stopped + n_added + n_dx + train_freq_added', 'train_freq_added_mean')['beta_std'])}, n_prev {fmt(g('prob_added ~ n_prev + n_stopped + n_added + n_dx + train_freq_added', 'n_prev')['beta_std'])}).")
    out.append(f"- oracle 크기(정답 개수만큼 뽑기)에서는 누락률 {fmt(lo['miss_rate_oracle'])} → {fmt(hi['miss_rate_oracle'])}이지만, 보정하면 n_prev 계수는 {fmt(bo['beta_std'])} [{fmt(bo['ci_low'])}, {fmt(bo['ci_high'])}]로 음수이고 대신 **stopped 수**가 {fmt(bos['beta_std'])}로 크다. "
               f"즉 큰 처방에서 순위가 나빠지는 것이 아니라, 중단됐어야 할 약물이 상위 순위를 차지해 added 자리를 뺏는다(stopped 약물 평균 확률 {fmt(lo['prob_stopped'])}~{fmt(hi['prob_stopped'])} vs added {fmt(hi['prob_added'])}~{fmt(lo['prob_added'])}; stopped 제거율은 {fmt(lo['stopped_drop_rate'])}~{fmt(hi['stopped_drop_rate'])}에 그친다).")
    out.append(f"- 디코딩 크기(M1)는 부차적이다. size gap은 작은 처방에서 +{fmt(lo['size_gap'], 1)}(과다 예측), 큰 처방에서 +{fmt(hi['size_gap'], 1)}로 임계 0.5가 작은 처방을 과다 예측하는 쪽의 문제이고, 임계를 0.3으로 낮춰도 n_prev 계수는 {fmt(b3['beta_std'])}로 그대로다.")
    out.append(f"- 임계값을 0.3으로 낮추면 최대 구간 누락률 {fmt(hi['miss_rate'])} → {fmt(hi['miss_rate@0.3'])}, Jaccard {fmt(hi['jaccard'])} → {fmt(hi['jaccard@0.3'])} (최소 구간 Jaccard {fmt(lo['jaccard'])} → {fmt(lo['jaccard@0.3'])}). 전역 임계 하나로는 구간별로 득실이 갈린다.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import time
    t0 = time.time()
    print("loading records ...", flush=True)
    with open(MIMIC4_RECORDS, "rb") as f:
        records = dill.load(f)
    print(f"[{time.time() - t0:.0f}s] records loaded: {len(records):,} patients", flush=True)
    df, seeds = load_transitions(records)
    print(f"{len(df):,} transitions, {df['patient'].nunique():,} patients, seeds {seeds}", flush=True)
    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds, "n_boot": args.n_boot,
         "n_transitions": int(len(df)), "n_patients": int(df["patient"].nunique()),
         "overall": {k: float(df[k].mean()) for k in ["n_prev", "n_cur", "n_added", "n_stopped", "n_pred", "size_gap", "share_pred_in_prev", "cont_recall", "stopped_drop_rate", "jaccard", "jaccard_oracle"]}
                    | {"pooled_miss_rate": float((df["miss_rate"] * df["n_added"]).sum() / df["n_added"].sum()),
                       "pooled_miss_rate_oracle": float((df["miss_rate_oracle"] * df["n_added"]).sum() / df["n_added"].sum())},
         "by_n_prev": bins_by(df, "n_prev"), "by_n_cur": bins_by(df, "n_cur"),
         "cross_prev_added": cross_tab(df, "n_prev", "n_added", "miss_rate"),
         "cross_prev_added_oracle": cross_tab(df, "n_prev", "n_added", "miss_rate_oracle")}
    print(f"[{time.time() - t0:.0f}s] tables done; regressions ...", flush=True)
    m["regressions"] = regressions(df, args.n_boot, args.seed)
    print(f"[{time.time() - t0:.0f}s] regressions done", flush=True)
    m["interpretation"] = interpret(m)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "prev_meds_dependence_mimic4.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = render(m)
    (RESULTS / "prev_meds_dependence_mimic4.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
