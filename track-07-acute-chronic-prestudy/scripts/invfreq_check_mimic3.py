"""Did track-04's inverse-frequency BCE weighting (Intervention A) fix rare-drug recall?

MIMIC-III, SafeDrug cohort, seed0 only (the only seed run for arm A):
  baseline  : SafeDrug_군집별평가_20260904/개입실험/B_thresholds/seed0/global_0.5/  (threshold 0.5)
  inv-freq  : SafeDrug_군집별평가_20260904/개입실험/A_inverse_freq/seed0/         (per-drug BCE weight = 1/freq, cap 10)

Both dumps hold y_prob / y_pred for the same test visits (2,264). Rarity = drug
training frequency (share of train visits containing the code, SafeDrug
records_final.pkl train split), split into code-level tertiles.

Measured on the test split:
  - per-drug recall / precision pooled by rarity tertile, both arms;
  - added-drug miss rate by rarity of the added drug and by previous-regimen
    size, chronological pairing (visits re-ordered by ADMITTIME, as in the
    chrono analysis; the model itself saw HADM_ID order);
  - visit-level Jaccard / F1 paired difference with patient bootstrap.
Single seed -> descriptive only; seed noise on Jaccard is ~0.003.
Outputs: results/invfreq_check_mimic3.{md,json}
"""
from __future__ import annotations

import ast
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
from change_visit_split import MIMIC3_RECORDS, chronological_visit_index, split_patients  # noqa: E402
from cvsplit_lib import multihot_rows_to_sets  # noqa: E402
from stop_decision_mimic4 import decompose, jac  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
BASE_DIR = Path(r"C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds\seed0\global_0.5")
INV_DIR = Path(r"C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\A_inverse_freq\seed0")
VOC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\voc_final.pkl")


def train_freq_mimic3(records: list, n_med: int) -> np.ndarray:
    train_idx, _, _ = split_patients(records)
    cnt = np.zeros(n_med); n = 0
    for pi in train_idx:
        for adm in records[pi]:
            cnt[list(adm[2])] += 1; n += 1
    return cnt / n


def load_dump(d: Path) -> dict:
    with np.load(d / "per_visit_predictions.npz") as npz:
        z = {k: npz[k] for k in npz.files}
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    off = m["official_metrics"]; off = ast.literal_eval(off) if isinstance(off, str) else off
    z["official_test"] = off["test"]; z["best_epoch"] = m["best_epoch"]; z["drug_weight"] = m.get("drug_weight")
    return z


def per_drug(z: dict, keep: np.ndarray, n_med: int) -> pd.DataFrame:
    gt, pr = z["y_gt"][keep].astype(bool), z["y_pred"][keep].astype(bool)
    tp = (gt & pr).sum(0); fn = (gt & ~pr).sum(0); fp = (~gt & pr).sum(0)
    return pd.DataFrame({"med": np.arange(n_med), "n_true": gt.sum(0), "tp": tp, "fn": fn, "fp": fp})


def rarity_of(train_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(train_freq[train_freq > 0], [1 / 3, 2 / 3])
    return np.digitize(train_freq, edges), edges


def pooled_by_tertile(pdrug: pd.DataFrame, tert: np.ndarray) -> list[dict]:
    out = []
    for t in range(3):
        s = pdrug[tert[pdrug["med"].to_numpy()] == t]
        tp, fn, fp = s["tp"].sum(), s["fn"].sum(), s["fp"].sum()
        out.append({"tertile": ["rare", "mid", "common"][t], "n_codes": int(len(s)), "n_true": int(s["n_true"].sum()),
                    "recall": float(tp / (tp + fn)) if tp + fn else float("nan"), "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
                    "n_pred": int(tp + fp)})
    return out


def transitions(z: dict, order_visit: np.ndarray, train_freq: np.ndarray, tert: np.ndarray) -> tuple[pd.DataFrame, Counter, Counter]:
    gt, pr = multihot_rows_to_sets(z["y_gt"]), multihot_rows_to_sets(z["y_pred"])
    order = np.lexsort((order_visit, z["patient_index"]))
    by_p: dict = defaultdict(list)
    for i in order:
        by_p[int(z["patient_index"][i])].append(i)
    rows, added_by_t, missed_by_t = [], Counter(), Counter()
    for p, idxs in by_p.items():
        for a, b in zip(idxs[:-1], idxs[1:]):
            if z["split"][b] != "test":
                continue
            d = decompose(gt[a], gt[b], pr[b]); d.update({"patient": p, "hadm": int(z["HADM_ID"][b])})
            added = gt[b] - gt[a]
            d["train_freq_added"] = float(np.mean([train_freq[m] for m in added])) if added else float("nan")
            for m in added:
                added_by_t[int(tert[m])] += 1
                if m not in pr[b]:
                    missed_by_t[int(tert[m])] += 1
            rows.append(d)
    return pd.DataFrame(rows), added_by_t, missed_by_t


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    with open(MIMIC3_RECORDS, "rb") as f:
        records = dill.load(f)
    voc = dill.load(open(VOC, "rb")); medw = voc["med_voc"].idx2word; n_med = len(medw)
    train_freq = train_freq_mimic3(records, n_med)
    tert, edges = rarity_of(train_freq)
    base, inv = load_dump(BASE_DIR), load_dump(INV_DIR)
    if not np.array_equal(base["HADM_ID"], inv["HADM_ID"]) or not np.array_equal(base["y_gt"], inv["y_gt"]):
        raise AssertionError("baseline and inv-freq dumps are not row-aligned")
    keep = base["split"] == "test"
    chrono = chronological_visit_index(base["patient_index"], base["visit_index"])

    arms = {}
    for name, z in [("baseline", base), ("inv_freq", inv)]:
        pdrug = per_drug(z, keep, n_med)
        tr, added_by_t, missed_by_t = transitions(z, chrono, train_freq, tert)
        gt, pr = z["y_gt"][keep].astype(bool), z["y_pred"][keep].astype(bool)
        inter, union = (gt & pr).sum(1), (gt | pr).sum(1)
        arms[name] = {
            "official_test": z["official_test"], "best_epoch": z["best_epoch"],
            "visit_jaccard_mean": float(np.mean(inter / np.maximum(union, 1))), "visit_n_pred_mean": float(pr.sum(1).mean()),
            "by_rarity": pooled_by_tertile(pdrug, tert),
            "added_miss_by_rarity": [{"tertile": ["rare", "mid", "common"][t], "n_added": added_by_t[t], "miss_rate": (missed_by_t[t] / added_by_t[t]) if added_by_t[t] else float("nan")} for t in range(3)],
            "added_miss_pooled": float(tr["fn_added"].sum() / tr["n_added"].sum()),
            "stale_keep_rate": float(tr["stale_keep_rate"].mean()),
            "n_transitions": int(len(tr)),
            "_tr": tr, "_pdrug": pdrug,
        }
        d = tr.copy(); d["_b"] = pd.qcut(d["n_prev"], 4, labels=False, duplicates="drop")
        arms[name]["added_miss_by_n_prev"] = [{"lo": int(s["n_prev"].min()), "hi": int(s["n_prev"].max()), "n": int(len(s)), "miss_rate": float(s["fn_added"].sum() / s["n_added"].sum()), "jaccard": float(s["jaccard"].mean())} for _, s in d.groupby("_b")]

    # paired visit-level differences (same visits, same order)
    gt_b, pr_b, pr_i = base["y_gt"][keep].astype(bool), base["y_pred"][keep].astype(bool), inv["y_pred"][keep].astype(bool)
    jb = (gt_b & pr_b).sum(1) / np.maximum((gt_b | pr_b).sum(1), 1); ji = (gt_b & pr_i).sum(1) / np.maximum((gt_b | pr_i).sum(1), 1)
    dfv = pd.DataFrame({"patient": base["SUBJECT_ID"][keep], "d": ji - jb})
    boot = A.cluster_bootstrap(lambda s: s["d"].mean(), dfv, "patient", n_boot=500, seed=0)
    paired = {"d_jaccard_inv_minus_base": float(dfv["d"].mean()), "ci_low": boot["ci_low"][0], "ci_high": boot["ci_high"][0], "n_visits": int(len(dfv))}
    trb, tri = arms["baseline"]["_tr"], arms["inv_freq"]["_tr"]
    mt = trb.merge(tri, on=["patient", "hadm"], suffixes=("_b", "_i"))
    dfa = pd.DataFrame({"patient": mt["patient"], "d": (mt["fn_added_i"] - mt["fn_added_b"])})
    boot_a = A.cluster_bootstrap(lambda s: s["d"].mean(), dfa, "patient", n_boot=500, seed=0)
    paired["d_missed_added_per_transition"] = {"mean": float(dfa["d"].mean()), "ci_low": boot_a["ci_low"][0], "ci_high": boot_a["ci_high"][0]}
    for a in arms.values():
        a.pop("_tr"); a.pop("_pdrug")

    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": "seed0 only", "rarity_edges": edges.tolist(),
         "n_codes_by_tertile": [int((tert == t).sum()) for t in range(3)], "weight_cap": (inv["drug_weight"] or {}).get("cap"),
         "arms": arms, "paired": paired}
    b, i = arms["baseline"], arms["inv_freq"]
    L = ["# inverse-frequency 가중(트랙 04 개입 A) 점검 — MIMIC-III seed0", "",
         f"생성 {m['generated_at_utc']} · test 방문 {paired['n_visits']:,} · 전이 {b['n_transitions']:,} (시간순 짝짓기) · 단일 seed(Jaccard seed 편차 ≈ 0.003)", "",
         "## 1. 공식 지표 (test)", "", "| 지표 | baseline | inv-freq (cap 10) | Δ |", "|---|---|---|---|"]
    for k in ["ja", "prauc", "avg_p", "avg_r", "avg_f1", "ddi_rate", "avg_med"]:
        L.append(f"| {k} | {fmt(b['official_test'][k], 4)} | {fmt(i['official_test'][k], 4)} | {fmt(i['official_test'][k] - b['official_test'][k], 4)} |")
    L += [f"| best_epoch | {b['best_epoch']} | {i['best_epoch']} | |",
          f"| 방문별 Jaccard 짝 차이 (inv − base) | | | {fmt(paired['d_jaccard_inv_minus_base'], 4)} [{fmt(paired['ci_low'], 4)}, {fmt(paired['ci_high'], 4)}] |", "",
          f"## 2. 약물 희귀도 3분위별 pooled recall / precision (코드 단위 3분위, 학습 빈도 경계 {', '.join(fmt(e) for e in edges)}; 코드 수 {m['n_codes_by_tertile']})", "",
          "| 희귀도 | 정답 건수 | recall base | recall inv | Δ | precision base | precision inv | 예측 건수 base → inv |", "|---|---|---|---|---|---|---|---|"]
    for rb, ri in zip(b["by_rarity"], i["by_rarity"]):
        L.append(f"| {rb['tertile']} | {rb['n_true']:,} | {fmt(rb['recall'])} | {fmt(ri['recall'])} | {fmt(ri['recall'] - rb['recall'])} | {fmt(rb['precision'])} | {fmt(ri['precision'])} | {rb['n_pred']:,} → {ri['n_pred']:,} |")
    L += ["", "## 3. added 약물 누락률 (시간순 짝짓기)", "",
          f"전체 pooled: base {fmt(b['added_miss_pooled'])} → inv {fmt(i['added_miss_pooled'])}; 전이당 누락 개수 짝 차이 {fmt(paired['d_missed_added_per_transition']['mean'], 3)} [{fmt(paired['d_missed_added_per_transition']['ci_low'], 3)}, {fmt(paired['d_missed_added_per_transition']['ci_high'], 3)}]; 중단 약물 유지율 base {fmt(b['stale_keep_rate'])} → inv {fmt(i['stale_keep_rate'])}", "",
          "| added 약물 희귀도 | added 건수 | 누락률 base | inv | Δ |", "|---|---|---|---|---|"]
    for rb, ri in zip(b["added_miss_by_rarity"], i["added_miss_by_rarity"]):
        L.append(f"| {rb['tertile']} | {rb['n_added']:,} | {fmt(rb['miss_rate'])} | {fmt(ri['miss_rate'])} | {fmt(ri['miss_rate'] - rb['miss_rate'])} |")
    L += ["", "| 직전 약물 수 | 전이 | 누락률 base | inv | Jaccard base | inv |", "|---|---|---|---|---|---|"]
    for rb, ri in zip(b["added_miss_by_n_prev"], i["added_miss_by_n_prev"]):
        L.append(f"| {rb['lo']}–{rb['hi']} | {rb['n']:,} | {fmt(rb['miss_rate'])} | {fmt(ri['miss_rate'])} | {fmt(rb['jaccard'])} | {fmt(ri['jaccard'])} |")
    rr = b["by_rarity"][0], i["by_rarity"][0]
    L += ["", "## 해석", "",
          f"- 드문 약물(하위 1/3 코드) recall {fmt(rr[0]['recall'])} → {fmt(rr[1]['recall'])}, precision {fmt(rr[0]['precision'])} → {fmt(rr[1]['precision'])}; 흔한 약물 recall {fmt(b['by_rarity'][2]['recall'])} → {fmt(i['by_rarity'][2]['recall'])}.",
          f"- added 누락률 pooled {fmt(b['added_miss_pooled'])} → {fmt(i['added_miss_pooled'])} (드문 added {fmt(b['added_miss_by_rarity'][0]['miss_rate'])} → {fmt(i['added_miss_by_rarity'][0]['miss_rate'])}). 방문별 Jaccard 짝 차이 {fmt(paired['d_jaccard_inv_minus_base'], 4)} [{fmt(paired['ci_low'], 4)}, {fmt(paired['ci_high'], 4)}], 약물 수 {fmt(b['official_test']['avg_med'], 1)} → {fmt(i['official_test']['avg_med'], 1)}.",
          "- 단일 seed·MIMIC-III·cap 10 한 설정의 결과다. 방향만 읽고 크기는 seed 편차(≈0.003)와 견줘 읽는다.", ""]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "invfreq_check_mimic3.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (RESULTS / "invfreq_check_mimic3.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
