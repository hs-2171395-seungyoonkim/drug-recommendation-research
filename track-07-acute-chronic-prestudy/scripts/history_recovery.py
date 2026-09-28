"""How much of SafeDrug's error headroom does medication-history input (GAMENet) recover?

Same records, same split, same seeds; SafeDrug (no medication history) vs
GAMENet (dynamic memory over previous prescriptions). Per test transition
(visit i >= 1), seed-averaged, the four error bins of stop_decision_mimic4:
stale FP, novel FP, added FN, continued FN; plus rarity-stratified keep/miss
rates, stop-decision AUC and paired visit-level dJaccard with patient bootstrap.

    py -3.12 scripts/history_recovery.py --dataset mimic3chrono
    py -3.12 scripts/history_recovery.py --dataset mimic4
Outputs: results/history_recovery_<dataset>.{md,json}
"""
from __future__ import annotations

import argparse
import ast
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
from stop_decision_mimic4 import auc, decompose  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
CFG = {
    "mimic3chrono": {"records": ROOT / "out" / "mimic3_chrono_data" / "records_final.pkl",
                     "safedrug": sorted((ROOT / "out" / "mimic3_chrono_safedrug").glob("seed[0-9]/per_visit_predictions.npz")),
                     "gamenet_dir": ROOT / "out" / "mimic3_chrono_gamenet", "gamenet_glob": "seed[0-9]/per_visit_predictions.npz",
                     "label_sd": "SafeDrug (시간순 records 재학습, 4 seed)", "label_gn": "GAMENet (같은 records)"},
    "mimic4": {"records": MIMIC4_RECORDS, "safedrug": sorted(MIMIC4_DUMP.glob("baseline-seed-*.npz")),
               "gamenet_dir": ROOT / "out" / "mimic4_gamenet", "gamenet_glob": "seed-*/per_visit_predictions.npz",
               "label_sd": "SafeDrug baseline (5 seed)", "label_gn": "GAMENet (final5, train-split ehr_adj)"},
}
ERR = ["fp_stale", "fp_novel", "fn_added", "fn_cont"]
FIX = ["j_fix_stale", "j_fix_novel", "j_fix_fn_added", "j_fix_fn_cont"]


def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def oracle_sets(prob: np.ndarray, gt_sets: list[set]) -> list[set]:
    """Top-|true set| of each row by probability: removes set-size effects, keeps ranking."""
    out = []
    for i, g in enumerate(gt_sets):
        k = len(g)
        out.append(set(int(j) for j in np.argsort(-prob[i], kind="stable")[:k]) if k else set())
    return out


def arm_frames(paths: list[Path], train_freq: np.ndarray, tert: np.ndarray, arm: str, oracle: bool = False):
    """Returns (transition frame seed-averaged, drug-level frame pooled over seeds, official metrics per seed).
    oracle=True re-decodes every visit as top-|true set| by probability (both arms get the same size)."""
    trans, drugs, official = [], [], {}
    for path in paths:
        z = load_npz(path)
        gt = multihot_rows_to_sets(z["y_gt"])
        pr = oracle_sets(z["y_prob"], gt) if oracle else multihot_rows_to_sets(z["y_pred"])
        order = np.lexsort((z["visit_index"], z["patient_index"]))
        by_p: dict = defaultdict(list)
        for i in order:
            by_p[int(z["patient_index"][i])].append(i)
        for p, idxs in by_p.items():
            for a, b in zip(idxs[:-1], idxs[1:]):
                if z["split"][b] != "test":
                    continue
                prev, cur, pred, prob = gt[a], gt[b], pr[b], z["y_prob"][b]
                d = decompose(prev, cur, pred); d.update({"patient": p, "visit_pos": int(z["visit_index"][b]), "seed": path.parent.name if path.name == "per_visit_predictions.npz" else path.stem})
                trans.append(d)
                for m in prev:
                    drugs.append(("prev", int(tert[m]), 1 if m in cur else 0, 1 if m in pred else 0, float(prob[m]), len(prev)))
                for m in cur - prev:
                    drugs.append(("added", int(tert[m]), 1, 1 if m in pred else 0, float(prob[m]), len(prev)))
        man = path.parent / "run_manifest.json"
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8")); off = m.get("official_metrics")
            off = ast.literal_eval(off) if isinstance(off, str) else off
            official[path.parent.name] = {"best_epoch": m.get("best_epoch"), **{k: off["test"][k] for k in ("ja", "ddi_rate", "avg_med")}} if off else None
        else:
            meta = path.with_suffix(".meta.json")
            official[path.stem] = {"best_epoch": json.loads(meta.read_text())["checkpoint_selection"]["epoch"]} if meta.exists() else None
    t = pd.DataFrame(trans)
    num = [c for c in t.columns if c not in ("patient", "visit_pos", "seed")]
    t_avg = t.groupby(["patient", "visit_pos"], as_index=False)[num].mean()
    t_avg["n_seeds"] = t.groupby(["patient", "visit_pos"]).size().to_numpy()
    d = pd.DataFrame(drugs, columns=["role", "tert", "label", "pred", "prob", "n_prev"])
    d["arm"] = arm
    return t_avg, d, official


def arm_summary(t: pd.DataFrame, d: pd.DataFrame) -> dict:
    prev = d[d["role"] == "prev"]; added = d[d["role"] == "added"]
    out = {"n_transitions": int(len(t)), "n_seeds": int(t["n_seeds"].iloc[0]), "jaccard": float(t["jaccard"].mean()), "n_pred": float(t["n_pred"].mean()), "n_cur": float(t["n_cur"].mean()),
           **{k: float(t[k].mean()) for k in ERR}, **{k: float(t[k].mean()) for k in FIX},
           "stale_keep_rate": float(t["stale_keep_rate"].mean()), "cont_drop_rate": float(t["cont_drop_rate"].mean()),
           "added_miss_pooled": float(t["fn_added"].sum() / t["n_added"].sum()),
           "stop_auc": auc(prev["prob"].to_numpy(), prev["label"].to_numpy()),
           "keep_rate_stopped": float(prev.loc[prev["label"] == 0, "pred"].mean()), "keep_rate_continued": float(prev.loc[prev["label"] == 1, "pred"].mean())}
    for name, sub in [("cont_keep", prev[prev["label"] == 1]), ("stopped_keep", prev[prev["label"] == 0]), ("added_hit", added)]:
        out[f"{name}_by_rarity"] = {["rare", "mid", "common"][k]: float(s["pred"].mean()) for k, s in sub.groupby("tert")}
        out[f"{name}_n_by_rarity"] = {["rare", "mid", "common"][k]: int(len(s)) for k, s in sub.groupby("tert")}
    return out


def by_n_prev(t_sd: pd.DataFrame, t_gn: pd.DataFrame) -> list[dict]:
    m = t_sd.merge(t_gn, on=["patient", "visit_pos"], suffixes=("_sd", "_gn"))
    m["_b"] = pd.qcut(m["n_prev_sd"], 5, labels=False, duplicates="drop")
    out = []
    for b, s in m.groupby("_b"):
        out.append({"lo": int(s["n_prev_sd"].min()), "hi": int(s["n_prev_sd"].max()), "n": int(len(s)),
                    "jaccard_sd": float(s["jaccard_sd"].mean()), "jaccard_gn": float(s["jaccard_gn"].mean()),
                    "added_miss_sd": float(s["fn_added_sd"].sum() / s["n_added_sd"].sum()), "added_miss_gn": float(s["fn_added_gn"].sum() / s["n_added_gn"].sum()),
                    "stale_sd": float(s["fp_stale_sd"].mean()), "stale_gn": float(s["fp_stale_gn"].mean()),
                    "novel_sd": float(s["fp_novel_sd"].mean()), "novel_gn": float(s["fp_novel_gn"].mean()),
                    "fn_cont_sd": float(s["fn_cont_sd"].mean()), "fn_cont_gn": float(s["fn_cont_gn"].mean())})
    return out


def paired(t_sd: pd.DataFrame, t_gn: pd.DataFrame, n_boot: int, seed: int) -> dict:
    m = t_sd.merge(t_gn, on=["patient", "visit_pos"], suffixes=("_sd", "_gn"))
    if len(m) != len(t_sd) or len(m) != len(t_gn):
        raise AssertionError(f"transition sets differ: sd {len(t_sd)}, gn {len(t_gn)}, joined {len(m)}")
    if not np.allclose(m["n_cur_sd"], m["n_cur_gn"]) or not np.allclose(m["n_added_sd"], m["n_added_gn"]):
        raise AssertionError("ground truth differs between arms")
    cols = ["jaccard"] + ERR
    def _stat(s):
        return np.array([(s[f"{c}_gn"] - s[f"{c}_sd"]).mean() for c in cols])
    bb = A.cluster_bootstrap(_stat, m, "patient", n_boot=n_boot, seed=seed)
    return {c: {"delta": pt, "ci_low": lo, "ci_high": hi} for c, pt, lo, hi in zip(cols, bb["point"], bb["ci_low"], bb["ci_high"])} | {"n": int(len(m))}


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render(m: dict) -> str:
    sd, gn, p = m["safedrug"], m["gamenet"], m["paired"]
    L = [f"# 처방 이력 입력이 회수하는 오류 — {m['dataset']} (SafeDrug vs GAMENet)", "",
         f"생성 {m['generated_at_utc']} · test 전이 {p['n']:,}건 · SafeDrug {sd['n_seeds']} seed / GAMENet {gn['n_seeds']} seed · 같은 records·분할, 방문별 시드 평균", "",
         f"공식 test 지표(run_manifest): SafeDrug {m['official_sd']} · GAMENet {m['official_gn']}", "",
         "## 1. Arm 수준 (test 전이)", "",
         f"| 항목 | {m['label_sd']} | {m['label_gn']} | Δ(GN − SD) [95% CI] |", "|---|---|---|---|",
         f"| Jaccard | {fmt(sd['jaccard'], 4)} | {fmt(gn['jaccard'], 4)} | {fmt(p['jaccard']['delta'], 4)} [{fmt(p['jaccard']['ci_low'], 4)}, {fmt(p['jaccard']['ci_high'], 4)}] |",
         f"| 예측 크기 / 정답 크기 | {fmt(sd['n_pred'], 2)} / {fmt(sd['n_cur'], 2)} | {fmt(gn['n_pred'], 2)} / {fmt(gn['n_cur'], 2)} | |"]
    names = {"fp_stale": "stale FP (중단됐는데 예측)", "fp_novel": "novel FP (근거 없는 예측)", "fn_added": "added 누락", "fn_cont": "유지 약물 누락"}
    for k in ERR:
        L.append(f"| {names[k]} (전이당) | {fmt(sd[k], 2)} | {fmt(gn[k], 2)} | {fmt(p[k]['delta'], 3)} [{fmt(p[k]['ci_low'], 3)}, {fmt(p[k]['ci_high'], 3)}] |")
    L += [f"| 중단 약물 유지율 / 유지 약물 누락률 | {fmt(sd['stale_keep_rate'])} / {fmt(sd['cont_drop_rate'])} | {fmt(gn['stale_keep_rate'])} / {fmt(gn['cont_drop_rate'])} | |",
          f"| added 누락률 (pooled) | {fmt(sd['added_miss_pooled'])} | {fmt(gn['added_miss_pooled'])} | |",
          f"| 중단 판단 AUC (직전 약물, 유지 vs 중단) | {fmt(sd['stop_auc'])} | {fmt(gn['stop_auc'])} | |", "",
          "## 2. 남은 여유 (각 오류 하나만 고쳤을 때 Jaccard)", "",
          "| 고치는 오류 | SafeDrug | GAMENet |", "|---|---|---|"]
    for k, f in zip(ERR, FIX):
        L.append(f"| {names[k]} | +{fmt(sd[f] - sd['jaccard'])} | +{fmt(gn[f] - gn['jaccard'])} |")
    L += ["", "## 3. 희귀도별 (코드 단위 3분위)", "",
          "| 약물 집합 | 희귀도 | n | SafeDrug 예측률 | GAMENet 예측률 |", "|---|---|---|---|---|"]
    for key, label in [("cont_keep", "유지된 직전 약물 (맞게 유지)"), ("stopped_keep", "중단된 직전 약물 (잘못 유지)"), ("added_hit", "added 약물 (맞게 추가)")]:
        for r in ["rare", "mid", "common"]:
            L.append(f"| {label} | {r} | {sd[f'{key}_n_by_rarity'].get(r, 0):,} | {fmt(sd[f'{key}_by_rarity'].get(r))} | {fmt(gn[f'{key}_by_rarity'].get(r))} |")
    L += ["", "## 4. 직전 약물 수 5분위별", "",
          "| 직전 약물 수 | 전이 | Jaccard SD → GN | added 누락 SD → GN | stale FP SD → GN | novel FP SD → GN | 유지 누락 SD → GN |", "|---|---|---|---|---|---|---|"]
    for r in m["by_n_prev"]:
        L.append(f"| {r['lo']}–{r['hi']} | {r['n']:,} | {fmt(r['jaccard_sd'])} → {fmt(r['jaccard_gn'])} | {fmt(r['added_miss_sd'])} → {fmt(r['added_miss_gn'])} | {fmt(r['stale_sd'], 2)} → {fmt(r['stale_gn'], 2)} | {fmt(r['novel_sd'], 2)} → {fmt(r['novel_gn'], 2)} | {fmt(r['fn_cont_sd'], 2)} → {fmt(r['fn_cont_gn'], 2)} |")
    if "oracle" in m:
        o = m["oracle"]; osd, ogn, op = o["safedrug"], o["gamenet"], o["paired"]
        L += ["", "## 5. 크기를 맞춘 비교 — 두 arm 모두 정답 개수만큼 확률 상위에서 뽑음(oracle size; 순위 품질만 비교)", "",
              "| 항목 | SafeDrug | GAMENet | Δ(GN − SD) [95% CI] |", "|---|---|---|---|",
              f"| Jaccard | {fmt(osd['jaccard'], 4)} | {fmt(ogn['jaccard'], 4)} | {fmt(op['jaccard']['delta'], 4)} [{fmt(op['jaccard']['ci_low'], 4)}, {fmt(op['jaccard']['ci_high'], 4)}] |"]
        for k in ERR:
            L.append(f"| {names[k]} (전이당) | {fmt(osd[k], 2)} | {fmt(ogn[k], 2)} | {fmt(op[k]['delta'], 3)} [{fmt(op[k]['ci_low'], 3)}, {fmt(op[k]['ci_high'], 3)}] |")
        L += [f"| 중단 약물 유지율 / 유지 약물 누락률 | {fmt(osd['stale_keep_rate'])} / {fmt(osd['cont_drop_rate'])} | {fmt(ogn['stale_keep_rate'])} / {fmt(ogn['cont_drop_rate'])} | |",
              f"| added 누락률 (pooled) | {fmt(osd['added_miss_pooled'])} | {fmt(ogn['added_miss_pooled'])} | |", "",
              "| 약물 집합 (oracle size) | 희귀도 | SafeDrug 예측률 | GAMENet 예측률 |", "|---|---|---|---|"]
        for key, label in [("cont_keep", "유지된 직전 약물"), ("stopped_keep", "중단된 직전 약물 (잘못 유지)"), ("added_hit", "added 약물")]:
            for r in ["rare", "mid", "common"]:
                L.append(f"| {label} | {r} | {fmt(osd[f'{key}_by_rarity'].get(r))} | {fmt(ogn[f'{key}_by_rarity'].get(r))} |")
        L += ["", "| 직전 약물 수 | Jaccard SD → GN (oracle) | added 누락 SD → GN | stale FP SD → GN | 유지 누락 SD → GN |", "|---|---|---|---|---|"]
        for r in o["by_n_prev"]:
            L.append(f"| {r['lo']}–{r['hi']} | {fmt(r['jaccard_sd'])} → {fmt(r['jaccard_gn'])} | {fmt(r['added_miss_sd'])} → {fmt(r['added_miss_gn'])} | {fmt(r['stale_sd'], 2)} → {fmt(r['stale_gn'], 2)} | {fmt(r['fn_cont_sd'], 2)} → {fmt(r['fn_cont_gn'], 2)} |")
    L += ["", "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    sd, gn, p = m["safedrug"], m["gamenet"], m["paired"]
    names = {"fp_stale": "stale FP", "fp_novel": "novel FP", "fn_added": "added 누락", "fn_cont": "유지 누락"}
    out = [f"- Jaccard {fmt(sd['jaccard'], 4)} → {fmt(gn['jaccard'], 4)} (Δ {fmt(p['jaccard']['delta'], 4)} [{fmt(p['jaccard']['ci_low'], 4)}, {fmt(p['jaccard']['ci_high'], 4)}])."]
    out.append("- 오류 갈래별 변화(전이당): " + ", ".join(f"{names[k]} {fmt(sd[k], 2)} → {fmt(gn[k], 2)}" for k in ERR) + ".")
    ck_sd, ck_gn = sd["cont_keep_by_rarity"], gn["cont_keep_by_rarity"]
    out.append(f"- 유지된 드문 약물 예측률 {fmt(ck_sd.get('rare'))} → {fmt(ck_gn.get('rare'))} (mid {fmt(ck_sd.get('mid'))} → {fmt(ck_gn.get('mid'))}); 중단 판단 AUC {fmt(sd['stop_auc'])} → {fmt(gn['stop_auc'])}; added 누락 {fmt(sd['added_miss_pooled'])} → {fmt(gn['added_miss_pooled'])}.")
    if "oracle" in m:
        o = m["oracle"]; osd, ogn, op = o["safedrug"], o["gamenet"], o["paired"]
        out.append(f"- **크기를 맞추면(oracle size)** Jaccard {fmt(osd['jaccard'], 4)} → {fmt(ogn['jaccard'], 4)} (Δ {fmt(op['jaccard']['delta'], 4)} [{fmt(op['jaccard']['ci_low'], 4)}, {fmt(op['jaccard']['ci_high'], 4)}]); "
                   + ", ".join(f"{names[k]} {fmt(osd[k], 2)} → {fmt(ogn[k], 2)}" for k in ERR)
                   + f"; 유지된 드문 약물 {fmt(osd['cont_keep_by_rarity'].get('rare'))} → {fmt(ogn['cont_keep_by_rarity'].get('rare'))}, added 누락 {fmt(osd['added_miss_pooled'])} → {fmt(ogn['added_miss_pooled'])}.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--gamenet-glob", default=None, help="override, e.g. 'seed-1203-smoke/per_visit_predictions.npz'")
    ap.add_argument("--other-dir", default=None, help="compare SafeDrug against dumps under this directory instead of GAMENet")
    ap.add_argument("--other-glob", default="*/per_visit_predictions.npz")
    ap.add_argument("--other-label", default=None)
    ap.add_argument("--tag", default=None, help="results file suffix (default: dataset)")
    ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = dict(CFG[args.dataset])
    if args.other_dir:
        cfg["gamenet_dir"] = Path(args.other_dir); cfg["gamenet_glob"] = args.other_glob
        cfg["label_gn"] = args.other_label or Path(args.other_dir).name
    gn_paths = sorted(cfg["gamenet_dir"].glob(args.gamenet_glob or cfg["gamenet_glob"]))
    if not args.gamenet_glob:
        gn_paths = [p for p in gn_paths if "smoke" not in p.parent.name]
    if not gn_paths or not cfg["safedrug"]:
        raise FileNotFoundError(f"safedrug {len(cfg['safedrug'])} dumps, gamenet {len(gn_paths)} dumps")
    with open(cfg["records"], "rb") as f:
        records = dill.load(f)
    n_med = max(m for p in records for adm in p for m in adm[2]) + 1
    train_freq = train_frequencies(records, n_med)
    edges = np.quantile(train_freq[train_freq > 0], [1 / 3, 2 / 3]); tert = np.digitize(train_freq, edges)
    t_sd, d_sd, off_sd = arm_frames(cfg["safedrug"], train_freq, tert, "safedrug")
    t_gn, d_gn, off_gn = arm_frames(gn_paths, train_freq, tert, "gamenet")
    print(f"safedrug {len(cfg['safedrug'])} seeds / gamenet {len(gn_paths)} seeds; transitions {len(t_sd):,} vs {len(t_gn):,}", flush=True)
    m = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset, "label_sd": cfg["label_sd"], "label_gn": cfg["label_gn"],
         "gamenet_dumps": [str(p) for p in gn_paths], "safedrug_dumps": [str(p) for p in cfg["safedrug"]], "rarity_edges": edges.tolist(),
         "official_sd": off_sd, "official_gn": off_gn,
         "safedrug": arm_summary(t_sd, d_sd), "gamenet": arm_summary(t_gn, d_gn),
         "paired": paired(t_sd, t_gn, args.n_boot, args.seed), "by_n_prev": by_n_prev(t_sd, t_gn)}
    # oracle-size decode for both arms: same set size per visit, ranking quality only
    o_sd, od_sd, _ = arm_frames(cfg["safedrug"], train_freq, tert, "safedrug", oracle=True)
    o_gn, od_gn, _ = arm_frames(gn_paths, train_freq, tert, "gamenet", oracle=True)
    m["oracle"] = {"safedrug": arm_summary(o_sd, od_sd), "gamenet": arm_summary(o_gn, od_gn),
                   "paired": paired(o_sd, o_gn, args.n_boot, args.seed), "by_n_prev": by_n_prev(o_sd, o_gn)}
    m["interpretation"] = interpret(m)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (args.dataset + ("_smoke" if args.gamenet_glob and "smoke" in args.gamenet_glob else ""))
    (RESULTS / f"history_recovery_{tag}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    md = render(m); (RESULTS / f"history_recovery_{tag}.md").write_text(md, encoding="utf-8"); print(md)


if __name__ == "__main__":
    main()
