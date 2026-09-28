"""Official SafeDrug metrics: original (HADM_ID-ordered) vs chronological MIMIC-III records.

Original runs: track-04 baseline seeds 0..3, whose official metrics are carried in
SafeDrug_군집별평가_20260904/개입실험/B_thresholds/seed{k}/global_0.5/run_manifest.json
(threshold 0.5 = the plain baseline). Chronological runs: out/mimic3_chrono_safedrug/seed{k}/.
Both come from the same wrapper, same seeds, same epochs, same split; only the visit
order differs. Metrics are the wrapper's `official_metrics` (SafeDrug's own
multi_label_metric / ddi_rate_score on data_test).

Writes results/mimic3_chrono_vs_original.{json,md}.
"""
from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIG = Path(r"C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds")
CHRONO = ROOT / "out" / "mimic3_chrono_safedrug"
RESULTS = ROOT / "results"
METRICS = ["ja", "prauc", "avg_f1", "ddi_rate", "avg_med"]


def load(manifest: Path) -> dict:
    m = json.loads(manifest.read_text(encoding="utf-8"))
    off = m["official_metrics"]
    if isinstance(off, str):
        off = ast.literal_eval(off)
    return {"best_epoch": m["best_epoch"], "seeds": m["seeds"],
            "records_sha256": str(m.get("data_files_sha256", ""))[:200],
            "test": off["test"], "eval": off["eval"], "train_seconds": m.get("train_seconds")}


def main():
    rows = {}
    for k in range(4):
        o = ORIG / f"seed{k}" / "global_0.5" / "run_manifest.json"
        c = CHRONO / f"seed{k}" / "run_manifest.json"
        rows[k] = {"original": load(o) if o.exists() else None, "chrono": load(c) if c.exists() else None}
    seeds_done = [k for k, r in rows.items() if r["original"] and r["chrono"]]
    summary = {}
    for split in ["test", "eval"]:
        summary[split] = {}
        for met in METRICS:
            o = np.array([rows[k]["original"][split][met] for k in seeds_done])
            c = np.array([rows[k]["chrono"][split][met] for k in seeds_done])
            d = c - o
            summary[split][met] = {"original_mean": float(o.mean()), "original_sd": float(o.std(ddof=1)) if len(o) > 1 else None,
                                   "chrono_mean": float(c.mean()), "chrono_sd": float(c.std(ddof=1)) if len(c) > 1 else None,
                                   "paired_diff_mean": float(d.mean()), "paired_diff_sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                                   "per_seed_diff": d.tolist()}
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "seeds_compared": seeds_done,
           "per_seed": rows, "summary": summary}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mimic3_chrono_vs_original.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    L = [f"# SafeDrug MIMIC-III: HADM_ID 순서(원본) vs 시간순 records — 공식 지표", "",
         f"생성 {out['generated_at_utc']} · 비교 시드 {seeds_done} · 같은 래퍼·시드·50 에포크·환자 분할, records 순서만 다름", ""]
    for split in ["test", "eval"]:
        L += [f"## {split} split", "", "| 지표 | 원본 (mean ± sd) | 시간순 (mean ± sd) | Δ(시간순 − 원본) mean ± sd | 시드별 Δ |", "|---|---|---|---|---|"]
        for met in METRICS:
            s = summary[split][met]
            sd = lambda v: "" if v is None else f" ± {v:.4f}"
            L.append(f"| {met} | {s['original_mean']:.4f}{sd(s['original_sd'])} | {s['chrono_mean']:.4f}{sd(s['chrono_sd'])} "
                     f"| {s['paired_diff_mean']:+.4f}{sd(s['paired_diff_sd'])} | {', '.join(f'{x:+.4f}' for x in s['per_seed_diff'])} |")
        L.append("")
    L += ["## 시드별 (test)", "", "| 시드 | 원본 best_epoch / ja / ddi / avg_med | 시간순 best_epoch / ja / ddi / avg_med |", "|---|---|---|"]
    for k, r in rows.items():
        f = lambda x: "—" if not x else f"{x['best_epoch']} / {x['test']['ja']:.4f} / {x['test']['ddi_rate']:.4f} / {x['test']['avg_med']:.2f}"
        L.append(f"| seed{k} | {f(r['original'])} | {f(r['chrono'])} |")
    L.append("")
    L.append("원본 수치는 트랙 04 baseline(임계 0.5) run_manifest의 official_metrics. 시간순 수치는 문헌값과 직접 비교 불가.")
    (RESULTS / "mimic3_chrono_vs_original.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
