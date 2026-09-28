"""Change-visit split re-aggregation on per-visit SafeDrug dumps (proposal 3.3-A).

    python scripts/change_visit_split.py --dataset mimic3 [--splits test eval]
    python scripts/change_visit_split.py --dataset mimic4 [--splits test]

Inputs
  mimic3: SafeDrug_군집별평가_20260904/개입실험/B_thresholds/seed{0..3}/global_0.5/
          per_visit_predictions.npz  (baseline SafeDrug, threshold 0.5, 4 seeds)
          + SOTA/SafeDrug/data/records_final.pkl for the train-split constant predictor
  mimic4: out/mimic4/baseline-seed-*.npz written by dump_mimic4_predictions.py
          + ServerityMed/data/mimic-iv/records_final5.pkl

Outputs (aggregate only, safe to keep):
  results/change_visit_split_<dataset>_<splits>.json
  results/change_visit_split_<dataset>_<splits>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cvsplit_lib as L  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"

MIMIC3_DUMP = Path(r"C:\Users\Administrator\Desktop\SafeDrug_군집별평가_20260904\개입실험\B_thresholds")
MIMIC3_RECORDS = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output\records_final.pkl")  # sha256 321684d9... = run_manifest
MIMIC3_MASTER = Path(r"C:\Users\Administrator\Desktop\unfair\out\acute_driver_audit\safedrug_mimic3_cohort\master_visits.csv")


def chronological_visit_index(patient_index: np.ndarray, visit_index: np.ndarray) -> np.ndarray:
    """SafeDrug's MIMIC-III records order visits by HADM_ID, not by time (50% of
    transitions go backwards). Re-rank each patient's visits by ADMITTIME from the
    unfair track's aligned sidecar so 'previous' means the chronologically previous
    visit. Model predictions per visit are unchanged; only the pairing moves."""
    import pandas as pd
    mv = pd.read_csv(MIMIC3_MASTER, parse_dates=["ADMITTIME"])
    mv["rank"] = mv.groupby("safedrug_patient_index")["ADMITTIME"].rank(method="first").astype(int) - 1
    key = {(int(p), int(v)): int(r) for p, v, r in zip(mv["safedrug_patient_index"], mv["safedrug_visit_index"], mv["rank"])}
    return np.array([key[(int(p), int(v))] for p, v in zip(patient_index, visit_index)], dtype=np.int64)
MIMIC4_DUMP = ROOT / "out" / "mimic4"
MIMIC4_RECORDS = Path(r"C:\Users\Administrator\Desktop\ServerityMed\data\mimic-iv\records_final5.pkl")
# SafeDrug retrained on the chronologically re-ordered MIMIC-III records (run_safedrug_chrono.py)
MIMIC3CHRONO_DUMP = ROOT / "out" / "mimic3_chrono_safedrug"
MIMIC3CHRONO_RECORDS = ROOT / "out" / "mimic3_chrono_data" / "records_final.pkl"

K_CONST = 15


def split_patients(records):
    """SafeDrug's split: 2/3 train, then half test / half eval, in list order."""
    split_point = int(len(records) * 2 / 3)
    eval_len = int((len(records) - split_point) / 2)
    return (list(range(0, split_point)),
            list(range(split_point, split_point + eval_len)),
            list(range(split_point + eval_len, len(records))))


def load_records(path: Path):
    import dill
    with path.open("rb") as f:
        return dill.load(f)


def dumps_for(dataset: str) -> dict[str, Path]:
    if dataset == "mimic3":
        return {f"seed{s}": MIMIC3_DUMP / f"seed{s}" / "global_0.5" / "per_visit_predictions.npz"
                for s in range(4)}
    if dataset == "mimic4":
        return {p.stem.replace("baseline-", ""): p for p in sorted(MIMIC4_DUMP.glob("baseline-seed-*.npz"))}
    if dataset == "mimic3chrono":
        return {d.name: d / "per_visit_predictions.npz" for d in sorted(MIMIC3CHRONO_DUMP.glob("seed*"))
                if (d / "per_visit_predictions.npz").exists() and (d / "run_manifest.json").exists()}
    raise ValueError(dataset)


def records_path(dataset: str) -> Path:
    return {"mimic3": MIMIC3_RECORDS, "mimic4": MIMIC4_RECORDS, "mimic3chrono": MIMIC3CHRONO_RECORDS}[dataset]


def check_gt_against_records(z, records, n_check: int = 500):
    """The dump's y_gt must equal the records' medication sets (same cohort,
    same ordering). Checks the first n_check rows and 200 random ones."""
    rng = np.random.default_rng(0)
    n = len(z["patient_index"])
    idx = list(range(min(n_check, n))) + rng.integers(0, n, size=min(200, n)).tolist()
    for i in idx:
        p, v = int(z["patient_index"][i]), int(z["visit_index"][i])
        expected = set(int(c) for c in records[p][v][2])
        got = set(int(c) for c in np.flatnonzero(z["y_gt"][i]))
        if expected != got:
            raise AssertionError(f"y_gt mismatch at row {i} (patient {p}, visit {v})")


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render_md(dataset, splits, seeds, pooled, per_seed, spread, meta) -> str:
    lines = []
    lines.append(f"# change-visit 분리 재집계 — {dataset.upper()} ({'+'.join(splits)} split, 방문 순서: {meta['visit_order']})")
    lines.append("")
    lines.append(f"생성 {meta['generated_at_utc']} · 시드 {len(seeds)}개 ({', '.join(map(str, seeds))}) · "
                 f"상수 예측기 = train 최빈 top-{K_CONST}")
    if meta["visit_order"] == "chronological" and dataset == "mimic3":
        lines.append("")
        lines.append("주의: SafeDrug MIMIC-III records는 방문을 HADM_ID 순으로 두어 전이의 50%가 시간상 역행한다. "
                     "이 표는 각 방문을 **시간상 직전** 방문과 짝지은 것이다(모델 예측은 방문별로 고정, 짝만 바뀜). "
                     "모델은 학습·추론 모두 HADM_ID 순 이력을 입력받았으므로, 여기서의 copy-previous는 모델이 본 '직전'과 다르다.")
    lines.append("")
    lines.append("전이 = 직전 방문이 있는 방문. continuation = 정답 처방 집합이 직전 방문과 동일, "
                 "change = 추가/중단이 하나라도 있음. copy-previous = 직전 방문 **정답** 집합 복사. "
                 "Δ = model − copy-previous, 환자 군집 부트스트랩 95% CI (시드별 지표를 전이 단위로 평균한 뒤 계산).")
    lines.append("")
    lines.append("## 1. 전이 구성")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|---|---|")
    lines.append(f"| 전이 수 | {pooled['n_transitions']:,} |")
    lines.append(f"| 환자 수 | {pooled['n_patients']:,} |")
    lines.append(f"| **change-visit 비율** | **{pooled['change_share']:.3f}** |")
    lines.append(f"| continuation 비율 | {1 - pooled['change_share']:.3f} |")
    lines.append("")
    lines.append("## 2. 계층별 성능 (시드 평균 ± 시드 SD)")
    lines.append("")
    lines.append("| 계층 | n | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model F1 | copy F1 | model added-J | model stopped-J | 평균 추가/중단 수 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for st in ["all_transitions", "continuation", "change"]:
        P = pooled["strata"][st]; S = spread[st]
        d = P.get("model_minus_copy")
        dtxt = f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]" if d else "—"
        lines.append(
            f"| {st} | {P['n_transitions']:,} | {S['model_jaccard']['mean']:.4f} ± {fmt(S['model_jaccard']['sd'])} "
            f"| {P['copy_previous_jaccard']:.4f} | {fmt(P['constant_jaccard'])} | {dtxt} "
            f"| {S['model_f1']['mean']:.4f} ± {fmt(S['model_f1']['sd'])} | {P['copy_previous_f1']:.4f} "
            f"| {P['model_added_jaccard']:.4f} | {P['model_stopped_jaccard']:.4f} "
            f"| {P['mean_n_added']:.2f} / {P['mean_n_stopped']:.2f} |")
    lines.append("")
    lines.append("## 3. 변화 크기 구간별 (Jaccard(직전 정답, 현재 정답) 구간)")
    lines.append("")
    lines.append("| 구간 | n | 비율 | model J | copy-prev J | 상수 J | Δ(model−copy) [95% CI] | model added-J | model stopped-J |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, _, _ in L.CHANGE_BINS:
        P = pooled["strata"][f"bin:{name}"]
        d = P.get("model_minus_copy")
        dtxt = f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]" if d else "—"
        share = P["n_transitions"] / pooled["n_transitions"]
        lines.append(f"| {name} | {P['n_transitions']:,} | {share:.3f} | {fmt(P['model_jaccard'])} "
                     f"| {fmt(P['copy_previous_jaccard'])} | {fmt(P['constant_jaccard'])} | {dtxt} "
                     f"| {fmt(P['model_added_jaccard'])} | {fmt(P['model_stopped_jaccard'])} |")
    lines.append("")
    lines.append("## 4. 시드별 (model J)")
    lines.append("")
    lines.append("| 시드 | all | continuation | change | all-visits J (첫 방문 포함, 공식 지표 대조용) |")
    lines.append("|---|---|---|---|---|")
    for s in seeds:
        ps = per_seed[str(s)]
        lines.append(f"| {s} | {ps['strata']['all_transitions']['model_jaccard']:.4f} "
                     f"| {ps['strata']['continuation']['model_jaccard']:.4f} "
                     f"| {ps['strata']['change']['model_jaccard']:.4f} "
                     f"| {ps['all_visits_model_jaccard']:.4f} |")
    lines.append("")
    lines.append("## 입력")
    lines.append("")
    for s, info in meta["inputs"].items():
        lines.append(f"- {s}: `{info['path']}` ({info['n_rows']:,} rows)")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mimic3", "mimic4", "mimic3chrono"])
    ap.add_argument("--splits", nargs="*", default=["test"], choices=["test", "eval"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--order", default="safedrug", choices=["safedrug", "chronological"],
                    help="mimic3 only: pair each visit with the chronologically previous visit "
                         "instead of SafeDrug's HADM_ID-ordered previous visit")
    args = ap.parse_args()
    if args.order == "chronological" and args.dataset != "mimic3":
        raise SystemExit("--order chronological applies to mimic3 only (mimic4 records are chronological)")

    records = load_records(records_path(args.dataset))
    train_idx, test_idx, eval_idx = split_patients(records)
    const_set = L.constant_topk_from_records([records[i] for i in train_idx], K_CONST)
    expected_patients = {"test": set(test_idx), "eval": set(eval_idx)}

    dumps = dumps_for(args.dataset)
    if not dumps:
        raise FileNotFoundError(f"no dumps for {args.dataset}")
    scored_by_seed, per_seed, inputs = {}, {}, {}
    for seed, path in dumps.items():
        z = np.load(path)
        check_gt_against_records(z, records)
        # the dump's split labels must agree with the SafeDrug split rule
        for split in args.splits:
            pts = set(int(p) for p in np.unique(z["patient_index"][z["split"] == split]))
            if pts != expected_patients[split]:
                raise AssertionError(f"{seed}: {split} patients in dump differ from split rule")
        keep = np.isin(z["split"], args.splits)
        visit_index = z["visit_index"]
        if args.order == "chronological":
            visit_index = chronological_visit_index(z["patient_index"], z["visit_index"])
        rows = L.build_transitions(z["patient_index"], visit_index, z["y_gt"], z["y_pred"],
                                   seed=seed, keep_mask=keep)
        scored = L.score_rows(rows, const_set)
        scored_by_seed[seed] = scored
        summ = L.summarize(scored, n_boot=args.n_boot)
        # all-visits Jaccard (first visits included) for cross-checking official numbers
        gt_sets = L.multihot_rows_to_sets(z["y_gt"][keep])
        pr_sets = L.multihot_rows_to_sets(z["y_pred"][keep])
        summ["all_visits_model_jaccard"] = float(np.mean([L.jaccard(p, t) for p, t in zip(pr_sets, gt_sets)]))
        summ["all_visits_n"] = int(keep.sum())
        per_seed[seed] = summ
        inputs[seed] = {"path": str(path), "n_rows": int(len(z["patient_index"]))}
        print(f"[{seed}] transitions {summ['n_transitions']:,}  change share {summ['change_share']:.3f}  "
              f"model J all/cont/change = {summ['strata']['all_transitions']['model_jaccard']:.4f}/"
              f"{summ['strata']['continuation']['model_jaccard']:.4f}/{summ['strata']['change']['model_jaccard']:.4f}  "
              f"copy J = {summ['strata']['all_transitions']['copy_previous_jaccard']:.4f}  "
              f"all-visits J = {summ['all_visits_model_jaccard']:.4f}")

    pooled_rows = L.average_over_seeds(scored_by_seed)
    pooled = L.summarize(pooled_rows, n_boot=args.n_boot)
    strata_names = ["all_transitions", "continuation", "change"] + [f"bin:{n}" for n, _, _ in L.CHANGE_BINS]
    spread = {st: {m: L.seed_spread(per_seed, st, m) for m in ["model_jaccard", "model_f1", "model_added_jaccard", "model_stopped_jaccard"]}
              for st in strata_names}

    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
            "visit_order": args.order if args.dataset == "mimic3" else "chronological (records order)",
            "splits": args.splits, "seeds": list(dumps), "constant_k": K_CONST,
            "records": str(records_path(args.dataset)),
            "n_boot": args.n_boot, "inputs": inputs}
    result = {"meta": meta, "pooled_over_seeds": pooled, "per_seed": per_seed, "seed_spread": spread}
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{'+'.join(args.splits)}" + ("_chrono" if args.order == "chronological" else "")
    (RESULTS / f"change_visit_split_{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_md(args.dataset, args.splits, list(dumps), pooled, per_seed, spread, meta)
    (RESULTS / f"change_visit_split_{tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
