"""Stage-0 gate: ground-truth prescription DDI rate per chief-complaint (CC) cluster.

Question. Does the ground-truth DDI rate of prescriptions differ between CC
clusters? If not, the hypothesis "the DDI penalty systematically hurts patient
groups that need many interacting drugs" has no premise and is dropped before
any model is trained.

Inputs (read only; no existing pipeline file is modified)
  - CC cluster labels  : out/cc_cluster_assignments.csv, column clusterCC_bert
                         (Bio_ClinicalBERT -> SVD50 -> k-means, k=24, track B);
                         stratum no_CC (1,737 visits) becomes the extra layer NO_CC.
  - cluster names      : out/40_cc_map_meta.json named_clusters (fig29 concept
                         dictionary, >=60% rule; 10 nameable clusters).
  - official cohort    : SOTA/SafeDrug/data/output/records_final.pkl (6,350 patients,
                         15,032 visits, 112 ATC3). data_final.pkl is NOT used.
  - HADM_ID mapping    : out/acute_driver_audit/safedrug_mimic3_cohort/master_visits.csv
                         (records flattening order <-> SUBJECT_ID, HADM_ID).
  - DDI matrix         : SOTA/SafeDrug/data/output/ddi_A_final.pkl (112x112, nnz 674).

DDI rate. Same definition as SafeDrug's util.ddi_rate_score: DDI pairs / all
unordered medication pairs. Per visit that is dd/pairs (NaN when the visit has
fewer than two medications: 0/0). The pooled rate over the whole cohort is
verified against util.ddi_rate_score itself and against the reference value
0.0826 (paper: 0.0808) to three decimals; the run aborts otherwise.

Gate rule (fixed before any result was seen, 2026-09-18):
  PASS  iff  permutation p (range statistic, per-visit ddi_rate, raw) < 0.05
        AND  permutation p (range statistic, ddi_rate residualised on
             n_med + n_med^2) < 0.05
  where the permutation test is the existing patient-level reassignment of
  scripts/safedrug_cluster_gap.py (10,000 draws, seed 0, min 30 visits per
  group) over the 24 CC clusters; NO_CC is tabulated but excluded from the
  test. Weighted-SD, DDI-pair-count, and n_med-quartile results are reported
  alongside but do not decide the gate. Any later change to this rule must be
  labelled post hoc.

Outputs
  data/manifests/cluster-ddi-rate-stage0.json   every number, join stats, seeds, time
  out/cluster_ddi_rate_stage0.md                tables + interpretation + PASS/FAIL

Run:  py -3.12 scripts/measure_cluster_ddi_rate.py [--n-perm 10000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SAFEDRUG_SRC = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\src")
SAFEDRUG_DATA = Path(r"C:\Users\Administrator\Desktop\SOTA\SafeDrug\data\output")
for p in (SCRIPTS, SAFEDRUG_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from safedrug_cluster_gap import MIN_VISITS, N_PERM, PERM_SEED, permutation_p  # noqa: E402

RECORDS = SAFEDRUG_DATA / "records_final.pkl"
VOC = SAFEDRUG_DATA / "voc_final.pkl"
DDI_A = SAFEDRUG_DATA / "ddi_A_final.pkl"
MASTER = ROOT / "out" / "acute_driver_audit" / "safedrug_mimic3_cohort" / "master_visits.csv"
CC_LABELS = ROOT / "out" / "cc_cluster_assignments.csv"
CC_NAMES = ROOT / "out" / "40_cc_map_meta.json"
MANIFEST = ROOT / "data" / "manifests" / "cluster-ddi-rate-stage0.json"
REPORT = ROOT / "out" / "cluster_ddi_rate_stage0.md"

CLUSTER_COL = "clusterCC_bert"
K = 24
NO_CC = "NO_CC"
REFERENCE_DDI_RATE = 0.0826        # measured on this cohort (paper reports 0.0808)
PAPER_DDI_RATE = 0.0808
JOIN_RATE_FLOOR = 0.95
ALPHA = 0.05
GATE_RULE = ("PASS iff p_range(ddi_rate raw) < 0.05 AND p_range(ddi_rate residualised on "
             "n_med + n_med^2) < 0.05; patient-level permutation, 10,000 draws, seed 0, "
             "min 30 visits/group, 24 CC clusters, NO_CC excluded. Fixed before results.")


# ------------------------------------------------------------------ loading
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    with path.open("rb") as f:
        return dill.load(f)


def load_master_visits() -> pd.DataFrame:
    mv = pd.read_csv(MASTER, usecols=["SUBJECT_ID", "HADM_ID", "safedrug_patient_index", "safedrug_visit_index"])
    return mv.rename(columns={"safedrug_patient_index": "patient_index", "safedrug_visit_index": "visit_index"})


def load_cc_labels() -> pd.DataFrame:
    """Keys and labels only; the free-text CC column is never read."""
    cc = pd.read_csv(CC_LABELS, usecols=["SUBJECT_ID", "HADM_ID", "stratum", CLUSTER_COL])
    is_no_cc = cc["stratum"].eq("no_CC")
    if (cc[CLUSTER_COL].isna() != is_no_cc).any():
        raise AssertionError("clusterCC_bert is NaN exactly for the no_CC stratum; that no longer holds")
    cc["cluster_label"] = np.where(is_no_cc, NO_CC, cc[CLUSTER_COL].fillna(-1).astype(int).map("{:02d}".format))
    return cc[["SUBJECT_ID", "HADM_ID", "cluster_label"]]


def load_cluster_names() -> dict[str, str]:
    meta = json.loads(CC_NAMES.read_text(encoding="utf-8"))
    named = {f"{int(k):02d}": v for k, v in meta["named_clusters"].items()}
    return {f"{k:02d}": (f"{k:02d} {named[f'{k:02d}']}" if f"{k:02d}" in named else f"{k:02d}") for k in range(K)} | {NO_CC: NO_CC}


def flatten_records(records: list) -> pd.DataFrame:
    rows = []
    for pi, patient in enumerate(records):
        for vi, adm in enumerate(patient):
            rows.append({"patient_index": pi, "visit_index": vi, "n_dx": len(adm[0]),
                         "n_proc": len(adm[1]), "n_med": len(adm[2]), "meds": list(adm[2])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ DDI counting
def visit_ddi_counts(meds, ddi_A: np.ndarray) -> tuple[int, int]:
    """(DDI pairs, all unordered pairs) for one visit -- the inner loop of
    util.ddi_rate_score, verbatim in semantics (j > i, either direction)."""
    dd = pairs = 0
    for i, mi in enumerate(meds):
        for j, mj in enumerate(meds):
            if j <= i:
                continue
            pairs += 1
            if ddi_A[mi, mj] == 1 or ddi_A[mj, mi] == 1:
                dd += 1
    return dd, pairs


def visit_rate(dd: int, pairs: int) -> float:
    return dd / pairs if pairs > 0 else float("nan")


def ddi_pairs_of_visit(meds, ddi_A: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for i, mi in enumerate(meds):
        for j, mj in enumerate(meds):
            if j <= i:
                continue
            if ddi_A[mi, mj] == 1 or ddi_A[mj, mi] == 1:
                out.append((min(mi, mj), max(mi, mj)))
    return out


# ------------------------------------------------------------------ join
def join_official_with_notes(master: pd.DataFrame, cc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = master.merge(cc.rename(columns={"SUBJECT_ID": "SUBJECT_ID_notes"}), on="HADM_ID", how="inner")
    if (joined["SUBJECT_ID"] != joined["SUBJECT_ID_notes"]).any():
        raise AssertionError("SUBJECT_ID disagrees between the official cohort and the note cohort")
    stats = {
        "join_key": "HADM_ID",
        "n_official": int(len(master)),
        "n_notes": int(len(cc)),
        "n_matched": int(len(joined)),
        "n_official_only": int(len(set(master["HADM_ID"]) - set(cc["HADM_ID"]))),
        "n_notes_only": int(len(set(cc["HADM_ID"]) - set(master["HADM_ID"]))),
        "join_rate_vs_official": float(len(joined) / len(master)),
        "join_rate_vs_notes": float(len(joined) / len(cc)),
        "floor": JOIN_RATE_FLOOR,
    }
    return joined, stats


# ------------------------------------------------------------------ aggregation
def aggregate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("cluster_label", sort=True)
    table = pd.DataFrame({
        "n_visits": g.size(),
        "n_patients": g["SUBJECT_ID"].nunique(),
        "ddi_rate_mean": g["ddi_rate"].mean(),
        "ddi_rate_median": g["ddi_rate"].median(),
        "ddi_rate_sd": g["ddi_rate"].std(ddof=1),
        "ddi_rate_pooled": g["dd_cnt"].sum() / g["all_cnt"].sum(),
        "n_med_mean": g["n_med"].mean(),
        "n_dx_mean": g["n_dx"].mean(),
        "dd_cnt_mean": g["dd_cnt"].mean(),
        "n_visits_rate_defined": g["ddi_rate"].count(),
    }).reset_index()
    return table


def residualise(y: np.ndarray, n_med: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS residual of y on [1, n_med, n_med^2]; returns (residuals, beta)."""
    X = np.column_stack([np.ones(len(y)), n_med, n_med ** 2]).astype(float)
    beta, *_ = np.linalg.lstsq(X, y.astype(float), rcond=None)
    return y - X @ beta, beta


def run_permutation(df: pd.DataFrame, value_col: str, n_perm: int, seed: int, min_visits: int = MIN_VISITS) -> dict:
    """Existing patient-level permutation test (safedrug_cluster_gap.permutation_p)
    over the 24 CC clusters; rows with NaN value or NO_CC are dropped."""
    sub = df[(df["cluster_label"] != NO_CC) & df[value_col].notna()]
    group = sub["cluster_label"].astype(int).to_numpy()
    res = permutation_p(group, sub["SUBJECT_ID"].to_numpy(), sub[value_col].to_numpy(dtype=float),
                        n_groups=K, min_visits=min_visits, n_perm=n_perm, seed=seed)
    res.update({"value": value_col, "n_visits": int(len(sub)), "n_patients": int(sub["SUBJECT_ID"].nunique()),
                "n_perm": n_perm, "seed": seed, "min_visits": min_visits})
    return res


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def top_pairs(df: pd.DataFrame, flat: pd.DataFrame, ddi_A: np.ndarray, med_names: dict, cluster_label: str,
              b01a_idx: int | None, k: int = 10) -> dict:
    sub = df[df["cluster_label"] == cluster_label]
    meds_by_key = flat.set_index(["patient_index", "visit_index"])["meds"]
    counter: Counter = Counter()
    for key in zip(sub["patient_index"], sub["visit_index"]):
        counter.update(ddi_pairs_of_visit(meds_by_key[key], ddi_A))
    total = sum(counter.values())
    with_b01a = sum(c for (i, j), c in counter.items() if b01a_idx in (i, j)) if b01a_idx is not None else None
    rows = [{"pair": f"{med_names[i]}-{med_names[j]}", "count": int(c), "share_of_cluster_ddi_pairs": c / total if total else float("nan"),
             "involves_B01A": bool(b01a_idx in (i, j)) if b01a_idx is not None else None}
            for (i, j), c in counter.most_common(k)]
    return {"cluster_label": cluster_label, "n_visits": int(len(sub)), "n_ddi_pair_occurrences": int(total),
            "share_involving_B01A": (with_b01a / total) if (total and with_b01a is not None) else float("nan"),
            "top": rows}


# ------------------------------------------------------------------ main
def fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_report(m: dict) -> str:
    names = m["cluster_names"]
    L = ["# CC 군집별 정답 처방 DDI율 — TWOSIDES 가설 0단계 게이트", "",
         f"생성 {m['generated_at_utc']} · 순열 {m['permutation']['ddi_rate_raw']['n_perm']:,}회 · 시드 {m['permutation']['ddi_rate_raw']['seed']} · 모델 학습 없음", "",
         f"**판정 기준(결과 확인 전 고정):** {m['gate']['rule']}", "",
         f"## 판정: **게이트 {'통과' if m['gate']['passed'] else '미통과'}**", "",
         f"- p_range(DDI율, raw) = {fmt(m['gate']['p_range_raw'])}, p_range(DDI율, 약물 수 보정 잔차) = {fmt(m['gate']['p_range_residual'])} (α = {ALPHA})", "",
         "## 1. 조인·검증", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 공식 코호트 방문 (records_final.pkl) | {m['join']['n_official']:,} |",
         f"| 노트 코호트 방문 (cc_cluster_assignments.csv) | {m['join']['n_notes']:,} |",
         f"| HADM_ID 매칭 | {m['join']['n_matched']:,} (공식 대비 {m['join']['join_rate_vs_official']:.4f}, 노트 대비 {m['join']['join_rate_vs_notes']:.4f}) |",
         f"| 공식에만 / 노트에만 | {m['join']['n_official_only']:,} / {m['join']['n_notes_only']:,} |",
         f"| 전체 코호트 pooled DDI율 (본 계산 / util.ddi_rate_score) | {m['verification']['pooled_ddi_rate_ours']:.5f} / {m['verification']['pooled_ddi_rate_util']:.5f} |",
         f"| 참조값 (실측 0.0826 / 논문 0.0808) 소수 3자리 일치 | {m['verification']['matches_reference_3dp']} |",
         f"| 매칭 방문 pooled DDI율 / 방문별 평균 | {m['verification']['pooled_ddi_rate_matched']:.5f} / {m['verification']['mean_visit_ddi_rate_matched']:.5f} |",
         f"| 약물 < 2 (DDI율 정의 불가) 방문 | {m['verification']['n_visits_rate_undefined']} |", "",
         "## 2. 군집별 정답 처방 DDI율 (DDI율 평균 내림차순, NO_CC는 맨 아래)", "",
         "| 군집 | 방문 | 환자 | DDI율 mean | median | SD | pooled | 약물 수 | 진단 수 | DDI 쌍 수 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(m["cluster_table"], key=lambda r: (r["cluster_label"] == NO_CC, -r["ddi_rate_mean"]))
    for r in rows:
        L.append(f"| {names[r['cluster_label']]} | {r['n_visits']:,} | {r['n_patients']:,} | {fmt(r['ddi_rate_mean'])} | {fmt(r['ddi_rate_median'])} "
                 f"| {fmt(r['ddi_rate_sd'])} | {fmt(r['ddi_rate_pooled'])} | {fmt(r['n_med_mean'], 2)} | {fmt(r['n_dx_mean'], 2)} | {fmt(r['dd_cnt_mean'], 2)} |")
    L += ["", "## 3. 게이트 검정 — 환자 단위 순열검정 (24개 CC 군집, NO_CC 제외)", "",
          "| 값 | 방문 | 관측 range | null mean | null p95 | z | p(range) | 관측 wSD | p(wSD) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, label in [("ddi_rate_raw", "DDI율 raw"), ("ddi_rate_residual", "DDI율 잔차 (n_med + n_med²)"),
                       ("dd_cnt_raw", "DDI 쌍 개수 raw"), ("dd_cnt_residual", "DDI 쌍 개수 잔차 (n_med + n_med²)")]:
        p = m["permutation"][key]
        L.append(f"| {label} | {p['n_visits']:,} | {fmt(p['observed_range'])} | {fmt(p['null_mean_range'])} | {fmt(p['null_p95_range'])} "
                 f"| {fmt(p['z_range'], 2)} | {fmt(p['p_raw_range'])} | {fmt(p['observed_wsd'])} | {fmt(p['p_raw_wsd'])} |")
    c = m["confounding"]
    L += ["", "## 4. 약물 수 교란", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 군집 평균 DDI율 vs 군집 평균 약물 수, Pearson r (24 군집) | {fmt(c['r_cluster_ddi_rate_vs_n_med'], 3)} |",
          f"| 군집 평균 DDI 쌍 수 vs 군집 평균 약물 수, Pearson r | {fmt(c['r_cluster_dd_cnt_vs_n_med'], 3)} |",
          f"| 방문 수준 DDI율 vs 약물 수, Pearson r | {fmt(c['r_visit_ddi_rate_vs_n_med'], 3)} |",
          f"| 잔차 회귀 계수 (절편, n_med, n_med²) | {', '.join(fmt(b, 5) for b in c['residual_beta_ddi_rate'])} |",
          f"| 잔차 군집 평균 vs 약물 수, Pearson r | {fmt(c['r_cluster_residual_ddi_rate_vs_n_med'], 3)} |", "",
          "약물 수 사분위 층화 (각 층 안에서 24 군집 range 순열검정, 최소 30 방문 군집만):", "",
          "| 사분위 (약물 수 구간) | 방문 | 사용 군집 | 관측 range | p(range) | 관측 wSD | p(wSD) |", "|---|---|---|---|---|---|---|"]
    for q in m["quartile_stratified"]:
        L.append(f"| Q{q['quartile']} [{q['n_med_min']}, {q['n_med_max']}] | {q['n_visits']:,} | {q['n_groups_used']} | {fmt(q['observed_range'])} | {fmt(q['p_raw_range'])} | {fmt(q['observed_wsd'])} | {fmt(q['p_raw_wsd'])} |")
    L += ["", "## 5. DDI율 상위 3개 군집의 기여 상위 ATC3 쌍", ""]
    for t in m["top_pairs"]:
        L += [f"**{names[t['cluster_label']]}** — 방문 {t['n_visits']:,}, DDI 쌍 발생 {t['n_ddi_pair_occurrences']:,}, B01A 관여 비율 {fmt(t['share_involving_B01A'], 3)}", "",
              "| 순위 | 쌍 | 발생 수 | 군집 DDI 쌍 중 비율 | B01A |", "|---|---|---|---|---|"]
        for i, r in enumerate(t["top"], 1):
            L.append(f"| {i} | {r['pair']} | {r['count']:,} | {fmt(r['share_of_cluster_ddi_pairs'], 3)} | {'Y' if r['involves_B01A'] else ''} |")
        L.append("")
    L += [f"코호트 전체 B01A 관여 비율: {fmt(m['cohort_top_pairs']['share_involving_B01A'], 3)} "
          f"(상위 쌍: {', '.join(r['pair'] for r in m['cohort_top_pairs']['top'][:5])})", "",
          "## 해석", ""] + m["interpretation"] + [""]
    return "\n".join(L)


def interpret(m: dict) -> list[str]:
    g, c = m["gate"], m["confounding"]
    P = m["permutation"]
    rows = [r for r in m["cluster_table"] if r["cluster_label"] != NO_CC]
    hi = max(rows, key=lambda r: r["ddi_rate_mean"]); lo = min(rows, key=lambda r: r["ddi_rate_mean"])
    names = m["cluster_names"]
    out = [f"- **게이트 {'통과' if g['passed'] else '미통과'}.** 24개 CC 군집의 정답 처방 DDI율 평균은 {fmt(lo['ddi_rate_mean'])}({names[lo['cluster_label']]})에서 "
           f"{fmt(hi['ddi_rate_mean'])}({names[hi['cluster_label']]}) 사이이고, range {fmt(P['ddi_rate_raw']['observed_range'])}의 환자 단위 순열 p는 {fmt(P['ddi_rate_raw']['p_raw_range'])}이다."]
    out.append(f"- 군집 평균 DDI율과 평균 약물 수의 상관은 r = {fmt(c['r_cluster_ddi_rate_vs_n_med'], 2)}. 약물 수(n_med + n_med²)를 보정한 잔차의 range 순열 p는 "
               f"{fmt(P['ddi_rate_residual']['p_raw_range'])}로, 군집 간 차이가 {'약물 수만으로는 설명되지 않는다' if g['p_range_residual'] < ALPHA else '약물 수를 고정하면 사라진다'}.")
    qs = [q for q in m["quartile_stratified"] if not np.isnan(q["p_raw_range"])]
    if qs:
        out.append("- 약물 수 사분위 층화(각 층 안에서 군집 range 검정): "
                   + "; ".join(f"Q{q['quartile']} [{q['n_med_min']}–{q['n_med_max']}개] range {fmt(q['observed_range'])}, p = {fmt(q['p_raw_range'])}" for q in qs) + ".")
    out.append(f"- DDI 쌍 절대 개수로도 같은 검정: raw p = {fmt(P['dd_cnt_raw']['p_raw_range'])}, 약물 수 보정 잔차 p = {fmt(P['dd_cnt_residual']['p_raw_range'])}.")
    tp = m["top_pairs"]
    out.append("- 상위 3개 군집의 DDI 쌍 중 B01A 관여 비율은 " + ", ".join(f"{fmt(t['share_involving_B01A'], 2)}" for t in tp)
               + f" (코호트 전체 {fmt(m['cohort_top_pairs']['share_involving_B01A'], 2)}). 최상위 쌍은 각각 "
               + ", ".join(t["top"][0]["pair"] if t["top"] else "—" for t in tp) + ".")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=PERM_SEED)
    args = ap.parse_args(argv)
    t0 = time.time()

    records, voc, ddi_A = _load(RECORDS), _load(VOC), np.asarray(_load(DDI_A))
    med_names = voc["med_voc"].idx2word
    b01a_idx = voc["med_voc"].word2idx.get("B01A")
    flat = flatten_records(records)
    if ddi_A.shape != (len(med_names), len(med_names)):
        raise AssertionError(f"ddi_A shape {ddi_A.shape} vs medication vocabulary {len(med_names)}")

    master = load_master_visits()
    if len(master) != len(flat) or master["patient_index"].nunique() != len(records):
        raise AssertionError("master_visits.csv does not align with records_final.pkl")
    cc = load_cc_labels()
    names = load_cluster_names()

    # --- per-visit DDI counts on the official cohort (all 15,032 visits)
    counts = [visit_ddi_counts(m, ddi_A) for m in flat["meds"]]
    flat["dd_cnt"] = [c[0] for c in counts]
    flat["all_cnt"] = [c[1] for c in counts]
    flat["ddi_rate"] = [visit_rate(*c) for c in counts]
    pooled_ours = flat["dd_cnt"].sum() / flat["all_cnt"].sum()
    from util import ddi_rate_score  # SafeDrug's own implementation, imported unchanged
    pooled_util = ddi_rate_score([[adm[2] for adm in p] for p in records], path=str(DDI_A))
    if abs(pooled_ours - pooled_util) > 1e-12:
        raise AssertionError(f"per-visit counts disagree with util.ddi_rate_score: {pooled_ours} vs {pooled_util}")
    matches_ref = round(pooled_ours, 3) == round(REFERENCE_DDI_RATE, 3)
    print(f"cohort pooled DDI rate {pooled_ours:.5f} (util {pooled_util:.5f}); reference 0.0826 -> 3dp match {matches_ref}", flush=True)
    if not matches_ref:
        raise SystemExit(f"ABORT: pooled DDI rate {pooled_ours:.5f} does not match the reference 0.0826 to 3 decimals")

    # --- join
    joined, join_stats = join_official_with_notes(master, cc)
    print(f"join: {join_stats['n_matched']:,}/{join_stats['n_official']:,} official visits matched "
          f"({join_stats['join_rate_vs_official']:.4f}); notes-only {join_stats['n_notes_only']}, official-only {join_stats['n_official_only']}", flush=True)
    if join_stats["join_rate_vs_official"] < JOIN_RATE_FLOOR:
        raise SystemExit(f"ABORT: join rate {join_stats['join_rate_vs_official']:.4f} < {JOIN_RATE_FLOOR}")
    df = joined.merge(flat.drop(columns=["meds"]), on=["patient_index", "visit_index"], how="inner")
    if len(df) != len(joined):
        raise AssertionError("joined rows lost when attaching per-visit counts")

    # --- cluster table
    table = aggregate_clusters(df)
    verification = {
        "pooled_ddi_rate_ours": float(pooled_ours), "pooled_ddi_rate_util": float(pooled_util),
        "reference_measured": REFERENCE_DDI_RATE, "reference_paper": PAPER_DDI_RATE, "matches_reference_3dp": bool(matches_ref),
        "mean_visit_ddi_rate_all": float(flat["ddi_rate"].mean()),
        "pooled_ddi_rate_matched": float(df["dd_cnt"].sum() / df["all_cnt"].sum()),
        "mean_visit_ddi_rate_matched": float(df["ddi_rate"].mean()),
        "n_visits_rate_undefined": int(df["ddi_rate"].isna().sum()),
        "n_visits_rate_undefined_all": int(flat["ddi_rate"].isna().sum()),
    }

    # --- residuals on n_med (visit level, CC clusters + NO_CC all included in the fit)
    ok = df["ddi_rate"].notna()
    resid, beta_rate = residualise(df.loc[ok, "ddi_rate"].to_numpy(), df.loc[ok, "n_med"].to_numpy())
    df["ddi_rate_residual"] = np.nan
    df.loc[ok, "ddi_rate_residual"] = resid
    df["dd_cnt_residual"], beta_cnt = residualise(df["dd_cnt"].to_numpy(float), df["n_med"].to_numpy())

    # --- permutation tests (existing method)
    perm = {}
    for out_key, column in [("ddi_rate_raw", "ddi_rate"), ("ddi_rate_residual", "ddi_rate_residual"),
                            ("dd_cnt_raw", "dd_cnt"), ("dd_cnt_residual", "dd_cnt_residual")]:
        t1 = time.time()
        perm[out_key] = run_permutation(df, column, args.n_perm, args.seed)
        print(f"permutation {out_key}: p_range={perm[out_key]['p_raw_range']:.4f} ({time.time() - t1:.0f}s)", flush=True)

    # --- quartile stratification by n_med
    cc_only = df[(df["cluster_label"] != NO_CC) & df["ddi_rate"].notna()].copy()
    cc_only["q"] = pd.qcut(cc_only["n_med"], 4, labels=False, duplicates="drop")
    quart = []
    for q, sub in cc_only.groupby("q", sort=True):
        r = run_permutation(sub, "ddi_rate", args.n_perm, args.seed)
        quart.append({"quartile": int(q) + 1, "n_med_min": int(sub["n_med"].min()), "n_med_max": int(sub["n_med"].max()),
                      "n_visits": int(len(sub)), "n_groups_used": r["n_groups_used"], "observed_range": r["observed_range"],
                      "p_raw_range": r["p_raw_range"], "observed_wsd": r["observed_wsd"], "p_raw_wsd": r["p_raw_wsd"]})

    # --- confounding correlations at cluster level (24 CC clusters)
    ct = table[table["cluster_label"] != NO_CC]
    resid_means = df[df["cluster_label"] != NO_CC].groupby("cluster_label")["ddi_rate_residual"].mean().reindex(ct["cluster_label"])
    confounding = {
        "r_cluster_ddi_rate_vs_n_med": pearson(ct["ddi_rate_mean"], ct["n_med_mean"]),
        "r_cluster_dd_cnt_vs_n_med": pearson(ct["dd_cnt_mean"], ct["n_med_mean"]),
        "r_visit_ddi_rate_vs_n_med": pearson(df["ddi_rate"], df["n_med"]),
        "residual_beta_ddi_rate": [float(b) for b in beta_rate],
        "residual_beta_dd_cnt": [float(b) for b in beta_cnt],
        "r_cluster_residual_ddi_rate_vs_n_med": pearson(resid_means.to_numpy(), ct["n_med_mean"].to_numpy()),
    }

    # --- top contributing pairs for the top-3 clusters by mean DDI rate
    top3 = ct.sort_values("ddi_rate_mean", ascending=False)["cluster_label"].head(3).tolist()
    tp = [top_pairs(df, flat, ddi_A, med_names, lab, b01a_idx) for lab in top3]
    cohort_tp = top_pairs(df.assign(cluster_label="ALL"), flat, ddi_A, med_names, "ALL", b01a_idx)

    p_raw, p_res = perm["ddi_rate_raw"]["p_raw_range"], perm["ddi_rate_residual"]["p_raw_range"]
    gate = {"rule": GATE_RULE, "alpha": ALPHA, "p_range_raw": p_raw, "p_range_residual": p_res,
            "passed": bool(p_raw < ALPHA and p_res < ALPHA), "post_hoc_changes": []}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "inputs": {p.name: {"path": str(p), "sha256": sha256(p)} for p in [RECORDS, VOC, DDI_A, MASTER, CC_LABELS, CC_NAMES]},
        "cluster_source": {"column": CLUSTER_COL, "k": K, "method": "Bio_ClinicalBERT -> TruncatedSVD(50) -> k-means (track B)",
                           "no_cc_layer": NO_CC, "names_from": "40_cc_map_meta.json named_clusters (fig29 concept dictionary, >=60%)"},
        "cluster_names": names,
        "ddi_matrix": {"shape": list(ddi_A.shape), "nnz": int((ddi_A > 0).sum())},
        "join": join_stats,
        "verification": verification,
        "gate": gate,
        "cluster_table": table.to_dict("records"),
        "permutation": perm,
        "confounding": confounding,
        "quartile_stratified": quart,
        "top_pairs": tp,
        "cohort_top_pairs": cohort_tp,
    }
    manifest["interpretation"] = interpret(manifest)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    REPORT.write_text(render_report(manifest), encoding="utf-8")
    print(f"gate {'PASS' if gate['passed'] else 'FAIL'}: p_range raw {p_raw:.4f}, residual {p_res:.4f}; wrote {MANIFEST} and {REPORT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
