"""§6 산출물: 표 A/B + 결론 문단 → out/REPORT_BHC.md"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

sweep = pd.read_csv(OUT / "tableA_bhc_sweep.csv")
perf = pd.read_csv(OUT / "tableB_bhc_cluster_performance.csv")
prep = json.load(open(OUT / "13_bhc_prep.json", encoding="utf-8"))
cmeta = json.load(open(OUT / "15_bhc_cluster_meta.json", encoding="utf-8"))
pmeta = json.load(open(OUT / "16_bhc_perf_meta.json", encoding="utf-8"))
emb = json.load(open(OUT / "14_embed_masked.json", encoding="utf-8"))


def md(d, floatfmt="{:.4g}"):
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = []
    for _, r in d.iterrows():
        cells = ["—" if (isinstance(v, float) and pd.isna(v)) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


k15 = sweep[sweep.k == 15].set_index("rep")
cbm = k15.loc["CB_masked"]
tfm = k15.loc["TFIDF_masked"]

REN_A = {"rep": "표현", "silhouette": "실루엣", "ARI_lendummy": "ARI(길이더미)",
         "eta2_notelen": "η²(길이)", "eta2_diag_count": "η²(진단수)", "eta2_drug_count": "η²(약물수)",
         "ARI_C1id15": "ARI(C1_id k15)", "NMI_C1id15": "NMI(C1_id k15)",
         "ARI_P1_full": "ARI(P1 전체)", "ARI_P1_labeled_only": "ARI(P1 기타제외)",
         "NMI_P1_labeled_only": "NMI(P1 기타제외)"}
cols_A = list(REN_A)

tabA = sweep[sweep.k == 15][cols_A].rename(columns=REN_A)
tabA_k = sweep[(sweep.rep == "CB_masked") & sweep.k.isin([5, 10, 15, 20, 25, 30])][
    ["k", "silhouette", "ARI_lendummy", "eta2_notelen", "eta2_diag_count", "ARI_C1id15", "ARI_P1_labeled_only"]
].rename(columns={"silhouette": "실루엣", "ARI_lendummy": "ARI(길이더미)", "eta2_notelen": "η²(길이)",
                  "eta2_diag_count": "η²(진단수)", "ARI_C1id15": "ARI(C1_id k15)",
                  "ARI_P1_labeled_only": "ARI(P1 기타제외)"})

prof = pd.DataFrame(cmeta["profiles"]["CB_masked_k15"]).rename(
    columns={"cluster": "B", "visits": "방문", "mean_n_diag": "평균진단", "mean_n_drug": "평균약물",
             "mean_words": "평균단어", "top_dx": "상위 진단 3개"})

perf2 = perf[perf.test_visits >= 30].copy()
perf2["CI(상수)"] = perf2.apply(lambda r: f"[{r.jac_const_lo:.3f}, {r.jac_const_hi:.3f}]", axis=1)
perf2["CI(prev)"] = perf2.apply(lambda r: f"[{r.jac_prev_lo:.3f}, {r.jac_prev_hi:.3f}]", axis=1)
perf2 = perf2[["cluster", "test_visits", "mean_n_diag", "mean_n_drug", "jac_const", "CI(상수)",
               "jac_prev", "CI(prev)", "top3_dx"]].rename(
    columns={"cluster": "B", "test_visits": "test 방문", "mean_n_diag": "평균진단",
             "mean_n_drug": "평균약물", "jac_const": "상수 Jac", "jac_prev": "copy-prev Jac",
             "top3_dx": "상위 진단 3개"})

sens = cmeta["sensitivity_ari"]
stab = cmeta["seed_stability"]
lk = prep["leakage_extended_plus_brands"]

doc = f"""# BHC ClinicalBERT 임베딩 클러스터링 — 판정

전처리 통과 {prep['n_kept']:,}방문 (제외 {prep['n_excluded_total']}건) · Bio_ClinicalBERT · 슬라이딩 윈도우 stride 256 · mean pooling · GPU {emb['elapsed_sec']}초
스크립트: `08_bhc_prep.py` → `09_bhc_embed.py` → `10_bhc_cluster.py` → `11_bhc_eval.py` → `12_bhc_figs.py`

## Leakage 실측 (§1)

| 지표 | strict(성분 151) | +브랜드 {prep['brand_dict']['adopted_ingredient_in_vocab']}종 |
|---|---|---|
| 약물 언급 있는 노트 | {prep['leakage_strict_ingredient_only']['pct_notes_with_any_mention']}% | {lk['pct_notes_with_any_mention']}% |
| 방문당 언급 약물 수(중앙) | {prep['leakage_strict_ingredient_only']['mentions_per_visit_median']:.0f} | {lk['mentions_per_visit_median']:.0f} |
| **재현율 중앙 / 평균 / p90** | {prep['leakage_strict_ingredient_only']['recall_median']} / {prep['leakage_strict_ingredient_only']['recall_mean']} / {prep['leakage_strict_ingredient_only']['recall_p90']} | **{lk['recall_median']} / {lk['recall_mean']} / {lk['recall_p90']}** |
| 재현율 ≥0.5 인 방문 | {prep['leakage_strict_ingredient_only']['pct_recall_ge_50']}% | {lk['pct_recall_ge_50']}% |
| 정밀도 중앙 | {prep['leakage_strict_ingredient_only']['precision_median']} | {lk['precision_median']} |

**BHC는 처방 목록을 담고 있지 않다.** 실제 처방 평균 23개 중 노트에 언급되는 건 2~3개(재현율 중앙 0.10),
언급된 것은 거의 다 실제 처방이다(정밀도 1.0). 상의 조건(재현율 중앙 ≥0.5) 미달 → 마스킹으로 충분.
마스킹: 약물 스팬 {prep['masking']['drug_spans_replaced']:,} + 클래스 표현 {prep['masking']['class_spans_replaced']:,}개 치환,
노트당 평균 토큰의 {prep['masking']['pct_tokens_masked_mean']}% 삭제. 한계: {prep['masking']['limitation']}

## 표 A. 판정 4종 × 표현 4종 (k=15)

{md(tabA)}

### CB_masked k 스윕 (판정 지표의 k 안정성)

{md(tabA_k)}

### 민감도 (같은 k에서 파티션 간 ARI)

| k | CB: 마스킹 vs 원본 | CB vs TF-IDF (마스킹) | TF-IDF: 마스킹 vs 원본 |
|---|---|---|---|
""" + "\n".join(
    f"| {k} | {v['ARI_CBmasked_vs_CBclean']} | {v['ARI_CBmasked_vs_TFIDFmasked']} | {v['ARI_TFIDFmasked_vs_TFIDFclean']} |"
    for k, v in sens.items()
) + f"""

### seed 안정성 (5 seeds, 쌍별 ARI 평균/최소)

| 설정 | 평균 | 최소 |
|---|---|---|
""" + "\n".join(f"| {k} | {v['mean']} | {v['min']} |" for k, v in stab.items()) + f"""

## 표 B. CB_masked k=15 클러스터 프로파일

{md(prof)}

## §5. 클러스터별 자명 baseline 성능 (test ≥30)

{md(perf2, "{:.3f}")}

격차: 상수 {pmeta['jac_const_spread']} (diagnose C1_id k15: 0.085), copy-prev {pmeta['jac_prev_spread']} (diagnose: 0.059)
약물수 상관: 상수 r={pmeta['corr_jacconst_vs_ndrug']}, copy-prev r={pmeta['corr_jacprev_vs_ndrug']}

## 그림

- `figs/fig6_bhc_umap_clusters.png` — BHC 임베딩 UMAP, k=15 클러스터
- `figs/fig7_bhc_umap_dxoverlay.png` — 같은 좌표에 diagnose 클러스터
- `figs/fig8_bhc_dx_heatmap.png` — BHC × diagnose 교차표 히트맵
"""

(OUT / "REPORT_BHC.md").write_text(doc, encoding="utf-8")
print("written REPORT_BHC.md")
