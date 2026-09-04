"""§9 순열검정 리포트 생성 → out/REPORT_PERM.md (14_permutation.py / 15_perm_check.py 산출물 사용)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

m10 = json.load(open(OUT / "17_perm_meta_10000.json", encoding="utf-8"))
m2 = json.load(open(OUT / "17_perm_meta_2000.json", encoding="utf-8"))
chk = json.load(open(OUT / "18_perm_check.json", encoding="utf-8"))
gap = pd.read_csv(OUT / "table4_gap_summary.csv")
bhc_meta = json.load(open(OUT / "16_bhc_perf_meta.json", encoding="utf-8"))

sweep = pd.read_csv(OUT / "tableA_bhc_sweep.csv")
_sw = sweep[(sweep["rep"] == "CB_masked") & (sweep["k"] == 15)].iloc[0]
ari_bhc_dx = float(_sw["ARI_C1id15"])
eta2_drug_bhc = float(_sw["eta2_drug_count"])

dx_row = gap[(gap.partition == "C1_id") & (gap.k == 15)].iloc[0]
T10 = {(r["partition"], r["predictor"], r["statistic"]): r for r in m10["table"]}
T2 = {(r["partition"], r["predictor"], r["statistic"]): r for r in m2["table"]}
P_DX, P_BHC = "diagnose C1_id k=15", "BHC CB_masked k=15"
FLOOR10 = 1 / 10001


def pf(p):
    return "**≤ 0.0001**" if p <= 1.5 * FLOOR10 else (f"**{p:.4f}**" if p < 0.05 else f"{p:.4f}")


def sig(p):
    return "유의" if p < 0.05 else "비유의"


L = []
A = L.append
A("# §9 클러스터별 성능 격차 순열검정")
A("")
A(f"환자 단위 순열 **{m10['n_perm']:,}회** (seed {m10['seed']}, 경계 p값 재확인용 상향 — 1차 {m2['n_perm']:,}회 결과는 §5). "
  f"스크립트: `14_permutation.py` `15_perm_check.py` `16_perm_report.py`")
A("")
A("## 1. 순열 설계")
A("")
A("| 요건 | 구현 |")
A("|---|---|")
A("| 환자 단위 셔플 | 한 환자의 모든 방문을 한 덩어리로 같은 무작위 클러스터에 배정 (train+test 전체 우주에서 배정 후 test 만 평가) |")
A("| 크기 분포 보존 | 관측 파티션의 클러스터별 **환자 수** 분포를 최대잉여법으로 정수 배분해 그대로 사용 |")
A("| 동일 필터 | 귀무 표본에도 `클러스터 test 방문 ≥30` 을 적용한 뒤 격차 계산 |")
A("| copy-prev 부분집합 | 관측과 동일하게 클러스터 내 '직전 방문 있는 방문'만으로 평균 계산 (필터는 test 방문 수 기준) |")
A("")
for p in m10["partitions"]:
    A(f"- **{p['partition']}**: 우주 {p['universe_visits']:,}방문 / {p['universe_subjects']:,}명, "
      f"test {p['test_visits']:,}방문 (직전 방문 있음 {p['test_visits_with_prev']:,}), 클러스터 {p['n_clusters']}개. "
      f"≥30 통과 클러스터 = 관측 {p['n_clusters_ge30_observed']}개 / 귀무 평균 {p['n_clusters_ge30_null_mean']}개 (최소 {p['n_clusters_ge30_null_min']}).")
A("")
A("> ≥30 필터는 두 파티션 모두에서 실질적으로 구속하지 않았다(클러스터당 test 방문이 평균 ~190개라 "
  "모든 순열에서 15개 클러스터 전부 통과). 필터는 적용했으나 결과를 좌우하지 않는다.")
A("")
A("## 2. 결과 (8개 p값)")
A("")
A("p는 단측 = (1 + #{귀무 ≥ 관측}) / (1 + 순열수). 하한은 1/10,001 ≈ 0.0001.")
A("")
A("| 파티션 | 예측기 | 통계 | 관측 | 귀무 평균 | 귀무 95%p | p값 | z | 판정 |")
A("|---|---|---|---|---|---|---|---|---|")
for part in [P_DX, P_BHC]:
    for pred in ["상수", "copy-prev"]:
        for stat in ["범위(max-min)", "가중 SD"]:
            r = T10[(part, pred, stat)]
            A(f"| {part} | {pred} | {stat} | **{r['observed']:.4f}** | {r['null_mean']:.4f} | "
              f"{r['null_p95']:.4f} | {pf(r['p_value'])} | {r['z_vs_null']:+.1f} | {sig(r['p_value'])} |")
A("")
A("## 3. 그림")
A("")
A("| 파일 | 내용 |")
A("|---|---|")
A("| `figs/fig9_perm_dx_const.png` | diagnose C1_id k=15 · 상수 — 범위/가중SD 귀무 히스토그램 + 관측 수직선 |")
A("| `figs/fig10_perm_dx_prev.png` | diagnose C1_id k=15 · copy-prev |")
A("| `figs/fig11_perm_bhc_const.png` | BHC CB_masked k=15 · 상수 |")
A("| `figs/fig12_perm_bhc_prev.png` | BHC CB_masked k=15 · copy-prev |")
A("")
A("## 4. 결론")
A("")
r1r, r1s = T10[(P_DX, "상수", "범위(max-min)")], T10[(P_DX, "상수", "가중 SD")]
q1r, q1s = T10[(P_DX, "copy-prev", "범위(max-min)")], T10[(P_DX, "copy-prev", "가중 SD")]
r2r, r2s = T10[(P_BHC, "상수", "범위(max-min)")], T10[(P_BHC, "상수", "가중 SD")]
q2r, q2s = T10[(P_BHC, "copy-prev", "범위(max-min)")], T10[(P_BHC, "copy-prev", "가중 SD")]
A(f"1. **diagnose 클러스터 격차는 우연 수준이 아니다** — 상수 예측기 격차 {r1r['observed']:.4f}는 "
  f"귀무 평균 {r1r['null_mean']:.4f}·95백분위 {r1r['null_p95']:.4f}를 크게 넘는다(p ≤ 0.0001, z={r1r['z_vs_null']:+.1f}; "
  f"가중 SD도 동일: {r1s['observed']:.4f} vs {r1s['null_mean']:.4f}, p ≤ 0.0001). "
  f"다만 copy-prev 격차 {q1r['observed']:.4f}은 귀무 평균 {q1r['null_mean']:.4f}보다도 **낮아** 완전한 우연 수준이다(p={q1r['p_value']:.2f}).")
A(f"2. **BHC 클러스터 격차도 우연 수준이 아니다** — 상수 예측기 격차 {r2r['observed']:.4f}, "
  f"귀무 평균 {r2r['null_mean']:.4f}, p ≤ 0.0001, z={r2r['z_vs_null']:+.1f} (가중 SD {r2s['observed']:.4f}, p ≤ 0.0001). "
  f"copy-prev 격차 {q2r['observed']:.4f}는 p={q2r['p_value']:.4f}(가중 SD p={q2s['p_value']:.4f})로 α=0.05 기준으로는 유의하지만, "
  f"8개 검정 Bonferroni 기준(0.00625)은 통과하지 못한다.")
A(f"3. **네 갈래 중 「둘 다 유의」** — 성능 이질성은 실재하고 여러 축에서 잡힌다. "
  f"진단 축(C1_id)과 진단과 거의 겹치지 않는 축(BHC, ARI {ari_bhc_dx:.4f}) 양쪽에서 "
  f"상수 예측기 격차가 귀무분포를 {r1r['z_vs_null']:.1f}~{r2s['z_vs_null']:.1f} 표준편차 넘어섰다. "
  f"나머지 세 갈래(진단만 유의 / BHC만 유의 / 둘 다 비유의)는 배제된다.")
A("")
A("## 5. 검증과 한계")
A("")
A("### 5-1. 순열 설계 타당성 (`18_perm_check.json`, diagnose C1_id k=15)")
A("")
A("방문 단위 셔플과 비교해 환자 단위 셔플이 귀무를 실제로 넓히는지 확인했다.")
A("")
A("| 통계 | 관측 | 환자셔플 귀무평균(SD) | p | 방문셔플 귀무평균(SD) | p | 귀무 SD 비 |")
A("|---|---|---|---|---|---|---|")
NM = {"const_range": "상수·범위", "const_wsd": "상수·가중SD", "prev_range": "copy-prev·범위", "prev_wsd": "copy-prev·가중SD"}
for k, v in chk["by_stat"].items():
    A(f"| {NM[k]} | {v['observed']:.4f} | {v['patient_shuffle_null_mean']:.4f} ({v['patient_shuffle_null_sd']:.4f}) | "
      f"{v['patient_shuffle_p']:.4f} | {v['visit_shuffle_null_mean']:.4f} ({v['visit_shuffle_null_sd']:.4f}) | "
      f"{v['visit_shuffle_p']:.4f} | {v['null_sd_ratio_patient_over_visit']:.2f}× |")
A("")
A(f"환자 단위 셔플의 귀무 SD가 방문 단위의 **1.19~1.25배**다. 방문 단위로 섞었다면 copy-prev·범위의 p가 "
  f"{chk['by_stat']['prev_range']['visit_shuffle_p']:.2f}로 나와 {chk['by_stat']['prev_range']['patient_shuffle_p']:.2f} 대신 "
  f"'경계 유의'처럼 보였을 것이다 — 환자 단위 선택이 실제로 결과를 바꿨다. "
  f"방문셔플 귀무평균({chk['by_stat']['const_range']['visit_shuffle_null_mean']:.4f})은 iid 근사 "
  f"{chk['iid_approx']['const_range_iid_approx']:.4f}와 거의 일치해, 구현이 의도대로 동작함을 뒷받침한다.")
A("")
A("### 5-2. 한계 (해석에 영향을 주는 기록 사항)")
A("")
A(f"- **상수 예측기 격차는 처방 수와 연동된다.** 클러스터 평균 처방 수와 `jac_const`의 상관은 "
  f"BHC {bhc_meta['corr_jacconst_vs_ndrug']}, diagnose C1_id k=15 {dx_row['corr_jacconst_vs_ndrug']}. "
  f"상수 예측기 Jaccard는 예측 집합이 고정(23개)이라 정답 집합 크기에 기계적으로 연동된다. "
  f"순열검정이 말해주는 것은 '클러스터가 성능과 무관하지 않다'까지이고, 그 축이 무엇인지는 말해주지 않는다.")
A(f"- **copy-prev 쪽 상관은 낮다** (BHC {bhc_meta['corr_jacprev_vs_ndrug']}, diagnose {dx_row['corr_jacprev_vs_ndrug']}). "
  f"그리고 copy-prev 격차는 diagnose에서 비유의, BHC에서 약한 유의로 두 예측기의 결론이 갈린다.")
A(f"- **다중비교.** 8개 검정을 했다. 상수 예측기 4개(p ≤ 0.0001)는 Bonferroni(0.00625)를 통과하고, "
  f"BHC copy-prev 2개(p={q2r['p_value']:.4f}, {q2s['p_value']:.4f})는 통과하지 못한다.")
A(f"- **귀무가 관측보다 환자 내 상관이 크다 → p값은 보수적이다.** 관측 파티션은 한 환자의 방문을 여러 클러스터에 쪼갠다"
  f"(diagnose {chk['subjects_spanning_multiple_observed_clusters']:,}명 / {chk['n_subjects']:,}명, "
  f"BHC {m10['partitions'][1]['subjects_spanning_multiple_clusters']:,}명). "
  f"귀무는 환자를 통째로 배정하므로 클러스터 내 상관이 더 크고 귀무분포가 더 넓다. 이 방향의 편의는 p값을 크게 만든다.")
A(f"- **격차는 여전히 작다.** 유의성과 크기는 별개다. 상수 예측기 전체 평균은 0.2762이고 클러스터 간 가중 SD는 "
  f"{r2s['observed']:.4f}(BHC) / {r1s['observed']:.4f}(diagnose) 수준이다.")
A("")
A("### 5-3. 순열 수 2,000 → 10,000 재확인")
A("")
A("§4 지침대로 1차 2,000회에서 경계(0.01~0.10)에 걸린 값을 10,000회로 재확인했다.")
A("")
A("| 파티션 · 예측기 · 통계 | p (2,000) | p (10,000) |")
A("|---|---|---|")
for part in [P_DX, P_BHC]:
    for pred in ["상수", "copy-prev"]:
        for stat in ["범위(max-min)", "가중 SD"]:
            a, b = T2[(part, pred, stat)], T10[(part, pred, stat)]
            mark = " ←경계 재확인" if 0.01 <= a["p_value"] <= 0.10 else ""
            A(f"| {part} · {pred} · {stat} | {a['p_value']:.4f} | {b['p_value']:.4f}{mark} |")
A("")
A("BHC copy-prev 두 통계가 경계였고, 10,000회에서도 각각 "
  f"{q2r['p_value']:.4f} / {q2s['p_value']:.4f}로 유지됐다(2,000회 대비 변화 미미). 결론은 바뀌지 않는다.")
A("")
A("## 산출물")
A("")
A("`table5_perm_10000.csv` · `table5_perm_2000.csv` · `17_perm_meta_{2000,10000}.json` · "
  "`18_perm_check.json` · `perm_null_{2000,10000}.npz` · `figs/fig9~12_perm_*.png`")

(OUT / "REPORT_PERM.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
