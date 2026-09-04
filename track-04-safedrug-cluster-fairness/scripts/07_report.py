"""§10 산출물: 표 1/2 + 한 장 요약을 out/REPORT.md 로 조립."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def md(df, cols=None, rename=None, floatfmt="{:.4g}"):
    d = df[cols] if cols else df
    if rename:
        d = d.rename(columns=rename)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("—" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


t1 = pd.read_csv(OUT / "table1_partition_metrics.csv")
t2 = pd.read_csv(OUT / "table2_cluster_performance.csv")
t3 = pd.read_csv(OUT / "table3_power.csv")
t3b = pd.read_csv(OUT / "table3b_power_prev.csv")
t4 = pd.read_csv(OUT / "table4_gap_summary.csv")
meta = json.load(open(OUT / "11_cluster_meta.json", encoding="utf-8"))
pmeta = json.load(open(OUT / "12_power_meta.json", encoding="utf-8"))

T1COLS = ["partition", "k", "n_clusters", "coverage_pct", "min_size", "max_size", "gini_size",
          "silhouette", "eta2_diag_count", "eta2_drug_count"]
T1REN = {"partition": "분할", "n_clusters": "클러스터", "coverage_pct": "커버리지%",
         "min_size": "최소", "max_size": "최대", "gini_size": "지니", "silhouette": "실루엣",
         "eta2_diag_count": "η²(진단수)", "eta2_drug_count": "η²(약물수)"}

t1p = t1[t1.partition.str.startswith("P1")]
t1c = t1[(~t1.partition.str.startswith("P1")) & (t1.k.isin([5, 10, 15, 20, 25, 30]))]

T2COLS = ["cluster", "test_visits", "mean_n_diag", "mean_n_drug", "jac_const", "ci_const",
          "jac_prev", "n_prev", "ci_prev", "top3_dx"]
T2REN = {"cluster": "클러스터", "test_visits": "test 방문", "mean_n_diag": "평균진단",
         "mean_n_drug": "평균약물", "jac_const": "상수 Jac", "ci_const": "95% CI",
         "jac_prev": "copy-prev Jac", "n_prev": "n(prev)", "ci_prev": "95% CI ", "top3_dx": "상위 진단 3개"}


def t2block(pn, k, min_visits=30):
    s = t2[(t2.partition == pn) & (t2.k == k) & (t2.test_visits >= min_visits)].copy()
    s = s.sort_values("jac_const", ascending=False)
    s["ci_const"] = s.apply(lambda r: f"[{r.jac_const_lo:.3f}, {r.jac_const_hi:.3f}]", axis=1)
    s["ci_prev"] = s.apply(
        lambda r: "—" if pd.isna(r.jac_prev) else f"[{r.jac_prev_lo:.3f}, {r.jac_prev_hi:.3f}]", axis=1)
    s["cluster"] = s["cluster"].map(lambda c: "기타" if c == -1 else f"C{c}")
    return md(s, T2COLS, T2REN, "{:.3f}")


kmax_const = pmeta["kmax_by_threshold"]
kmax_prev = {}
for thr in [0.05, 0.08, 0.10]:
    kmax_prev[f"MDD<={thr}"] = {
        pn: (int(t3b[(t3b.partition == pn) & (t3b.worst_mdd_prev <= thr)].k.max())
             if len(t3b[(t3b.partition == pn) & (t3b.worst_mdd_prev <= thr)]) else None)
        for pn in ["C1_id", "C2_id", "C1_name", "C2_name"]
    }

bl = pmeta["baseline"]
sp = pmeta["split"]

doc = f"""# diagnose 기반 방문 군집화 — 금요일 논의 자료

데이터 `data4LLM_with_note.csv` · 방문 14,541 / 환자 6,310 · 노트 임베딩 없음(진단 코드만)
스크립트 `scripts/02_gate.py` → `03_cluster.py` → `04_power_baseline.py` → `06_figures.py`

---

## 한 장 요약

**1. 클러스터는 나온다. 단 "주 진단"이 아니라 "동반질환 프로파일"로 갈린다.**
k-means k=15의 15개 클러스터 중 13개가 특정 진단을 100% 가까이 공유한다 — 그런데 그 진단은
Hyperlipidemia(100%), Hypertension(100%), Atrial Fibrillation(100%), CKD(92%), GERD 처럼
**만성 동반질환**이지 입원 사유가 아니다. 첫 진단(주 진단) 라벨과는 축이 다르다.
실제로 첫 진단 분할 P1은 이 벡터공간에서 실루엣이 **음수**(id -0.044 / name -0.046)다 —
진단 코드 벡터공간은 주 진단을 기하학적으로 분리하지 않는다.

**2. 진단 개수가 클러스터를 상당 부분 설명한다.** k=15에서 η²(진단수) = {t1[(t1.partition=='C1_id')&(t1.k==15)].eta2_diag_count.iloc[0]:.3f},
k=30에서 {t1[(t1.partition=='C1_id')&(t1.k==30)].eta2_diag_count.iloc[0]:.3f}. C7(n=1,552)은 평균 진단 8.2개에 지배 진단이 없는
"진단 적은 방문" 덩어리다. 즉 클러스터 축의 일부는 질환이 아니라 **기록 복잡도**다.

**3. 클러스터별 성능 격차는 실재한다. 그런데 두 baseline이 서로 다른 이야기를 한다.**
- 상수 top-23 예측기: P1 클러스터 간 격차 0.206→0.346 (**0.140**). 그러나 클러스터 평균 약물 수와의
  상관이 **r = +0.80** — 이 격차는 사실상 "약을 몇 개 받았나"를 재고 있다. §5 경고가 그대로 실현됨.
- copy-previous 예측기: 격차 0.268→0.475 (**0.207**), 약물 수와의 상관 **r = −0.03**.
  **이쪽은 약물 개수로 설명되지 않는 진짜 진단군 차이다.**

**4. 통계적으로 가능한 k 상한은 12~15다.** copy-previous 기준(가장 보수적, test n=1,618)
MDD ≤ 0.05를 만족하는 최대 k는 k-means {kmax_prev['MDD<=0.05']['C1_id']}(id) / {kmax_prev['MDD<=0.05']['C1_name']}(name), Ward {kmax_prev['MDD<=0.05']['C2_id']}(id).
관측된 C-분할 격차가 0.059 수준이므로 **k=15를 넘기면 격차를 유의하게 보여줄 수 없다.**

**5. id 벡터화와 name 벡터화의 차이는 유의하지 않다.** 두 결과의 ARI는 k=10에서 0.506, k=25에서 0.367인데,
**같은 벡터화에서 seed만 바꿨을 때의 ARI가 k=10에서 평균 0.687(최소 0.509), k=25에서 0.507**이다.
즉 벡터화 차이가 seed 흔들림보다 크지 않다. 실루엣도 사실상 동일(0.0426 vs 0.0426 @k=15).
**이후 단계는 id 버전으로 간다** — 성능이 나아서가 아니라 원 코드에 가깝고 해석이 명확해서다.

---

## 표 1. 분할 × k — 구조와 교란

**주의: P1과 C1/C2는 커버리지가 다르다.** P1_ge100은 방문의 36.5%만 실라벨로 덮고 나머지 63.5%가
"기타"에 들어간다(P1_ge30은 66.4% / 33.6%). C1/C2는 정의상 100%를 덮는다. 아래 지표를 나란히 읽을 때
이 비대칭을 반드시 감안해야 한다 — **커버리지 자체가 결과다.**

### P1 (첫 진단 라벨 직접 분할)

{md(t1p, T1COLS, T1REN)}

*P1의 실루엣은 벡터공간에 의존하지 않는 분할이라 표에서 비움. 참고로 id 공간 {meta['p1_silhouette_reference']['P1_ge100']['id']} /
name 공간 {meta['p1_silhouette_reference']['P1_ge100']['name']} (P1_ge100), 즉 **음수**다.*

### C1(k-means) / C2(Ward)

{md(t1c, T1COLS, T1REN)}

읽는 법:
- **실루엣이 전부 0.04 이하**다. 진단 multi-hot 공간에는 뚜렷한 밀도 분리가 없다. k 선택을 실루엣으로 할 수 없다.
- **η²(진단수)가 η²(약물수)보다 일관되게 크다.** k=30 k-means에서 진단수 분산의 25%를 클러스터가 설명한다.
- Ward(C2)는 k≥15부터 **n=64짜리 미세 클러스터를 계속 떼어내고 더 이상 쪼개지 않는다**(최소 크기가 k=15~30 내내 64로 고정). 지니계수도 k-means(0.16~0.24)보다 높다(0.31~0.43). 검정력 관점에서 Ward가 불리하다.

---

## 표 2. 클러스터별 성능 — 자명 baseline

분할: 환자 단위 8:2 (train {sp['train_visits']:,}방문/{sp['train_subjects']:,}명, test {sp['test_visits']:,}방문/{sp['test_subjects']:,}명, 겹치는 환자 {sp['leakage_subjects_in_both']}명).
상수 예측기 = train 최빈 약물 **{bl['K_const']}개**(= train 평균 약물 수)를 모든 방문에 예측.
copy-previous = 같은 환자 직전 방문 처방 그대로. 전체 test 중 직전 방문 있는 방문 {bl['test_visits_with_prev']:,}건.
전체 평균: 상수 **{bl['overall_jac_const']}**, copy-previous **{bl['overall_jac_prev']}**.
CI는 환자 단위 부트스트랩 1,000회. test 방문 30건 미만 클러스터는 표에서 제외(전체는 `table2_cluster_performance.csv`).

### 2-A. P1_ge100 (첫 진단 라벨)

{t2block('P1_ge100', 0)}

### 2-B. C1_id k=15 (k-means)

{t2block('C1_id', 15)}

---

## 표 3. 클러스터 간 격차와 약물 개수 교란

{md(t4[t4.k.isin([0,10,15,20,25])], ['partition','k','n_clusters_ge30_test','jac_const_spread','corr_jacconst_vs_ndrug','jac_prev_spread','corr_jacprev_vs_ndrug'],
   {'partition':'분할','n_clusters_ge30_test':'비교가능 클러스터','jac_const_spread':'상수 격차','corr_jacconst_vs_ndrug':'상수 격차↔약물수 r','jac_prev_spread':'copy-prev 격차','corr_jacprev_vs_ndrug':'copy-prev 격차↔약물수 r'}, '{:.3f}')}

**⚠ 경고문 (§5 요구사항)**: 상수 top-k baseline의 클러스터별 성능은 클러스터 평균 약물 수와
**r = +0.80**(P1 기준)으로 강하게 연동된다. Jaccard가 정답 집합 크기에 기계적으로 좌우되기 때문이다.
학습 모델로 넘어갔을 때 나올 "클러스터별 성능 차이"도 같은 방식으로 오염될 수 있으므로,
**약물 개수를 통제하지 않은 클러스터별 성능 비교는 fairness 근거로 쓸 수 없다.**
반면 copy-previous에서는 이 상관이 사라진다(r = −0.03) — 격차의 성격이 다르다는 뜻이고,
fairness 서사는 copy-previous 쪽 격차 위에 세워야 방어된다.

---

## 표 4. 검정력 — k 상한 역산 (§6, 금요일 핵심 숫자)

α=.05 양측, power=.80, 두 독립 평균 비교. MDD = 2.80 × sd × √(1/n₁+1/n₂).
sd는 test 방문별 Jaccard의 표준편차 (상수 {bl['sd_jac_const']}, copy-previous {bl['sd_jac_prev']}).
"worst"는 **가장 작은 클러스터를 나머지 전체와 비교**할 때의 MDD.

### 상수 top-23 기준 (test n = {bl['test_visits']:,})

{md(t3[(~t3.partition.str.startswith('P1')) & (t3.k.isin([5,10,15,20,25,30]))],
   ['partition','k','test_min_cluster','n_clusters_lt_30','worst_mdd'],
   {'partition':'분할','test_min_cluster':'test 최소 클러스터','n_clusters_lt_30':'test<30인 클러스터','worst_mdd':'worst MDD'}, '{:.4f}')}

### copy-previous 기준 (test n = {bl['test_visits_with_prev']:,}) — 더 보수적

{md(t3b[t3b.k.isin([5,10,15,20,25,30])], ['partition','k','min_cluster','worst_mdd_prev'],
   {'partition':'분할','min_cluster':'test 최소 클러스터','worst_mdd_prev':'worst MDD'}, '{:.4f}')}

### 임계값별 최대 k

| 기준 | 임계 MDD | C1_id | C1_name | C2_id | C2_name |
|---|---|---|---|---|---|
| 상수 top-23 | 0.03 | {kmax_const['MDD<=0.03']['C1_id']} | {kmax_const['MDD<=0.03']['C1_name']} | {kmax_const['MDD<=0.03']['C2_id']} | {kmax_const['MDD<=0.03']['C2_name']} |
| 상수 top-23 | 0.05 | {kmax_const['MDD<=0.05']['C1_id']} | {kmax_const['MDD<=0.05']['C1_name']} | {kmax_const['MDD<=0.05']['C2_id']} | {kmax_const['MDD<=0.05']['C2_name']} |
| copy-previous | 0.05 | {kmax_prev['MDD<=0.05']['C1_id']} | {kmax_prev['MDD<=0.05']['C1_name']} | {kmax_prev['MDD<=0.05']['C2_id']} | {kmax_prev['MDD<=0.05']['C2_name']} |
| copy-previous | 0.08 | {kmax_prev['MDD<=0.08']['C1_id']} | {kmax_prev['MDD<=0.08']['C1_name']} | {kmax_prev['MDD<=0.08']['C2_id']} | {kmax_prev['MDD<=0.08']['C2_name']} |

**결론: k ≤ 15.** copy-previous 기준으로 관측 격차(0.059)를 유의하게 잡으려면 k-means k≤12~13,
상수 기준으로도 k≤15에서 test 최소 클러스터가 115~120건이라 안정적이다. k=20을 넘으면 꼬리 클러스터가
60건 아래로 떨어지고, k=25에서는 12건짜리 클러스터가 나와 MDD가 0.075로 뛴다.
**P1_ge30은 104개 라벨 중 88개가 test 30건 미만이라 클러스터별 성능 비교가 불가능하다.**

---

## 안정성 (참고)

{md(pd.DataFrame([{'설정': k, '평균 쌍별 ARI': v['mean_pairwise_ARI'], '최소': v['min']} for k, v in meta['kmeans_seed_stability'].items()]), None, None, '{:.3f}')}

seed 5개 재군집 결과. k가 커질수록 안정성이 떨어진다(k=25에서 0.51). k≤15 권고의 두 번째 근거.

---

## 그림

| 파일 | 내용 |
|---|---|
| `figs/fig1_svd_clusters.png` | SVD 2차원, k-means k=15 클러스터별 위치 (소형 다중 패널) |
| `figs/fig2_cluster_jaccard_P1.png` | P1 첫 진단 라벨별 baseline 성능 + 95% CI |
| `figs/fig3_cluster_jaccard_C1.png` | C1_id k=15 클러스터별 baseline 성능 + 95% CI |
| `figs/fig4_confound_drugcount.png` | 성능 vs 약물 개수 — 교란 점검 |
| `figs/fig5_power_vs_k.png` | k별 MDD 곡선 — k 상한 |

---

## 노트 트랙 연결 (§8)

- `out/cluster_assignments.csv` — `SUBJECT_ID, HADM_ID, partition, k, cluster_id` (전 분할·전 k).
- `scripts/03_cluster.py`의 `describe_partition(labels, Z, ...)`, `scripts/04_power_baseline.py`의
  `evaluate_partition(partition, k, labels)` 는 **라벨 벡터/행렬만 받는다.** 노트 임베딩 Z가 생기면
  `embed()` 결과를 바꿔 끼우기만 하면 같은 표가 나온다.
- 노트 군집이 나오면 평가는 **"이 diagnose 군집을 기준 파티션으로 두고 ARI"** 로 한다.
  diagnose 군집을 진단 라벨로 평가하는 것은 동어반복이므로 하지 않았다.
"""

(OUT / "REPORT.md").write_text(doc, encoding="utf-8")
print(f"written: {OUT/'REPORT.md'}  ({len(doc):,} chars)")
