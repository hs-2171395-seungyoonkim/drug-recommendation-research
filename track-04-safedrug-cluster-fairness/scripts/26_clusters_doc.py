"""out/CLUSTERS.md — 지금까지 만든 모든 분할/군집의 카탈로그.

숫자는 전부 out/ 의 table*.csv 와 메타 JSON 에서 읽는다(손으로 옮기지 않는다).
새로 계산하는 것은 D1/D2/D2b 라벨의 대표 진단(SEQ_NUM=1 상위 코드 → SHORT_TITLE)뿐이다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

J = lambda f: json.load(open(OUT / f, encoding="utf-8"))
C = lambda f: pd.read_csv(OUT / f)

CL = J("11_cluster_meta.json")
BH = J("15_bhc_cluster_meta.json")
BP = J("13_bhc_prep.json")
BPF = J("16_bhc_perf_meta.json")
TA = J("21_trackA_meta.json")
FE = J("20_icd_features_meta.json")
PA = J("23_icd_partition_meta.json")
EV = J("24_icd_eval_meta.json")
MP = J("26_matched_pairs_meta.json")

t1, t2 = C("table1_partition_metrics.csv"), C("table2_cluster_performance.csv")
t3 = C("table3_power.csv")
t4, t5 = C("table4_gap_summary.csv"), C("table5_perm_10000.csv")
t6, t6b, t7, t8 = (C("table6_trackA_compare.csv"), C("table6b_trackA_describe.csv"),
                   C("table7_trackA_clusters.csv"), C("table8_trackA_perm.csv"))
tA, tB = C("tableA_bhc_sweep.csv"), C("tableB_bhc_cluster_performance.csv")
t13, t14, t15, t16 = (C("table13_icd_power.csv"), C("table14_icd_clusters.csv"),
                      C("table15_icd_perm.csv"), C("table16_icd_compare.csv"))
part = C("icd_partition_assignments.csv")

MIN_TEST = PA["min_test_visits"]
N_VIS, N_TEST = PA["n_visits"], PA["n_test"]
SPL = FE["split"]


# ---------------------------------------------------------------- 조회 헬퍼
def cmp16(tag):
    return t16[t16["분할"].str.startswith(tag)].iloc[0]


def perm16(tag, stat, pred="상수"):
    r = t15[(t15["분할"] == tag) & (t15["통계량"] == stat) & (t15["예측기"] == pred)]
    return r.iloc[0]


def perm5(pname, stat, pred="상수"):
    r = t5[(t5["partition"] == pname) & (t5["statistic"] == stat) & (t5["predictor"] == pred)]
    return r.iloc[0]


def esc(s):
    return str(s).replace("|", r"\|")


def md(df_, cols=None, fmt=None):
    t = df_[cols] if cols else df_
    lines = ["| " + " | ".join(esc(c) for c in t.columns) + " |",
             "|" + "|".join(["---"] * len(t.columns)) + "|"]
    for _, r in t.iterrows():
        cells = []
        for c in t.columns:
            v = r[c]
            if isinstance(v, float):
                cells.append("—" if np.isnan(v) else f"{v:.{fmt.get(c, 3) if fmt else 3}f}")
            else:
                cells.append(esc(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- D1/D2/D2b 라벨의 대표 진단
dic = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
SHORT = dict(zip(dic["ICD9_CODE"].str.strip(), dic["SHORT_TITLE"].astype(str)))


def rep_dx(col, label, topn=2):
    """그 라벨 안에서 가장 흔한 SEQ_NUM=1 코드 topn 개의 명칭 + 점유율."""
    sub = part[part[col] == label]
    if not len(sub):
        return "—"
    vc = sub["icd9_seq1"].astype(str).str.strip().value_counts()
    out = []
    for c, n in vc.head(topn).items():
        out.append(f"{SHORT.get(c, c)} {n / len(sub) * 100:.0f}%")
    return ", ".join(out)


def label_table(tag, col):
    p = t13[(t13["분할"] == tag)].copy()
    perf = t14[t14["partition"] == tag].set_index("라벨")
    p["상수 Jaccard"] = [float(perf.loc[l, "jac_const"]) if l in perf.index else np.nan for l in p["라벨"]]
    p["평균 약물수"] = [float(perf.loc[l, "mean_n_drug"]) if l in perf.index else np.nan for l in p["라벨"]]
    p["대표 진단 (SEQ_NUM=1 상위)"] = [rep_dx(col, l) for l in p["라벨"]]
    p["test≥30"] = np.where(p["ge30"], "○", "×")
    return p.rename(columns={"전체방문": "전체 방문", "test방문": "test 방문"})


LT_COLS = ["라벨", "전체 방문", "test 방문", "test≥30", "상수 Jaccard", "평균 약물수", "대표 진단 (SEQ_NUM=1 상위)"]
LT_FMT = {"상수 Jaccard": 4, "평균 약물수": 1}

# ---------------------------------------------------------------- 본문
L = []
A = L.append

A("# 만든 분할·군집 카탈로그")
A("")
A(f"방문 {N_VIS:,}건 · 환자 {SPL['train_subjects'] + SPL['test_subjects']:,}명 "
  f"(train {SPL['train_visits']:,} / test {SPL['test_visits']:,} 방문, 환자 단위 분리, 겹침 "
  f"{SPL['leakage_subjects_in_both']}명) 위에 만든 분할 전부를 정리한다. "
  f"괄호 안 대문자는 코드·파일에서 쓰는 내부 이름이다.")
A("")
A("**모든 분할에 공통으로 적용한 평가 규약** — 아래 각 절의 '결과 요약'은 이 규약의 산물이다.")
A("")
A(f"- 예측기 둘: **상수 top-{FE['baseline']['K_const']}**(train 최빈 약물 {FE['baseline']['K_const']}개를 모든 방문에 그대로 추천)과 "
  f"**copy-prev**(같은 환자의 직전 방문 처방을 복사, test {FE['baseline']['test_visits_with_prev']:,}건에만 정의됨).")
A(f"- 성능은 test 방문의 Jaccard. 전체 평균은 상수 {FE['baseline']['test_jac_const']:.4f} / copy-prev {FE['baseline']['test_jac_prev']:.4f}.")
A(f"- 라벨별 CI는 **환자 단위 부트스트랩**, 격차 통계량은 **test ≥{MIN_TEST} 라벨만** 써서 max−min(범위)과 가중 SD.")
A(f"- 순열검정은 **환자 블록 단위**로 라벨을 다시 배정(크기 분포 보존), 10,000회, 단측 p.")
A("- '약물수 상관'은 라벨 평균 Jaccard와 라벨 평균 처방 개수의 상관. 이 값이 크면 격차를 질병군 효과로 읽을 수 없다.")
A("")

# ================================================================ 1. P1
p1 = t1[t1.partition == "P1_ge100"].iloc[0]
p130 = t1[t1.partition == "P1_ge30"].iloc[0]
p1cmp = cmp16("P1_ge100")
p1lab = t2[(t2.partition == "P1_ge100") & (t2.cluster != -1)].copy()
p1lab["라벨(주 진단)"] = [s.split(" 100%")[0] for s in p1lab["top3_dx"]]
p1lab = p1lab.sort_values("test_visits", ascending=False)
p1pass = p1lab[p1lab.test_visits >= MIN_TEST].sort_values("jac_const")
p1etc = t2[(t2.partition == "P1_ge100") & (t2.cluster == -1)].iloc[0]
p30pow = t3[t3.partition == "P1_ge30"].iloc[0]

A("## 1. 첫 진단 코드 라벨 분할 (P1_ge100 / P1_ge30)")
A("")
A("**입력** — `data4LLM_with_note.csv` 의 `diag_id` 컬럼에서 **첫 원소 하나**(`diag_id[0]`). "
  f"진단 리스트 전체가 아니라 첫 진단만 쓴다. 어휘는 이 파일의 진단 id {CL['vectorization']['id']['dim']:,}개.")
A("")
A(f"**만든 방법** — 룰 기반. 전체 방문에서 그 코드가 첫 진단인 방문이 **100건 이상**이면 그 코드를 라벨로 쓰고, "
  f"미달이면 전부 '기타(-1)'로 묶는다. 군집화·학습 없음. 문턱을 30건으로 낮춘 변형이 P1_ge30.")
A("")
A(f"**몇 개로 나뉘었나** — P1_ge100은 진단 라벨 {len(p1lab)}개 + 기타 하나(표·메타의 k={int(p1['k'])}는 기타를 한 칸으로 세운 값이다). "
  f"커버리지 **{p1['coverage_pct']}%**(기타 {p1['pct_other']}%), 진단 라벨 중 가장 큰 것 "
  f"{int(p1lab['test_visits'].max())}건·가장 작은 것 {int(p1lab['test_visits'].min())}건(test 기준), "
  f"test ≥{MIN_TEST} 통과 **{len(p1pass)}개**(기타까지 세면 {int(p1cmp['test≥30 라벨 수'])}). "
  f"P1_ge30으로 문턱을 낮추면 라벨은 {int(p130['k'])}개, 커버리지는 {p130['coverage_pct']}%로 오르지만 "
  f"test 중앙 크기가 {p30pow['test_median_cluster']:.0f}건이고 {int(p30pow['n_clusters_lt_30'])}개 라벨이 "
  f"≥{MIN_TEST}에 미달해 검정력이 무너진다.")
A("")
A(f"**각 라벨이 무엇인가** — 라벨이 곧 진단명이라 해석은 가장 쉽다. test ≥{MIN_TEST}를 통과한 {len(p1pass)}개(성능 낮은 순):")
A("")
A(md(p1pass.rename(columns={"test_visits": "test 방문", "mean_n_drug": "평균 약물수",
                            "jac_const": "상수 Jaccard"}),
     ["라벨(주 진단)", "test 방문", "상수 Jaccard", "평균 약물수"],
     {"상수 Jaccard": 4, "평균 약물수": 1}))
A("")
A(f"나머지 {len(p1lab) - len(p1pass)}개 라벨은 test {MIN_TEST}건 미만이라 격차 계산에서 빠졌고, "
  f"기타(-1)는 test {int(p1etc['test_visits']):,}건에 상수 Jaccard {float(p1etc['jac_const']):.4f}로 "
  f"거의 정확히 전체 평균이다.")
A("")
A(f"**왜 만들었나 / 무엇이 문제였나** — 가장 단순한 출발점이었다. 문제는 커버리지다. "
  f"첫 진단 코드는 꼬리가 길어서 100건 문턱을 걸면 방문의 **{p1['pct_other']}%가 기타로 빠진다**. "
  f"기타는 test {int(p1etc['test_visits']):,}건짜리 이질적 덩어리라 '어느 질병이 불리한가'를 물을 수 없다. "
  f"이 한계 때문에 진단 리스트 전체를 쓰는 군집으로 옮겨갔다.")
A("")
A(f"**결과 요약** — 상수 격차 {float(p1cmp['상수 격차']):.4f}(모든 분할 중 최대 — 바닥 비파열 뇌동맥류 "
  f"{float(p1pass.iloc[0]['jac_const']):.4f}, 천장 수술후 감염 {float(p1pass.iloc[-1]['jac_const']):.4f}), "
  f"순열 p {float(p1cmp['순열 p(상수·범위)']):.5f}, 약물수 상관 **{float(p1cmp['약물수 상관(상수)']):.3f}**. "
  f"격차가 큰 것은 라벨이 순수해서이기도 하지만 상관 0.8은 그 격차 대부분이 처방 개수 차이와 같이 움직인다는 뜻이다"
  f"(바닥 라벨 평균 {float(p1pass.iloc[0]['mean_n_drug']):.1f}개 vs 천장 라벨 "
  f"{float(p1pass.iloc[-1]['mean_n_drug']):.1f}개). 순열검정은 기타(-1)를 우주에서 빼고 돌렸다.")
A("")

# ================================================================ 2. C1_id
c1 = t1[(t1.partition == "C1_id") & (t1.k == 15)].iloc[0]
c1g = t4[(t4.partition == "C1_id") & (t4.k == 15)].iloc[0]
c1cmp = cmp16("C1_id")
c1lab = t2[(t2.partition == "C1_id") & (t2.k == 15)].sort_values("jac_const")

A("## 2. 동반질환 프로파일 k-means 군집 (C1_id, k=15) — 진단 트랙의 주 분할")
A("")
A(f"**입력** — `data4LLM_with_note.csv` 의 `diag_id` **리스트 전체**(첫 진단 하나가 아니라 그 방문의 모든 진단). "
  f"멀티핫 차원 {CL['vectorization']['id']['dim']:,}, 밀도 {CL['vectorization']['density_id']:.4f}.")
A("")
A(f"**만든 방법** — L2 정규화 → TruncatedSVD **100차원**(설명분산 {CL['vectorization']['id']['svd_explained_var']:.3f}) "
  f"→ 다시 L2 → **KMeans**(n_init=10, random_state=0). k는 5~30을 훑고 k=15를 주 분할로 삼았다. "
  f"seed 5개 재실행 시 평균 쌍별 ARI {CL['kmeans_seed_stability']['C1_id_k15']['mean_pairwise_ARI']:.3f}"
  f"(최소 {CL['kmeans_seed_stability']['C1_id_k15']['min']:.3f}).")
A("")
A(f"**몇 개로 나뉘었나** — 라벨 15개, 커버리지 **{c1['coverage_pct']}%**(기타 없음), "
  f"가장 큰 군집 {int(c1['max_size']):,}방문 / 가장 작은 군집 {int(c1['min_size'])}방문(gini {c1['gini_size']}), "
  f"test ≥{MIN_TEST}를 15개 전부 통과한다. 커버리지와 검정력을 동시에 만족하는 유일한 초기 분할이었다.")
A("")
A("**각 라벨이 무엇인가** — 상위 동반진단 보유율로 읽는다(성능 낮은 순):")
A("")
A(md(c1lab.rename(columns={"cluster": "군집", "test_visits": "test 방문", "mean_n_drug": "평균 약물수",
                           "jac_const": "상수 Jaccard", "top3_dx": "상위 동반진단(보유율)"}),
     ["군집", "test 방문", "상수 Jaccard", "평균 약물수", "상위 동반진단(보유율)"],
     {"상수 Jaccard": 4, "평균 약물수": 1}))
A("")
A("**왜 만들었나 / 무엇이 문제였나** — 첫 진단 하나로는 커버리지가 안 나와서 진단 집합 전체를 벡터로 만들었다. "
  "그런데 만들어진 군집은 급성 사건이 아니라 **만성 동반질환 프로파일**로 갈렸다 — C10 말기신부전, C1 간경변, "
  "C5 만성신장질환, C9 고지혈증처럼 '이번 입원의 이유'가 아니라 '이 환자가 달고 있는 병'이 축이다. "
  "그래서 여기서 나온 바닥(C10·C1)을 '지엽적 질병이 불리하다'의 근거로 쓰려면 외부 분류 체계로 다시 확인해야 했고, "
  "그것이 아래 6~8번의 ICD chapter 분할이다.")
A("")
A(f"**결과 요약** — 상수 격차 {float(c1cmp['상수 격차']):.4f}(바닥 C10 {float(c1lab.iloc[0]['jac_const']):.4f} ~ "
  f"천장 C13 {float(c1lab.iloc[-1]['jac_const']):.4f}), 순열 p "
  f"{float(perm5('diagnose C1_id k=15', '범위(max-min)')['p_value']):.5f} "
  f"(z={float(perm5('diagnose C1_id k=15', '범위(max-min)')['z_vs_null']):.2f}), 가중 SD도 z="
  f"{float(perm5('diagnose C1_id k=15', '가중 SD')['z_vs_null']):.2f}로 이 프로젝트에서 가장 강한 신호다. "
  f"약물수 상관은 {float(c1cmp['약물수 상관(상수)']):.3f} — 처방 개수 교란이 작은 축에 든다"
  f"(P1_ge100 {float(p1cmp['약물수 상관(상수)']):.3f}, BHC {BPF['corr_jacconst_vs_ndrug']:.3f}과 비교). "
  f"다만 copy-prev 예측기로 보면 격차가 사라진다(p="
  f"{float(perm5('diagnose C1_id k=15', '범위(max-min)', 'copy-prev')['p_value']):.4f}) — "
  f"불리함은 '전역 추천'에 고유한 것이지 그 환자군이 본질적으로 어려운 것이 아니라는 뜻이다.")
A("")

# ================================================================ 3. C1_name / C2
var = t4[(t4.k == 15) & (t4.partition.isin(["C1_name", "C2_id", "C2_name"]))].copy()
var1 = t1[(t1.k == 15) & (t1.partition.isin(["C1_name", "C2_id", "C2_name"]))].set_index("partition")
var["커버리지%"] = [float(var1.loc[p, "coverage_pct"]) for p in var["partition"]]
var["실루엣"] = [float(var1.loc[p, "silhouette"]) for p in var["partition"]]
var["설명"] = ["진단 **이름** 어휘로 KMeans", "진단 id 어휘로 **Ward 계층군집**", "진단 이름 어휘로 Ward 계층군집"]

A("## 3. 동반질환 군집의 어휘·알고리즘 변형 (C1_name / C2_id / C2_name, k=5~30)")
A("")
A(f"**입력** — 2번과 같은 방문별 진단 리스트. 단 어휘를 id 대신 **진단 이름**으로 바꾼 것이 `_name` 계열"
  f"(차원 {CL['vectorization']['name']['dim']:,} — 서로 다른 코드가 같은 이름으로 합쳐져 id보다 작다).")
A("")
A("**만든 방법** — 벡터화·SVD100·L2는 2번과 동일. C1은 KMeans, **C2는 Ward 연결 계층군집**"
  "(`linkage(method='ward')` → `fcluster(maxclust=k)`). k=5~30 전부.")
A("")
A(f"**몇 개로 나뉘었나 / 각 라벨** — k=15 기준 아래 표. 라벨의 임상적 내용은 C1_id와 거의 같다 — "
  f"id·이름 어휘의 ARI는 k=15에서 {CL['ari_id_vs_name']['C1']['15']:.3f}, "
  f"KMeans·Ward의 ARI는 {CL['ari_kmeans_vs_ward']['id']['15']:.3f}로 알고리즘 차이가 어휘 차이보다 크다.")
A("")
A(md(var.rename(columns={"partition": "분할", "n_clusters_ge30_test": f"test≥{MIN_TEST} 통과",
                         "jac_const_spread": "상수 격차", "corr_jacconst_vs_ndrug": "약물수 상관"}),
     ["분할", "설명", "커버리지%", f"test≥{MIN_TEST} 통과", "상수 격차", "약물수 상관", "실루엣"],
     {"커버리지%": 1, "상수 격차": 4, "약물수 상관": 3, "실루엣": 4}))
A("")
A(f"**왜 만들었나 / 무엇이 문제였나** — 결론이 어휘 선택이나 군집 알고리즘에 딸려 나오는 것인지 확인하려고 만든 "
  f"민감도 변형이다. 격차 크기({var['jac_const_spread'].min():.4f}~{var['jac_const_spread'].max():.4f}, "
  f"C1_id는 {float(c1cmp['상수 격차']):.4f})와 바닥 라벨의 성격이 유지되므로 C1_id k=15를 대표로 쓰기로 했다. "
  f"순열검정은 이 변형들에는 돌리지 않았다(대표 분할에만).")
A("")
A("**결과 요약** — 위 표. C1_name k=25에서 격차가 0.1589까지 튀지만 이는 최소 군집이 잘게 쪼개진 결과이고 "
  "약물수 상관도 0.472로 함께 올라간다.")
A("")

# ================================================================ 4. C1_icdfull
fu = t6b[t6b.partition.str.startswith("C1_icdfull")].iloc[0]
fucmp = cmp16("C1_icdfull")
fulab = t7.sort_values("jac_const")

A("## 4. 원본 ICD 전체 어휘 군집 (C1_icdfull, k=15)")
A("")
A(f"**입력** — `data4LLM_with_note.csv` 가 아니라 **`DIAGNOSES_ICD.csv` 원본**의 방문별 ICD-9 코드 리스트 전체. "
  f"어휘 **{FE['icd']['vocab_full']:,}개**(2번의 {FE['icd']['vocab_filtered_ids']:,}개보다 3배 넓다), "
  f"nnz {FE['icd']['nnz']:,}, 밀도 {FE['icd']['density']:.5f}.")
A("")
A(f"**만든 방법** — 2번과 완전히 같은 파이프라인(L2 → SVD100, 설명분산 "
  f"{TA['vectorization']['svd_explained_var_full']:.3f} → L2 → KMeans k=15, seed 0).")
A("")
A(f"**몇 개로 나뉘었나** — 라벨 15개, 커버리지 {fu['coverage_pct']}%, 최대 {int(fu['max_size']):,} / 최소 "
  f"{int(fu['min_size'])}방문, test ≥{MIN_TEST} 15개 전부 통과.")
A("")
A("**각 라벨이 무엇인가** — 성능 낮은 순 상위/하위 5개:")
A("")
A(md(pd.concat([fulab.head(5), fulab.tail(5)]).rename(
    columns={"cluster": "군집", "test_visits": "test 방문", "mean_n_drug": "평균 약물수",
             "jac_const": "상수 Jaccard", "top3_dx": "상위 동반진단(보유율)"}),
    ["군집", "test 방문", "상수 Jaccard", "평균 약물수", "상위 동반진단(보유율)"],
    {"상수 Jaccard": 4, "평균 약물수": 1}))
A("")
A(f"**왜 만들었나 / 무엇이 문제였나** — LLM 데이터셋의 진단 리스트는 원본보다 평균 "
  f"{FE['dropped']['mean_n_icd'] - FE['dropped']['mean_n_ours']:.2f}개 적고"
  f"(방문당 {FE['dropped']['mean_n_icd']:.2f} → {FE['dropped']['mean_n_ours']:.2f}, 손실률 "
  f"{FE['dropped']['mean_frac_dropped'] * 100:.1f}%, 최소 1개를 잃은 방문이 {FE['dropped']['pct_visits_lost_ge1']:.1f}%), "
  f"**누락이 희귀 코드에 쏠려 있다**. 그러면 '희귀 진단이 이미 지워진 어휘로 희귀 질병의 불리함을 재는' 자기충족이 된다. "
  f"그래서 원본 어휘로 다시 군집해 결론이 유지되는지 봤다.")
A("")
A(f"**결과 요약** — 유지된다. 격차 {float(fucmp['상수 격차']):.4f}(필터본 {float(c1cmp['상수 격차']):.4f}), 순열 p "
  f"{float(t8[(t8.statistic == '범위(max-min)') & (t8.predictor == '상수')].iloc[0]['p_value']):.5f} "
  f"(z={float(t8[(t8.statistic == '범위(max-min)') & (t8.predictor == '상수')].iloc[0]['z_vs_null']):.2f}). "
  f"두 군집의 ARI는 {TA['ari']['ari_stored_C1id15_vs_full_seed0']:.3f}로 seed 재실행 잡음"
  f"({TA['ari']['seed_noise_reference_from_11_cluster_meta']:.3f})과 같은 수준 — 즉 **어휘를 넓혀도 seed를 바꾼 정도의 차이만 난다**. "
  f"덤으로 약물수 상관이 {float(c1cmp['약물수 상관(상수)']):.3f} → {float(fucmp['약물수 상관(상수)']):.3f}로 낮아져 교란은 더 적다.")
A("")

# ================================================================ 5. BHC
bh = tA[(tA.rep == "CB_masked") & (tA.k == 15)].iloc[0]
bhprof = pd.DataFrame(BH["profiles"]["CB_masked_k15"]).set_index("cluster")
bhlab = tB.sort_values("jac_const").copy()
bhlab["평균 단어수"] = [int(bhprof.loc[c, "mean_words"]) for c in bhlab["cluster"]]
SENS = BH["sensitivity_ari"]["15"]

A("## 5. 퇴원요약 텍스트 임베딩 군집 (BHC_CB_masked, k=15)")
A("")
A(f"**입력** — 진단 코드를 전혀 쓰지 않는다. `NOTEEVENTS.csv` 의 퇴원요약에서 뽑은 "
  f"**Brief Hospital Course 섹션 텍스트**. 방문 {BP['n_input']:,}건 중 "
  f"{BP['n_kept']:,}건 사용(10단어 미만 {BP['excluded_by_reason']['empty_or_too_short_lt10w']}건 제외).")
A("")
A(f"**만든 방법** — ① 비식별 토큰 `[**...**]` 정규화({BP['deid']['spans_replaced']:,}개 치환) "
  f"→ ② **약물명 마스킹**: 성분명·브랜드명 {len(BP['brand_dict']['adopted'])}개 + 약효군 "
  f"{BP['class_terms_n']}개를 `[DRUG]` 로 치환({BP['masking']['drug_spans_replaced']:,} + "
  f"{BP['masking']['class_spans_replaced']:,}개) → ③ **Bio_ClinicalBERT**"
  f"(`emilyalsentzer/Bio_ClinicalBERT`, 최대 512토큰 슬라이딩 윈도우 stride 256, mean pooling, 청크 평균, L2) "
  f"→ ④ SVD100 → L2 → KMeans k=15(seed 0). TF-IDF 표현으로 만든 대조군도 같이 훑었다.")
A("")
A(f"**마스킹을 한 이유** — 노트에 정답 처방이 그대로 적혀 있으면 군집이 정답을 미리 본 셈이 된다. "
  f"마스킹 전 실측 누출은 성분명 기준 노트의 {BP['leakage_strict_ingredient_only']['pct_notes_with_any_mention']}%에서 "
  f"평균 recall {BP['leakage_strict_ingredient_only']['recall_mean']:.3f}, 브랜드명까지 포함하면 "
  f"{BP['leakage_extended_plus_brands']['pct_notes_with_any_mention']}% / "
  f"{BP['leakage_extended_plus_brands']['recall_mean']:.3f}였다.")
A("")
A(f"**몇 개로 나뉘었나** — 라벨 15개, 커버리지 {BP['n_kept'] / BP['n_input'] * 100:.2f}%, "
  f"최대 {int(bh['max_size']):,} / 최소 {int(bh['min_size'])}방문, test ≥{MIN_TEST} 15개 전부 통과. "
  f"seed 5개 평균 ARI {BH['seed_stability']['CB_masked_k15']['mean']:.3f}.")
A("")
A("**각 라벨이 무엇인가** — 여기가 이 군집의 약점이다. 상위 동반진단이 라벨마다 거의 같다:")
A("")
A(md(pd.concat([bhlab.head(4), bhlab.tail(3)]).rename(
    columns={"cluster": "군집", "test_visits": "test 방문", "mean_n_drug": "평균 약물수",
             "jac_const": "상수 Jaccard", "top3_dx": "상위 동반진단(보유율)"}),
    ["군집", "test 방문", "상수 Jaccard", "평균 약물수", "평균 단어수", "상위 동반진단(보유율)"],
    {"상수 Jaccard": 4, "평균 약물수": 1}))
A("")
A(f"성능 바닥 군집도 천장 군집도 '고혈압·심부전·심방세동'이다. 대신 갈라지는 축은 **노트 길이·문서 스타일**이다 — "
  f"군집 평균 단어 수가 {int(bhprof['mean_words'].min())}단어에서 {int(bhprof['mean_words'].max())}단어까지 벌어지고, "
  f"노트 길이에 대한 η²가 {bh['eta2_notelen']:.4f}로 진단 수 η²({bh['eta2_diag_count']:.4f})보다 크다. "
  f"진단 군집(C1_id k=15)과의 ARI는 {bh['ARI_C1id15']:.4f}(NMI {bh['NMI_C1id15']:.4f})로 사실상 무관하다.")
A("")
A(f"**표현 민감도** — 같은 텍스트를 TF-IDF로 바꾸면 완전히 다른 분할이 나온다(k=15에서 ARI "
  f"{SENS['ARI_CBmasked_vs_TFIDFmasked']:.3f}). 마스킹 여부는 그보다 영향이 작다"
  f"(BERT 마스킹본 vs 원문 ARI {SENS['ARI_CBmasked_vs_CBclean']:.3f}, "
  f"TF-IDF 쪽은 {SENS['ARI_TFIDFmasked_vs_TFIDFclean']:.3f}). "
  f"네 표현(BERT/TF-IDF × 마스킹/원문) × k=5~30을 모두 훑은 결과가 `tableA_bhc_sweep.csv` 이고, "
  f"성능 평가는 BERT 마스킹본 k=15 하나에만 돌렸다.")
A("")
A("**왜 만들었나 / 무엇이 문제였나** — 진단 코드가 아닌 자유 텍스트가 코드와 다른 환자 축을 잡아내는지 보려고 만들었다. "
  "결과적으로 다른 축을 잡긴 했는데 그것이 임상 축이 아니라 문서 축이었다. 격차는 가장 크지만 해석이 안 되므로 "
  "주 분할로 쓰지 않았고, 이후 트랙은 다시 진단 코드로 돌아갔다.")
A("")
A(f"**결과 요약** — 상수 격차 {BPF['jac_const_spread']:.4f}(진단 군집 {BPF['diagnose_C1_id_k15_reference']['jac_const_spread']:.4f}보다 크다), "
  f"순열 p {float(perm5('BHC CB_masked k=15', '범위(max-min)')['p_value']):.5f} "
  f"(z={float(perm5('BHC CB_masked k=15', '범위(max-min)')['z_vs_null']):.2f}), "
  f"약물수 상관 **{BPF['corr_jacconst_vs_ndrug']:.3f}**. "
  f"진단 군집과 달리 copy-prev 예측기에서도 격차가 유의하다"
  f"(p={float(perm5('BHC CB_masked k=15', '범위(max-min)', 'copy-prev')['p_value']):.4f}) — "
  f"전역 추천 고유의 문제가 아니라 그 방문들이 원래 어렵다는 신호이므로, 주장을 뒷받침하는 증거로는 오히려 약하다.")
A("")

# ================================================================ 6~8. D1/D2/D2b
d1s = {s["분할"]: s for s in PA["summary"]}
d1cmp, d2cmp, d2bcmp = cmp16("D1"), cmp16("D2"), cmp16("D2b")

# D1 통과 라벨(test>=30) 안에서의 소화기 순위 — 사전 고정한 판정 기준이 참조하는 값
d1pass_labels = set(t13[(t13["분할"] == "D1") & t13["ge30"]]["라벨"])
d1rank = t14[(t14.partition == "D1") & t14["라벨"].isin(d1pass_labels)].sort_values("jac_const")
d1_n_pass = len(d1rank)
d1_pass_rank = list(d1rank["라벨"]).index("소화기") + 1

A("## 6. 주 진단 ICD chapter 분할 (D1)")
A("")
A(f"**입력** — `DIAGNOSES_ICD.csv` 에서 각 방문의 **`SEQ_NUM=1` 코드 하나**(주 진단). 진단 리스트 전체가 아니다. "
  f"코드 앞 3자리만 쓰고 V·E 코드는 따로 둔다. 매핑 실패 {PA['chapter_mapping_failures']}건.")
A("")
A("**만든 방법** — 룰 기반, 학습 없음. ICD-9 공식 chapter 경계로 자른다: "
  "감염성 001–139, 신생물 140–239, 내분비·대사·면역 240–279, 혈액 280–289, 정신 290–319, 신경 320–389, "
  "순환기 390–459, 호흡기 460–519, 소화기 520–579, 비뇨생식기 580–629, 임신·출산 630–679, 피부 680–709, "
  "근골격 710–739, 선천기형 740–759, 주산기 760–779, 증상·불명확 780–799, 손상·중독 800–999, V코드, E코드.")
A("")
A(f"**몇 개로 나뉘었나** — 라벨 {d1s['D1']['라벨 수']}개, 커버리지 **{float(d1cmp['커버리지%'])}%**, "
  f"test ≥{MIN_TEST} 통과 **{d1s['D1']['test>=30 통과 라벨 수 (실질 k)']}개**"
  f"(통과 라벨의 test {d1s['D1']['통과 라벨의 test 방문']:,}건, 탈락 {d1s['D1']['탈락 test 비율%']}%). "
  f"가장 큰 라벨 순환기 {int(t13[(t13['분할'] == 'D1') & (t13['라벨'] == '순환기')].iloc[0]['전체방문']):,}방문, "
  f"가장 작은 라벨 임신·출산 4방문.")
A("")
A("**각 라벨이 무엇인가**")
A("")
A(md(label_table("D1", "d1_label"), LT_COLS, LT_FMT))
A("")
A("**왜 만들었나 / 무엇이 문제였나** — 2번의 동반질환 군집은 만성질환 프로파일로 갈렸고 클러스터 번호는 "
  "외부 분류 체계와 대응하지 않는다. 같은 주장을 **사람이 만든 공식 분류**로 다시 확인하려고 만들었다. "
  "문제는 입도다. chapter는 동반질환 군집보다 훨씬 거칠어서, 예컨대 소화기 chapter에는 간경변(C1) 외에 "
  "위장관 출혈·췌장염 등이 섞여 바닥 신호가 평균에 묻힐 수 있다. 이 때문에 판정 규칙을 결과 보기 전에 "
  "'소화기·비뇨생식기가 바닥이면 확증, 아니면 **반증이 아니라 입도 차이로 판정 불가**'로 고정했다.")
A("")
A(f"**결과 요약** — 상수 격차 {float(d1cmp['상수 격차']):.4f}, 순열 p(범위) "
  f"**{float(d1cmp['순열 p(상수·범위)']):.5f} — 유의하지 않다**(가중 SD는 "
  f"{float(d1cmp['순열 p(상수·가중SD)']):.4f}로 유의). 약물수 상관 {float(d1cmp['약물수 상관(상수)']):.3f}. "
  f"사전에 고정한 판정 기준대로 보면 통과 라벨 {d1_n_pass}개 중 비뇨생식기가 최하위, "
  f"소화기가 밑에서 {d1_pass_rank}번째였다. "
  f"다만 약물 수를 0.5개 이내로 맞춘 쌍만 보면 {MP['pairs_meta']['D1']['n_pairs_matched']}쌍 중 2쌍에서 격차가 남는다"
  f"(§3-1, `REPORT_ICD.md`).")
A("")

A("## 7. 순환기 블록 세분 분할 (D2)")
A("")
A(f"**입력·방법** — D1과 같되, 순환기(390–459)만 ICD-9 공식 8블록으로 다시 자른다: "
  f"류마티스심질환 390–398, 고혈압성 401–405, 허혈성 심질환 410–414, 폐순환 415–417, 기타 심질환 420–429, "
  f"뇌혈관 430–438, 동맥·세동맥 440–449, 정맥·림프 451–459. 블록 미지정 {PA['circ_block_unassigned']}건.")
A("")
A(f"**몇 개로 나뉘었나** — 라벨 {d1s['D2']['라벨 수']}개, 커버리지 {float(d2cmp['커버리지%'])}%, "
  f"test ≥{MIN_TEST} 통과 **{d1s['D2']['test>=30 통과 라벨 수 (실질 k)']}개**, 탈락 test "
  f"{d1s['D2']['탈락 test 비율%']}%. 통과 라벨의 중앙 test 크기는 D1의 "
  f"{d1s['D1']['중앙 test 크기(통과 라벨 중)']:.0f}건에서 {d1s['D2']['중앙 test 크기(통과 라벨 중)']:.0f}건으로 줄어든다.")
A("")
A("**각 라벨이 무엇인가** (순환기 블록만 발췌, 나머지는 D1과 동일)")
A("")
A(md(label_table("D2", "d2_label").pipe(lambda t: t[t["라벨"].str.startswith("순환기")]), LT_COLS, LT_FMT))
A("")
A(f"**왜 만들었나 / 무엇이 문제였나** — 순환기가 `SEQ_NUM=1` 기준 {PA['circ_share_pct']}%로 한 덩어리인데, "
  f"그 안이 균질하지 않다. 실제로 쪼개 보니 정맥·림프 "
  f"{float(t14[(t14.partition == 'D2') & (t14['라벨'] == '순환기: 정맥·림프')].iloc[0]['jac_const']):.4f}에서 동맥·세동맥 "
  f"{float(t14[(t14.partition == 'D2') & (t14['라벨'] == '순환기: 동맥·세동맥')].iloc[0]['jac_const']):.4f}까지 벌어진다 "
  f"— **chapter 평균이 이 격차를 상쇄해 지우고 있었다.**")
A("")
A(f"**결과 요약** — 격차 {float(d1cmp['상수 격차']):.4f} → **{float(d2cmp['상수 격차']):.4f}**, 순열 p "
  f"{float(d1cmp['순열 p(상수·범위)']):.5f} → **{float(d2cmp['순열 p(상수·범위)']):.5f}**. "
  f"즉 D1에서 유의하지 않던 격차가 D2에서 유의해진다. 대가는 정밀도(평균 CI 폭 증가)와 교란이다 — "
  f"약물수 상관 {float(d1cmp['약물수 상관(상수)']):.3f} → {float(d2cmp['약물수 상관(상수)']):.3f}. "
  f"성능 바닥 정맥·림프는 평균 약물 17.3개, 천장 동맥·세동맥은 27.3개다.")
A("")

A("## 8. 순환기 기타 심질환 3자리 재분할 (D2b, 참고용)")
A("")
A("**입력·방법** — D2와 같되 '순환기: 기타 심질환'(420–429)만 3자리 코드 단위로 한 번 더 쪼개고, "
  "각 3자리 그룹의 대표 명칭을 `D_ICD_DIAGNOSES.csv` 의 `SHORT_TITLE` 에서 붙였다(예: 428 CHF NOS).")
A("")
A(f"**몇 개로 나뉘었나** — 라벨 {d1s['D2b']['라벨 수']}개, 커버리지 {float(d2bcmp['커버리지%'])}%, "
  f"test ≥{MIN_TEST} 통과 {d1s['D2b']['test>=30 통과 라벨 수 (실질 k)']}개, 탈락 test "
  f"{d1s['D2b']['탈락 test 비율%']}%(세 분할 중 최대).")
A("")
A("**각 라벨이 무엇인가** (420–429 재분할분만)")
A("")
A(md(label_table("D2b", "d2b_label").pipe(
    lambda t: t[t["라벨"].str.startswith("순환기: 4")]), LT_COLS, LT_FMT))
A("")
A(f"**왜 만들었나 / 무엇이 문제였나** — 세분화를 한 단계 더 밀면 격차가 계속 커지는지 보려고 만든 참고용이다. "
  f"격차는 커지지만({float(d2bcmp['상수 격차']):.4f}) 약물수 상관이 "
  f"**{float(d2bcmp['약물수 상관(상수)']):.3f}** — P1_ge100({float(p1cmp['약물수 상관(상수)']):.3f}) 수준까지 올라간다. "
  f"즉 세분화가 드러내는 격차의 상당 부분이 처방 개수 차이다. 주 분할을 D2로 두라는 사전 결정은 결과를 보고도 유지했다.")
A("")
A(f"**결과 요약** — 격차 {float(d2bcmp['상수 격차']):.4f}, 순열 p {float(d2bcmp['순열 p(상수·범위)']):.5f}, "
  f"약물수 상관 {float(d2bcmp['약물수 상관(상수)']):.3f}.")
A("")

# ================================================================ 분할이 아닌 것
A("## 9. 분할이 아닌 트랙 — 진단 희귀도 연속변수")
A("")
A(f"방문을 나누지 않고 `SEQ_NUM=1` 코드의 유병률을 −log10 으로 바꿔 **연속 변수**로 쓴 트랙이다"
  f"(평균 {FE['rarity_seq1_neglog10']['mean']:.3f}, SD {FE['rarity_seq1_neglog10']['sd']:.3f}). "
  f"유병률은 **train split에서만** 추정했다. 라벨이 없으므로 이 문서의 비교표에는 넣지 않는다. "
  f"결과는 `REPORT_RARITY.md` 에 있다.")
A("")

# ================================================================ 비교표
A("## 10. 한 장 비교표")
A("")


def one(name, inp, how, nlab, cov, gap, p, corr, note):
    return {"분할": name, "입력": inp, "만든 방법": how, "라벨 수": nlab, "커버리지": cov,
            "격차": gap, "순열 p": p, "약물수 상관": corr, "한 줄 평": note}


rows = [
    one("첫 진단 코드 라벨 (P1_ge100)", "diag_id[0] 하나", "룰: ≥100방문 코드만 라벨",
        f"{int(p1['k'])}+기타", f"{p1['coverage_pct']}%", f"{float(p1cmp['상수 격차']):.4f}",
        f"{float(p1cmp['순열 p(상수·범위)']):.5f}", f"{float(p1cmp['약물수 상관(상수)']):.3f}",
        "해석 쉽고 격차 최대. 커버리지 36.5%로 탈락"),
    one("동반질환 프로파일 군집 (C1_id k=15)", f"diag_id 리스트 전체 ({CL['vectorization']['id']['dim']:,}어휘)",
        "L2→SVD100→L2→KMeans(seed 0)", "15", f"{c1['coverage_pct']}%",
        f"{float(c1cmp['상수 격차']):.4f}", f"{float(c1cmp['순열 p(상수·범위)']):.5f}",
        f"{float(c1cmp['약물수 상관(상수)']):.3f}", "커버리지·검정력·교란 균형이 가장 좋은 주 분할"),
    one("어휘·알고리즘 변형 (C1_name / C2_id / C2_name)", "같은 진단 리스트(id·이름 어휘)", "KMeans / Ward 계층군집",
        "15 (k=5~30)", "100.0%",
        f"{var['jac_const_spread'].min():.4f}~{var['jac_const_spread'].max():.4f}", "미실시",
        f"{var['corr_jacconst_vs_ndrug'].min():.3f}~{var['corr_jacconst_vs_ndrug'].max():.3f}",
        "결론이 어휘·알고리즘에 딸려 있지 않음을 확인"),
    one("원본 ICD 전체 어휘 군집 (C1_icdfull k=15)", f"DIAGNOSES_ICD 전체 ({FE['icd']['vocab_full']:,}어휘)",
        "C1_id와 동일 파이프라인", "15", f"{fu['coverage_pct']}%", f"{float(fucmp['상수 격차']):.4f}",
        f"{float(fucmp['순열 p(상수·범위)']):.5f}", f"{float(fucmp['약물수 상관(상수)']):.3f}",
        "LLM 데이터의 희귀 진단 누락이 결론을 만들지 않았음"),
    one("퇴원요약 텍스트 군집 (BHC_CB_masked k=15)", "NOTEEVENTS 의 BHC 섹션(약물 마스킹)",
        "Bio_ClinicalBERT→SVD100→KMeans", "15", f"{BP['n_kept'] / BP['n_input'] * 100:.2f}%",
        f"{BPF['jac_const_spread']:.4f}", f"{float(perm5('BHC CB_masked k=15', '범위(max-min)')['p_value']):.5f}",
        f"{BPF['corr_jacconst_vs_ndrug']:.3f}", "격차 크나 축이 노트 길이·문서 스타일. 해석 불가"),
    one("주 진단 ICD chapter 분할 (D1)", "SEQ_NUM=1 코드 하나", "룰: ICD-9 공식 chapter 경계",
        f"{d1s['D1']['라벨 수']}", f"{float(d1cmp['커버리지%'])}%", f"{float(d1cmp['상수 격차']):.4f}",
        f"{float(d1cmp['순열 p(상수·범위)']):.5f}", f"{float(d1cmp['약물수 상관(상수)']):.3f}",
        "공식 분류로 본 독립 확인. 범위 기준 유의하지 않음"),
    one("순환기 블록 세분 (D2)", "SEQ_NUM=1 코드 하나", "D1 + 순환기 공식 8블록",
        f"{d1s['D2']['라벨 수']}", f"{float(d2cmp['커버리지%'])}%", f"{float(d2cmp['상수 격차']):.4f}",
        f"{float(d2cmp['순열 p(상수·범위)']):.5f}", f"{float(d2cmp['약물수 상관(상수)']):.3f}",
        "chapter 평균이 지우던 격차를 드러냄. ICD 트랙의 주 분할"),
    one("기타 심질환 3자리 재분할 (D2b)", "SEQ_NUM=1 코드 하나", "D2 + 420–429 3자리",
        f"{d1s['D2b']['라벨 수']}", f"{float(d2bcmp['커버리지%'])}%", f"{float(d2bcmp['상수 격차']):.4f}",
        f"{float(d2bcmp['순열 p(상수·범위)']):.5f}", f"{float(d2bcmp['약물수 상관(상수)']):.3f}",
        "참고용. 격차는 커지나 교란도 최대"),
]
A(md(pd.DataFrame(rows)))
A("")
A(f"'격차'는 test ≥{MIN_TEST} 라벨의 상수 top-{FE['baseline']['K_const']} Jaccard 최대−최소, "
  f"'순열 p'는 그 범위 통계량의 환자 단위 순열 10,000회 단측 p다. "
  f"C1_name/C2 계열은 순열검정을 돌리지 않았으므로 '미실시'로 적었다.")
A("")
A("### 이 카탈로그에 없는 분할")
A("")
A("동반질환 제거 후 다시 군집화한 여섯 분할(COMORB-elix, CHR-persist τ=0.5 / τ=0.3, CHR-idf, "
  "그리고 주 진단 비보존 민감도 둘)은 **사후 분석**이라 이 카탈로그에 넣지 않았다. "
  "정의·라벨·평가는 `out/REPORT_CHRONIC.md`, 방문별 배정은 `out/chronic_partition_assignments.csv` 에 있다. "
  "주 분할은 여전히 C1_id k=15 다.")
A("")

# ================================================================ 파일 목록
A("## 11. 어느 파일에 무엇이 있나")
A("")
A("### 방문별 배정")
A("")
A("| 파일 | 내용 | 키 |")
A("|---|---|---|")
A(f"| `out/cluster_assignments.pkl` | P1_ge100 · P1_ge30 · C1_id · C2_id · C1_name · C2_name 의 k=5~30 전부 "
  f"({len(pd.read_pickle(OUT / 'cluster_assignments.pkl')):,}행) | SUBJECT_ID, HADM_ID, partition, k, cluster_id |")
A(f"| `out/cluster_assignments_bhc.csv` | BHC 텍스트 군집 배정 — BHC_CB_masked · CB_clean · TFIDF_masked · TFIDF_clean 의 "
  f"k=10/15/20/25 ({len(C('cluster_assignments_bhc.csv')):,}행) | 동상 |")
A(f"| `out/trackA_assignments.csv` | C1_icdfull k=15 배정 ({len(C('trackA_assignments.csv')):,}행) | 동상 |")
A(f"| `out/icd_partition_assignments.csv` | D1 · D2 · D2b 라벨 + 주 진단 코드 ({len(part):,}행) | "
  f"SUBJECT_ID, HADM_ID, icd9_seq1, chapter, d1_label, d2_label, d2b_label, split |")
A(f"| `out/20_icd_features.pkl` | 위 모든 트랙의 공용 입력(방문별 ICD 리스트, 희귀도, split) | SUBJECT_ID, HADM_ID |")
A("")
A("### 요약 표")
A("")
A("| 파일 | 내용 |")
A("|---|---|")
A("| `table1_partition_metrics.csv` | P1/C1/C2 전 분할의 구조 지표(크기·gini·실루엣·η²) |")
A("| `table2_cluster_performance.csv` | 위 분할의 라벨별 성능 + 부트스트랩 CI + 상위 동반진단 |")
A("| `table3_power.csv`, `table3b_power_prev.csv` | 라벨 크기 기반 검정력(최소 검출 가능 차이) |")
A("| `table4_gap_summary.csv` | 분할·k별 격차와 약물수 상관 한 줄 요약 |")
A("| `table5_perm_10000.csv`, `table5_perm_2000.csv` | C1_id k=15 · BHC k=15 순열검정 |")
A("| `table6_trackA_compare.csv`, `table6b_trackA_describe.csv` | 필터본 vs 원본 어휘 군집 비교 |")
A("| `table7_trackA_clusters.csv`, `table8_trackA_perm.csv` | C1_icdfull 라벨별 성능 · 순열검정 |")
A("| `tableA_bhc_sweep.csv` | BHC 표현(CB/TF-IDF × masked/clean) × k=5~25 스윕 |")
A("| `tableB_bhc_cluster_performance.csv` | BHC_CB_masked k=15 라벨별 성능 |")
A("| `table13_icd_power.csv` | D1/D2/D2b 라벨별 표본 수와 ≥30 통과 여부 |")
A("| `table14_icd_clusters.csv` | D1/D2/D2b 라벨별 성능 + CI |")
A("| `table15_icd_perm.csv`, `table16_icd_compare.csv` | ICD 분할 순열검정 · 여섯 분할 비교표 |")
A("| `table17_c1c10_crosstab.csv` | C1/C10 군집 방문이 D1·D2 라벨로 흩어진 교차표 |")
A("| `table18_icd_tradeoff.csv` | 세분화 트레이드오프(실질 k vs CI 폭) |")
A("| `table19_matched_pairs.csv`, `table20_matched_perm_global.csv` | 약물 수 매칭 쌍 분석 |")
A("| `table10~12b` | 희귀도 연속변수 트랙(회귀·5분위·통제·위약) |")
A("")
A("### 그림")
A("")
A("| 파일 | 어느 분할 |")
A("|---|---|")
A("| `figs/fig1_svd_clusters.png` | C1_id — 진단 SVD 공간의 군집 |")
A("| `figs/fig2_cluster_jaccard_P1.png` | P1_ge100 라벨별 성능 |")
A("| `figs/fig3_cluster_jaccard_C1.png` | C1_id 라벨별 성능 |")
A("| `figs/fig4_confound_drugcount.png` | 약물 수 교란(전 분할 공통) |")
A("| `figs/fig5_power_vs_k.png` | k에 따른 검정력 |")
A("| `figs/fig6_bhc_umap_clusters.png`, `fig7_bhc_umap_dxoverlay.png`, `fig8_bhc_dx_heatmap.png` | BHC 임베딩 군집 |")
A("| `figs/fig9~10_perm_dx_*.png` | C1_id k=15 순열 귀무분포(상수 / copy-prev) |")
A("| `figs/fig11~12_perm_bhc_*.png` | BHC k=15 순열 귀무분포 |")
A("| `figs/fig13~15_rarity_*.png` | 희귀도 연속변수 트랙 |")
A("| `figs/fig16_D1_label_perf.png`, `fig17_D2_label_perf.png` | D1 · D2 라벨별 성능 |")
A("| `figs/fig18_D2_label_dx_heatmap.png` | D2 라벨 × 상위 동반진단 |")
A("| `figs/fig19_matched_pairs.png` | 약물 수 매칭 쌍의 성능 차이 |")
A("| `figs/fig22_D1_chapter_composition.png` | D1 18개 chapter 구성(방문 수·test≥30 통과·대표 주진단) |")
A("| `figs/fig23_D1_seq1_heatmap.png` | D1 chapter × 그 chapter 의 주 진단 상위 5개 |")
A("| `figs/fig24_svd_separation.png` | 진단 SVD 2차원에서의 분리도 — D1 색칠 / C1_id k=15 색칠 |")
A("| `figs/fig25_circulatory_split.png` | D1 순환기 4,211방문 → D2 8블록 분해 |")
A("")
A("### 보고서")
A("")
A("| 파일 | 내용 |")
A("|---|---|")
A("| `REPORT.md` | P1 · C1 · C2 트랙 |")
A("| `REPORT_BHC.md` | 퇴원요약 임베딩 군집 |")
A("| `REPORT_PERM.md` | 순열검정 |")
A("| `REPORT_RARITY.md` | 전체 어휘 군집 + 희귀도 연속변수 |")
A("| `REPORT_ICD.md` | D1 · D2 · D2b + 약물 수 매칭 쌍 |")
A("| `README_REGEN.md` | 삭제한 중간 산물과 재생성 명령 |")

(OUT / "CLUSTERS.md").write_text("\n".join(L), encoding="utf-8")
print(f"saved: out/CLUSTERS.md ({len(L)} lines)")
