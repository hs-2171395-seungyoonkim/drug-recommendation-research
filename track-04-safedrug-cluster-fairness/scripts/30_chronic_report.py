"""§6~§7 — REPORT_CHRONIC.md 조립 + 그림 2장.

계산은 하지 않는다. 27/28/29 가 만든 table*.csv 와 메타 JSON 을 읽어 붙일 뿐이다.
예외: 그림용 유병률 행렬은 여기서 센다.

산출: out/REPORT_CHRONIC.md, out/figs/fig20_chronic_heatmap.png, out/figs/fig21_chronic_perf.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
MIN_TEST = 30

C = lambda f: pd.read_csv(OUT / f)
J = lambda f: json.load(open(OUT / f, encoding="utf-8"))

PREP, SETS = J("27_chronic_prep_meta.json"), J("27_chronic_sets.json")
CLU, PERM = J("28_chronic_cluster_meta.json"), J("29_chronic_perm_meta.json")
t21, t22, t23 = C("table21_chronic_removal.csv"), C("table22_chronic_remaining.csv"), C("table23_chronic_removed_top.csv")
t24, t25, t26 = C("table24_chronic_overlap.csv"), C("table25_chronic_defdiff.csv"), C("table26_chronic_seq1_kept.csv")
t27, t28 = C("table27_chronic_acute_cats.csv"), C("table28_chronic_tau_scan.csv")
t29, t30, t31 = C("table29_chronic_describe.csv"), C("table30_chronic_ksweep.csv"), C("table31_chronic_seed.csv")
t32, t33, t34 = C("table32_chronic_axis.csv"), C("table33_chronic_clusters.csv"), C("table34_chronic_toplift.csv")
t35, t36, t37 = C("table35_chronic_seq1_sens.csv"), C("table36_chronic_perm.csv"), C("table37_chronic_gap.csv")

TAU_MAIN, TAU_SCALE = PREP["tau_main"], PREP["tau_scale_matched"]
MAIN, SENS = CLU["main_partitions"], CLU["sensitivity"]
CCI = SETS["cci"]["summary"]

# ---------------------------------------------------------------- 기존 분할 참조값
ref2, ref5 = C("table2_cluster_performance.csv"), C("table5_perm_10000.csv")
ref7, ref8 = C("table7_trackA_clusters.csv"), C("table8_trackA_perm.csv")
ref14, ref15 = C("table14_icd_clusters.csv"), C("table15_icd_perm.csv")


def gap_of(t, pcol="partition", pname=None, kk=15):
    t = t if pname is None else t[(t[pcol] == pname) & (t.get("k", pd.Series(kk, index=t.index)) == kk)]
    big = t[t["test_visits"] >= MIN_TEST]
    return {"n": len(big), "gap": float(big["jac_const"].max() - big["jac_const"].min()),
            "corr": float(np.corrcoef(big["mean_n_drug"], big["jac_const"])[0, 1])}


REF = {
    "C1_id k=15": {**gap_of(ref2, pname="C1_id"),
                   "p": float(ref5[(ref5.partition == "diagnose C1_id k=15") & (ref5.predictor == "상수")
                                   & (ref5.statistic == "범위(max-min)")]["p_value"].iloc[0])},
    "C1_icdfull k=15": {**gap_of(ref7, pname="C1_icdfull"),
                        "p": float(ref8[(ref8.predictor == "상수") & (ref8.statistic == "범위(max-min)")]["p_value"].iloc[0])},
    "D1 주진단 chapter": {**gap_of(ref14[ref14.partition == "D1"]),
                      "p": float(ref15[(ref15["분할"] == "D1") & (ref15["예측기"] == "상수")
                                       & (ref15["통계량"] == "범위(max-min)")]["p_value"].iloc[0])},
}

# ---------------------------------------------------------------- 마크다운 도우미
def esc(s):
    return str(s).replace("|", r"\|")


def md(df_, cols=None, fmt=None):
    df_ = df_ if cols is None else df_[cols]
    fmt = fmt or {}
    head = "| " + " | ".join(esc(c) for c in df_.columns) + " |"
    sep = "|" + "|".join("---" for _ in df_.columns) + "|"
    lines = [head, sep]
    for i in range(len(df_)):
        cells = []
        for c in df_.columns:
            v = df_[c].iloc[i]      # iterrows 는 행 단위로 dtype 을 올려버려 int 열이 float 로 찍힌다
            if isinstance(v, float) and not pd.isna(v):
                cells.append(f"{v:.{fmt.get(c, 3)}f}")
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(esc(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pget(part, pred="상수", stat="범위(max-min)", col="p_value"):
    r = t36[(t36.partition == part) & (t36.predictor == pred) & (t36.statistic == stat)]
    return float(r[col].iloc[0])


AX = t32.set_index("분할")
GP = t37.set_index("분할")

# ---------------------------------------------------------------- 그림 1: 정의별 상위 진단 히트맵
PANELS = ["C1_id k=15 (기준)", "COMORB-elix", f"CHR-persist{TAU_MAIN}", "CHR-idf"]
asg = pd.read_csv(OUT / "chronic_partition_assignments.csv")
feat = pd.read_pickle(OUT / "20_icd_features.pkl")
FULL = [set(str(c).strip() for c in l) for l in feat["icd_l"]]
LABCOL = {"C1_id k=15 (기준)": "c1_id_label", "COMORB-elix": "comorb_elix_label",
          f"CHR-persist{TAU_MAIN}": "chr_persist_label", "CHR-idf": "chr_idf_label"}

fig, axes = plt.subplots(2, 2, figsize=(21, 15))
for ax, pan in zip(axes.ravel(), PANELS):
    tl = t34[t34["분할"] == pan]
    top2 = tl.sort_values(["cluster", "lift_diff"], ascending=[True, False]).groupby("cluster").head(2)
    codes = list(dict.fromkeys(top2["code"].astype(str)))
    labels = asg[LABCOL[pan]].to_numpy()
    cls = sorted(c for c in set(labels) if c >= 0)
    M = np.zeros((len(cls), len(codes)))
    for i, g in enumerate(cls):
        idx = np.where(labels == g)[0]
        for j, code in enumerate(codes):
            M[i, j] = sum(code in FULL[x] for x in idx) / len(idx)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    nm = dict(zip(tl["code"].astype(str), tl["name"]))
    cc = dict(zip(tl["code"].astype(str), tl["cci"]))
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([f"{nm.get(c, c)[:22]}" for c in codes], rotation=75, ha="right", fontsize=7)
    for lab, c in zip(ax.get_xticklabels(), codes):
        lab.set_color({"급성": "#c0392b", "만성": "#2471a3"}.get(cc.get(c), "#7f8c8d"))
    ax.set_yticks(range(len(cls)))
    ax.set_yticklabels([f"C{g}" for g in cls], fontsize=8)
    ac = AX.loc[pan, "특징 진단 중 CCI 급성%"] if pan in AX.index else np.nan
    ax.set_title(f"{pan}   —   특징 진단 중 CCI 급성 {ac:.0f}%", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
fig.suptitle("정의별 클러스터 상위 진단 유병률 (제거 전 원 진단 리스트 기준)\n"
             "x축 라벨 색: 빨강 = AHRQ CCI 급성, 파랑 = 만성, 회색 = 미분류", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGS / "fig20_chronic_heatmap.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- 그림 2: 클러스터별 성능 + CI
PERF = ["C1_id k=15", "COMORB-elix", f"CHR-persist{TAU_MAIN}", "CHR-idf"]
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
for ax, pan in zip(axes, PERF):
    src = ref2[(ref2.partition == "C1_id") & (ref2.k == 15)] if pan == "C1_id k=15" else t33[t33.partition == pan]
    big = src[src.test_visits >= MIN_TEST].sort_values("jac_const").reset_index(drop=True)
    x = np.arange(len(big))
    err = np.vstack([big.jac_const - big.jac_const_lo, big.jac_const_hi - big.jac_const])
    sc = ax.bar(x, big.jac_const, yerr=err, capsize=2, color=plt.cm.viridis(
        (big.mean_n_drug - big.mean_n_drug.min()) / max(big.mean_n_drug.max() - big.mean_n_drug.min(), 1e-9)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(c)}\nn={int(v)}" for c, v in zip(big.cluster, big.test_visits)], fontsize=7)
    g = big.jac_const.max() - big.jac_const.min()
    pv = REF["C1_id k=15"]["p"] if pan == "C1_id k=15" else pget(pan)
    cr = REF["C1_id k=15"]["corr"] if pan == "C1_id k=15" else GP.loc[pan, "약물수-성능 상관(Pearson)"]
    ax.set_title(f"{pan}\n격차 {g:.4f} · 순열 p={pv:.4f} · 약물수 상관 {cr:.2f}", fontsize=10)
    ax.set_xlabel("클러스터 (막대 색 = 평균 약물 수)")
axes[0].set_ylabel("상수 top-23 Jaccard (test, 95% 부트스트랩 CI)")
fig.suptitle("클러스터별 성능과 신뢰구간 — 동반질환 제거가 격차를 없애지 못한다", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGS / "fig21_chronic_perf.png", dpi=130)
plt.close(fig)
print("[.] 그림 2장 저장", flush=True)

# ---------------------------------------------------------------- 비교표 (§6)
cmp_rows = []
for nm, r in REF.items():
    cmp_rows.append({"분할": nm, "입력": {"C1_id k=15": "협업자 진단 id 1,454어휘",
                                       "C1_icdfull k=15": "DIAGNOSES_ICD 4,514어휘",
                                       "D1 주진단 chapter": "SEQ_NUM=1 ICD chapter"}[nm],
                     "제거": "없음", "라벨 수": r["n"], "격차": round(r["gap"], 4),
                     "순열 p(상수·범위)": r["p"], "약물수 상관": round(r["corr"], 3),
                     "특징진단 CCI 급성%": AX.loc[{"C1_id k=15": "C1_id k=15 (기준)",
                                             "C1_icdfull k=15": "C1_icdfull k=15 (기준)",
                                             "D1 주진단 chapter": "D1 주진단 chapter (기준)"}[nm], "특징 진단 중 CCI 급성%"],
                     "ARI vs C1_id": AX.loc[{"C1_id k=15": "C1_id k=15 (기준)",
                                             "C1_icdfull k=15": "C1_icdfull k=15 (기준)",
                                             "D1 주진단 chapter": "D1 주진단 chapter (기준)"}[nm], "ARI vs C1_id k=15"]})
for nm in MAIN + SENS:
    rem = t21[t21["정의"].str.startswith(nm.split("(")[0][:11])]
    cmp_rows.append({"분할": nm, "입력": "DIAGNOSES_ICD 4,514어휘",
                     "제거": {"COMORB-elix": "Elixhauser 751코드(주진단 보존)",
                            f"CHR-persist{TAU_MAIN}": f"재출현률≥{TAU_MAIN} 52코드",
                            f"CHR-persist{TAU_SCALE}": f"재출현률≥{TAU_SCALE} 190코드(규모정합)",
                            "CHR-idf": "제거 없음, IDF 가중",
                            "COMORB-elix-rmseq1": "Elixhauser 751코드(주진단도 제거)",
                            f"CHR-persist{TAU_MAIN}-rmseq1": f"재출현률≥{TAU_MAIN}(주진단도 제거)"}[nm],
                     "라벨 수": int(GP.loc[nm, "test≥30 클러스터"]), "격차": GP.loc[nm, "jac_const 격차"],
                     "순열 p(상수·범위)": pget(nm), "약물수 상관": GP.loc[nm, "약물수-성능 상관(Pearson)"],
                     "특징진단 CCI 급성%": AX.loc[nm, "특징 진단 중 CCI 급성%"],
                     "ARI vs C1_id": AX.loc[nm, "ARI vs C1_id k=15"]})
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(OUT / "table38_chronic_compare.csv", index=False)

# ---------------------------------------------------------------- 본문
E_ACUTE = CCI["급성 우세 Elixhauser 카테고리(CCI)"]
elix_share = PREP["elix_removed_share_pct_keepseq1"]
axis_main = AX.loc["COMORB-elix"]
axis_c1 = AX.loc["C1_id k=15 (기준)"]
sens = t35.set_index("정의")

L = []
A = L.append
A("# 동반질환 제거 후 급성 축 군집화 — 사후 분석(post-hoc, 탐색적)")
A("")
A("> **이 트랙은 사전에 정한 분석이 아니다.** 사전 등록된 주 분할은 `C1_id k=15`(동반질환 프로파일 "
  "k-means)이고, 이 문서의 결과가 어떻게 나오든 주 분할을 바꾸지 않는다. 여기 실린 모든 수치는 탐색적 "
  "라벨을 달고 보고한다.")
A("")
A(f"작성 2026-08-12. 입력 `DIAGNOSES_ICD.csv` 원본 어휘 {PREP['vocab_size']:,}개, "
  f"방문 {PREP['n_visits']:,}건(train {PREP['n_train_visits']:,} / test {PREP['n_test_visits']:,}).")
A("")

A("## 0. 한 문단 요약")
A("")
A(f"Elixhauser 동반질환 {PREP['elix_n_removed_codes']}개 코드를 지우고(주 진단은 보존) 같은 파이프라인으로 "
  f"다시 군집화하면, 클러스터를 특징짓는 진단 중 급성(AHRQ CCI 기준) 비율이 "
  f"C1_id 의 {axis_c1['특징 진단 중 CCI 급성%']:.1f}% 에서 {axis_main['특징 진단 중 CCI 급성%']:.1f}% 로 오른다. "
  f"고혈압·심부전·AF·ESRD 로 정의되던 클러스터가 사라지고 폐렴·패혈증·요로감염·급성 출혈성 빈혈·급성 신부전 "
  f"클러스터가 그 자리를 채운다. **축은 부분적으로 옮겨갔다.** 다만 완전히 옮겨가지는 않았다 — 남은 클러스터의 "
  f"절반가량은 고지혈증·고콜레스테롤·관상동맥경화·흡연·GERD·장기 항응고제 같은 **Elixhauser 가 다루지 않는 "
  f"만성 표지**로 정의된다. 그리고 성능 격차는 사라지지 않는다: 격차 {GP.loc['COMORB-elix','jac_const 격차']:.4f} "
  f"(순열 p={pget('COMORB-elix'):.4f}), 약물 수 상관 {GP.loc['COMORB-elix','약물수-성능 상관(Pearson)']:.3f} 로 "
  f"C1_id({REF['C1_id k=15']['corr']:.3f})보다 오히려 **더 강한 약물 수 교란**을 보인다.")
A("")

A("## 1. 정의 — 무엇을 제거했나")
A("")
A("### 1-1. 이름과 프레이밍")
A("")
A("이 트랙을 처음에 '만성 제거'라고 불렀으나 **'동반질환 제거'가 맞다.** Elixhauser 는 만성 질환 목록이 아니라 "
  "동반질환(comorbidity) 목록이고, 급성 상태도 포함한다. 그래서 정의 이름도 `CHR-elix` → **`COMORB-elix`** 로 바꿨다.")
A("")
A(f"AHRQ 만성상태지표(CCI 2015, ICD-9-CM)로 31개 카테고리를 판정하면 **급성이 우세한 카테고리는 "
  f"{', '.join(E_ACUTE)} 세 개**다(건수 가중 만성 비율 50% 미만). "
  f"사전에 급성으로 예상했던 `coag`(응고장애)·`wloss`(체중감소)·`blane`(실혈성 빈혈)은 "
  f"CCI 기준으로는 만성으로 분류되고(각각 97.5% / 81.6% / 100%), 반대로 예상하지 않았던 "
  f"`dane`(결핍성 빈혈, 0%)이 급성으로 나왔다. 즉 급성 목록은 우리 판단이 아니라 외부 표준이 정한 것이다. "
  f"제거되는 진단 **건수** 기준으로 CCI 급성이 차지하는 비중은 "
  f"**{CCI['COMORB-elix 제거분 중 CCI 급성 건수%']}%**, 어휘 기준으로는 "
  f"{CCI['COMORB-elix 제거 어휘 중 CCI 급성 코드%']}% 다. "
  f"즉 '동반질환 제거'의 실질은 약 {100 - CCI['COMORB-elix 제거분 중 CCI 급성 건수%']:.0f}% 가 만성 제거이지만, "
  f"수분·전해질 장애(`fed`, 제거 건수의 9.7%)처럼 급성 사건이 섞여 들어가는 것이 사실이다.")
A("")
A(md(t27.rename(columns={"우리 표기": "사전 예상"}),
     ["카테고리", "사전 예상", "CCI 판정", "어휘 수", "train 진단 건수", "elix 제거 건수 중 비중%",
      "CCI 만성 건수가중%", "대표 코드"], {"elix 제거 건수 중 비중%": 1, "CCI 만성 건수가중%": 1}))
A("")
A("### 1-2. 구현")
A("")
A(f"- **COMORB-elix** — {SETS['elix']['reference']} 의 ICD-9-CM 코드 매핑({SETS['elix']['n_categories']}개 카테고리, "
  f"{SETS['elix']['n_prefixes_unique']}개 고유 프리픽스)을 접두 매칭한다. 구현체는 "
  f"`{SETS['elix']['source']}`. 손으로 만든 목록이 아니다. "
  f"**AHRQ SAS 판이 쓰는 DRG 기반 배제(입원 사유에 해당하는 동반질환을 DRG 로 걸러내는 단계)는 사용하지 않는다.** "
  f"그 자리를 주 진단 보존이 대신한다.")
A(f"- **CHR-persist(τ)** — train split 의 다방문 환자에서 같은 코드가 두 번 이상 나타난 환자 비율(재출현률)이 "
  f"τ 이상이면 제거. train 전용 추정이고, 환자 수 10명 미만인 코드는 추정하지 않는다"
  f"(추정 가능 코드 {PREP['n_codes_estimable_recurrence']:,} / {PREP['vocab_size']:,}). "
  f"주 임계 τ={TAU_MAIN}, 민감도 τ∈{{0.3, 0.5, 0.7}}, 그리고 COMORB-elix 와 **제거 규모를 맞춘** τ={TAU_SCALE}.")
A("- **CHR-idf** — 아무것도 제거하지 않고 train IDF `log((1+N)/(1+df))+1` 로 가중만 준다. 흔한 만성 코드의 "
  "기여를 줄이되 정보는 남기는 대조군.")
A("")
A("### 1-3. 주 진단 보존 (주 버전)")
A("")
A(f"**SEQ_NUM=1 코드는 Elixhauser 에 해당해도 제거하지 않는다.** 근거는 두 가지다. "
  f"(1) Elixhauser 는 정의상 이차 진단에 적용하는 지수이고 주 진단에는 적용하지 않는다. "
  f"(2) 비교 대상인 C1_id 는 주 진단을 항상 포함하므로, 주 진단을 지우면 공정한 비교가 아니다. "
  f"실제로 이 규칙이 걸리는 방문은 전체의 **{PREP['seq1_in_elix_pct']}%**"
  f"({PREP['seq1_in_elix_n']:,}건)이고, "
  f"제거 비율은 주 진단을 지울 때 {PREP['elix_removed_share_pct_rmseq1']}% → 보존할 때 **{elix_share}%** 로 내려간다.")
A("")
A("보존되는 주 진단 상위 10개 — 대동맥판막 질환·심부전·전이암·간경변 같은, 그 입원의 사유 자체인 진단들이다.")
A("")
A(md(t26, ["코드", "명칭", "방문 수", "비중%", "Elixhauser 카테고리"], {"비중%": 1}))
A("")

A("## 2. 제거 규모 점검")
A("")
A(f"주 진단 보존 덕에 **남은 진단이 0개인 방문은 주 버전 네 개 모두 0건**이다"
  f"(비보존 민감도에서만 COMORB-elix 105건 0.72%, CHR-persist{TAU_MAIN} 6건 0.04% 발생). "
  f"처리 방침은 '군집에서 빼되 파일에는 라벨 -1 로 남긴다'로 정했고, 주 버전에서는 발동하지 않았다.")
A("")
A(md(t21, ["정의", "주 진단 보존", "제거 코드 수", "남은 어휘", "제거된 진단 건수 비율%", "방문당 남은 진단 평균",
           "중앙", "남은 진단 0개 방문", "0개 비율%", "SEQ_NUM=1 제거 방문%"],
     {"제거된 진단 건수 비율%": 1, "방문당 남은 진단 평균": 2, "0개 비율%": 2, "SEQ_NUM=1 제거 방문%": 2}))
A("")
A(f"**τ 규모 정합.** 주 진단 보존 기준 COMORB-elix 의 제거 건수 비율은 {elix_share}% 이고, "
  f"τ 격자에서 이에 가장 가까운 값은 **τ={TAU_SCALE}(33.8%, 차이 0.3%p)** 였다. "
  f"주 진단을 지우던 이전 규칙(36.2%)에서도 τ=0.3 이 선택됐으므로 규모 정합 대조군은 바뀌지 않는다.")
A("")
A(md(t28, ["τ", "제거 코드 수", "제거된 진단 건수 비율%", "elix 와의 규모 차"],
     {"τ": 2, "제거된 진단 건수 비율%": 1, "elix 와의 규모 차": 2}))
A("")
A("**두 정의는 거의 겹치지 않는다.** 코드 집합 Jaccard 는 0.027(τ=0.5) / 0.080(τ=0.3)에 불과하지만, "
  "겹치는 소수의 코드가 고빈도라 제거 건수로는 12~19% 를 차지한다.")
A("")
A(md(t24.rename(columns={"|A|": "A 코드수", "|B|": "B 코드수"}),
     ["A", "B", "A 코드수", "B 코드수", "교집합", "Jaccard", "교집합이 차지한 진단 건수%"],
     {"Jaccard": 3, "교집합이 차지한 진단 건수%": 1}))
A("")
A("엇갈리는 코드가 무엇인지가 두 정의의 성격을 보여준다. Elixhauser 만 지우는 쪽에는 산증·저삼투압·혈소판감소 같은 "
  "재출현률이 낮은(=만성이 아닌) 코드가 들어 있고, 재출현 기준만 지우는 쪽에는 수면무호흡·악성종양 과거력·"
  "신이식 합병증처럼 Elixhauser 목록에 없는 지속 상태가 들어 있다.")
A("")
A(md(t25, ["쪽", "코드", "명칭", "train 방문%", "카테고리", "재출현률"], {"train 방문%": 1, "재출현률": 3}))
A("")

A("## 3. 군집화")
A("")
A(f"파이프라인은 C1_id 와 **완전히 동일**하다: `{CLU['pipeline']}`. 어휘 컬럼은 모든 정의에서 4,514 로 고정해 "
  f"SVD 차원이 같은 조건에서 비교되게 했다.")
A("")
A(md(t29, ["partition", "k", "n_clusters", "coverage_pct", "min_size", "max_size", "gini_size",
           "silhouette", "silhouette_excl_other", "calinski_harabasz", "eta2_diag_count", "eta2_drug_count"],
     {"coverage_pct": 1, "gini_size": 3, "silhouette": 4, "silhouette_excl_other": 4, "calinski_harabasz": 1}))
A("")
A("비보존 민감도 둘의 `silhouette` 가 음수인 것은 구조가 나빠서가 아니라 **남은 진단이 0개인 방문 때문**이다. "
  "그 방문들은 SVD 공간의 원점에 놓여 단위구 위의 모든 점에서 거리가 정확히 1.0 이 되고, 그 결과 대부분의 점에게 "
  "'가장 가까운 다른 클러스터'가 되어 버린다. 라벨 -1 을 빼고 계산한 `silhouette_excl_other`(0.0571 / 0.0479)가 "
  "주 버전들과 비교 가능한 값이다. 주 버전 네 개는 -1 이 없어 두 값이 같다.")
A("")
A("**seed 안정성** — C1_id k=15 의 seed 간 평균 ARI 0.599 가 기준선이다. COMORB-elix 와 CHR-persist0.5 는 "
  "그보다 약간 높고(0.627), CHR-idf 가 가장 불안정하다(0.510).")
A("")
A(md(t31, ["분할", "seed 쌍별 ARI 평균", "최소", "최대", "C1_id k=15 seed 노이즈 기준"]))
A("")
A("**k 스윕 (k=5~30)** — 요약만 싣는다. 전체는 `table30_chronic_ksweep.csv`.")
A("")
A(md(t30[t30["k"].isin([5, 10, 15, 20, 25, 30])],
     ["분할", "k", "최소 클러스터", "최대 클러스터", "test≥30 클러스터", "silhouette", "η²(약물수)", "ARI vs C1_id15"],
     {"silhouette": 4, "η²(약물수)": 4}))
A("")

A("## 4. 판정 — 축이 실제로 옮겨갔는가")
A("")
A("### 4-1. 클러스터 상위 진단 — 이 실험의 성패")
A("")
A("**판정 지표를 먼저 밝힌다.** 제거 후 남은 코드로 상위 진단을 뽑으면 지운 코드가 안 나오는 게 당연해서 "
  "순환논증이 된다. 그래서 **제거 전 원 진단 리스트**로 클러스터를 특징짓는 진단(클러스터 내 유병률 − 전체 유병률이 "
  "큰 상위 5개, 클러스터 내 유병률 5% 이상)을 뽑고, 그것이 AHRQ CCI 기준 급성인지를 센다. "
  "단순 빈도 상위 3개는 기저 유병률(고혈압 36.6%)에 지배되므로 판정에 쓰지 않는다.")
A("")
A(md(t32, ["분할", "특징 진단 중 CCI 급성%", "특징 진단이 급성 과반인 클러스터", "특징 진단 중 Elixhauser 동반질환%",
           "ARI vs D1(주진단 chapter)", "NMI vs D1", "ARI vs C1_id k=15", "ARI vs C1_icdfull k=15",
           "η²(진단수)", "η²(약물수)"],
     {"특징 진단 중 CCI 급성%": 1, "특징 진단 중 Elixhauser 동반질환%": 1, "η²(진단수)": 4, "η²(약물수)": 4}))
A("")
A(f"**답: 부분적으로 옮겨갔다.** COMORB-elix 의 특징 진단 중 급성 비율은 "
  f"{axis_main['특징 진단 중 CCI 급성%']:.1f}% 로 C1_id({axis_c1['특징 진단 중 CCI 급성%']:.1f}%)보다 "
  f"{axis_main['특징 진단 중 CCI 급성%'] - axis_c1['특징 진단 중 CCI 급성%']:.1f}%p 높고, "
  f"급성이 과반인 클러스터는 {int(axis_c1['특징 진단이 급성 과반인 클러스터'])}개 → "
  f"{int(axis_main['특징 진단이 급성 과반인 클러스터'])}개로 늘었다. "
  f"C1_id 에 있던 고혈압 전용(C12)·AF(C11)·심부전(C14)·ESRD(C10)·CKD(C5)·간경변(C1) 클러스터는 사라졌다.")
A("")
A("그러나 그 자리를 전부 급성이 채우지는 않았다. COMORB-elix 15개 클러스터를 사람이 읽으면 이렇게 갈린다.")
A("")
A("| | 클러스터 | 정의하는 진단 |")
A("|---|---|---|")
A("| **급성 사유** | C0 폐렴 · C2 중증 패혈증/패혈성 쇼크 · C3 요로감염 · C4 급성 출혈성 빈혈 · "
  "C9 흡인성 폐렴 · C10 급성 신부전 · C14 급성 호흡부전 | 486 폐렴, 99592 중증패혈증, 5990 요로감염, "
  "2851 급성출혈성빈혈, 5070 흡인성폐렴, 5849 급성신부전, 51881 급성호흡부전 |")
A("| **Elixhauser 밖 만성 표지** | C5 장기 항응고제 · C6 고콜레스테롤혈증 · C8 관상동맥경화 · C11 CABG 상태 · "
  "C12 흡연 · C13 고지혈증 · C7 GERD/빈혈 | V5861 장기항응고제 사용, 2720 순수고콜레스테롤, 41401 관상동맥경화, "
  "V4581 관상동맥우회술 상태, 3051 흡연, 2724 고지혈증, 53081 위식도역류 |")
A("| **잔여** | C1 (뚜렷한 특징 진단 없음, lift 최대 0.022) | — |")
A("")
A("**이것이 가장 중요한 관찰이다.** Elixhauser 를 지워도 만성 축이 사라지지 않고, "
  "Elixhauser 가 다루지 않는 만성 표지(고지혈증·고콜레스테롤·관상동맥경화·흡연·GERD·시술 상태 코드)로 "
  "**옮겨 앉았다.** 동반질환 목록 하나를 지우는 것으로는 만성 축을 제거할 수 없다.")
A("")
A(f"CHR-persist(τ={TAU_MAIN})는 급성 비율이 {AX.loc[f'CHR-persist{TAU_MAIN}', '특징 진단 중 CCI 급성%']:.1f}% 로 "
  f"C1_id 와 사실상 같다(52개 코드만 지우므로 당연하다). CHR-idf 는 오히려 "
  f"{AX.loc['CHR-idf', '특징 진단 중 CCI 급성%']:.1f}% 로 **가장 만성적**이다 — IDF 가중은 희귀 코드를 키우는데 "
  f"희귀 코드에는 만성 합병증 코드가 많다.")
A("")
A("**동반질환 정의와 지속성 정의가 갈리는가.** 갈린다. COMORB-elix 는 축을 움직였고 "
  f"CHR-persist{TAU_MAIN} 는 움직이지 않았다. 첫 번째 설명 후보는 **제거 규모**이지 급성/만성 성격 차이가 아니다 — "
  f"규모를 맞춘 CHR-persist{TAU_SCALE}(190코드, 33.8%)의 급성 비율은 "
  f"{AX.loc[f'CHR-persist{TAU_SCALE}', '특징 진단 중 CCI 급성%']:.1f}% 로 COMORB-elix("
  f"{axis_main['특징 진단 중 CCI 급성%']:.1f}%)와 τ=0.5({AX.loc[f'CHR-persist{TAU_MAIN}', '특징 진단 중 CCI 급성%']:.1f}%) "
  f"사이에 놓이고, 규모가 커질수록 급성 비율이 오른다. 다만 규모를 맞춰도 COMORB-elix 에 못 미치므로 "
  f"정의의 성격 차이도 일부 작용한다. 제거분 중 CCI 급성 비중이 COMORB-elix "
  f"{CCI['COMORB-elix 제거분 중 CCI 급성 건수%']}% vs CHR-persist{TAU_MAIN} "
  f"{CCI[f'CHR-persist(τ={TAU_MAIN}) 제거분 중 CCI 급성 건수%']}% 인 점(Elixhauser 쪽이 급성을 더 많이 지운다)은 "
  f"COMORB-elix 에 **불리하게** 작용하는데도 결과는 반대이므로, 급성 혼입이 결과를 만든 것은 아니다.")
A("")
A("![정의별 클러스터 상위 진단 히트맵](figs/fig20_chronic_heatmap.png)")
A("")
A("### 4-2. 주 진단과의 정합")
A("")
A(f"ARI(D1, 주 진단 chapter)는 C1_id {axis_c1['ARI vs D1(주진단 chapter)']:.3f} → COMORB-elix "
  f"{axis_main['ARI vs D1(주진단 chapter)']:.3f} 로 **전혀 변하지 않았다.** NMI 도 0.102 → 0.096 이다. "
  f"동반질환을 지워도 군집은 주 진단 chapter 와 정렬되지 않는다. "
  f"즉 이 군집은 '무슨 병으로 왔는가'를 잡는 것이 아니라 여전히 다른 축을 잡고 있다.")
A("")
A("### 4-3. C1_id 와의 ARI — seed 노이즈보다 큰 변화인가")
A("")
A(f"C1_id k=15 의 seed 간 평균 ARI 는 0.599 다. COMORB-elix 대 C1_id 는 "
  f"**{axis_main['ARI vs C1_id k=15']:.3f}** 으로 seed 노이즈의 절반에도 못 미친다. "
  f"CHR-persist{TAU_MAIN} {AX.loc[f'CHR-persist{TAU_MAIN}', 'ARI vs C1_id k=15']:.3f}, "
  f"CHR-idf {AX.loc['CHR-idf', 'ARI vs C1_id k=15']:.3f} 도 마찬가지다. "
  f"**세 정의 모두 C1_id 와 실질적으로 다른 분할을 만들었다** — 조작이 먹히긴 했다는 뜻이다.")
A("")
A("### 4-4. η²(진단 수) / η²(약물 수)")
A("")
A(f"η²(진단 수)는 C1_id {axis_c1['η²(진단수)']:.4f} → COMORB-elix {axis_main['η²(진단수)']:.4f}, "
  f"η²(약물 수)는 {axis_c1['η²(약물수)']:.4f} → {axis_main['η²(약물수)']:.4f} 로 둘 다 소폭 줄었다. "
  f"동반질환을 지우면 '진단이 많은 환자' 축이 조금 약해지지만 없어지지는 않는다. "
  f"규모 정합 CHR-persist{TAU_SCALE} 는 오히려 η²(약물수) "
  f"{AX.loc[f'CHR-persist{TAU_SCALE}', 'η²(약물수)']:.4f} 로 C1_id 보다 높다.")
A("")
A("### 4-5. 주 진단 보존 여부 민감도 — 30% 가 결과를 바꾸는가")
A("")
A(f"COMORB-elix 는 방문의 {PREP['seq1_in_elix_pct']}% 에서 주 진단이 Elixhauser 목록에 걸린다. "
  f"이 30% 를 지우느냐 마느냐가 실제로 결과를 바꾸는지 숫자로 답한다.")
A("")
A(md(t35, ["정의", "두 버전 ARI", "두 버전 NMI", "ARI vs D1 (보존)", "ARI vs D1 (비보존)",
           "ARI vs C1_id (보존)", "ARI vs C1_id (비보존)", "특징진단 CCI 급성% (보존)", "특징진단 CCI 급성% (비보존)"],
     {"특징진단 CCI 급성% (보존)": 1, "특징진단 CCI 급성% (비보존)": 1}))
A("")
A(f"**답: 바꾸지 않는다.** 두 버전의 ARI 는 {sens.loc['COMORB-elix', '두 버전 ARI']:.3f} 로 "
  f"C1_id 의 seed 간 노이즈(0.599)보다 **높다.** 즉 주 진단 보존 여부로 생기는 차이가 같은 파이프라인을 "
  f"seed 만 바꿔 돌렸을 때의 흔들림보다 작다. ARI(D1)은 {sens.loc['COMORB-elix', 'ARI vs D1 (보존)']:.3f} → "
  f"{sens.loc['COMORB-elix', 'ARI vs D1 (비보존)']:.3f}, ARI(C1_id)은 "
  f"{sens.loc['COMORB-elix', 'ARI vs C1_id (보존)']:.3f} → {sens.loc['COMORB-elix', 'ARI vs C1_id (비보존)']:.3f} 로 "
  f"소수점 둘째 자리 차이다. 우려했던 '보존이 ARI(D1)을 구조적으로 부풀린다'는 효과는 "
  f"실제로는 0.013 에 그쳤다. 급성 비율은 보존 쪽이 오히려 높다"
  f"({sens.loc['COMORB-elix', '특징진단 CCI 급성% (보존)']:.1f}% vs "
  f"{sens.loc['COMORB-elix', '특징진단 CCI 급성% (비보존)']:.1f}%).")
A("")

A("## 5. 평가 — 격차와 순열검정")
A("")
A("기존 배터리를 그대로 썼다: test ≥30 방문 클러스터만, 상수 top-23 / copy-prev Jaccard, "
  "환자 단위 부트스트랩 1,000회 CI, 클러스터 크기(환자 수) 보존 환자블록 순열 10,000회, "
  "단측 p = (1 + #{null ≥ obs}) / (1 + n).")
A("")
A(md(t37, ["분할", "test≥30 클러스터", "jac_const 최소", "jac_const 최대", "jac_const 격차",
           "jac_prev 격차", "약물수-성능 상관(Pearson)", "약물수-성능 상관(Spearman)"], {}))
A("")
A(md(t36, ["partition", "predictor", "statistic", "observed", "null_mean", "null_p95", "p_value", "z_vs_null"],
     {"p_value": 5, "z_vs_null": 2, "observed": 4, "null_mean": 4, "null_p95": 4}))
A("")
A(f"**격차는 사라지지 않는다.** 네 정의 모두 상수 예측기 격차가 순열 귀무분포보다 크다"
  f"(COMORB-elix p={pget('COMORB-elix'):.4f}, CHR-persist{TAU_MAIN} p={pget(f'CHR-persist{TAU_MAIN}'):.4f}, "
  f"CHR-idf p={pget('CHR-idf'):.4f}). 가중 SD 기준으로는 전부 p<0.001 이다. "
  f"동반질환을 지우고 급성 축으로 다시 나눠도 클러스터 간 성능 격차는 남는다.")
A("")
A(f"**copy-prev 해리 (중요).** COMORB-elix 의 copy-prev 격차는 순열 귀무분포와 구분되지 않는다"
  f"(범위 p={pget('COMORB-elix', 'copy-prev'):.3f}, 가중 SD p={pget('COMORB-elix', 'copy-prev', '가중 SD'):.3f}, "
  f"관측값이 귀무 평균보다 오히려 **작다**: z={pget('COMORB-elix', 'copy-prev', col='z_vs_null'):.2f}). "
  f"CHR-persist{TAU_MAIN}(p={pget(f'CHR-persist{TAU_MAIN}', 'copy-prev'):.3f})도 "
  f"경계 이상이다. 상수 예측기에서만 격차가 나오고 copy-prev 에서는 나오지 않는 이 해리는 "
  f"C1_id 에서 관찰된 것과 같은 패턴이며, 격차가 '지엽적 질병이라 불리하다'보다 "
  f"**약물 수/처방 길이 구조**에서 온다는 해석과 일관된다.")
A("")
A(f"**약물 수 상관은 오히려 더 강해졌다.** COMORB-elix 의 클러스터 평균 약물 수와 성능의 상관은 "
  f"Pearson {GP.loc['COMORB-elix', '약물수-성능 상관(Pearson)']:.3f} 로, C1_id({REF['C1_id k=15']['corr']:.3f})보다 "
  f"훨씬 높다. 급성 축으로 나누면 클러스터가 중증도(=약물 수)로 정렬되기 때문이다. "
  f"CHR-idf 만 상관이 {GP.loc['CHR-idf', '약물수-성능 상관(Pearson)']:.3f} 로 낮다.")
A("")
A("![클러스터별 성능과 CI](figs/fig21_chronic_perf.png)")
A("")

A("## 6. 비교표")
A("")
A(md(cmp, ["분할", "입력", "제거", "라벨 수", "격차", "순열 p(상수·범위)", "약물수 상관",
           "특징진단 CCI 급성%", "ARI vs C1_id"],
     {"격차": 4, "순열 p(상수·범위)": 5, "약물수 상관": 3, "특징진단 CCI 급성%": 1, "ARI vs C1_id": 3}))
A("")
A("| 분할 | 한 줄 평 |")
A("|---|---|")
A("| C1_id k=15 | 사전 등록된 주 분할. 이 트랙의 결과와 무관하게 유지된다 |")
A("| C1_icdfull k=15 | 어휘만 4,514로 넓힌 것. 축은 그대로 만성 |")
A("| D1 주진단 chapter | 급성 비율은 가장 높지만(54.7%) 격차 순열 p=0.088 로 유의하지 않음 |")
A("| COMORB-elix | 축을 가장 많이 옮겼다(급성 52%). 그러나 격차 유지, 약물수 상관 0.84로 최악 |")
A(f"| CHR-persist{TAU_MAIN} | 52개 코드만 지워 사실상 C1_id 와 같은 축. 대조군 역할 |")
A(f"| CHR-persist{TAU_SCALE} | 규모를 맞춘 대조군. 급성 44%로 중간 — 축 이동의 상당 부분이 '제거 규모' 효과임을 보여줌 |")
A("| CHR-idf | 제거 없이 가중만. 가장 만성적이고 격차가 가장 크며(0.1030) seed 안정성 최저 |")
A("| COMORB-elix-rmseq1 | 주 진단까지 지운 민감도. 보존판과 ARI 0.648 — 결론 불변 |")
A(f"| CHR-persist{TAU_MAIN}-rmseq1 | 동상 (ARI 0.505) |")
A("")

A("## 7. 결론")
A("")
A("1. **§4-1 판정: 축은 부분적으로 옮겨갔다.** 급성 특징 진단 비율 41.9% → 52.0%, 급성 과반 클러스터 5 → 8개. "
  "고혈압·AF·심부전·ESRD 전용 클러스터는 사라지고 폐렴·패혈증·요로감염·급성출혈·급성신부전 클러스터가 생겼다.")
A("2. **그러나 만성 축은 제거되지 않고 이동했다.** 남은 클러스터의 절반은 Elixhauser 밖 만성 표지"
  "(고지혈증·고콜레스테롤·관상동맥경화·흡연·GERD·시술 상태)로 정의된다. "
  "동반질환 목록 하나를 지우는 방식으로는 만성 축을 없앨 수 없다는 것이 이 실험의 실질적 결과다.")
A("3. **성능 격차는 남는다.** 네 정의 모두 상수 예측기 격차가 순열 유의. "
  "그리고 약물 수 상관이 0.84로 C1_id(0.24)보다 강해졌다 — 급성 축은 중증도 축이기도 하다.")
A("4. **copy-prev 해리가 재현된다.** 상수 예측기에서만 격차가 나오고 copy-prev 에서는 귀무와 구분되지 않는다.")
A("5. **주 진단 보존 여부는 결론을 바꾸지 않는다**(두 버전 ARI 0.648 > seed 노이즈 0.599).")
A("6. 이 트랙은 사후 분석이다. **주 분할은 C1_id k=15 로 유지한다.**")
A("")

A("## 8. 파일 목록")
A("")
A("| 파일 | 내용 |")
A("|---|---|")
A("| `out/chronic_partition_assignments.csv` | 방문별 배정 — SUBJECT_ID, HADM_ID, split, "
  "comorb_elix_label, chr_persist_label, chr_persist_scale_label, chr_idf_label, "
  "comorb_elix_rmseq1_label, chr_persist_rmseq1_label, n_remaining_dx(=COMORB-elix 기준), "
  "n_remaining_persist, n_remaining_persist_scale, c1_id_label, d1_label |")
A("| `out/27_chronic_sets.json` | 제거 코드 집합, Elixhauser 매핑 출처, CCI 만성/급성 판정, IDF 값 |")
A("| `out/27_chronic_recurrence.csv` | 코드별 train 재출현률 (n_pat_train_multi, n_pat_repeat, recur) |")
A("| `out/27_chronic_lists.pkl` | 정의별 방문당 남은 코드 리스트 |")
A("| `out/28_chronic_labels.pkl` | 정의별 라벨 + k=5~30 스윕 라벨 |")
A("| `out/table21~28_chronic_*.csv` | §1~§2 제거 규모·τ 스캔·겹침·급성 카테고리·주진단 보존 |")
A("| `out/table29~35_chronic_*.csv` | §3~§4 구조 지표·k 스윕·seed·축 판정·클러스터 성능·특징 진단·주진단 민감도 |")
A("| `out/table36~38_chronic_*.csv` | §5~§6 순열검정·격차 요약·비교표 |")
A("| `out/perm_null_chronic_*.npz` | 순열 귀무분포 원본 (정의별 10,000×4) |")
A("| `out/figs/fig20_chronic_heatmap.png` | 정의별 클러스터 상위 진단 히트맵 |")
A("| `out/figs/fig21_chronic_perf.png` | 클러스터별 성능 막대 + 95% CI |")
A("")
A("재생성: `27_chronic_prep.py` → `28_chronic_cluster.py` → `29_chronic_perm.py` → `30_chronic_report.py`")

(OUT / "REPORT_CHRONIC.md").write_text("\n".join(L), encoding="utf-8")
print(f"[.] REPORT_CHRONIC.md {len(L)} 줄", flush=True)
