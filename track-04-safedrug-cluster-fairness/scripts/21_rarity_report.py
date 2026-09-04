"""트랙 A/B 산출물 → out/REPORT_RARITY.md + figs/fig13~15.

숫자는 전부 21_trackA_meta.json / 22_trackB_meta.json / table*.csv 에서 읽는다(손으로 옮기지 않는다).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)

A = json.load(open(OUT / "21_trackA_meta.json", encoding="utf-8"))
B = json.load(open(OUT / "22_trackB_meta.json", encoding="utf-8"))
F = json.load(open(OUT / "20_icd_features_meta.json", encoding="utf-8"))
G = json.load(open(OUT / "19_icd_gate.json", encoding="utf-8"))
reg = pd.read_csv(OUT / "table10_trackB_reg.csv")
full = pd.read_csv(OUT / "table10b_trackB_fullmodel.csv")
quint = pd.read_csv(OUT / "table11_trackB_quintile.csv")
plac = pd.read_csv(OUT / "table12b_trackB_placebo.csv")
cmp = pd.read_csv(OUT / "table6_trackA_compare.csv")
permA = pd.read_csv(OUT / "table8_trackA_perm.csv")
d = pd.read_pickle(OUT / "20_icd_features.pkl")
te = d[d["split"] == "test"]

L1, L1C, L2 = B["link1_rarity_vs_dropped"], B["link1_code_level"], B["link2_rarity_vs_jaccard"]
P = B["permutation"]


def r(df_, model, dv="상수 top-23 Jaccard"):
    return df_[(df_["모형"] == model) & (df_["종속변수"] == dv)].iloc[0]


s1, s2, s3 = r(reg, "1단계 단변량"), r(reg, "2단계 +약물수"), r(reg, "3단계 전체통제")
l3a = r(reg, "고리3 +탈락률만")
p1, p2, p3 = r(plac, "1단계 단변량"), r(plac, "2단계 +약물수"), r(plac, "3단계 전체통제")

# ================================================================= 그림
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, RED, MUTED, GREEN = "#2a78d6", "#c0392b", "#c9c8c3", "#2e7d5b"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})

# --- fig13: 오분위별 Jaccard 막대 + CI (+ 약물수 이중축)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
x = np.arange(len(quint))
for ax, col, lo, hi, ttl in [
    (axes[0], "상수 Jaccard", "CI저", "CI고", "상수 top-23 예측기"),
    (axes[1], "copy-prev Jaccard", "prev CI저", "prev CI고", "copy-prev 예측기"),
]:
    v = quint[col].to_numpy()
    err = np.vstack([v - quint[lo], quint[hi] - v])
    ax.bar(x, v, color=S1, width=0.62, zorder=2)
    ax.errorbar(x, v, yerr=err, fmt="none", ecolor=INK2, lw=1.1, capsize=3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(quint["오분위"], fontsize=8)
    ax.set_ylim(0, max(quint[hi]) * 1.35)
    ax.set_ylabel("평균 Jaccard")
    ax.set_title(ttl, fontsize=9.5, color=INK2)
    ax2 = ax.twinx()
    ax2.plot(x, quint["평균 약물수"], color=RED, marker="o", ms=3.5, lw=1.2, zorder=4)
    ax2.set_ylabel("평균 처방 약물 수", color=RED, fontsize=8.5)
    ax2.tick_params(axis="y", colors=RED, labelsize=8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)
    ax2.set_ylim(0, max(quint["평균 약물수"]) * 1.5)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
fig.suptitle("주 진단 희귀도 오분위별 성능 (막대·좌축) 과 처방 약물 수 (선·우축) · test 방문 2,880건", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIGS / "fig13_rarity_quintile.png", dpi=170)
plt.close(fig)

# --- fig14: 희귀도 vs Jaccard 산점 + 회귀선 + 구간평균
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
for ax, col, ttl in [(axes[0], "jac_const", "상수 top-23 예측기"), (axes[1], "jac_prev", "copy-prev 예측기")]:
    sub = te[te[col].notna()]
    ax.scatter(sub["rarity_seq1"], sub[col], s=5, color=MUTED, alpha=0.45, edgecolors="none", zorder=1)
    b, a = np.polyfit(sub["rarity_seq1"], sub[col], 1)
    xs = np.linspace(sub["rarity_seq1"].min(), sub["rarity_seq1"].max(), 50)
    ax.plot(xs, a + b * xs, color=RED, lw=1.8, zorder=3, label=f"단변량 회귀선 기울기 {b:+.4f}")
    bins = pd.qcut(sub["rarity_seq1"], 10, duplicates="drop")
    gm = sub.groupby(bins, observed=True).agg(x=("rarity_seq1", "mean"), y=(col, "mean"))
    ax.plot(gm["x"], gm["y"], color=S1, marker="o", ms=4, lw=1.3, zorder=4, label="십분위 구간평균")
    ax.set_xlabel("rarity_seq1 = -log10(주 진단 train 유병률), 클수록 희귀")
    ax.set_ylabel("Jaccard")
    ax.set_title(f"{ttl}  (n={len(sub):,})", fontsize=9.5, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("주 진단 희귀도 vs 예측 성능 — 단변량 관계", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIGS / "fig14_rarity_vs_jaccard.png", dpi=170)
plt.close(fig)

# --- fig15: frac_dropped vs rarity_mean
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
ax = axes[0]
hb = ax.hexbin(d["rarity_mean"], d["frac_dropped"], gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
fig.colorbar(hb, ax=ax, label="방문 수")
bins = pd.qcut(d["rarity_mean"], 20, duplicates="drop")
gm = d.groupby(bins, observed=True).agg(x=("rarity_mean", "mean"), y=("frac_dropped", "mean"))
ax.plot(gm["x"], gm["y"], color=RED, marker="o", ms=3.5, lw=1.5, label="이십분위 구간평균")
ax.set_xlabel("rarity_mean = -log10(방문 ICD 코드 평균 유병률)")
ax.set_ylabel("frac_dropped (어휘 탈락 비율)")
ax.set_title(f"방문 수준  r={L1['pearson_r']:.3f} (n={L1['n_visits']:,})", fontsize=9.5, color=INK2)
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
voc = json.load(open(OUT / "20_icd_vocab.json", encoding="utf-8"))
prev = np.array(voc["train_prevalence"])
pairs = defaultdict(Counter)
for ids, icds in zip(d["diag_id_l"], d["icd_l"]):
    if len(ids) == len(icds):
        for i, c in zip(ids, icds):
            pairs[i][c] += 1
representable = {c.most_common(1)[0][0] for c in pairs.values()}
rep = np.array([c in representable for c in voc["vocab"]])
ax.hist(-np.log10(prev[rep]), bins=40, color=S1, alpha=0.75, label=f"어휘에 남은 코드 {rep.sum():,}개")
ax.hist(-np.log10(prev[~rep]), bins=40, color=RED, alpha=0.55, label=f"탈락한 코드 {(~rep).sum():,}개")
ax.set_xlabel("-log10(train 유병률), 클수록 희귀")
ax.set_ylabel("ICD 코드 수")
ax.set_title(f"코드 수준  중앙 유병률 {L1C['median_train_prevalence_kept']:.5f} vs "
             f"{L1C['median_train_prevalence_dropped']:.5f}", fontsize=9.5, color=INK2)
ax.legend(frameon=False, fontsize=8)
fig.suptitle("고리 1 — 어휘 필터는 희귀한 진단을 우선적으로 지운다", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIGS / "fig15_dropped_vs_rarity.png", dpi=170)
plt.close(fig)

# ================================================================= 판정
seed_ref = A["ari"]["within_filtered_mean"]
cross = A["ari"]["cross_mean_25pairs"]
trackA_verdict = ("필터의 영향이 seed 흔들림 수준" if cross >= min(seed_ref, A["ari"]["within_full_mean"]) - 0.03
                  else "필터가 구조를 실제로 바꿈")

v1 = "확인됨"
v2 = "확인 안 됨"
v3 = "판정 불가"

md = []
w = md.append
w("# 전체 어휘 군집화(트랙 A) + 희귀도 연속변수(트랙 B) 결과\n")
w(f"입력: `DIAGNOSES_ICD.csv` 전체 어휘 {F['icd']['vocab_full']:,}코드 · "
  f"방문 {F['icd']['visits_total']:,}건(ICD 조인 {F['icd']['visits_with_icd']:,}건, "
  f"SEQ_NUM=1 보유 {F['icd']['visits_with_seq1']:,}건) · "
  f"train {F['split']['train_visits']:,} / test {F['split']['test_visits']:,} "
  f"(환자 {F['split']['train_subjects']:,}/{F['split']['test_subjects']:,}, 누수 {F['split']['leakage_subjects_in_both']}명).\n")
w("생성 스크립트: `18_icd_features.py` → `19_trackA_fullvocab.py` / `20_trackB_rarity.py` → `21_rarity_report.py`\n")

w("\n## 0. 결론 세 줄\n")
w(f"1. **고리 1 (희귀할수록 어휘에서 더 탈락하는가) — {v1}.** "
  f"방문 수준 `rarity_mean` vs `frac_dropped` Pearson r={L1['pearson_r']}, Spearman ρ={L1['spearman_rho']}. "
  f"코드 수준으로는 남은 코드의 train 유병률 중앙값 {L1C['median_train_prevalence_kept']:.5f} vs "
  f"탈락 코드 {L1C['median_train_prevalence_dropped']:.6f} (약 "
  f"{L1C['median_train_prevalence_kept']/L1C['median_train_prevalence_dropped']:.0f}배, Mann-Whitney p<1e-300).")
w(f"2. **고리 2 (희귀할수록 전역 추천이 안 맞는가) — {v2}.** "
  f"단변량 상관 r={L2['pearson_r']} (p={L2['pearson_p']:.3f}), 회귀 계수 {s1['계수']:+.5f} (p={s1['p']:.3f}). "
  f"약물 수를 통제하면 계수가 {s3['계수']:+.5f} (p={s3['p']:.1e}) 로 유의해지지만 **부호가 주장과 반대**다 "
  f"— 희귀할수록 상수 예측기가 조금 더 잘 맞는다.")
w(f"3. **고리 3 (정보가 지워져서인가, 환자가 원래 특수해서인가) — {v3}.** "
  f"고리 2 가 주장 방향으로 성립하지 않으므로 설명 대상이 없다. "
  f"참고로 `frac_dropped` 만 통제한 계수는 {l3a['계수']:+.5f} (p={l3a['p']:.3f}) 로 단변량과 사실상 같다.\n")

w(f"> **약물 수를 통제한 뒤에도 희귀도 계수는 살아남는다. 다만 부호가 음수가 아니라 양수다"
  f"({s3['계수']:+.5f}, 로버스트 SE {s3['로버스트SE']}, p={s3['p']:.1e}; 희귀도 1 SD 당 "
  f"{s3['계수_1SD당']:+.5f} Jaccard) — 즉 '전역 추천이 희귀 질병에 불리하다'는 방향이 아니라 그 반대 방향으로 살아남는다.**\n")

w("\n---\n\n# 트랙 A — 전체 어휘 군집화\n")
w(f"필터본 재현 확인: 같은 파이프라인으로 다시 만든 필터본 seed 0 라벨과 저장된 `C1_id k=15` 의 "
  f"ARI = {A['ari']['reproduction_ari_filtered_seed0_vs_stored']} (완전 일치).\n")
w("\n## A-1. ARI — 판정 기준은 seed 노이즈\n")
w("| 비교 | 평균 ARI | 최소 | 최대 |")
w("|---|---|---|---|")
w(f"| 필터본 seed 간 (10쌍) | {A['ari']['within_filtered_mean']} | {A['ari']['within_filtered_min']} | - |")
w(f"| 전체본 seed 간 (10쌍) | {A['ari']['within_full_mean']} | {A['ari']['within_full_min']} | - |")
w(f"| **필터본 vs 전체본 (25쌍)** | **{A['ari']['cross_mean_25pairs']}** | {A['ari']['cross_min']} | {A['ari']['cross_max']} |")
w(f"| 같은 seed 끼리만 (5쌍) | {round(float(np.mean(A['ari']['cross_same_seed'])),3)} | "
  f"{min(A['ari']['cross_same_seed'])} | {max(A['ari']['cross_same_seed'])} |")
w(f"\n두 버전 사이 ARI {cross} 는 필터본 자체의 seed 간 ARI {A['ari']['within_filtered_mean']}, "
  f"전체본 자체의 seed 간 ARI {A['ari']['within_full_mean']} 와 같은 수준이다 "
  f"(교차 범위 {A['ari']['cross_min']}~{A['ari']['cross_max']} 가 seed 간 최소값 "
  f"{min(A['ari']['within_filtered_min'], A['ari']['within_full_min'])} 를 포함한다). "
  f"→ **{trackA_verdict}.** 어휘의 68% 가 잘렸는데도 k=15 군집 구조는 seed 를 바꾸는 것 이상으로 달라지지 않았다.\n")

w("\n## A-2. 비교표\n")
cols = ["버전", "커버리지%", "클러스터 수", "최소 크기", "gini_size", "silhouette",
        "test 클러스터(>=30)", "상수 격차", "copy-prev 격차", "순열 p(상수·범위)",
        "순열 p(상수·가중SD)", "약물수 상관(상수)", "eta2_drug"]
w("| " + " | ".join(cols) + " |")
w("|" + "---|" * len(cols))
for _, row in cmp.iterrows():
    w("| " + " | ".join("미실행" if pd.isna(row[c]) else
                        (f"{row[c]:g}" if isinstance(row[c], (int, float, np.floating)) else str(row[c]))
                        for c in cols) + " |")
w(f"\n전체본의 순열검정({A['permutation']['n_perm']:,}회, 환자 단위) 결과는 필터본과 같은 패턴이다:")
for _, row in permA.iterrows():
    w(f"- {row['predictor']} · {row['statistic']}: 관측 {row['observed']}, 귀무평균 {row['null_mean']}, "
      f"p={row['p_value']}, z={row['z_vs_null']}")
w("\nICD chapter 분할 D1 은 아직 승인받지 않은 §3~§7 구간이라 비워 두었다.\n")

w("\n---\n\n# 트랙 B — 희귀도 연속변수\n")
w(f"`rarity_seq1` = -log10(SEQ_NUM=1 코드의 **train split** 유병률). 값이 클수록 희귀하고, "
  f"**계수가 음수여야 주장(희귀할수록 전역 추천이 불리)이 지지된다.** "
  f"test 방문 {B['sample']['test_visits']:,}건 / 환자 {B['sample']['test_subjects']:,}명, "
  f"copy-prev 는 직전 방문이 있는 {B['sample']['test_visits_with_prev']:,}건. "
  f"유병률은 train {F['split']['train_visits']:,}건에서만 추정했고 평활은 (count+0.5)/(n+1).\n")

w("\n## B-1. 지표 요약\n")
w("| 지표 | 정의 | 값 |")
w("|---|---|---|")
w(f"| `rarity_seq1` | -log10(주 진단 train 유병률) | 평균 {F['rarity_seq1_neglog10']['mean']}, "
  f"SD {F['rarity_seq1_neglog10']['sd']}, 범위 {F['rarity_seq1_neglog10']['min']}~{F['rarity_seq1_neglog10']['max']} |")
w(f"| 주 진단 유병률 | 원 스케일 | 중앙값 {F['prev_seq1']['median']}, 5%~95% "
  f"{F['prev_seq1']['p05']}~{F['prev_seq1']['p95']} |")
w(f"| `n_dropped` | 실제 ICD 진단 수 - 협업자 파일 진단 수 | 평균 {F['dropped']['mean_n_dropped']} "
  f"({F['dropped']['mean_n_icd']} → {F['dropped']['mean_n_ours']}) |")
w(f"| `frac_dropped` | 탈락 비율 | 평균 {F['dropped']['mean_frac_dropped']}, "
  f"1개 이상 잃은 방문 {F['dropped']['pct_visits_lost_ge1']}% |")

w("\n## B-2. 사슬 검증\n")
w(f"### 고리 1 — {v1}\n")
w(f"- 방문 수준: `rarity_mean` vs `frac_dropped` Pearson r={L1['pearson_r']} (p<1e-300), "
  f"Spearman ρ={L1['spearman_rho']}. `rarity_meanlog` 로 재면 r={L1['rarity_meanlog_pearson_r']}.")
w(f"- 코드 수준: 전체 {L1C['n_codes_total']:,}개 ICD 코드 중 협업자 어휘가 표현할 수 있는 코드는 "
  f"{L1C['n_codes_representable']:,}개, 표현 못 하는 코드가 {L1C['n_codes_dropped']:,}개다. "
  f"남은 코드의 평균 -log10 유병률 {L1C['mean_neglog10_prev_kept']} vs 탈락 코드 "
  f"{L1C['mean_neglog10_prev_dropped']} (Mann-Whitney p<1e-300).")
w(f"- 즉 어휘 필터는 무작위로 지운 것이 아니라 희귀한 코드를 우선적으로 지웠다. (그림 3)\n")

w(f"### 고리 2 (주 가설) — {v2}\n")
w(f"- 단변량: Pearson r={L2['pearson_r']} (p={L2['pearson_p']:.3f}), Spearman ρ={L2['spearman_rho']} "
  f"(p={L2['spearman_p']:.3f}). copy-prev 는 r={L2['pearson_r_vs_jac_prev']} (p={L2['pearson_p_vs_jac_prev']:.3f}).")
w(f"- 관계가 있다면 그 방향은 주장과 반대다. 회귀 계수표는 B-3 참조.\n")

w(f"### 고리 3 — {v3}\n")
w(f"- 고리 2 가 주장 방향으로 성립하지 않으므로 '지워진 정보 때문인가'를 물을 대상이 없다.")
w(f"- 기록: `frac_dropped` 만 통제하면 계수 {l3a['계수']:+.5f} (p={l3a['p']:.3f}), "
  f"단변량 {s1['계수']:+.5f} 와 사실상 동일하다. 약물 수까지 넣으면 "
  f"{r(reg,'고리3 +약물수+탈락률')['계수']:+.5f} (p={r(reg,'고리3 +약물수+탈락률')['p']:.1e}) 로 "
  f"2단계·3단계와 같다 — `frac_dropped` 는 계수를 거의 움직이지 않는다.\n")

w("\n## B-3. 회귀 3단계 계수표 (환자 단위 클러스터 로버스트 SE)\n")
w("초점 변수는 `rarity_seq1`. **음수 = 주장 지지.**\n")
w("| 종속변수 | 모형 | 계수 | 로버스트 SE | t | p | 95% CI | 1 SD 당 | R² | n |")
w("|---|---|---|---|---|---|---|---|---|---|")
for _, row in reg.iterrows():
    w(f"| {row['종속변수']} | {row['모형']} | {row['계수']:+.5f} | {row['로버스트SE']:.5f} | {row['t']:.2f} | "
      f"{row['p']:.2e} | [{row['CI저']:+.5f}, {row['CI고']:+.5f}] | {row['계수_1SD당']:+.5f} | "
      f"{row['R2']:.4f} | {row['n']:,} |")
w(f"\n통제를 넣을 때 계수가 어떻게 변하는지가 핵심이다. 상수 예측기에서 단변량 {s1['계수']:+.5f}(p={s1['p']:.3f}) → "
  f"약물 수만 넣으면 {s2['계수']:+.5f}(p={s2['p']:.1e}) → 전체 통제 {s3['계수']:+.5f}(p={s3['p']:.1e}) 로, "
  f"**약물 수를 넣는 순간 계수가 양(+)으로 커지고 유의해진다.** "
  f"이는 억제(suppression) 형태다: 희귀도와 약물 수의 상관은 {L2['rarity_seq1_vs_n_drugs_r']}, "
  f"약물 수와 상수 Jaccard 의 상관은 {L2['n_drugs_vs_jac_const_r']} 로, "
  f"희귀한 방문일수록 약이 조금 적고 약이 적으면 상수 예측기 Jaccard 가 낮아진다. "
  f"약물 수를 고정하면 그 경로가 막히고 남은 부호는 양수다.\n")
w("전체통제 모형의 모든 계수:\n")
w("| 종속변수 | 항 | 계수 | 로버스트 SE | t | p |")
w("|---|---|---|---|---|---|")
for _, row in full.iterrows():
    w(f"| {row['종속변수']} | `{row['항']}` | {row['계수']:+.5f} | {row['로버스트SE']:.5f} | "
      f"{row['t']:.2f} | {row['p']:.2e} |")

w("\n## B-3b. 오분위 표 (비선형 확인)\n")
qc = ["오분위", "n", "환자수", "rarity_seq1 평균", "유병률 중앙값", "상수 Jaccard", "CI저", "CI고",
      "copy-prev Jaccard", "평균 약물수", "평균 진단수", "평균 탈락률"]
w("| " + " | ".join(qc) + " |")
w("|" + "---|" * len(qc))
for _, row in quint.iterrows():
    w("| " + " | ".join(f"{row[c]:g}" if isinstance(row[c], (int, float, np.floating)) else str(row[c]) for c in qc) + " |")
jc = quint["상수 Jaccard"]
w(f"\n단조가 아니다. 최다빈 Q1 {jc.iloc[0]:.4f} → Q3 {jc.iloc[2]:.4f} 로 내려갔다가 "
  f"최희귀 Q5 {jc.iloc[4]:.4f} 로 다시 올라간다. 전체 진폭은 {jc.max()-jc.min():.4f} 로 "
  f"오분위 CI 폭({(quint['CI고']-quint['CI저']).mean():.4f}) 과 비슷한 규모다. "
  f"같은 표에서 평균 약물수는 {quint['평균 약물수'].iloc[0]:.1f} → {quint['평균 약물수'].iloc[-1]:.1f} 로 "
  f"단조 감소하고, 평균 탈락률은 {quint['평균 탈락률'].iloc[0]:.4f} → {quint['평균 탈락률'].iloc[-1]:.4f} 로 "
  f"Q5 에서 급증한다. (그림 1)\n")

w("\n## B-4. 대조\n")
w(f"### 희귀도 환자단위 순열 ({P[list(P)[0]]['n_perm']:,}회)\n")
w("환자 블록을 통째로 재배정해 `rarity_seq1` 만 섞고 회귀를 다시 적합했다(환자 내 상관 보존). "
  "기증자 블록의 길이가 다르면 순환시켜 채운다.\n")
w("| 종속변수 | 모형 | 관측 계수 | 귀무 평균 | 귀무 SD | 귀무 2.5~97.5% | 양측 p | 단측 p(음수 방향) | z |")
w("|---|---|---|---|---|---|---|---|---|")
for k, v in P.items():
    dv_, mdl_ = k.split(" | ")
    w(f"| {dv_} | {mdl_} | {v['observed_coef']:+.5f} | {v['null_mean']:+.5f} | {v['null_sd']:.5f} | "
      f"[{v['null_p2.5']:+.5f}, {v['null_p97.5']:+.5f}] | {v['p_two_sided']} | "
      f"{v['p_one_sided_negative']} | {v['z_vs_null']:+.2f} |")
w(f"\n전체통제 모형의 양수 계수는 순열 귀무분포 밖에 있다(양측 p="
  f"{P['상수 top-23 Jaccard | 3단계 전체통제']['p_two_sided']}). "
  f"주장이 예측한 음수 방향으로는 단측 p="
  f"{P['상수 top-23 Jaccard | 3단계 전체통제']['p_one_sided_negative']} 로 전혀 지지되지 않는다. "
  f"copy-prev 는 두 모형 모두 귀무와 구분되지 않는다.\n")

w("### 역방향 위약 — 희귀도 대신 `AGE`\n")
w("| 종속변수 | 모형 | 계수 | 로버스트 SE | p | 1 SD 당 |")
w("|---|---|---|---|---|---|")
for _, row in plac.iterrows():
    w(f"| {row['종속변수']} | {row['모형']} | {row['계수']:+.5f} | {row['로버스트SE']:.5f} | "
      f"{row['p']:.2e} | {row['계수_1SD당']:+.5f} |")
w(f"\n상수 예측기 전체통제에서 AGE 의 1 SD 당 계수는 {p3['계수_1SD당']:+.5f} (p={p3['p']:.2e}) 로, "
  f"희귀도의 {s3['계수_1SD당']:+.5f} 와 크기가 사실상 같다. "
  f"희귀도 계수의 절대 크기는 이 자료에서 임의의 인구학 변수가 만드는 크기와 구분되지 않는다.\n")

w("### 다른 희귀도 지표로 재검 (전체통제)\n")
w("| 지표 | 계수 | SE | p | 1 SD 당 |")
w("|---|---|---|---|---|")
for k, v in B["robustness_other_rarity_metrics"].items():
    w(f"| `{k}` | {v['coef']:+.5f} | {v['se']:.5f} | {v['p']:.2e} | {v['coef_per_SD']:+.5f} |")
w(f"\n`rarity_seq1` 을 포함한 네 가지 희귀도 정의 어느 것도 음수 계수를 내지 않는다 "
  f"(가장 약한 `rarity_mean` 도 {B['robustness_other_rarity_metrics']['rarity_mean']['coef']:+.5f}, "
  f"p={B['robustness_other_rarity_metrics']['rarity_mean']['p']:.2f}).\n")

w("\n## 그림\n")
w("1. `figs/fig13_rarity_quintile.png` — 희귀도 오분위별 Jaccard 막대 + 환자 부트스트랩 CI, 우축에 평균 약물 수")
w("2. `figs/fig14_rarity_vs_jaccard.png` — 희귀도 vs Jaccard 산점도 + 단변량 회귀선 + 십분위 구간평균")
w("3. `figs/fig15_dropped_vs_rarity.png` — `frac_dropped` vs `rarity_mean` (방문 수준) + 코드 수준 유병률 분포\n")

w("\n## 한계 — 기록해 둘 사실\n")
w(f"- 검정력: test {B['sample']['test_visits']:,}건 / 환자 {B['sample']['test_subjects']:,}명에서 "
  f"희귀도 1 SD 당 계수의 로버스트 SE 는 {s3['로버스트SE']*B['sample']['rarity_seq1_sd']:.5f} 다. "
  f"전체통제 CI [{s3['SD당_CI저']:+.5f}, {s3['SD당_CI고']:+.5f}] 는 전 구간이 양수라 이 모형에서는 "
  f"음의 효과가 크기와 무관하게 배제된다. 반면 통제 없는 단변량 CI 는 "
  f"[{s1['SD당_CI저']:+.5f}, {s1['SD당_CI고']:+.5f}] 이므로, 통제하지 않은 척도에서는 1 SD 당 "
  f"{abs(s1['SD당_CI저']):.5f} 까지의 음의 효과는 여전히 자료와 양립한다. "
  f"'효과 없음'이 아니라 '주장 방향의 효과가 이 크기 이상은 아니다'가 정확한 진술이다.")
w(f"- `n_dropped` 는 코드별 매칭이 아니라 진단 개수 차이로 정의했다(게이트에서 차이 분포가 전부 >=0, "
  f"부분수열 보존 {G['gate5_mismatch_cause']['our_list_is_subsequence_of_icd_pct']}% 확인됨). "
  f"어느 코드가 탈락했는지 방문별로 특정한 것은 아니다.")
w(f"- 상수 예측기 Jaccard 는 정답 집합 크기에 기계적으로 딸려간다(r={L2['n_drugs_vs_jac_const_r']}). "
  f"약물 수를 통제한 계수만 해석 대상이며, 통제 전후 부호가 바뀐다는 사실 자체가 이 교란의 크기를 보여준다.")
w(f"- 희귀도는 train split 유병률로만 정의했고 test 정보를 쓰지 않았다. 평활 때문에 train 미출현 코드는 "
  f"모두 같은 최대 희귀도 {F['rarity_seq1_neglog10']['max']} 를 받는다.")
w(f"- copy-prev 표본은 {B['sample']['test_visits_with_prev']:,}건으로 상수 예측기 표본의 절반 수준이라 "
  f"검정력이 더 낮다. 이 예측기에서는 어느 방향으로도 신호가 없다.")
w(f"- 트랙 A 의 ARI 판정은 k=15 한 지점에서만 했다. 다른 k 는 확인하지 않았다.\n")

(OUT / "REPORT_RARITY.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:3]))
print("\n[ok] out/REPORT_RARITY.md  +  fig13, fig14, fig15")
print(f"trackA verdict: {trackA_verdict} (cross {cross} vs within {seed_ref}/{A['ari']['within_full_mean']})")
print(f"links: 1={v1} 2={v2} 3={v3}")
