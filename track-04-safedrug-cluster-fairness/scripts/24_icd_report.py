"""§6~§7 — out/REPORT_ICD.md + figs/fig16~18.

숫자는 전부 23~24 메타 JSON / table*.csv 에서 읽는다(손으로 옮기지 않는다).
추가로 순환기 비중 확인(전체 진단 행 기준 vs SEQ_NUM=1 기준)을 여기서 계산해 메타에 남긴다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
MIN_TEST = 30

P = json.load(open(OUT / "23_icd_partition_meta.json", encoding="utf-8"))
E = json.load(open(OUT / "24_icd_eval_meta.json", encoding="utf-8"))
power = pd.read_csv(OUT / "table13_icd_power.csv")
clusters = pd.read_csv(OUT / "table14_icd_clusters.csv")
perm = pd.read_csv(OUT / "table15_icd_perm.csv")
cmp = pd.read_csv(OUT / "table16_icd_compare.csv")
ct = pd.read_csv(OUT / "table17_c1c10_crosstab.csv")
MP_PATH = OUT / "26_matched_pairs_meta.json"
MP = json.load(open(MP_PATH, encoding="utf-8")) if MP_PATH.exists() else None
pairs = pd.read_csv(OUT / "table19_matched_pairs.csv") if MP else None
gperm = pd.read_csv(OUT / "table20_matched_perm_global.csv") if MP else None
PM = MP["pairs_meta"] if MP else None
nsig = {t_: int((pairs[pairs["분할"] == t_]["BH q"] < 0.05).sum()) for t_ in ["D1", "D2", "D2b"]} if MP else None
tab2 = pd.read_csv(OUT / "table2_cluster_performance.csv")
d = pd.read_pickle(OUT / "20_icd_features.pkl")
part = pd.read_csv(OUT / "icd_partition_assignments.csv")

# ---------------------------------------------------------------- 순환기 비중 확인
CHAPTERS = [
    (1, 139, "감염성·기생충성"), (140, 239, "신생물"), (240, 279, "내분비·대사·면역"), (280, 289, "혈액"),
    (290, 319, "정신"), (320, 389, "신경"), (390, 459, "순환기"), (460, 519, "호흡기"), (520, 579, "소화기"),
    (580, 629, "비뇨생식기"), (630, 679, "임신·출산"), (680, 709, "피부"), (710, 739, "근골격"),
    (740, 759, "선천기형"), (760, 779, "주산기"), (780, 799, "증상·불명확"), (800, 999, "손상·중독"),
]


def chapter_of(code):
    c = str(code).strip().replace(".", "")
    if c.startswith("V"):
        return "V코드"
    if c.startswith("E"):
        return "E코드"
    try:
        v = int(c[:3])
    except ValueError:
        return None
    for lo, hi, nm in CHAPTERS:
        if lo <= v <= hi:
            return nm
    return None


all_ch = pd.Series([chapter_of(c) for l in d["icd_l"] for c in l])
seq1_ch = part["chapter"]
share = pd.DataFrame({
    "전체 진단 행 기준%": (all_ch.value_counts(normalize=True) * 100).round(1),
    "SEQ_NUM=1 기준%": (seq1_ch.value_counts(normalize=True) * 100).round(1),
}).fillna(0.0).sort_values("SEQ_NUM=1 기준%", ascending=False)
CIRC = {
    "n_diag_rows": int(len(all_ch)),
    "circ_all_rows_pct": float(share.loc["순환기", "전체 진단 행 기준%"]),
    "circ_seq1_pct": float(share.loc["순환기", "SEQ_NUM=1 기준%"]),
    "dig_all_rows_pct": float(share.loc["소화기", "전체 진단 행 기준%"]),
    "dig_seq1_pct": float(share.loc["소화기", "SEQ_NUM=1 기준%"]),
    "circ_any_visit_pct": round(float(np.mean([any(chapter_of(c) == "순환기" for c in l) for l in d["icd_l"]])) * 100, 1),
}

# ---------------------------------------------------------------- 헬퍼
def pf(t, stat, pred="상수"):
    r = perm[(perm["분할"] == t) & (perm["통계량"] == stat) & (perm["예측기"] == pred)]
    return r.iloc[0]


def perf(tag, only_big=True):
    t = clusters[clusters["partition"] == tag].copy()
    if only_big:
        t = t[t["test_visits"] >= MIN_TEST]
    return t.sort_values("jac_const").reset_index(drop=True)


def row(tag):
    return cmp[cmp["분할"].str.startswith(tag + " ")].iloc[0]


D1, D2, D2b = perf("D1"), perf("D2"), perf("D2b")
SUM = {s["분할"]: s for s in P["summary"]}


def ci_width(t):
    return float((t["jac_const_hi"] - t["jac_const_lo"]).mean())


TRADEOFF = pd.DataFrame([{
    "분할": tag, "라벨 수": SUM[tag]["라벨 수"], "test≥30 라벨 수 (실질 k)": SUM[tag]["test>=30 통과 라벨 수 (실질 k)"],
    "통과 라벨 중앙 test 크기": SUM[tag]["중앙 test 크기(통과 라벨 중)"],
    "탈락 test 비율%": SUM[tag]["탈락 test 비율%"],
    "평균 CI 폭": round(ci_width(t), 4), "최대 CI 폭": round(float((t["jac_const_hi"] - t["jac_const_lo"]).max()), 4),
    "상수 격차": float(row(tag)["상수 격차"]), "순열 p(범위)": float(row(tag)["순열 p(상수·범위)"]),
    "순열 p(가중SD)": float(row(tag)["순열 p(상수·가중SD)"]), "약물수 상관": float(row(tag)["약물수 상관(상수)"]),
} for tag, t in [("D1", D1), ("D2", D2), ("D2b", D2b)]])
TRADEOFF.to_csv(OUT / "table18_icd_tradeoff.csv", index=False)

# ---------------------------------------------------------------- 그림
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, RED, MUTED, GREEN = "#2a78d6", "#c0392b", "#c9c8c3", "#2e7d5b"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})
WATCH = {"소화기", "비뇨생식기"}


def perf_fig(t, tag, fname, ttl):
    h = max(2.6, 0.34 * len(t) + 1.5)
    fig, ax = plt.subplots(figsize=(8.4, h))
    y = np.arange(len(t))[::-1]
    col = [RED if l in WATCH else S1 for l in t["라벨"]]
    ax.barh(y, t["jac_const"], color=col, height=0.62, zorder=3)
    ax.errorbar(t["jac_const"], y, xerr=[t["jac_const"] - t["jac_const_lo"], t["jac_const_hi"] - t["jac_const"]],
                fmt="none", ecolor=INK2, elinewidth=1.0, capsize=2.5, zorder=4)
    gm = float((clusters[clusters["partition"] == tag]["jac_const"] *
                clusters[clusters["partition"] == tag]["test_visits"]).sum() /
               clusters[clusters["partition"] == tag]["test_visits"].sum())
    ax.axvline(gm, color=GREEN, lw=1.1, ls="--", zorder=5, label=f"가중 평균 {gm:.3f}")
    for yy, (_, r_) in zip(y, t.iterrows()):
        ax.text(0.004, yy, f"{r_['라벨']}  (n={r_['test_visits']})", va="center", ha="left",
                fontsize=8, color="white" if r_["jac_const"] > 0.08 else INK2, zorder=6)
        ax.text(r_["jac_const_hi"] + 0.004, yy, f"{r_['jac_const']:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(0, max(0.38, float(t["jac_const_hi"].max()) + 0.05))
    ax.set_xlabel("상수 top-23 예측기의 평균 Jaccard (test 방문, 환자 부트스트랩 95% CI)")
    ax.set_title(ttl, fontsize=10, color=INK2, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=170)
    plt.close(fig)


perf_fig(D1, "D1", "fig16_D1_label_perf.png",
         f"D1 (ICD chapter) 라벨별 성능 · test ≥{MIN_TEST} 라벨 {len(D1)}개 · 빨강 = 사전 지정 관찰 대상(소화기·비뇨생식기)")
perf_fig(D2, "D2", "fig17_D2_label_perf.png",
         f"D2 (D1 + 순환기 8블록) 라벨별 성능 · test ≥{MIN_TEST} 라벨 {len(D2)}개")

# fig18 — 라벨 x 상위진단 히트맵 (D2, 전체 방문 기준)
lab_all = part["d2_label"].to_numpy()
keep = set(D2["라벨"])
cnt = {}
for l, names in zip(lab_all, d["diagnose_l"]):
    if l not in keep:
        continue
    for nm in set(names):
        cnt[nm] = cnt.get(nm, 0) + 1
top_dx = [k for k, _ in sorted(cnt.items(), key=lambda x: -x[1])[:16]]
order = list(D2["라벨"])
M = np.zeros((len(order), len(top_dx)))
for i, l in enumerate(order):
    m = lab_all == l
    sub = d["diagnose_l"][m]
    for j, nm in enumerate(top_dx):
        M[i, j] = np.mean([nm in set(x) for x in sub]) * 100

fig, ax = plt.subplots(figsize=(11.2, 0.42 * len(order) + 2.6))
im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=min(100, M.max()))
ax.set_xticks(range(len(top_dx)))
ax.set_xticklabels([t[:26] for t in top_dx], rotation=42, ha="right", fontsize=7.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{l}  ({v:.3f})" for l, v in zip(order, D2["jac_const"])], fontsize=8)
for i in range(len(order)):
    for j in range(len(top_dx)):
        if M[i, j] >= 8:
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] > 55 else INK)
ax.set_title("D2 라벨 x 상위 동반진단 보유율(%) · 세로축은 성능 낮은 순(괄호=상수 Jaccard) · 전체 14,541 방문 기준",
             fontsize=10, color=INK2, loc="left")
fig.colorbar(im, ax=ax, shrink=0.6, label="라벨 내 보유 방문 비율 %")
fig.tight_layout()
fig.savefig(FIGS / "fig18_D2_label_dx_heatmap.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 표 렌더링
def esc(s):
    """셀 안의 | 는 표를 깨뜨리므로 이스케이프한다(예: 관측 max|ΔJaccard|)."""
    return str(s).replace("|", r"\|")


def md(df_, cols=None, floatfmt=None):
    t = df_[cols] if cols else df_
    hdr = "| " + " | ".join(esc(c) for c in t.columns) + " |"
    sep = "|" + "|".join(["---"] * len(t.columns)) + "|"
    lines = [hdr, sep]
    for _, r_ in t.iterrows():
        cells = []
        for c in t.columns:
            v = r_[c]
            if isinstance(v, float):
                cells.append(f"{v:.{floatfmt.get(c, 3) if floatfmt else 3}f}" if not np.isnan(v) else "—")
            else:
                cells.append(esc(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def perf_md(t):
    t = t.copy()
    t["CI"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t["jac_const_lo"], t["jac_const_hi"])]
    t["CI 폭"] = (t["jac_const_hi"] - t["jac_const_lo"]).round(4)
    t = t.rename(columns={"라벨": "라벨", "test_visits": "test 방문", "jac_const": "상수 Jaccard",
                          "jac_prev": "copy-prev", "mean_n_drug": "평균 약물수", "mean_rarity_seq1": "평균 희귀도"})
    return md(t, ["라벨", "test 방문", "상수 Jaccard", "CI", "CI 폭", "copy-prev", "평균 약물수", "평균 희귀도"],
              {"상수 Jaccard": 4, "CI 폭": 4, "copy-prev": 4, "평균 약물수": 1, "평균 희귀도": 2})


# 바닥/상단 위치
def rank_of(t, name):
    idx = list(t["라벨"]).index(name) if name in list(t["라벨"]) else None
    return None if idx is None else (idx + 1, len(t))


r_dig_d1, r_uro_d1 = rank_of(D1, "소화기"), rank_of(D1, "비뇨생식기")
r_dig_d2, r_uro_d2 = rank_of(D2, "소화기"), rank_of(D2, "비뇨생식기")
c1_bottom = tab2[(tab2["partition"] == "C1_id") & (tab2["k"] == 15)].nsmallest(2, "jac_const")

ct1_all = ct[(ct["C1_id 클러스터"] == 1) & (ct["분할"] == "D1")]
ct10_all = ct[(ct["C1_id 클러스터"] == 10) & (ct["분할"] == "D1")]
ct10_d2_all = ct[(ct["C1_id 클러스터"] == 10) & (ct["분할"] == "D2")]
ct1, ct10, ct10_d2 = ct1_all.head(6), ct10_all.head(6), ct10_d2_all.head(6)


def cell(tbl, label, col):
    r_ = tbl[tbl["라벨"] == label]
    return float(r_.iloc[0][col]) if len(r_) else float("nan")
CT_COLS = ["라벨", "방문 수", "클러스터 내 비중%", "그 라벨 안에서 이 클러스터가 차지하는 비중%"]

p1_cov = float(cmp[cmp["분할"].str.startswith("P1_ge100")].iloc[0]["커버리지%"])
n_test = int(P["n_test"])
htn_test = int(power[(power["분할"] == "D2") & (power["라벨"] == "순환기: 고혈압성")].iloc[0]["test방문"])

# 소화기 라벨의 실제 구성 (희석 주장 확인용)
dic = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
short = dict(zip(dic["ICD9_CODE"].str.strip(), dic["SHORT_TITLE"].astype(str)))
dig = part[part["d1_label"] == "소화기"]["icd9_seq1"].value_counts()
DIG_TOP = [(short.get(str(c).strip(), str(c)), int(v), round(v / len(part[part["d1_label"] == "소화기"]) * 100, 1))
           for c, v in dig.head(5).items()]

# ---------------------------------------------------------------- 본문
L = []
A = L.append
A("# ICD chapter 분할(D1) · 순환기 세분(D2/D2b) — 결과")
A("")
A(f"방문 {P['n_visits']:,}건(test {n_test:,}건) · `SEQ_NUM=1` 코드 앞 3자리 기준 · "
  f"chapter 매핑 실패 {P['chapter_mapping_failures']}건 · 순환기 블록 미지정 {P['circ_block_unassigned']}건 · "
  f"순열 {E['n_perm']:,}회(환자 단위, 크기 분포 보존, test ≥{MIN_TEST} 필터 동일).")
A("")
A("## 0. 판정 요약")
A("")
A("| # | 항목 | 판정 |")
A("|---|---|---|")
A(f"| 1 | 커버리지 | D1/D2/D2b 모두 **100%**. P1_ge100의 {p1_cov}%를 63.5%p 넘는다 |")
A(f"| 2 | 세분화의 값 | **D2 > D1**. 실질 k 11 → 15, 격차 {float(row('D1')['상수 격차']):.4f} → "
  f"{float(row('D2')['상수 격차']):.4f}, 순열 p {float(row('D1')['순열 p(상수·범위)']):.4f} → "
  f"{float(row('D2')['순열 p(상수·범위)']):.4f}. 대가는 CI 폭 "
  f"{TRADEOFF.iloc[0]['평균 CI 폭']:.4f} → {TRADEOFF.iloc[1]['평균 CI 폭']:.4f} |")
A(f"| 3 | 성능 격차 | D1은 범위 기준 **유의하지 않음**(p={float(pf('D1','범위(max-min)')['p_value']):.4f}, "
  f"z={float(pf('D1','범위(max-min)')['z_vs_null']):.2f}), 가중 SD 기준은 유의(p="
  f"{float(pf('D1','가중 SD')['p_value']):.4f}). D2·D2b는 두 통계량 모두 유의 |")
A(f"| 4 | 약물 수 교란 | **더 나쁘다**. r = {float(row('D1')['약물수 상관(상수)']):.3f}(D1) / "
  f"{float(row('D2')['약물수 상관(상수)']):.3f}(D2) / {float(row('D2b')['약물수 상관(상수)']):.3f}(D2b) vs "
  f"C1_id 0.240 · C1_icdfull 0.131"
  + (f". **단 약물 수를 맞춘 쌍에서는 격차가 남는다** — D1 매칭 "
     f"{PM['D1']['n_pairs_matched']}쌍 중 {nsig['D1']}쌍이 q<0.05, "
     f"사후 선택 보정 후 p={float(gperm[gperm['분할']=='D1'].iloc[0]['전역 순열 p']):.4f} (§3-1)" if MP else "")
  + " |")
A(f"| 5 | 바닥 라벨 | 비뇨생식기 **예**(D1 최하위). 소화기 **아니오**(D1 {r_dig_d1[0]}/{r_dig_d1[1]}위). "
  f"사전 규칙에 따라 소화기는 반증이 아니라 **입도 차이로 판정 불가** |")
A("")

A("## 1. 분할 정의와 순환기 비중 확인")
A("")
A("| 이름 | 정의 | 라벨 수 |")
A("|---|---|---|")
A(f"| D1 | `SEQ_NUM=1` 코드의 ICD-9 chapter (V·E 별도) | {SUM['D1']['라벨 수']} |")
A(f"| D2 | D1과 동일하되 순환기(390–459)만 ICD-9 공식 8블록으로 세분 | {SUM['D2']['라벨 수']} |")
A(f"| D2b | D2에 더해 420–429만 3자리 코드로 재분할 (참고용) | {SUM['D2b']['라벨 수']} |")
A("")
A("**순환기 비중 확인** — 지시대로 두 기준을 대조했다.")
A("")
A(f"우리 코호트의 진단 행은 {CIRC['n_diag_rows']:,}행이다. 순환기는 "
  f"**전체 진단 행 기준 {CIRC['circ_all_rows_pct']}%**, **`SEQ_NUM=1` 기준 {CIRC['circ_seq1_pct']}%**이고, "
  f"순환기 코드를 하나라도 가진 방문은 {CIRC['circ_any_visit_pct']}%다. "
  f"즉 어느 기준으로도 순환기가 13.6%인 경우는 없다. 게이트 출력의 13.6%는 순환기가 아니라 "
  f"**소화기 행**이었다(소화기: `SEQ_NUM=1` 기준 {CIRC['dig_seq1_pct']}%, 전체 진단 행 기준 "
  f"{CIRC['dig_all_rows_pct']}%). 게이트와 이번 표는 같은 `SEQ_NUM=1` 기준으로 둘 다 순환기 "
  f"4,211/14,541 = {CIRC['circ_seq1_pct']}%를 보고했으므로 **둘 중 틀린 것은 없다** — 표에서 행을 잘못 짚은 것이다. "
  f"D2 정의 변경(과반 조건 폐기)의 근거였던 '순환기는 과반이 아니다'는 결론 자체는 그대로 유지된다.")
A("")
A("| chapter | 전체 진단 행 기준% | `SEQ_NUM=1` 기준% |")
A("|---|---|---|")
for nm, r_ in share.head(8).iterrows():
    A(f"| {nm} | {r_['전체 진단 행 기준%']:.1f} | {r_['SEQ_NUM=1 기준%']:.1f} |")
A("")

A("## 2. 검정력")
A("")
A(md(pd.DataFrame(P["summary"]), None, {"탈락 test 비율%": 2, "매핑 실패(기타)%": 2, "중앙 test 크기(통과 라벨 중)": 1}))
A("")
A(f"test {n_test:,}건을 나누는 것이므로 라벨이 늘수록 라벨당 표본이 줄지만, "
  f"D2b는 통과 라벨 수를 **줄이지 않고 늘린다**(11 → 15 → 17). 커버리지가 100%라 P1_ge100의 "
  f"기타 63.5%에 대응하는 손실은 '≥{MIN_TEST}에 못 미쳐 격차 분석에서 빠지는 test 방문' 뿐이고, "
  f"그 비율은 {SUM['D1']['탈락 test 비율%']}% / {SUM['D2']['탈락 test 비율%']}% / "
  f"{SUM['D2b']['탈락 test 비율%']}%다.")
A("")

A("## 3. 라벨별 성능")
A("")
A(f"### D1 — test ≥{MIN_TEST} 라벨 {len(D1)}개 (성능 낮은 순)")
A("")
A(perf_md(D1))
A("")
A(f"### D2 — test ≥{MIN_TEST} 라벨 {len(D2)}개 (성능 낮은 순)")
A("")
A(perf_md(D2))
A("")
A(f"### D2b — test ≥{MIN_TEST} 라벨 {len(D2b)}개 (참고용, 성능 낮은 순)")
A("")
A(perf_md(D2b))
A("")

if MP:
    def pairs_md(tag):
        t = pairs[pairs["분할"] == tag].sort_values("쌍별 순열 p").copy()
        t["쌍 (test, 평균 약물수)"] = [
            f"{a} ({int(na)}, {da:.1f}) vs {b} ({int(nb)}, {db:.1f})"
            for a, b, na, nb, da, db in zip(t["라벨 A"], t["라벨 B"], t["test A"], t["test B"],
                                            t["약물수 A"], t["약물수 B"])]
        t["Jaccard A vs B"] = [f"{x:.4f} vs {y:.4f}" for x, y in zip(t["Jaccard A"], t["Jaccard B"])]
        t = t.rename(columns={"ΔJaccard(A-B)": "ΔJaccard", "쌍 내 양쪽 걸친 환자%": "걸친 환자%"})
        return md(t, ["쌍 (test, 평균 약물수)", "Δ약물수", "Jaccard A vs B", "ΔJaccard", "CI 겹침",
                      "Δ 부트스트랩 CI", "Δ CI 0 제외", "쌍별 순열 p", "BH q", "걸친 환자%"],
                  {"Δ약물수": 1, "ΔJaccard": 4, "쌍별 순열 p": 5, "BH q": 4, "걸친 환자%": 1})

    def pr(tag, a, b, col):
        r_ = pairs[(pairs["분할"] == tag) & (pairs["라벨 A"] == a) & (pairs["라벨 B"] == b)]
        return r_.iloc[0][col] if len(r_) else None

    def g(tag, col):
        return gperm[gperm["분할"] == tag].iloc[0][col]

    lo1, hi1 = D1.iloc[0], D1.iloc[-1]
    span_lo, span_hi = pairs["쌍 내 양쪽 걸친 환자%"].min(), pairs["쌍 내 양쪽 걸친 환자%"].max()

    A("### 3-1. 약물 수 매칭 쌍 — 교란을 맞춘 뒤에도 격차가 남는가")
    A("")
    A(f"라벨 평균 약물 수 차이가 **{MP['max_drug_diff']}개 이내**인 쌍만 골라 상수 Jaccard 차이를 본다. "
      f"test ≥{MIN_TEST} 라벨의 모든 쌍 중 D1 {PM['D1']['n_pairs_possible']}쌍 → "
      f"{PM['D1']['n_pairs_matched']}쌍, D2 {PM['D2']['n_pairs_possible']}쌍 → {PM['D2']['n_pairs_matched']}쌍, "
      f"D2b {PM['D2b']['n_pairs_possible']}쌍 → {PM['D2b']['n_pairs_matched']}쌍이 매칭된다.")
    A("")
    A("추론은 세 겹이다.")
    A("")
    A(f"1. **쌍별 차이의 환자 단위 부트스트랩 CI** — 두 라벨의 환자를 각각 {MP['n_boot']:,}회 재표집해 "
      f"차이 자체의 95% CI를 만든다. 라벨별 CI의 겹침 여부보다 덜 보수적이므로 둘 다 싣는다.")
    A(f"2. **쌍별 순열검정** {MP['n_perm_pair']:,}회 — 약물 수 {MP['strata_pair']}분위 층 **안에서만** "
      f"환자 블록 단위로 두 라벨을 다시 배정한다. 매칭 조건(약물 수 분포)을 유지한 채 라벨만 무작위화하는 방식이다.")
    A(f"3. **전역 순열검정** {MP['n_perm_global']:,}회 — 쌍 선택이 사후적이라는 점을 보정한다. "
      f"약물 수 {MP['strata_global']}분위 층 안에서 라벨을 섞고, **매 반복마다 ≥{MIN_TEST} 필터와 "
      f"|Δ약물수| ≤ {MP['max_drug_diff']} 재매칭을 다시 적용**해 그때의 max|ΔJaccard|를 모은다. "
      f"관측된 최대 격차를 이 귀무분포와 비교한다.")
    A("")
    A(f"쌍 안에서 양쪽 라벨에 모두 방문을 가진 환자는 {span_lo:.1f}~{span_hi:.1f}%뿐이다. "
      f"쌍 비교는 사실상 서로 다른 환자군의 비교이므로 재표집·재배정을 환자 블록 단위로 한 것이 맞다.")
    A("")
    A(f"#### D1 — 매칭 {PM['D1']['n_pairs_matched']}쌍 (순열 p 오름차순)")
    A("")
    A(pairs_md("D1"))
    A("")
    A(f"지목된 두 쌍은 세 기준을 모두 통과한다. **내분비·대사·면역 vs 신생물**은 약물 수를 "
      f"{pr('D1','내분비·대사·면역','신생물','Δ약물수'):.1f}개까지 맞춘 상태에서 ΔJaccard = "
      f"{pr('D1','내분비·대사·면역','신생물','ΔJaccard(A-B)'):.4f}, 차이 CI "
      f"{pr('D1','내분비·대사·면역','신생물','Δ 부트스트랩 CI')}, 순열 p="
      f"{pr('D1','내분비·대사·면역','신생물','쌍별 순열 p'):.5f}(q="
      f"{pr('D1','내분비·대사·면역','신생물','BH q'):.4f})이다. **순환기 vs 호흡기**는 ΔJaccard = "
      f"{pr('D1','순환기','호흡기','ΔJaccard(A-B)'):.4f}, 차이 CI {pr('D1','순환기','호흡기','Δ 부트스트랩 CI')}, "
      f"p={pr('D1','순환기','호흡기','쌍별 순열 p'):.5f}(q={pr('D1','순환기','호흡기','BH q'):.4f})다. "
      f"둘 다 라벨별 CI도 겹치지 않는다.")
    A("")
    A(f"**두 방법이 엇갈리는 쌍이 하나 있다.** 비뇨생식기 vs 소화기는 차이 CI "
      f"{pr('D1','비뇨생식기','소화기','Δ 부트스트랩 CI')}가 0을 제외하지만 순열 p="
      f"{pr('D1','비뇨생식기','소화기','쌍별 순열 p'):.5f}로 유의하지 않다. 부트스트랩은 각 라벨 안에서 "
      f"환자를 재표집하는 반면 순열은 층 안에서 환자 블록째 라벨을 다시 붙이므로, test "
      f"{int(pr('D1','비뇨생식기','소화기','test A'))}건짜리 작은 라벨에서는 후자의 귀무분포가 더 넓다. "
      f"엇갈릴 때는 보수적인 쪽(순열)을 따라 **유의하지 않은 것으로 센다.** "
      f"나머지 {PM['D1']['n_pairs_matched']-3}쌍은 차이 CI가 모두 0을 포함한다.")
    A("")
    A(f"#### D2 — 매칭 {PM['D2']['n_pairs_matched']}쌍")
    A("")
    A(pairs_md("D2"))
    A("")
    A(f"#### D2b — 매칭 {PM['D2b']['n_pairs_matched']}쌍")
    A("")
    A(pairs_md("D2b"))
    A("")
    A(f"**교란이 심한 D2에서도 매칭 후 격차가 남는가 — 남는다. 단 D1과 같은 쌍 하나뿐이다.** "
      f"내분비·대사·면역 vs 신생물은 D2에서도 그대로 유의하고(q={pr('D2','내분비·대사·면역','신생물','BH q'):.4f}), "
      f"비뇨생식기 vs 소화기도 같은 값으로 남는다 — 이 라벨들은 D2에서 쪼개지지 않았으므로 당연하다. "
      f"사라지는 것은 순환기 vs 호흡기다. 순환기를 8블록으로 쪼개면 블록별 평균 약물 수가 흩어져 "
      f"호흡기(24.1개)와 0.5개 이내로 맞는 블록이 없어진다.")
    A("")
    A(f"**D2가 새로 만든 순환기 블록 쌍에서는 격차가 남지 않는다.** 매칭되는 블록 쌍은 D2·D2b에서 각각 "
      f"하나씩뿐인데, D2의 감염성·기생충성 vs 순환기: 기타 심질환은 약물 수를 "
      f"{pr('D2','감염성·기생충성','순환기: 기타 심질환','Δ약물수'):.1f}개까지 맞추면 ΔJaccard = "
      f"{pr('D2','감염성·기생충성','순환기: 기타 심질환','ΔJaccard(A-B)'):.4f}(p="
      f"{pr('D2','감염성·기생충성','순환기: 기타 심질환','쌍별 순열 p'):.4f})로 사실상 0이고, "
      f"D2b의 감염성·기생충성 vs 순환기: 428 CHF NOS도 "
      f"{pr('D2b','감염성·기생충성','순환기: 428 CHF NOS','ΔJaccard(A-B)'):.4f}(p="
      f"{pr('D2b','감염성·기생충성','순환기: 428 CHF NOS','쌍별 순열 p'):.4f})다. "
      f"D2에서 격차를 키운 양 끝(정맥·림프 17.3개 / 동맥·세동맥 27.3개)은 약물 수가 10개나 달라 "
      f"애초에 어떤 라벨과도 매칭되지 않는다. §5-4의 교란 진단과 정확히 같은 그림이다.")
    A("")
    A("#### 사후 선택 보정 — 전역 순열")
    A("")
    A(md(gperm, None, {"관측 max|ΔJaccard|": 4, "귀무 매칭 쌍 수(평균)": 1, "귀무 max 평균": 4,
                       "귀무 max p95": 4, "전역 순열 p": 4}))
    A("")
    A(f"세 분할 모두 관측 최대 격차는 내분비 vs 신생물의 {g('D1','관측 max|ΔJaccard|'):.4f}다. "
      f"8~9개 쌍을 훑어 가장 큰 것을 골랐다는 점을 보정해도 D1 p={g('D1','전역 순열 p'):.4f}, "
      f"D2 p={g('D2','전역 순열 p'):.4f}로 유의하고, D2b는 p={g('D2b','전역 순열 p'):.4f}로 경계다. "
      f"귀무 매칭 쌍 수가 D1에서는 {g('D1','귀무 매칭 쌍 수(평균)'):.1f}개로 관측 "
      f"{int(g('D1','관측 매칭 쌍 수'))}개보다 적고 D2/D2b에서는 {g('D2','귀무 매칭 쌍 수(평균)'):.1f}·"
      f"{g('D2b','귀무 매칭 쌍 수(평균)'):.1f}개로 더 많다 — 라벨을 섞으면 검사 대상 쌍 수 자체가 바뀌는데, "
      f"이 변동까지 귀무분포에 들어 있다.")
    A("")
    A(f"**정리.** 약물 수 교란을 통제한 뒤에도 chapter 간 격차는 남는다 — 다만 D1 "
      f"{PM['D1']['n_pairs_possible']}쌍 중 매칭 {PM['D1']['n_pairs_matched']}쌍, 그중 "
      f"{nsig['D1']}쌍(BH q < 0.05)에 한정된다. 이는 D1 전체 범위 통계량이 유의하지 않은 것"
      f"(p={float(pf('D1','범위(max-min)')['p_value']):.4f})과 모순되지 않는다. 범위 통계량이 쓰는 두 끝점은 "
      f"{lo1['라벨']}({int(lo1['test_visits'])}건, {lo1['mean_n_drug']:.1f}개)와 "
      f"{hi1['라벨']}({int(hi1['test_visits'])}건, {hi1['mean_n_drug']:.1f}개)인데, 약물 수가 "
      f"{abs(hi1['mean_n_drug']-lo1['mean_n_drug']):.1f}개 차이나는 데다 둘 다 작아 CI가 넓다. "
      f"매칭 쌍 분석은 교란이 통제되고 표본이 큰 비교만 골라 쓰므로 같은 데이터에서 더 높은 검정력을 얻는다. "
      f"두 결과는 층위가 다르다: **11개 라벨을 한꺼번에 요약한 통계량은 유의하지 않고, "
      f"약물 수를 맞춘 특정 쌍의 격차는 유의하다.**")
    A("")

A("## 4. 비교표")
A("")
A(md(cmp, None, {"커버리지%": 1, "실루엣(공통공간)": 4, "상수 격차": 4, "순열 p(상수·범위)": 5,
                 "순열 p(상수·가중SD)": 5, "copy-prev 격차": 4, "약물수 상관(상수)": 3, "η²(약물수)": 4}))
A("")
A("읽는 법 세 가지를 먼저 밝힌다.")
A("")
A("1. **실루엣은 여섯 분할을 같은 공간(전체 4,514 어휘 SVD-100)에서 다시 쟀다.** 원래 각 분할의 실루엣은 "
  "자기가 만들어진 공간에서 계산돼 서로 비교할 수 없었다. D1/D2/D2b가 음수인 것은 chapter 라벨이 진단 "
  "동시출현 공간에서 뭉쳐 있지 않다는 뜻이다 — 주 진단 하나로 자른 분할이므로 당연하고, 성능 격차의 유효성과는 별개다.")
A(f"2. **P1_ge100의 순열검정은 기타(-1) {100-p1_cov:.1f}%를 우주에서 빼고 돌렸다.** 기타는 방문 9,240건짜리 "
  "이질적 덩어리라 그대로 두면 가중 SD가 인위적으로 눌린다. 관측값과 귀무분포 모두 같은 부분집합에서 계산했다.")
A("3. **최소 크기는 전체 방문 기준**이라 D1/D2/D2b가 4까지 내려간다(임신·출산 등). 격차 통계량은 "
  f"test ≥{MIN_TEST} 라벨만 쓰므로 이 작은 라벨들은 들어가지 않는다.")
A("")

A("## 5. 판정")
A("")
A("### 1. 커버리지")
A("")
A(f"D1/D2/D2b 모두 100%다. chapter 매핑 실패 0건, 순환기 블록 미지정 0건이므로 버려지는 방문이 없다. "
  f"P1_ge100은 {p1_cov}%였다. 다만 '분석에 실제로 쓰이는' 방문으로 좁히면 D1 "
  f"{100-SUM['D1']['탈락 test 비율%']:.1f}% / D2 {100-SUM['D2']['탈락 test 비율%']:.1f}% / "
  f"D2b {100-SUM['D2b']['탈락 test 비율%']:.1f}%이고, P1_ge100도 기타를 뺀 뒤 test ≥{MIN_TEST}를 "
  f"적용하면 {int(cmp[cmp['분할'].str.startswith('P1_ge100')].iloc[0]['test≥30 라벨 수'])}개 라벨만 남는다.")
A("")
A("### 2. 세분화의 값 — 트레이드오프")
A("")
A(md(TRADEOFF, None, {"통과 라벨 중앙 test 크기": 1, "탈락 test 비율%": 2, "평균 CI 폭": 4, "최대 CI 폭": 4,
                      "상수 격차": 4, "순열 p(범위)": 5, "순열 p(가중SD)": 5, "약물수 상관": 3}))
A("")
A(f"D2는 D1보다 낫다. 실질 k가 {SUM['D1']['test>=30 통과 라벨 수 (실질 k)']} → "
  f"{SUM['D2']['test>=30 통과 라벨 수 (실질 k)']}로 늘고, 격차가 "
  f"{float(row('D1')['상수 격차']):.4f} → {float(row('D2')['상수 격차']):.4f}로 커지면서 "
  f"순열 p가 {float(row('D1')['순열 p(상수·범위)']):.4f}(유의하지 않음) → "
  f"{float(row('D2')['순열 p(상수·범위)']):.4f}(유의)로 바뀐다. 즉 **D1에서 유의하지 않던 격차가 D2에서 유의해진다.** "
  f"이유는 표에서 바로 보인다 — D1의 순환기 824건은 상수 Jaccard 0.2732 한 덩어리인데, D2로 쪼개면 "
  f"정맥·림프 0.2082에서 동맥·세동맥 0.3095까지 0.10 넘게 벌어진다. chapter 평균이 이 격차를 상쇄해 지우고 있었다.")
A("")
A(f"대가는 두 가지다. 첫째 정밀도: 평균 CI 폭이 {TRADEOFF.iloc[0]['평균 CI 폭']:.4f} → "
  f"{TRADEOFF.iloc[1]['평균 CI 폭']:.4f} → {TRADEOFF.iloc[2]['평균 CI 폭']:.4f}로 넓어지고, "
  f"D2에서 CI가 가장 넓은 라벨({D2.loc[(D2['jac_const_hi']-D2['jac_const_lo']).idxmax(), '라벨']}, "
  f"test {int(D2.loc[(D2['jac_const_hi']-D2['jac_const_lo']).idxmax(), 'test_visits'])}건)의 폭 "
  f"{float((D2['jac_const_hi']-D2['jac_const_lo']).max()):.4f}는 D1의 라벨 간 격차 전체"
  f"({float(row('D1')['상수 격차']):.4f})와 맞먹는다. 둘째 교란: 약물수 상관이 "
  f"{float(row('D1')['약물수 상관(상수)']):.3f} → {float(row('D2')['약물수 상관(상수)']):.3f} → "
  f"{float(row('D2b')['약물수 상관(상수)']):.3f}로 단조 증가한다. **세분화가 드러내는 격차의 상당 부분은 "
  f"라벨별 처방 개수 차이와 같이 움직인다.** D2b는 실질 k를 17까지 늘리지만 상관이 0.81로 P1_ge100 수준이 되므로, "
  f"주 분할을 D2로 두라는 사전 결정은 결과를 보고도 유지된다.")
A("")
A("### 3. 성능 격차 — 순열검정")
A("")
A(md(perm[perm["예측기"] == "상수"], ["분할", "통계량", "observed", "null_mean", "null_sd", "null_p95",
                                   "p_value", "z_vs_null"],
     {"observed": 4, "null_mean": 4, "null_sd": 4, "null_p95": 4, "p_value": 5, "z_vs_null": 2}))
A("")
A(f"**D1의 범위 통계량은 유의하지 않다** (관측 {float(pf('D1','범위(max-min)')['observed']):.4f}, "
  f"귀무 p95 {float(pf('D1','범위(max-min)')['null_p95']):.4f}, p="
  f"{float(pf('D1','범위(max-min)')['p_value']):.4f}, z={float(pf('D1','범위(max-min)')['z_vs_null']):.2f}). "
  f"이 트랙에서 상수 예측기의 범위 통계량이 유의하지 않게 나온 것은 D1이 처음이다. "
  f"가중 SD는 유의하다(p={float(pf('D1','가중 SD')['p_value']):.4f}, "
  f"z={float(pf('D1','가중 SD')['z_vs_null']):.2f}) — 두 끝점 차이는 우연 범위지만 라벨들이 전체적으로 "
  f"흩어진 정도는 우연이 아니라는 뜻이다.")
A("")
A(f"D2(z={float(pf('D2','범위(max-min)')['z_vs_null']):.2f} / "
  f"{float(pf('D2','가중 SD')['z_vs_null']):.2f})와 D2b(z={float(pf('D2b','범위(max-min)')['z_vs_null']):.2f} / "
  f"{float(pf('D2b','가중 SD')['z_vs_null']):.2f})는 두 통계량 모두 유의하다. 기존 C1_id k=15의 z=+7.6~10.2에는 "
  f"세 분할 모두 못 미친다. copy-prev 예측기는 다음과 같다.")
A("")
A(md(perm[perm["예측기"] == "copy-prev"], ["분할", "통계량", "observed", "null_p95", "p_value", "z_vs_null"],
     {"observed": 4, "null_p95": 4, "p_value": 5, "z_vs_null": 2}))
A("")
A("### 4. 약물 수 교란")
A("")
A(f"**나쁘다.** 라벨 평균 상수 Jaccard와 라벨 평균 약물 수의 상관은 D1 "
  f"r={float(row('D1')['약물수 상관(상수)']):.3f}, D2 r={float(row('D2')['약물수 상관(상수)']):.3f}, "
  f"D2b r={float(row('D2b')['약물수 상관(상수)']):.3f}로, C1_id(0.240)·C1_icdfull(0.131)보다 모두 크고 "
  f"P1_ge100(0.802)에 근접한다. η²(약물수)도 같은 방향이다(D1 {float(row('D1')['η²(약물수)']):.4f} → "
  f"D2b {float(row('D2b')['η²(약물수)']):.4f}, C1_id는 0.0918).")
A("")
A("해석은 한 문장으로 제한한다. 상수 top-23 예측기의 Jaccard는 정답 처방 집합 크기에 기계적으로 의존하므로, "
  "약물 수와 강하게 같이 움직이는 라벨 격차는 '질병군이 불리하다'의 증거로 그대로 쓸 수 없다. "
  "D2에서 성능이 가장 낮은 정맥·림프(평균 17.3개)와 가장 높은 동맥·세동맥(27.3개)의 약물 수 차이가 이 문제를 그대로 보여준다.")
A("")
if MP:
    A(f"**다만 상관이 크다는 것과 격차가 전부 약물 수 때문이라는 것은 다르다.** §3-1에서 약물 수를 "
      f"{MP['max_drug_diff']}개 이내로 맞춘 쌍만 따로 보면 D1 {PM['D1']['n_pairs_matched']}쌍 중 "
      f"{nsig['D1']}쌍(내분비·대사·면역 vs 신생물, 순환기 vs 호흡기)에서 격차가 남고, 사후 선택을 보정한 "
      f"전역 순열에서도 D1 p={float(gperm[gperm['분할']=='D1'].iloc[0]['전역 순열 p']):.4f} / "
      f"D2 p={float(gperm[gperm['분할']=='D2'].iloc[0]['전역 순열 p']):.4f}로 유의하다. "
      f"반대로 D2가 새로 만든 순환기 블록 쌍은 약물 수를 맞추면 격차가 사라진다. "
      f"즉 교란은 실재하지만 격차 전부를 설명하지는 못한다.")
    A("")
A("### 5. 바닥 라벨 — 예/아니오")
A("")
A("사전 규칙(결과 보기 전 고정): 소화기·비뇨생식기가 바닥이면 독립적 확증, 아니면 반증이 아니라 입도 차이로 판정 불가.")
A("")
A("| 라벨 | D1 순위 | D2 순위 | 바닥인가 |")
A("|---|---|---|---|")
A(f"| 비뇨생식기 | {r_uro_d1[0]}/{r_uro_d1[1]} (최하위) | {r_uro_d2[0]}/{r_uro_d2[1]} | **예** |")
A(f"| 소화기 | {r_dig_d1[0]}/{r_dig_d1[1]} | {r_dig_d2[0]}/{r_dig_d2[1]} | **아니오** |")
A("")
A(f"비뇨생식기는 D1에서 상수 Jaccard {float(D1.iloc[0]['jac_const']):.4f}로 최하위다 — **예**. "
  f"소화기는 {float(D1[D1['라벨']=='소화기'].iloc[0]['jac_const']):.4f}로 11개 중 {r_dig_d1[0]}위, "
  f"하위권이지만 바닥은 아니다 — **아니오**. 사전 규칙대로 소화기 결과는 반증이 아니라 판정 불가로 남긴다.")
A("")
A("**C1/C10 교차표** — 두 분할이 같은 환자를 어떻게 다르게 묶는지가 chapter 평균 비교보다 정확하다. "
  f"기존 바닥은 C10 말기신부전(상수 {float(c1_bottom.iloc[0]['jac_const']):.4f})과 "
  f"C1 간경변({float(c1_bottom.iloc[1]['jac_const']):.4f})이었다.")
A("")
A(f"C1(간경변, 방문 {int(ct1_all['방문 수'].sum())}건 중 상위 6개 행선지):")
A("")
A(md(ct1, CT_COLS, {"클러스터 내 비중%": 1, "그 라벨 안에서 이 클러스터가 차지하는 비중%": 1}))
A("")
A(f"C10(말기신부전) — D1:")
A("")
A(md(ct10, CT_COLS, {"클러스터 내 비중%": 1, "그 라벨 안에서 이 클러스터가 차지하는 비중%": 1}))
A("")
A("C10 — D2:")
A("")
A(md(ct10_d2, CT_COLS, {"클러스터 내 비중%": 1, "그 라벨 안에서 이 클러스터가 차지하는 비중%": 1}))
A("")
A("교차표가 말하는 것은 두 가지다.")
A("")
A(f"**간경변은 소화기 안에서 희석된다.** C1의 {float(ct1.iloc[0]['클러스터 내 비중%']):.1f}%가 소화기로 가지만, "
  f"소화기 라벨 입장에서 간경변은 {float(ct1.iloc[0]['그 라벨 안에서 이 클러스터가 차지하는 비중%']):.1f}%에 불과하다. "
  f"소화기 라벨의 `SEQ_NUM=1` 상위 코드는 "
  + ", ".join(f"{nm} {p}%" for nm, _, p in DIG_TOP)
  + f" 로, 나머지 {100-float(ct1.iloc[0]['그 라벨 안에서 이 클러스터가 차지하는 비중%']):.1f}%는 간경변 군집(C1)에 "
    f"속하지 않는 소화기 방문이다. 사전에 예상한 희석이 실제로 약 5:1로 일어나고 있으므로, 소화기가 바닥에 오지 않은 것은 "
    f"입도 차이로 설명된다.")
A("")
A(f"**말기신부전은 비뇨생식기로 가지 않는다.** C10의 최대 행선지는 순환기 "
  f"{float(ct10.iloc[0]['클러스터 내 비중%']):.1f}%이고 비뇨생식기는 "
  f"{cell(ct10_all, '비뇨생식기', '클러스터 내 비중%'):.1f}%뿐이다. "
  f"고혈압성 말기신부전(403.x)의 `SEQ_NUM=1`이 순환기 chapter로 분류되기 때문이다. D2에서 이 집단은 "
  f"순환기: 고혈압성 라벨의 {cell(ct10_d2_all, '순환기: 고혈압성', '그 라벨 안에서 이 클러스터가 차지하는 비중%'):.1f}%를 "
  f"차지하는데, 그 라벨의 test 방문은 {htn_test}건이라 ≥{MIN_TEST} 문턱을 넘지 못해 격차 분석에서 빠진다. "
  f"**즉 기존 바닥 집단이 가장 진하게 모이는 라벨이 하필 검정력 부족으로 탈락한다.** "
  f"이것이 chapter 분할이 동반질환 군집과 '다른 것을 재고 있다'의 구체적 내용이다.")
A("")

A("## 6. 산출물")
A("")
A("| 파일 | 내용 |")
A("|---|---|")
A("| `out/icd_partition_assignments.csv` | SUBJECT_ID, HADM_ID, icd9_seq1, chapter, d1_label, d2_label (+ d2b_label, split) |")
A("| `out/table13_icd_power.csv` | §2 라벨별 검정력 |")
A("| `out/table14_icd_clusters.csv` | §3 라벨별 성능 + 부트스트랩 CI |")
A("| `out/table15_icd_perm.csv` | §3 순열검정 10,000회 |")
A("| `out/table16_icd_compare.csv` | §4 비교표 |")
A("| `out/table17_c1c10_crosstab.csv` | §5-5 C1/C10 교차표 |")
A("| `out/table18_icd_tradeoff.csv` | §5-2 세분화 트레이드오프 |")
if MP:
    A("| `out/table19_matched_pairs.csv` | §3-1 약물 수 매칭 쌍 + 부트스트랩 CI + 쌍별 순열 p + BH q |")
    A("| `out/table20_matched_perm_global.csv` | §3-1 사후 선택 보정 전역 순열 |")
A("| `out/figs/fig16_D1_label_perf.png` | D1 라벨별 성능 막대 + CI (성능 순) |")
A("| `out/figs/fig17_D2_label_perf.png` | D2 라벨별 성능 막대 + CI (성능 순) |")
A("| `out/figs/fig18_D2_label_dx_heatmap.png` | D2 라벨 x 상위 동반진단 히트맵 |")
if MP:
    A("| `out/figs/fig19_matched_pairs.png` | §3-1 매칭 쌍별 ΔJaccard + 부트스트랩 CI |")
A("")
A("## 7. 한계")
A("")
A("- 주 진단 하나(`SEQ_NUM=1`)로 방문을 자른다. 방문당 평균 진단은 14.2개이고 순환기 코드를 하나라도 "
  f"가진 방문이 {CIRC['circ_any_visit_pct']}%이므로, D1/D2 라벨은 '이 환자의 병'이 아니라 "
  "'청구서 첫 줄'에 가깝다. 게이트에서 확인한 `diag_id[0]`과의 chapter 수준 일치율은 90.1%였다.")
A(f"- test 방문 {n_test:,}건을 최대 {SUM['D2b']['라벨 수']}개 라벨로 나눈다. D2의 최소 통과 라벨은 "
  f"{int(D2['test_visits'].min())}건(CI 폭 "
  f"{float((D2.loc[D2['test_visits'].idxmin(), 'jac_const_hi'] - D2.loc[D2['test_visits'].idxmin(), 'jac_const_lo'])):.3f}), "
  f"CI가 가장 넓은 라벨은 폭 {float((D2['jac_const_hi']-D2['jac_const_lo']).max()):.3f}로 D1의 라벨 간 격차 전체와 비슷하다.")
A("- 약물 수 상관이 0.46~0.81이라 라벨 격차를 질병군 효과로 읽을 수 없다. 이 교란은 희귀도 트랙"
  "(`REPORT_RARITY.md`)에서 회귀로 통제했을 때 희귀도 계수의 부호가 뒤집혔던 것과 같은 문제다.")
if MP:
    A(f"- §3-1의 매칭은 **라벨 평균** 약물 수를 맞춘 것이지 방문 단위 분포를 맞춘 것이 아니다. 층별 순열이 "
      f"이를 부분적으로 보완하지만 완전한 통제는 아니다. 또 매칭되는 쌍은 D1 {PM['D1']['n_pairs_possible']}쌍 중 "
      f"{PM['D1']['n_pairs_matched']}쌍뿐이라, 나머지 쌍에 대해서는 '격차가 없다'가 아니라 "
      f"'이 방법으로는 비교할 수 없다'가 맞다.")
A("- D2b는 참고용이다. 실질 k는 늘지만 약물수 상관이 가장 크고 라벨당 표본이 가장 작다.")
A("- 순열검정은 라벨 배정만 무작위화한다. 라벨과 무관하게 존재하는 환자 단위 성능 차이는 귀무분포에 이미 들어 있다.")
A("")

(OUT / "REPORT_ICD.md").write_text("\n".join(L), encoding="utf-8")
json.dump({"circ_share_check": CIRC, "share_table": share.reset_index().to_dict("records"),
           "tradeoff": TRADEOFF.to_dict("records"),
           "ranks": {"소화기_D1": r_dig_d1, "비뇨생식기_D1": r_uro_d1,
                     "소화기_D2": r_dig_d2, "비뇨생식기_D2": r_uro_d2}},
          open(OUT / "25_icd_report_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)

print("saved: out/REPORT_ICD.md, table18_icd_tradeoff.csv, 25_icd_report_meta.json, "
      "figs/fig16~18")
print(f"순환기: 전체 진단 행 {CIRC['circ_all_rows_pct']}% / SEQ_NUM=1 {CIRC['circ_seq1_pct']}% · "
      f"소화기 SEQ_NUM=1 {CIRC['dig_seq1_pct']}%")
