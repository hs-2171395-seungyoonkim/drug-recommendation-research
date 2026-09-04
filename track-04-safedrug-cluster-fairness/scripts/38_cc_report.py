"""§주증상 군집 보고 — REPORT_CC.md 와 그림 세 장.

37 이 두 표현(A=TF-IDF, B=ClinicalBERT)을 각각 군집화만 해 두었다. 여기서 둘을 비교해
어느 쪽을 본 분석에 쓸지 정하고, 그 근거를 표로 남긴다.

비교 기준 (37 의 docstring 에 미리 적어 둔 가설을 그대로 검정한다):
  "A 는 축약어를 문자 그대로 본다. sob 와 shortness of breath 가 갈릴 수 있다.
   B 는 동의어를 붙일 수 있다."
  -> 동의어 8쌍이 같은 군집에 들어가는지 센다. 질환군을 만드는 것이 목적이므로
     sob 와 shortness of breath 가 갈리면 그 자체가 실패다.

최빈 CC 점유율(순도)은 A 에 유리하게 치우친 지표다. B 가 동의어를 붙이면 한 문자열의
점유율은 내려가는데 그것이 바로 원하던 동작이다. 그래서 순도는 참고로만 싣고 판정은
동의어 검정과 최대 군집의 내용으로 한다.

산출: out/REPORT_CC.md, out/table48_cc_synonym.csv, out/table49_cc_clusters_bert.csv,
      out/figs/fig28_cc_sizes.png, out/figs/fig29_cc_heatmap.png,
      out/figs/fig30_cc_kpick.png, out/38_cc_report_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score as ari

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, RED, MUTED, GREEN = "#2a78d6", "#c0392b", "#c9c8c3", "#2e7d5b"
FAINT = "#a9a8a3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})

# 동의어/축약 쌍 — 임상 표기 관행에서 같은 주소견을 가리키는 것만 골랐다.
# dyspnea/shortness of breath 는 사전적으로 같은 뜻이라 넣는다.
PAIRS = [("shortness of breath", "sob"), ("dyspnea", "shortness of breath"),
         ("abdominal pain", "abd pain"), ("fever", "fevers"),
         ("gi bleed", "gib"), ("gi bleed", "gastrointestinal bleed"),
         ("unresponsive", "unresponsiveness"),
         ("altered mental status", "mental status changes")]

ex = json.load(open(OUT / "36_cc_extract_meta.json", encoding="utf-8"))
cl = json.load(open(OUT / "37_cc_cluster_meta.json", encoding="utf-8"))
prep = json.load(open(OUT / "32_hpi_prep_meta.json", encoding="utf-8"))
t45 = pd.read_csv(OUT / "table45_cc_extract.csv")
t47 = pd.read_csv(OUT / "table47_cc_ksweep.csv")
asg = pd.read_csv(OUT / "cc_cluster_assignments.csv")
z = np.load(OUT / "37_cc_svd.npz")

asg["cc_demog_stripped"] = asg["cc_demog_stripped"].fillna("")
s = asg[asg.stratum == "CC"].reset_index(drop=True)
s["t"] = s["cc_demog_stripped"].str.lower()
KA, KB = int(cl["A"]["k"]), int(cl["B"]["k"])
K_GRID, SVD = cl["k_grid"], int(cl["A"]["reducer"].split("(")[1].rstrip(")"))
MIN_CLUSTER = 100                # 37 의 k 규칙 하한과 같은 값
PURE = 25                        # 이 미만이면 최빈 CC 하나로 군집 이름을 붙이지 않는다
N = len(asg)

# ---------------------------------------------------------------- 동의어 검정
rows = []
for x, y in PAIRS:
    nx, ny = int((s.t == x).sum()), int((s.t == y).sum())
    if nx == 0 or ny == 0:
        continue
    r = {"표현1": x, "n1": nx, "표현2": y, "n2": ny}
    for col, tag in [("clusterCC", "A"), ("clusterCC_bert", "B")]:
        cx = int(s.loc[s.t == x, col].mode().iloc[0])
        cy = int(s.loc[s.t == y, col].mode().iloc[0])
        r[f"{tag}_군집1"], r[f"{tag}_군집2"] = cx, cy
        r[f"{tag}_병합"] = "O" if cx == cy else "X"
    rows.append(r)
syn = pd.DataFrame(rows)
syn.to_csv(OUT / "table48_cc_synonym.csv", index=False, encoding="utf-8-sig")
mergeA = int((syn["A_병합"] == "O").sum())
mergeB = int((syn["B_병합"] == "O").sum())
NP = len(syn)


def cluster_table(col, k):
    g = s.groupby(col)
    tab = pd.DataFrame({
        "cluster": g.size().index.astype(int), "n": g.size().values,
        "pct": (g.size() / len(s) * 100).round(1).values,
        "words_median": g["n_word"].median().values,
        "n_icd_median": g["n_icd"].median().values,
        "top_cc": g["t"].agg(lambda v: v.value_counts().index[0]).values,
        "top_cc_share": g["t"].agg(
            lambda v: round(v.value_counts().iloc[0] / len(v) * 100, 1)).values,
        "top5": g["t"].agg(lambda v: "; ".join(
            f"{a}({b})" for a, b in v.value_counts().head(5).items())).values,
    })
    return tab.sort_values("n", ascending=False).reset_index(drop=True)


tabA = cluster_table("clusterCC", KA)
tabB = cluster_table("clusterCC_bert", KB)
tabB.to_csv(OUT / "table49_cc_clusters_bert.csv", index=False, encoding="utf-8-sig")

# 본 분석 트랙 판정 — 동의어 병합 수가 많은 쪽. 동수면 최대 군집이 작은 쪽.
if mergeB != mergeA:
    PRIMARY = "B" if mergeB > mergeA else "A"
else:
    PRIMARY = "B" if tabB.pct.max() < tabA.pct.max() else "A"
PT, PK = (tabB, KB) if PRIMARY == "B" else (tabA, KA)
PCOL = "clusterCC_bert" if PRIMARY == "B" else "clusterCC"
PNAME = "ClinicalBERT" if PRIMARY == "B" else "TF-IDF"

# ---------------------------------------------------------------- 층외(CC 없음)
noc = asg[asg.stratum == "no_CC"]
noc_route = noc["route"].value_counts(dropna=False)
noc_hpi = noc["clusterA"].value_counts().head(3)
sp = asg.groupby("stratum")["split"].value_counts().unstack(fill_value=0)

# 군집이 split 을 고르게 나눠 갖는지 — 군집별 평가가 가능한지 본다.
pk = s.groupby([PCOL, "split"]).size().unstack(fill_value=0)
n_eval = int((pk["test"] >= 30).sum())
thin = pk[pk["test"] < 30]


def eta2(lab, x):
    """군집이 그 변수를 얼마나 설명하는지. REPORT_BHC 의 실패선이 η²(길이)=0.2484 였다."""
    x = np.asarray(x, float)
    gm = x.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2
               for _, g in pd.Series(x).groupby(np.asarray(lab)))
    return round(float(ss_b / ((x - gm) ** 2).sum()), 4)


eta_A = eta2(s["clusterCC"].values, s["n_word"].values)
eta_B = eta2(s["clusterCC_bert"].values, s["n_word"].values)
eta_P = eta_B if PRIMARY == "B" else eta_A
# 단어 중앙 1 인 군집은 6개인데 그중 셋만 뜻이 안 통한다. 자동 판정 기준을 못 세워서
# top5 를 눈으로 보고 골랐다 — 아래가 그 목록이고, 나머지 셋(C0 hematemesis/hemoptysis/
# hematochezia, C7 fever/sepsis/fevers, C10 dyspnea/DOE/tachypnea)은 뜻이 통한다.
SHORT_INCOHERENT = {"melena", "sob", "cc"}   # 각 군집의 최빈 CC 로 지정
short = PT[(PT.words_median <= 1) & PT.top_cc.isin(SHORT_INCOHERENT)]
longest = PT.loc[PT.words_median.idxmax()]
# 빈 템플릿("Chief Complaint:" 다음 줄이 "CC:" 뿐) 이 그대로 남은 건수.
n_tmpl = int(s.t.isin(["cc", "c/c", "chief complaint"]).sum())

# ---------------------------------------------------------------- fig28 군집 크기
def clip(t, n):
    """단어 중간에서 자르지 않는다. 'esophageal cance' 같은 라벨을 막는다."""
    return t if len(t) <= n else (t[:n].rsplit(" ", 1)[0] if " " in t[:n] else t[:n])


fig, axes = plt.subplots(1, 2, figsize=(12.4, 8.2))
for ax, (tab, k, name) in zip(axes, [(tabA, KA, "A · TF-IDF"), (tabB, KB, "B · ClinicalBERT")]):
    y = np.arange(len(tab))[::-1]
    prim = (name.startswith("B")) == (PRIMARY == "B")
    col = [S1 if prim else MUTED] * len(tab)
    col[int(tab.pct.idxmax())] = RED if prim else FAINT
    ax.barh(y, tab["n"], color=col, height=0.74, zorder=3)
    for i, (yy, r) in enumerate(zip(y, tab.itertuples())):
        # 최빈 CC 하나로 이름 붙이면 점유율 3% 짜리가 1,285건 군집을 대표하게 된다.
        # 점유율이 PURE 미만이면 상위 3개를 같이 적어 "지배적인 주소견이 없다"를 보이게 한다.
        if r.top_cc_share >= PURE:
            lab = f"{r.top_cc[:30]}  {r.top_cc_share:.0f}%"
        else:
            top3 = " · ".join(clip(t.split("(")[0].strip(), 21)
                              for t in r.top5.split(";")[:3])
            lab = f"{top3} …  최빈 {r.top_cc_share:.0f}%"
        ax.text(r.n + len(s) * 0.006, yy, lab, va="center", ha="left", fontsize=6.4,
                color=(INK if r.top_cc_share >= PURE else INK2) if prim else FAINT,
                zorder=5)
    ax.set_yticks([])
    ax.set_xlim(0, tab["n"].max() * 2.15)
    ax.grid(axis="x", lw=0.7, zorder=0)
    ax.set_title(f"{name}  k={k}" + ("   ← 본 분석" if prim else ""),
                 fontsize=10.5, color=INK if prim else INK2, loc="left")
    ax.set_xlabel("방문 수")
fig.suptitle(f"fig28 · 주증상(Chief Complaint) 군집 크기 — CC 있는 {len(s):,} 방문\n"
             f"최빈 CC 가 {PURE}% 이상이면 그 이름 하나로, 미만이면 상위 3개를 적었다"
             f"(지배적인 주소견이 없는 군집이다). 빨강은 최대 군집.",
             fontsize=10.5, color=INK2, x=0.012, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGS / "fig28_cc_sizes.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- fig29 개념 히트맵
# CC 를 문자열 그대로 세면 상위 30개가 방문의 26.8% 밖에 못 덮는다(꼬리가 길다).
# 그래서 임상 표기 관행으로 같은 주소견을 가리키는 표현을 손으로 묶었다.
# 손으로 만든 사전이다 — 군집화에는 안 쓰였고, 만들어진 군집을 읽기 위한 것뿐이다.
#
# 1차 사전에서 두 곳을 고쳤다. 그림을 보고 고친 것이라 그대로 적어 둔다.
#  - "respiratory distress" 를 호흡곤란에 넣었더니 C15(respiratory distress 132 ·
#    respiratory failure 59)가 '호흡곤란'으로 이름 붙었다. 이건 환자가 호소하는 증상이
#    아니라 관찰된 호흡부전 상태다. 저산소·호흡부전으로 옮긴다.
#  - "의식·정신상태" 가 AMS 와 unresponsive 를 한 덩어리로 묶어 C9(altered mental status
#    201)와 C17(unresponsive 24 · unresponsiveness 21)에 같은 이름이 붙었다. 임상적으로
#    구분되는 상태라 둘로 나눈다.
# 이 수정으로도 호흡곤란은 여전히 둘(C5 shortness of breath / C10 dyspnea)이다. 그것은
# 사전의 문제가 아니라 군집화 결과다 — table48 의 동의어 검정에서 이미 X 로 기록돼 있다.
CONCEPTS = [
    ("호흡곤란",     r"shortness of breath|\bsob\b|dyspnea|\bdoe\b"),
    ("흉통",         r"chest pain|\bcp\b|substernal"),
    ("복통",         r"abdominal pain|abd pain|\bepigastric"),
    ("발열·감염",    r"\bfever|febrile|\bsepsis|septic|infection|cellulitis|pneumonia"),
    ("의식변화",     r"altered mental status|mental status|lethargy|lethargic|confusion|obtunded"),
    ("무반응",       r"unrespons|responsiveness"),
    ("소화관 출혈",  r"\bgi bleed|\bgib\b|brbpr|melena|hematemesis|hematochezia|blood per rectum|coffee ground"),
    ("저혈압·쇼크",  r"hypotension|hypotensive|\bshock\b|bradycardia"),
    ("저산소·호흡부전", r"hypox|respiratory distress|respiratory failure|respiratory arrest|hypercarbic|intubat"),
    ("낙상·외상",    r"\bfall\b|\bfalls\b|trauma|motor vehicle|\bmvc\b|found down"),
    ("오심·구토",    r"nausea|vomiting|\bn/v\b|emesis|diarrhea"),
    ("신경 증상",    r"weakness|seizure|syncope|\bstroke\b|headache|numbness|\bcva\b|aphasia"),
    ("심장·부정맥",  r"palpitation|atrial fib|\bafib\b|tachycard|cardiac arrest|\bchf\b|myocardial|\bstemi\b|angina"),
    ("대사·신장",    r"hyperglycemia|hypoglycemia|\bdka\b|hyperkalemia|renal failure|\barf\b|hyponatremia"),
    ("암·종괴",      r"cancer|carcinoma|\bmass\b|tumor|lymphoma|leukemia|metasta"),
    ("전원·예정입원", r"transfer|elective|scheduled|\bpost-?op|\bs/p .*(?:resection|repair|surgery)"),
]
CM = pd.DataFrame({nm: s.t.str.contains(rx, regex=True, na=False).to_numpy()
                   for nm, rx in CONCEPTS}, index=s.index)
hit_any = round(float(CM.any(axis=1).mean() * 100), 1)
H = (CM.groupby(s[PCOL].astype(int)).mean() * 100).reindex(PT.cluster.astype(int))
H["(해당 없음)"] = (~CM.any(axis=1)).groupby(s[PCOL].astype(int)).mean().reindex(H.index) * 100

# 한 개념이 60% 이상을 차지하면 그 주소견으로 군집 이름을 붙일 수 있다고 본다.
CLEAN = 60
top_share = H[[nm for nm, _ in CONCEPTS]].max(axis=1)
clean_ids = top_share.index[top_share >= CLEAN]
n_clean = len(clean_ids)
n_clean_v = int(PT.set_index(PT.cluster.astype(int)).loc[clean_ids, "n"].sum())

fig, ax = plt.subplots(figsize=(11.8, 7.4))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("s1", [SURF, "#bcd7f2", S1])
ax.imshow(H.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=100)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.iat[i, j]
        if v >= 5:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.2,
                    color=SURF if v >= 55 else INK)
ax.set_xticks(range(H.shape[1]))
ax.set_xticklabels(H.columns, rotation=42, ha="right", fontsize=8.4,
                   color=INK2)
ax.get_xticklabels()[-1].set_color(RED)
ax.set_yticks(range(H.shape[0]))
ax.set_yticklabels([f"C{c}  ({n:,})" for c, n in zip(PT.cluster.astype(int), PT.n)],
                   fontsize=8.0, color=INK2)
ax.set_xticks(np.arange(H.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(H.shape[0] + 1) - 0.5, minor=True)
ax.grid(which="minor", color=SURF, lw=1.6)
ax.tick_params(which="minor", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title(f"fig29 · 군집이 무엇으로 뭉쳤나 — 군집 {PK}개 × 주소견 개념 {len(CONCEPTS)}개 "
             f"({PRIMARY}·{PNAME})\n"
             f"칸 = 그 군집 방문 중 해당 표현을 포함한 비율(%). 5% 미만은 숫자를 지웠다. "
             f"행 합은 100%가 아니다(한 CC 가 여러 개념에 걸린다).\n"
             f"개념 묶음은 손으로 만든 사전이다 — 군집화에는 쓰이지 않았고 결과를 읽기 위한 "
             f"것이다. 방문의 {hit_any}%가 최소 한 개념에 걸린다.",
             fontsize=10, color=INK2, loc="left", pad=12)
fig.tight_layout()
fig.savefig(FIGS / "fig29_cc_heatmap.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- fig30 k 선택
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
for ax, (metric, ylab) in zip(axes, [("min_size", "최소 군집 크기 (방문)"),
                                     ("largest_pct", "최대 군집 비중 (%)")]):
    for tag, colr, kk in [("A", MUTED, KA), ("B", S1, KB)]:
        d = t47[t47.track == tag].sort_values("k")
        ax.plot(d["k"], d[metric], marker="o", ms=4.5, lw=1.6, color=colr,
                label=f"{tag} · {'TF-IDF' if tag == 'A' else 'ClinicalBERT'}", zorder=4)
        pick = d[d.k == kk]
        ax.scatter(pick["k"], pick[metric], s=110, facecolor="none", edgecolor=RED,
                   lw=1.8, zorder=6)
        ax.annotate(f"k={kk}", (kk, float(pick[metric].iloc[0])),
                    xytext=(0, 13), textcoords="offset points", ha="center",
                    fontsize=8.4, color=RED, zorder=7)
    if metric == "min_size":
        ax.axhline(MIN_CLUSTER, color=RED, lw=1.0, ls="--", zorder=3)
        ax.text(K_GRID[-1], MIN_CLUSTER, f" 하한 {MIN_CLUSTER}", va="bottom", ha="right",
                fontsize=8.2, color=RED)
    ax.set_xlabel("k")
    ax.set_ylabel(ylab)
    ax.set_xticks(K_GRID)
    ax.grid(axis="y", lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
fig.suptitle(f"fig30 · k 는 규칙으로 정했다 — 최소 군집 {MIN_CLUSTER}건을 지키는 가장 큰 k\n"
             f"시드는 트랙·k 마다 inertia 최소를 골랐다. 빨간 동그라미가 채택한 k.",
             fontsize=10.5, color=INK2, x=0.008, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.savefig(FIGS / "fig30_cc_kpick.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- REPORT_CC.md
def mdtab(d, index=False):
    """tabulate 없이 마크다운 표를 만든다."""
    d = d.reset_index() if index else d
    cell = lambda v: (f"{v:g}" if isinstance(v, float) and not pd.isna(v)
                      else "" if pd.isna(v) else str(v)).replace("|", "\\|")
    head = [str(c) if not isinstance(c, tuple) else " ".join(map(str, c)) for c in d.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in d.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)


L = []
A = L.append
A("# 주증상(Chief Complaint) 기반 방문 군집\n")
A(f"입력 {N:,} 방문 · 군집 대상 {len(s):,} · 본 분석 트랙 **{PRIMARY}({PNAME}) k={PK}**\n")

A("## 1. 왜 Chief Complaint 인가\n")
A("HPI 산문에서 급성 절을 복원하는 3단계 파이프라인(32~35)을 먼저 만들었다. 그 결과 "
  "군집 24개 중 5개(18.6%)가 주소견이 아니라 인구학 서두(\"year-old male\")로 뭉쳤다. "
  "원인을 따라가 보니 입력이 문제였다 — 해당 방문의 HPI 에는 급성 발현이 아예 없고 "
  "만성 병력만 적혀 있어서 앵커가 찾을 것이 없었다.\n")
A("그 다음에 `NOTEEVENTS.csv` 원본 퇴원요약에 `Chief Complaint:` 섹션이 그대로 있다는 "
  "것을 확인했다. 복원하려던 것이 처음부터 적혀 있었다. 설계 단계에서 원본 노트를 "
  "확인하지 않고 가공본(`data4LLM_with_note.csv`, CC 가 0.4% 뿐)만 본 것이 원인이다.\n")
A("세 입력을 같은 잣대로 비교하면 이렇다.\n")
A(mdtab(pd.DataFrame({
    "입력": ["HPI 원문", "HPI 급성 절(32~35)", "**Chief Complaint(무전처리)**"],
    "단어 중앙": [24, 11, int(ex["words"]["median"])],
    "만성 표현 %": [51.0, 6.0, ex["contamination_pct"]["chronic_marker"]],
    "인구학 표현 %": [72.4, 24.1, ex["contamination_pct"]["demographic_before_strip"]],
})))
A("\n전처리를 하나도 하지 않은 원문이 3단계 파이프라인을 이긴다. HPI 트랙은 버리지 않고 "
  "층외(CC 없음) 방문의 라벨로 남겨 두었다.\n")

A("## 2. 추출\n")
A(f"`NOTEEVENTS.csv` {ex['rows_scanned']:,}행을 훑어 코호트 퇴원요약 "
  f"{ex['cohort_notes']:,}건을 모으고, 방문당 한 건을 고른다"
  f"(규칙: {ex['selection_rule']}). 방문당 노트가 여러 건인 경우가 있어 규칙이 필요한데, "
  f"규칙 3종을 비교했더니 커버리지 87.8~88.2% · 문자열 일치 99.31% 로 영향은 미미했다.\n")
A(mdtab(t45))
A(f"\nCC 는 중앙 {int(ex['words']['median'])}단어, 고유 {ex['unique_strings']:,}종이고, "
  f"CC 가 있는 것 중 {ex['words']['le3_pct']}% 가 3단어 이하다(위 표의 56.7% 는 전 방문 "
  f"기준). 전처리는 비식별화 토큰 `[**...**]` 제거와 인구학 서두 제거"
  f"({ex['contamination_pct']['demographic_before_strip']}% 에 해당) 둘뿐이다.\n")
A(f"추출이 완전하지는 않다. `Chief Complaint:` 다음 줄이 빈 템플릿(`CC:`)뿐인 노트가 있어 "
  f"내용 없이 \"CC\" 만 남은 것이 {n_tmpl}건({n_tmpl / len(s) * 100:.1f}%) 있다. "
  f"아래 §6 의 약어 군집에 그대로 들어가 있다. 층외로 뺐어야 맞다.\n")

A("## 3. 층화\n")
A(f"CC 가 있는 {len(s):,}건만 군집화한다. 없는 {len(noc):,}건"
  f"({len(noc) / N * 100:.1f}%)에 HPI 급성 절을 채워 넣지 않는다. 3단어 텍스트와 11단어 "
  f"텍스트를 한 판에 섞으면 길이가 축이 되기 때문이다 — `REPORT_BHC.md` 에서 "
  f"η²(길이)=0.2484 로 이미 겪은 실패다.\n")
A(f"대신 층외 {len(noc):,}건은 HPI 트랙 라벨(`clusterA`)을 달아 `cc_cluster_assignments.csv` "
  f"에 그대로 남긴다. 전 방문 {N:,}건이 추적된다.\n")
A(mdtab(sp, index=True))
A("")

A("## 4. 두 표현을 겨루게 했다\n")
A("| | A | B |\n|---|---|---|")
A(f"| 표현 | TF-IDF ({cl['A']['vocab']:,}어휘) | Bio_ClinicalBERT 평균풀링 |")
A(f"| 축소 | {cl['A']['reducer']} (설명분산 {cl['A']['svd_explained_variance_pct']}%) "
  f"| {cl['B']['reducer']} |")
A(f"| k | {KA} | {KB} |")
A(f"| 최대 군집 | {tabA.pct.max()}% | {tabB.pct.max()}% |")
A(f"| 최소 군집 | {cl['A']['min_size']} | {cl['B']['min_size']} |")
A(f"| 최빈 CC 점유율 중앙 | {cl['A']['top_cc_share_median_pct']}% "
  f"| {cl['B']['top_cc_share_median_pct']}% |")
A(f"| 동의어 병합 | {mergeA}/{NP}쌍 | {mergeB}/{NP}쌍 |")
A("")
A("### 4.1 동의어 검정이 판정한다\n")
A("37 을 쓸 때 미리 적어 둔 가설이 있다 — \"A 는 축약어를 문자 그대로 본다. sob 와 "
  "shortness of breath 가 갈릴 수 있다. B 는 동의어를 붙일 수 있다.\" 결과를 보기 전에 "
  "적은 문장이고, 같은 주소견을 가리키는 표기 쌍이 한 군집에 들어가는지로 검정한다.\n")
A(mdtab(syn[["표현1", "n1", "표현2", "n2", "A_병합", "B_병합"]]))
A(f"\nB 가 {mergeB}쌍, A 가 {mergeA}쌍을 붙였다. B 는 오타까지 붙인다 — hematemesis 군집에 "
  "`hemetemesis`(7) `hematemasis`(5) 가 같이 들어간다. TF-IDF 로는 원리상 불가능하다.\n")
A("다만 B 가 이긴 것이 아니라 A 가 진 것에 가까운 대목이 있다. **가설을 세울 때 이름을 "
  "댄 그 쌍(shortness of breath / sob)은 A 도 B 도 못 붙였다.** B 는 `sob` 를 "
  "seizure(69) fall(35) confusion(34) 과 같은 군집에 넣었는데, 뜻이 아니라 \"단어 1개짜리 "
  "짧은 표현\"이라는 성질로 뭉친 자리다. dyspnea / shortness of breath 도 둘 다 갈렸다. "
  f"그리고 gi bleed / gastrointestinal bleed 는 A 만 붙였다. {mergeB}:{mergeA} 는 방향을 "
  "말해 줄 뿐 B 가 동의어 문제를 푼 것은 아니다.\n")
A("**최빈 CC 점유율(순도)은 A 에 유리한 지표라 판정에 쓰지 않았다.** B 가 동의어를 붙이면 "
  "한 문자열의 점유율이 내려가는데, 그게 바로 원하던 동작이다. 예를 들어 B 의 GI 출혈 "
  "군집은 gi bleed(89) brbpr(62) upper gi bleed(24) gib(23) lower gi bleed(12) "
  "gi bleeding(12) 를 한 군집에 담고도 순도는 14.2% 로 찍힌다.\n")
A("### 4.2 최대 군집의 내용\n")
A(f"두 트랙 다 10% 안팎의 큰 군집을 하나씩 갖는다. 안을 보면 성격이 다르다.\n")
A(f"A 의 최대 군집은 {tabA.pct.max()}%({tabA.n.max():,}건, 단어 중앙 1)인데 서로 무관한 "
  "단발 표현이 모여 있다 — brbpr · melena · hemoptysis · syncope · confusion · anemia · "
  "cough · dka · stemi. 묶은 원리가 \"문자 수준에서 아무와도 안 닮았다\" 말고는 없다.\n")
A(f"B 의 최대 군집은 {tabB.pct.max()}%({tabB.n.max():,}건, 단어 중앙 3)이고 여기에는 "
  "원리가 있다 — nausea/vomiting · nausea, vomiting, diarrhea · fever, chills · "
  "fever, cough · abdominal pain, fever 처럼 **증상을 여러 개 쉼표로 나열한 CC** 다. "
  "다만 그 원리는 주소견이 아니라 적는 형식이다.\n")
A(f"B 의 2위 군집({tabB.pct.iloc[1]}%, {int(tabB.n.iloc[1]):,}건, 단어 중앙 8)도 형식으로 "
  "묶였는데, 이쪽은 형식이 뜻과 맞아떨어진다 — \"elective admission for coiling\" "
  "\"transferred to for bronchoscopy\" \"direct admit for chf management\" 처럼 CC 를 "
  "문장으로 길게 쓴 것은 대개 **예정 입원·전원**이다. 증상이 아니라 입원 경위를 적은 CC 다.\n")
A(f"즉 B 의 상위 두 군집 {tabB.pct.iloc[0] + tabB.pct.iloc[1]:.1f}%"
  f"({int(tabB.n.iloc[0] + tabB.n.iloc[1]):,}건)는 단일 주소견 군집이 아니라 CC 서술 형식 "
  "군집이다. 아래 4.3 의 η² 와 같은 이야기다.\n")
A("### 4.3 길이가 축이 되지는 않았는가\n")
A(f"`REPORT_BHC.md` 에서 노트 길이가 군집을 지배해 η²(길이)=0.2484 를 찍은 적이 있다. "
  f"같은 잣대로 재면 A 는 **η²(단어수)={eta_A}**, B 는 **{eta_B}** 다. 둘 다 그 실패선보다 "
  f"낮지만 B 가 A 보다 높다.\n")
A(f"B 안에서 길이 성분이 어디 있는지는 짚어 둘 만하다. 한쪽 끝은 4.2 에서 본 "
  f"C{int(longest.cluster)}(단어 중앙 {longest.words_median:.0f}, {int(longest.n):,}건)의 "
  f"긴 서술문이다. 반대쪽 끝에는 단어 중앙 1 인 "
  f"군집이 6개 있는데, 그중 {len(short)}개"
  f"({int(short.n.sum()):,}건, {short.n.sum() / len(s) * 100:.1f}%)는 뜻이 안 통한다 — "
  + " · ".join(f"C{int(r.cluster)}({r.top5.split(';')[0].split('(')[0].strip()}…)"
               for r in short.itertuples()) +
  ". 각각 melena/headache/weakness/fatigue, sob/seizure/fall/confusion, "
  "cc/ams/dka/esrd 처럼 서로 무관한 단발 표현이 \"짧다\"는 이유로 모였다. 나머지 셋"
  "(hematemesis·hemoptysis·hematochezia / fever·sepsis·fevers / dyspnea·DOE·tachypnea)은 "
  "단어 중앙이 1 이어도 뜻이 통한다. 어느 쪽인지는 자동으로 가르지 못해 top5 를 눈으로 "
  "보고 정했다.\n")
A(f"판정: **{PRIMARY}({PNAME})** 를 본 분석에 쓴다. 동의어를 더 붙이고"
  f"({mergeB}:{mergeA}), 큰 군집에 최소한의 조직 원리가 있기 때문이다. 압도적인 차이는 "
  f"아니다 — B 도 상위 두 군집은 서술 형식으로 뭉쳤고 sob 문제는 못 풀었다. A 라벨은 "
  "`cc_cluster_assignments.csv` 에 `clusterCC` 로 남긴다. 이 판정은 결과를 보고 내린 것이 "
  "아니라 37 의 docstring 에 적어 둔 가설을 검정한 것이지만, **판정 기준(동의어 병합 쌍 "
  "수)을 숫자로 못박은 것은 결과를 본 뒤다.** 그대로 적어 둔다.\n")

A("## 5. 규칙이 바뀐 이력\n")
A("사전에 정한 규칙이 퇴화한 해를 고른 것이 두 번 있었다. 규칙과 결과를 그대로 남긴다.\n")
for h in cl.get("method_history", []):
    A(f"- **{h['when']} — {h['rule']}**  \n  {h['result']}  \n  → `{h['superseded_by']}`")
A(f"\n지금 쓰는 규칙은 이렇다.\n")
A(f"- 시드: {cl['seed_rule']}. k-means 가 실제로 최소화하는 값이다.")
A(f"- k: {cl['k_rule']}. 군집이 작아질수록 주소견은 안 섞이지만 군집별 SafeDrug 성능을 "
  f"못 잰다. 그 경계를 100건으로 뒀다.")
A("")
A(mdtab(t47.pivot(index="k", columns="track",
                  values=["largest_pct", "min_size", "top_cc_share_median_pct"]), index=True))
A(f"\nA 는 k={KA} 까지, B 는 k={KB} 까지 최소 군집 {MIN_CLUSTER}건을 지킨다. "
  f"그림은 `figs/fig30_cc_kpick.png`.\n")

A(f"## 6. 군집 {PK}개 ({PRIMARY}·{PNAME})\n")
A("크기와 최빈 CC 는 `figs/fig28_cc_sizes.png` 에 두 트랙을 나란히 그렸다.\n")
# 임베딩 좌표 산점도(구 fig29)는 뺐다. SVD 50차원에서 나눈 것을 2차원에 눌러 담으면
# 겹쳐 보이는 것이 당연한데, 그림만 보면 "군집이 안 갈라졌다"로 읽힌다.
# 캡션으로 방어해야 하는 그림이면 없는 편이 낫다. 대신 개념 히트맵을 넣는다.
A(f"군집이 무엇으로 뭉쳤는지는 `figs/fig29_cc_heatmap.png` 에 군집 {PK}개 × 주소견 개념 "
  f"{len(CONCEPTS)}개 표로 그렸다. 개념 묶음은 손으로 만든 사전이고 군집화에는 쓰이지 "
  f"않았다 — 결과를 읽기 위한 것이다. 방문의 {hit_any}% 가 최소 한 개념에 걸린다.\n")
A(f"한 개념이 그 군집의 60% 이상을 차지하는 군집은 {int(n_clean)}개 "
  f"({n_clean_v:,} 방문 · CC 층의 {n_clean_v / len(s) * 100:.1f}%, 전체 방문의 "
  f"{n_clean_v / N * 100:.1f}%) 다. 이 군집들은 하나의 주소견으로 이름을 붙일 수 있다. "
  f"나머지 {PK - int(n_clean)}개는 그렇지 않다 — 군집별 성능 차이를 재더라도 "
  f"\"이 주소견 그룹이 불리하다\"고 말할 수 있는 것은 앞의 {int(n_clean)}개까지다.\n")
A(mdtab(PT[["cluster", "n", "pct", "words_median", "n_icd_median", "top_cc",
                "top_cc_share", "top5"]]))
A("")

A("## 7. 다른 분할과의 관계\n")
A(f"- ARI(A, B) = {cl['compare']['ARI_A_vs_B']} — 같은 텍스트를 두 방식으로 나눈 결과가 "
  f"이만큼만 겹친다. 표현 선택이 분할을 크게 바꾼다.")
A(f"- ARI(A, HPI 트랙) = {cl['compare']['ARI_A_vs_HPI']}")
A(f"- ARI(B, HPI 트랙) = {cl['compare']['ARI_B_vs_HPI']}")
A("\nHPI 트랙과의 ARI 가 낮지만 두 분할이 무관한 것은 아니다. HPI 의 증상 군집은 CC 와 "
  "정확히 맞고(chest pain 군집의 CC 최빈이 chest pain 182건, shortness of breath 군집은 "
  "214건), 인구학 서두 군집만 CC 가 기저율대로 흩어진다. 즉 HPI 트랙은 증상을 잡은 "
  "부분에서는 맞았고 못 잡은 부분에서 인구학으로 뭉쳤다.\n")

A("## 8. 군집별 평가가 가능한가\n")
A(f"이 데이터의 split 은 train {int(sp['train'].sum()):,} / test {int(sp['test'].sum()):,} "
  f"둘뿐이다(val 없음). test 가 30건 이상인 군집이 {PK}개 중 **{n_eval}개**다. "
  + (f"미달은 C{int(thin.index[0])} 하나(test {int(thin['test'].iloc[0])}건)뿐이다.\n"
     if len(thin) == 1 else f"미달 {len(thin)}개는 군집별 성능을 따로 내기에 표본이 얇다.\n"))
A(mdtab(pk.assign(합=pk.sum(axis=1)).sort_values("합", ascending=False).head(12),
        index=True))
A("")

A("## 9. 한계\n")
A(f"- CC 가 없는 방문이 {len(noc):,}건({len(noc) / N * 100:.1f}%)이다. 별도 층으로 두었을 "
  f"뿐 아직 군집화하지 않았다.")
A("- CC 는 증상만 적힌 것이 아니다. 질환명(esophageal cancer, abdominal aortic aneurysm), "
  "입원 경위(elective admission for), 검사 수치(elevated creatinine on labs)가 섞여 있다. "
  "B 는 이것들을 각각 다른 군집으로 모았지만 \"증상 군집\"과 \"질환명 군집\"을 라벨로 "
  "구분하지는 않았다.")
A("- k-means 는 시드 편차가 크다. 같은 k 에서 순도가 6.6~33.5% 로 흔들린 것을 확인했다. "
  "inertia 최소로 고정했으므로 재현은 되지만, 이 분할이 유일한 좋은 분할은 아니다.")
A(f"- 표기 길이가 완전히 빠지지는 않았다(η²(단어수)={eta_P}). 단어 1개짜리 표현이 뜻과 "
  f"무관하게 한 군집에 모인 자리가 {len(short)}개 있다.")
A(f"- 빈 템플릿 \"CC\" {n_tmpl}건을 층외로 빼지 않고 군집화에 넣었다.")
A(f"- 군집이 임상적으로 뜻이 통하는지는 최빈 CC 로만 확인했다. 군집 안의 나머지 "
  f"{100 - PT.top_cc_share.median():.0f}% 가 무엇인지는 표의 top5 까지만 실었다.")
A("- SafeDrug 성능을 군집별로 재는 것은 아직 하지 않았다. HADM_ID 로 이어 붙이면 된다.\n")

A("## 10. 산출 파일\n")
A("""| 파일 | 내용 |
|---|---|
| `36_cc_text.pkl` | 방문별 CC 원문·인구학 제거본 |
| `table45_cc_extract.csv` | 추출 커버리지 |
| `cc_cluster_assignments.csv` | 전 방문 라벨(`clusterCC` A / `clusterCC_bert` B / `stratum`) |
| `table46_cc_clusters.csv` | A 군집표 |
| `table49_cc_clusters_bert.csv` | B 군집표 |
| `table47_cc_ksweep.csv` | k 스윕(두 트랙) |
| `table48_cc_synonym.csv` | 동의어 병합 검정 |
| `37_cc_svd.npz` | SVD 좌표 |
| `figs/fig28_cc_sizes.png` | 두 트랙 군집 크기·최빈 CC |
| `figs/fig29_cc_heatmap.png` | 군집 × 주소견 개념 구성비 |
| `figs/fig30_cc_kpick.png` | k 선택 규칙 |""")

txt = "\n".join(L) + "\n"
(OUT / "REPORT_CC.md").write_text(txt, encoding="utf-8")

meta = {"primary_track": PRIMARY, "primary_repr": PNAME, "primary_k": PK,
        "decision_rule": "동의어 병합 쌍 수가 많은 쪽 (37 docstring 의 사전 가설을 검정)",
        "synonym_merge": {"A": mergeA, "B": mergeB, "pairs": NP},
        "largest_pct": {"A": float(tabA.pct.max()), "B": float(tabB.pct.max())},
        "clusters_with_test_ge_30": n_eval,
        "eta2_word_count": {"A": eta_A, "B": eta_B},
        "length_driven_clusters": int(len(short)), "empty_template_cc": n_tmpl,
        "n_visits": N, "n_clustered": len(s), "n_no_cc": int(len(noc)),
        "report_chars": len(txt)}
with open(OUT / "38_cc_report_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"[=] REPORT_CC.md {len(txt):,}자 | 본 분석 {PRIMARY}({PNAME}) k={PK}")
print(f"[=] 동의어 병합 A {mergeA}/{NP} · B {mergeB}/{NP} | test>=30 군집 {n_eval}/{PK}")
print(f"[=] fig28_cc_sizes.png, fig29_cc_heatmap.png, fig30_cc_kpick.png")
print(f"[=] 한 개념 60%+ 군집 {n_clean}/{PK} · {n_clean_v:,} 방문")
