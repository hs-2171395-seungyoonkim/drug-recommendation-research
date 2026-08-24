"""§외부 채점 그림 — 실루엣 순위와 실제 성능 순위가 뒤집힌다.

59_external_check.py 의 결과를 그린다. 핵심 주장 네 개를 패널 하나씩 맡긴다.
  A 실루엣 1위와 외부 채점 1위가 다르다 (범프 차트)
  B 무작위 대비 추가 설명력 (막대) — 같은 k 조회표를 기준선으로
  C 같은 k 순수 조회표 대비 이득 — 임베딩이 실제로 더하는 양
  D 상한선(SEQ1 코드 그대로) 대비 회수율 — 압축의 대가

실루엣은 한 공간에서만 잰다(가중치_concise). 공간을 바꾸면 승자가 바뀌므로
"어느 공간에서 잰 실루엣이냐" 를 고정하지 않으면 A 패널이 성립하지 않는다.

산출: out/figs/fig44_external_check.png (+ 바탕화면 사본)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
DESK = Path.home() / "Desktop"
BG, FG = "#1a1a2e", "white"

tb = pd.read_csv(OUT / "table81_external_check.csv")
tb["짧은이름"] = tb["분할"].str.replace(r"\(.*\)", "", regex=True).str.strip()

SHORT = {"CCS(주분석)": "CCS\nk=22",
         "ICD9_chapter": "ICD9 챕터\nk=18",
         "기본_kmeans_k25(출하판)": "기본 임베딩\nk-means k=25",
         "가중치_kmeans_k30": "가중치 임베딩\nk-means k=30",
         "SEQ1최빈24+기타(조회표,k=25)": "순수 조회표\n최빈24+기타 k=25",
         "SEQ1코드_그대로(k=1400)": "SEQ1 코드 그대로\nk=1,400 (상한선)"}
COL = {"CCS(주분석)": "#4ecdc4", "ICD9_chapter": "#95a5a6",
       "기본_kmeans_k25(출하판)": "#5b9bd5", "가중치_kmeans_k30": "#ff9f43",
       "SEQ1최빈24+기타(조회표,k=25)": "#e056a0",
       "SEQ1코드_그대로(k=1400)": "#7f8c8d"}
tb["c"] = tb["분할"].map(COL)
LOOK = "SEQ1최빈24+기타(조회표,k=25)"
CEIL = "SEQ1코드_그대로(k=1400)"
O = {"O1_처방약물": "처방 약물", "O2_시술": "시술",
     "O3_30일재입원": "30일 재입원", "O4_나이": "나이"}

# ── 실루엣: 한 공간(가중치_concise)에서 전부 다시 잰다 ──────────────────────
df = pd.read_pickle(OUT / "40_dxtext.pkl")
HADM = df["HADM_ID"].to_numpy()
z = np.load(OUT / "emb_dxtext_weight_concise.npz")
assert (z["HADM_ID"] == HADM).all()
P = PCA(n_components=50, random_state=0).fit_transform(z["E"]).astype(np.float32)


def load(path, col):
    s = pd.read_csv(OUT / path, usecols=["HADM_ID", col])
    return s.set_index("HADM_ID").reindex(HADM)[col].to_numpy()


seq1 = df["seq1_code"].fillna("NA")
top24 = set(seq1.value_counts().index[:24])
LABS = {
    "CCS(주분석)": pd.factorize(load("ccs_assignments.csv", "group"))[0],
    "ICD9_chapter": pd.factorize(load("dxtext_chapter_assignments.csv", "chapter"))[0],
    "기본_kmeans_k25(출하판)": load("dxtext_cluster_assignments.csv", "primary_cluster"),
    "가중치_kmeans_k30": load("dxtext_weight_cluster_assignments.csv", "primary_cluster"),
    LOOK: pd.factorize(np.where(seq1.isin(top24), seq1, "__기타__"))[0],
}
SIL = {n: float(silhouette_score(P, np.asarray(l), sample_size=5000, random_state=0))
       for n, l in LABS.items()}
for n, v in SIL.items():
    print(f"[.] 실루엣(가중치공간) {n:28s} {v:+.4f}", flush=True)

# ── 그림 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(21, 15), facecolor=BG)
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.20,
                      left=0.055, right=0.975, top=0.855, bottom=0.075)


def style(ax, title, sub=None):
    ax.set_facecolor(BG)
    ax.set_title(title + (f"\n{sub}" if sub else ""), fontsize=13.5,
                 color=FG, pad=12, fontweight="bold")
    ax.tick_params(colors="#bbb", labelsize=10)
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.grid(axis="y", color="#333", lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# A ─ 범프 차트: 실루엣 순위 vs 외부 채점 순위 ───────────────────────────────
axA = fig.add_subplot(gs[0, 0])
style(axA, "A. 실루엣이 1등이라고 실제로 1등이 아니다",
      "세 잣대의 순위 (1위가 위) — 선이 교차하면 잣대끼리 답이 다르다는 뜻")
names = list(LABS)
treat = tb.set_index("분할").loc[names, ["O1_처방약물_delta%", "O2_시술_delta%"]].mean(1)
pat = tb.set_index("분할").loc[names, ["O3_30일재입원_delta%", "O4_나이_delta%"]]
pat = (pat / pat.max()).mean(1)                       # 척도가 달라 정규화 후 평균
cols = [("실루엣\n(가중치 공간)", pd.Series(SIL)[names]),
        ("외부 채점 ①\n치료 (약물·시술)", treat),
        ("외부 채점 ②\n환자 (나이·재입원)", pat)]
n = len(names)
rank = {t: v.rank(ascending=False).to_dict() for t, v in cols}
for nm in names:
    ys = [n + 1 - rank[t][nm] for t, _ in cols]
    axA.plot(range(3), ys, "-o", color=COL[nm], lw=3.2, ms=13, zorder=3,
             mec=BG, mew=2, alpha=0.92)
    axA.text(-0.10, ys[0], SHORT[nm].replace("\n", " "), ha="right", va="center",
             fontsize=10.5, color=COL[nm], fontweight="bold")
    axA.text(2.10, ys[2], SHORT[nm].replace("\n", " "), ha="left", va="center",
             fontsize=10.5, color=COL[nm], fontweight="bold")
axA.set_xticks(range(3)); axA.set_xticklabels([t for t, _ in cols], fontsize=11.5, color=FG)
axA.set_yticks(range(1, n + 1)); axA.set_yticklabels([f"{n + 1 - i}위" for i in range(1, n + 1)])
axA.set_xlim(-1.55, 3.55); axA.set_ylim(0.45, n + 0.55)
axA.grid(axis="y", color="#333", lw=0.6, alpha=0.7)
axA.annotate("기본 임베딩: 실루엣 꼴찌 → 환자 특성 1위",
             xy=(2, n + 1 - rank[cols[2][0]]["기본_kmeans_k25(출하판)"]),
             xytext=(0.62, 4.62), textcoords="data", fontsize=11,
             color="#7fc0f5", fontweight="bold", ha="center",
             bbox=dict(boxstyle="round,pad=0.35", fc="#16233a", ec="#5b9bd5", alpha=0.95),
             arrowprops=dict(arrowstyle="->", color="#5b9bd5", lw=1.7,
                             connectionstyle="arc3,rad=0.30", shrinkA=6, shrinkB=12))

# B ─ 무작위 대비 추가 설명력 ────────────────────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
style(axB, "B. 군집을 만들 때 안 쓴 정보로 채점하면",
      "eta² - 무작위(같은 k·같은 크기, 20회) — 0 이면 아무것도 설명 못 한다는 뜻")
keys = [k for k in tb["분할"] if k != CEIL]
outs = ["O1_처방약물", "O2_시술", "O4_나이"]
x = np.arange(len(outs)); w = 0.155
for i, nm in enumerate(keys):
    r = tb[tb["분할"] == nm].iloc[0]
    v = [r[f"{o}_delta%"] for o in outs]
    hatch = "//" if nm == LOOK else None
    axB.bar(x + (i - 2) * w, v, w * 0.92, color=COL[nm], label=SHORT[nm].replace("\n", " "),
            zorder=3, hatch=hatch, edgecolor=BG, lw=0.8)
    for xi, vi in zip(x + (i - 2) * w, v):
        axB.text(xi, vi + 0.22, f"{vi:.1f}", ha="center", fontsize=8.4,
                 color=COL[nm], fontweight="bold", zorder=4)
r = tb[tb["분할"] == CEIL].iloc[0]
for xi, o in zip(x, outs):
    axB.plot([xi - 0.42, xi + 0.42], [r[f"{o}_delta%"]] * 2, "--", color="#7f8c8d", lw=2, zorder=2)
    axB.text(xi + 0.44, r[f"{o}_delta%"], f"{r[f'{o}_delta%']:.1f}", fontsize=8.6,
             color="#95a5a6", va="center", fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels([O[o] for o in outs], fontsize=12, color=FG)
axB.set_ylabel("무작위 대비 추가 설명력 (%p)", color="#ccc", fontsize=11)
axB.set_ylim(0, 23)
axB.text(0.985, 0.965, "회색 점선 = SEQ1 코드 1,400개 그대로 (상한선)",
         transform=axB.transAxes, ha="right", va="top", fontsize=9.5, color="#95a5a6")
axB.legend(fontsize=9, loc="upper left", facecolor="#20203a", edgecolor="#444",
           labelcolor=FG, framealpha=0.95)

# C ─ 같은 k 조회표 대비 이득 ────────────────────────────────────────────────
axC = fig.add_subplot(gs[1, 0])
style(axC, "C. 임베딩은 '진단 코드 조회표'보다 나은가",
      "같은 k=25 순수 조회표(최빈24+기타) 를 0 으로 놓았을 때의 차이")
base = tb[tb["분할"] == LOOK].iloc[0]
keys2 = [k for k in tb["분할"] if k not in (LOOK, CEIL)]
for i, nm in enumerate(keys2):
    r = tb[tb["분할"] == nm].iloc[0]
    v = [r[f"{o}_delta%"] - base[f"{o}_delta%"] for o in outs]
    axC.bar(x + (i - 1.5) * 0.20, v, 0.185, color=COL[nm], zorder=3,
            label=SHORT[nm].replace("\n", " "), edgecolor=BG, lw=0.8)
    for xi, vi in zip(x + (i - 1.5) * 0.20, v):
        axC.text(xi, vi + (0.13 if vi >= 0 else -0.42), f"{vi:+.1f}", ha="center",
                 fontsize=8.8, color=COL[nm], fontweight="bold", zorder=4)
axC.axhline(0, color="#e056a0", lw=2.2, zorder=2)
axC.text(2.46, 0.14, "순수 조회표", fontsize=10, color="#e056a0", fontweight="bold", ha="right")
axC.set_xticks(x); axC.set_xticklabels([O[o] for o in outs], fontsize=12, color=FG)
axC.set_ylabel("조회표 대비 (%p)", color="#ccc", fontsize=11)
axC.set_ylim(-1.9, 11.5)
axC.legend(fontsize=9.5, loc="upper left", facecolor="#20203a", edgecolor="#444",
           labelcolor=FG, framealpha=0.95)
axC.text(0.055, 0.615,
         "CCS·가중치 임베딩은 세 항목 모두 조회표 위 - 코드 조회로\n"
         "설명 안 되는 것을 잡고 있다. 반면 ICD9 챕터는 약물 -0.2·시술 -1.4 로\n"
         "조회표만 못하고, 기본 임베딩도 시술은 -1.4 다(대신 나이 +9.7).\n"
         "실루엣으로는 조회표 0.155 < k-means 0.383 이라 안 보이던 구분이다.",
         transform=axC.transAxes, ha="left", va="top", fontsize=10,
         color="#ffd479", linespacing=1.55)

# D ─ 상한선 회수율 ──────────────────────────────────────────────────────────
axD = fig.add_subplot(gs[1, 1])
style(axD, "D. 압축의 대가 — 상한선의 몇 %를 회수하나",
      "SEQ1 코드 1,400개 그대로 쓴 값을 100 으로 놓았을 때")
for i, nm in enumerate(keys):
    r = tb[tb["분할"] == nm].iloc[0]
    v = [r[f"{o}_delta%"] / tb[tb["분할"] == CEIL].iloc[0][f"{o}_delta%"] * 100 for o in outs]
    hatch = "//" if nm == LOOK else None
    axD.bar(x + (i - 2) * w, v, w * 0.92, color=COL[nm], zorder=3,
            hatch=hatch, edgecolor=BG, lw=0.8)
    for xi, vi in zip(x + (i - 2) * w, v):
        axD.text(xi, vi + 0.9, f"{vi:.0f}", ha="center", fontsize=8.4,
                 color=COL[nm], fontweight="bold", zorder=4)
axD.axhline(100, color="#7f8c8d", ls="--", lw=2, zorder=2)
axD.text(2.48, 101.5, "상한선 = 100 (라벨 1,400개)", fontsize=9.5, color="#95a5a6",
         ha="right", fontweight="bold")
axD.set_xticks(x); axD.set_xticklabels([O[o] for o in outs], fontsize=12, color=FG)
axD.set_ylabel("상한선 대비 회수율 (%)", color="#ccc", fontsize=11)
axD.set_ylim(0, 118)
axD.text(0.018, 0.945,
         "라벨 22~30개로 상한선의 20~68% 를 회수한다.\n"
         "압축하면 반드시 잃는다 — 대신 사람이 읽을 수 있게 된다.\n"
         "군집화의 값어치는 정보량이 아니라 거기에 있다.",
         transform=axD.transAxes, ha="left", va="top", fontsize=10.5,
         color="#ffd479", linespacing=1.6)

fig.suptitle("실루엣 대신 '만들 때 안 쓴 정보' 로 채점하면 순위가 뒤집힌다",
             fontsize=21, color=FG, y=0.975, fontweight="bold")
fig.text(0.5, 0.935,
         "채점 재료: 처방 약물 151종 · 시술 467종 · 30일 재입원 10.8% · 나이 — "
         "전부 진단 텍스트에 한 글자도 안 들어갔다",
         ha="center", fontsize=13, color="#4ecdc4")
fig.text(0.5, 0.028,
         "eta² = 1 - SS_within/SS_total, 같은 k·같은 군집 크기로 라벨만 섞은 무작위 분할 20회 평균을 뺀 값 · "
         "방문 14,444 · 30일 재입원은 어느 분할도 못 설명한다(상한선조차 +0.67%p)",
         ha="center", fontsize=11, color="#999")

FIGS.mkdir(parents=True, exist_ok=True)
for p in (FIGS / "fig44_external_check.png", DESK / "fig44_external_check.png"):
    plt.savefig(p, dpi=150, facecolor=BG)
plt.close()
print("[+] out/figs/fig44_external_check.png (+ 바탕화면)")
