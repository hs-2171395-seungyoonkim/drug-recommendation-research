"""§CCS 트랙 결과 그림 — 4패널.

57_ccs_track.py 가 만든 CCS 분할을 한 장으로 보여 준다. 보여야 할 것은 네 가지다.

  A 범주 크기 — 임상적으로 말이 되는 20개가 나오지만 '기타(소규모)' 가 35% 다.
  B 롱테일 — 193개 범주 중 몇 개가 코호트를 얼마나 덮는지. 문턱 200/100 을 같이 긋는다.
  C 지도 — 기본 임베딩 UMAP 에 CCS 색을 얹은 것. 이게 "진짜 Track 1" 이다.
    (이전 그림은 가중치 임베딩 좌표에 챕터 색을 얹고 baseline 이라고 적어 두었다.)
  D 라벨이 내용과 맞는가 — 가로축 고유 주진단 코드 수, 세로축 top-1 코드 비중.
    CCS 범주와 가중치 k-means 군집을 같은 축에 놓는다. 왼쪽 위가 좋은 것이다.

D 가 요점이다. 가중치 k-means 는 양극단으로 흩어진다 — 코드 1개짜리(군집이 아니라
코드)와 코드 175개짜리(이름표가 무의미한 잔여 덩어리)가 같이 있다. CCS 는 그 사이에
모인다. 코드를 여러 개 묶되 묶는 이유가 임상적으로 정의돼 있기 때문이다.

산출: out/figs/fig43_ccs_track.png (+ 바탕화면 사본)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
DESK = Path.home() / "Desktop"
BG, FG, GRID = "#1a1a2e", "white", "#3a3a52"
SMALL, SMALL_ALT = 200, 100
ETC, NA = "기타(소규모)", "미지정/매핑불가"

KO = {
    "Septicemia": "패혈증", "Complic devi": "기기·이식편 합병증",
    "Adlt resp fl": "성인 호흡부전", "Acute MI": "급성 심근경색",
    "chf;nonhp": "심부전(비고혈압성)", "Complic proc": "처치 합병증",
    "GI hemorrhag": "위장관 출혈", "Coron athero": "관상동맥 죽상경화",
    "Acute CVD": "급성 뇌혈관질환", "Hrt valve dx": "심장판막질환",
    "Pneumonia": "폐렴", "2ndary malig": "전이암",
    "DiabMel w/cm": "당뇨(합병증 동반)", "Aneurysm": "동맥류",
    "Dysrhythmia": "부정맥", "Oth liver dx": "기타 간질환",
    "Intracrn inj": "두개내 손상", "Alcohol-related disorders": "알코올 관련",
    "Asp pneumon": "흡인성 폐렴", "Ac renl fail": "급성 신부전",
    ETC: ETC, NA: "미지정",
}

gt = pd.read_csv(OUT / "table80_ccs_groups.csv")
au = pd.read_csv(OUT / "table79_weight_cluster_audit.csv")
asg = pd.read_csv(OUT / "ccs_assignments.csv")
proj = np.load(OUT / "42_dxtext_proj.npz")
df = pd.read_pickle(OUT / "40_dxtext.pkl")
assert (proj["HADM_ID"] == df["HADM_ID"].to_numpy()).all()
xy = proj["umap2"]
N = len(df)

gt["ko"] = gt["CCS범주"].map(KO).fillna(gt["CCS범주"])
real = gt[~gt["CCS범주"].isin([ETC, NA])].sort_values("n", ascending=False)
pal = dict(zip(real["CCS범주"], sns.color_palette("husl", len(real))))
GREY = (0.55, 0.55, 0.60)

fig = plt.figure(figsize=(21, 16.5), facecolor=BG)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.22, wspace=0.19)


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=14, color=FG, pad=11)
    ax.tick_params(colors=FG, labelsize=10)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)


# ── A. 범주 크기 ────────────────────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
order = gt.sort_values("n")
cols = [(0.85, 0.33, 0.33) if r == ETC else
        (0.4, 0.4, 0.45) if r == NA else pal[r]
        for r in order["CCS범주"]]
axA.barh(order["ko"], order["n"], color=cols, edgecolor=GRID, linewidth=0.5)
for y, (n, r) in enumerate(zip(order["n"], order["CCS범주"])):
    axA.text(n + 60, y, f"{n:,}", va="center", fontsize=9,
             color="#ffd479" if r == ETC else FG,
             fontweight="bold" if r == ETC else "normal")
axA.set_xlim(0, order["n"].max() * 1.18)
style(axA, f"A. CCS 주진단 범주별 방문 수  (문턱 {SMALL}명 미만은 '기타'로 병합)")
axA.set_xlabel("방문 수")
axA.annotate(f"코호트의 {order.loc[order['CCS범주'] == ETC, 'n%'].iloc[0]:.1f}%가\n"
             "여기로 몰린다",
             xy=(5054, len(order) - 1), xytext=(3100, len(order) - 7.2),
             fontsize=11, color="#ffd479", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#ffd479", lw=1.6))

# ── B. 롱테일 ───────────────────────────────────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
sizes = asg["ccs_desc"].value_counts().to_numpy()
rank = np.arange(1, len(sizes) + 1)
cum = np.cumsum(sizes) / N * 100
axB.bar(rank, sizes, color="#5ab9ea", width=1.0, linewidth=0)
axB.set_yscale("log")
axB.set_xlabel(f"CCS 범주 (크기 내림차순, 총 {len(sizes)}개)")
axB.set_ylabel("방문 수 (로그)")
for th, c, ls in ((SMALL, "#ff6b6b", "-"), (SMALL_ALT, "#ffd479", "--")):
    k = int((sizes >= th).sum())
    axB.axhline(th, color=c, ls=ls, lw=1.4)
    axB.axvline(k, color=c, ls=ls, lw=1.4)
    axB.text(k + 3, th * 1.35, f"{th}명 문턱 → {k}개 범주",
             color=c, fontsize=10.5, fontweight="bold")
ax2 = axB.twinx()
ax2.plot(rank, cum, color="#b088f9", lw=2.2)
ax2.set_ylim(0, 100); ax2.set_ylabel("누적 커버리지 %", color="#b088f9")
ax2.tick_params(colors="#b088f9", labelsize=10)
for s in ax2.spines.values():
    s.set_edgecolor(GRID)
ax2.text(len(sizes) * 0.52, cum[19] - 13,
         f"상위 20개 = {cum[19]:.0f}%", color="#b088f9",
         fontsize=11, fontweight="bold")
style(axB, "B. 롱테일 — 193개 범주 중 큰 것 20개가 코호트의 3분의 2")

# ── C. UMAP ─────────────────────────────────────────────────────────────────
axC = fig.add_subplot(gs[1, 0])
lab = asg.set_index("HADM_ID").reindex(df["HADM_ID"])["group"].to_numpy()
m_rest = ~np.isin(lab, real["CCS범주"].to_numpy())
axC.scatter(xy[m_rest, 0], xy[m_rest, 1], c=[GREY], s=4, alpha=0.28,
            linewidths=0, rasterized=True, label=f"{ETC}·미지정")
for r, ko in zip(real["CCS범주"], real["ko"]):
    m = lab == r
    axC.scatter(xy[m, 0], xy[m, 1], c=[pal[r]], s=6, alpha=0.72,
                linewidths=0, rasterized=True, label=ko)
axC.set_xticks([]); axC.set_yticks([])
style(axC, "C. 기본 임베딩(진단목록 통째, long) UMAP 에 CCS 색 — 진짜 Track 1")
# 범례를 따로 두지 않는다. 색은 A 패널의 막대와 같은 팔레트라 A 가 범례 노릇을 한다.
axC.text(0.985, 0.975, "색 = A 패널과 동일 (회색 = 기타·미지정)",
         transform=axC.transAxes, fontsize=10, color="#9aa0b5",
         ha="right", va="top")
axC.text(0.015, 0.02,
         "겹쳐 있다 — 주진단이 같아도 임베딩은 동반질환 목록을 따라간다\n"
         "(같은 CCS 라벨의 실루엣: 이 공간 -0.060 / 가중치 공간 +0.208)",
         transform=axC.transAxes, fontsize=10.5, color="#ffd479", va="bottom")

# ── D. 라벨 vs 내용 ─────────────────────────────────────────────────────────
axD = fig.add_subplot(gs[1, 1])
ccs_pts = gt[~gt["CCS범주"].isin([NA])]
axD.scatter(ccs_pts["고유_SEQ1코드수"], ccs_pts["top1_비중%"],
            s=ccs_pts["n"] / 3.2, c="#5ab9ea", alpha=0.72,
            edgecolors="white", linewidths=0.9, label="CCS 범주 (의학 표준)", zorder=3)
axD.scatter(au["고유_SEQ1코드수"], au["top1_순도%_군집n기준"],
            s=au["n"] / 3.2, c="#ff9f43", alpha=0.72, marker="^",
            edgecolors="white", linewidths=0.9,
            label="가중치 k-means k=30 (알고리즘)", zorder=3)
axD.set_xscale("log")
axD.set_xlabel("군집 안의 고유 주진단 코드 수 (로그)")
axD.set_ylabel("top-1 주진단 코드 비중 %")
axD.set_ylim(-4, 108)
axD.grid(alpha=0.16, color=GRID, zorder=0)
style(axD, "D. 이름표가 내용과 맞는가 — 왼쪽 위가 좋은 것")

# 글자 위치는 데이터 좌표로 직접 준다. offset 으로 주면 점 위에 겹친다.
ANN = [("가중치", 7, "c7·c23 — 코드 1개짜리\n(군집이 아니라 코드 하나다)\nCCS 는 둘을 '성인 호흡부전'으로 묶는다",
        55, 99, "left"),
       ("CCS", "Coron athero", "관상동맥 죽상경화\n코드 5개, top-1 98.5%", 11, 90, "left"),
       ("가중치", 18, "c18 '위장관 출혈2'\n코드 175개, top-1 9.0%", 45, 48, "left"),
       ("가중치", 9, "c9 '당뇨성 케토산증'\n실제 DKA 7.9%", 240, 33, "left"),
       ("CCS", ETC, "기타(소규모)\n코드 1,027개, top-1 2.3%", 300, 16, "left")]
for kind, key, txt, tx, ty, ha in ANN:
    if kind == "가중치":
        r = au[au["cluster"] == key].iloc[0]
        x, y, c = r["고유_SEQ1코드수"], r["top1_순도%_군집n기준"], "#ff9f43"
    else:
        r = gt[gt["CCS범주"] == key].iloc[0]
        x, y, c = r["고유_SEQ1코드수"], r["top1_비중%"], "#5ab9ea"
    axD.annotate(txt, xy=(x, y), xytext=(tx, ty), textcoords="data",
                 fontsize=9.5, color=c, fontweight="bold", zorder=4, ha=ha,
                 va="center",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.3,
                                 connectionstyle="arc3,rad=0.12",
                                 shrinkA=6, shrinkB=6))
axD.axhspan(0, 25, color="#ff6b6b", alpha=0.07, zorder=0)
axD.text(0.012, 0.035, "이 아래는 이름표가 4명 중 1명도 못 맞힌다",
         transform=axD.transAxes, ha="left", fontsize=10.5, color="#ff6b6b")
axD.legend(fontsize=10.5, framealpha=0.3, labelcolor=FG, facecolor="#222",
           edgecolor=GRID, loc="lower left", bbox_to_anchor=(0.0, 0.09),
           markerscale=0.55)

fig.suptitle("CCS 주진단 트랙 — 원안(CCS 매핑)을 실제로 구현한 결과",
             fontsize=21, color=FG, y=0.973)
fig.text(0.5, 0.006,
         f"AHRQ HCUP Single-Level CCS for ICD-9-CM 2015 · 방문 {N:,} 중 "
         f"{len(asg) - int(asg['ccs'].isna().sum()):,}건 매핑(매핑불가 0) · "
         "주진단 = SEQ_NUM=1 · 문턱 200명은 사전 고정",
         ha="center", fontsize=11, color="#9aa0b5")
plt.tight_layout(rect=(0, 0.016, 1, 0.962))
FIGS.mkdir(parents=True, exist_ok=True)
for p in (FIGS / "fig43_ccs_track.png", DESK / "fig43_ccs_track.png"):
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[+] out/figs/fig43_ccs_track.png (+ 바탕화면)")
