"""§10 그림. 정적 PNG(라이트 서페이스), 카테고리 색은 검증된 슬롯 1~4만 사용."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED = "#c9c8c3"

plt.rcParams.update(
    {
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    }
)

Z = np.load(OUT / "embeddings.npz")["Z_id"]
assign = pd.read_pickle(OUT / "cluster_assignments.pkl")
df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
t2 = pd.read_csv(OUT / "table2_cluster_performance.csv")
p_const = pd.read_csv(OUT / "table3_power.csv")
p_prev = pd.read_csv(OUT / "table3b_power_prev.csv")


def labels_of(pn, k):
    s = assign[(assign.partition == pn) & (assign.k == k)]
    return df["HADM_ID"].map(dict(zip(s.HADM_ID, s.cluster_id))).to_numpy()


# ---------------------------------------------------------------- 그림 1
K = 15
lab = labels_of("C1_id", K)
x, y = Z[:, 0], Z[:, 1]
top = {}
for g in range(K):
    c = {}
    for l in df.loc[lab == g, "diagnose_l"]:
        for d in set(l):
            c[d] = c.get(d, 0) + 1
    top[g] = max(c, key=c.get) if c else ""

ncol = 5
nrow = int(np.ceil(K / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow), sharex=True, sharey=True)
for g, ax in enumerate(axes.ravel()):
    if g >= K:
        ax.axis("off")
        continue
    m = lab == g
    ax.scatter(x[~m], y[~m], s=1.5, c=MUTED, lw=0, alpha=0.5, rasterized=True)
    ax.scatter(x[m], y[m], s=2.5, c=S1, lw=0, alpha=0.75, rasterized=True)
    ax.set_title(f"C{g} · n={m.sum():,}\n{top[g][:34]}", fontsize=8, color=INK, loc="left")
    ax.tick_params(labelsize=7)
fig.suptitle(
    "그림 1. 진단 multi-hot → SVD 2차원, k-means(k=15) 클러스터별 위치 (id 벡터화)",
    fontsize=11, x=0.01, ha="left", color=INK,
)
fig.supxlabel("SVD 1", color=INK2, fontsize=9)
fig.supylabel("SVD 2", color=INK2, fontsize=9)
fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
fig.savefig(FIG / "fig1_svd_clusters.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 그림 2
def dotci(ax, sub, col, lo, hi, color, title, labeler):
    sub = sub.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    yy = np.arange(len(sub))
    ax.hlines(yy, sub[lo], sub[hi], color=color, lw=2.2, alpha=0.45)
    ax.plot(sub[col], yy, "o", color=color, ms=5.5, mec=SURF, mew=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([labeler(r) for _, r in sub.iterrows()], fontsize=7.5)
    ax.grid(axis="x", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel("방문별 Jaccard 평균 (95% CI, 환자 단위 부트스트랩)", fontsize=8)


p1 = t2[(t2.partition == "P1_ge100") & (t2.test_visits >= 30)].copy()
p1["nm"] = p1["top3_dx"].str.split("|").str[0].str.strip().str.replace(r"\s+\d+%$", "", regex=True)
p1.loc[p1.cluster == -1, "nm"] = "기타(라벨 <100)"
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
dotci(axes[0], p1, "jac_const", "jac_const_lo", "jac_const_hi", S1,
      "상수 top-23 예측기", lambda r: f"{r.nm[:34]}  (n={r.test_visits}, 약{r.mean_n_drug:.0f})")
dotci(axes[1], p1, "jac_prev", "jac_prev_lo", "jac_prev_hi", S2,
      "copy-previous 예측기", lambda r: f"{r.nm[:34]}  (n={r.n_prev})")
fig.suptitle("그림 2. P1(첫 진단 라벨) 클러스터별 자명 baseline 성능 — test 방문 30건 이상",
             fontsize=11, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig2_cluster_jaccard_P1.png", dpi=150)
plt.close(fig)

c1 = t2[(t2.partition == "C1_id") & (t2.k == 15)].copy()
c1["nm"] = c1["top3_dx"].str.split("|").str[0].str.strip().str.replace(r"\s+\d+%$", "", regex=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
dotci(axes[0], c1, "jac_const", "jac_const_lo", "jac_const_hi", S1,
      "상수 top-23 예측기", lambda r: f"C{r.cluster} {r.nm[:30]} (n={r.test_visits}, 약{r.mean_n_drug:.0f})")
dotci(axes[1], c1, "jac_prev", "jac_prev_lo", "jac_prev_hi", S2,
      "copy-previous 예측기", lambda r: f"C{r.cluster} {r.nm[:30]} (n={r.n_prev})")
fig.suptitle("그림 3. C1_id k=15 클러스터별 자명 baseline 성능",
             fontsize=11, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig3_cluster_jaccard_C1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 그림 4: 교란
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (col, color, ttl) in zip(axes, [("jac_const", S1, "상수 top-23"), ("jac_prev", S2, "copy-previous")]):
    for sub, mark, lbl in [(p1, "o", "P1 첫진단"), (c1, "s", "C1_id k=15")]:
        s = sub.dropna(subset=[col])
        ax.scatter(s["mean_n_drug"], s[col], s=46, marker=mark,
                   facecolor=color if mark == "o" else "none",
                   edgecolor=color, lw=1.4, alpha=0.85, label=lbl)
    s = p1.dropna(subset=[col])
    r = np.corrcoef(s["mean_n_drug"], s[col])[0, 1]
    ax.set_title(f"{ttl} — P1 클러스터 r = {r:+.2f}", fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel("클러스터 평균 처방 약물 수")
    ax.set_ylabel("클러스터 평균 Jaccard")
    ax.grid(lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("그림 4. 클러스터별 성능 vs 약물 개수 — 교란 점검", fontsize=11, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / "fig4_confound_drugcount.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 그림 5: 검정력
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
cols = {"C1_id": S1, "C2_id": S2, "C1_name": S3, "C2_name": S4}
for ax, (tab, key, ttl, spread) in zip(
    axes,
    [(p_const, "worst_mdd", "상수 top-23 (sd=0.093, test n=2,880)", 0.085),
     (p_prev, "worst_mdd_prev", "copy-previous (sd=0.138, test n=1,618)", 0.059)],
):
    for pn, c in cols.items():
        s = tab[tab.partition == pn].sort_values("k")
        ax.plot(s["k"], s[key], color=c, lw=1.9, label=pn)
        last = s.iloc[-1]
        ax.annotate(pn, (last["k"], last[key]), color=c, fontsize=7.5,
                    xytext=(3, 0), textcoords="offset points", va="center")
    ax.axhline(spread, color=INK2, lw=1.1, ls="--")
    ax.annotate(f"C 분할에서 실제 관측된 클러스터 간 격차 = {spread:.3f}",
                (32.6, spread), color=INK2, fontsize=7.5, xytext=(0, 4), textcoords="offset points", ha="right")
    ax.set_title(ttl, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel("k")
    ax.set_ylabel("검출 가능한 최소 격차 (MDD, α=.05 power=.80)")
    ax.grid(lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(4, 33)
fig.suptitle("그림 5. k가 커질수록 꼬리 클러스터가 작아져 검출력이 무너진다 — 점선 아래여야 비교 가능",
             fontsize=11, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / "fig5_power_vs_k.png", dpi=150)
plt.close(fig)

print("saved figs:", sorted(p.name for p in FIG.glob("*.png")))
