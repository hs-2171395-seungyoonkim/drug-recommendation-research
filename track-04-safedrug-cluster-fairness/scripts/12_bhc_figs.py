"""BHC 트랙 그림: UMAP 나란히(임베딩 클러스터 vs diagnose 클러스터) + 교차표 히트맵."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIG = OUT / "figs"

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1 = "#2a78d6"
MUTED = "#c9c8c3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})

bhc = pd.read_pickle(OUT / "bhc_prep.pkl").reset_index(drop=True)
U = np.load(OUT / "umap_bhc_masked.npz")["U"]
bassign = pd.read_csv(OUT / "cluster_assignments_bhc.csv")
dassign = pd.read_pickle(OUT / "cluster_assignments.pkl")

s = bassign[(bassign.partition == "BHC_CB_masked") & (bassign.k == 15)]
lab_bhc = bhc["HADM_ID"].map(dict(zip(s.HADM_ID, s.cluster_id))).to_numpy()
s = dassign[(dassign.partition == "C1_id") & (dassign.k == 15)]
lab_dx = bhc["HADM_ID"].map(dict(zip(s.HADM_ID, s.cluster_id))).to_numpy()

parsed = pd.read_pickle(OUT / "parsed.pkl")[["HADM_ID", "diagnose_l"]]
diag_l = bhc[["HADM_ID"]].merge(parsed, on="HADM_ID")["diagnose_l"]


def top_dx(labels, g):
    cnt = {}
    for l in diag_l[labels == g]:
        for d in set(l):
            cnt[d] = cnt.get(d, 0) + 1
    return max(cnt, key=cnt.get) if cnt else ""


# ---------------- 그림 6: 소형 다중 패널 — BHC 클러스터 (fig1 과 같은 형식) ----------------
K = 15
ncol, nrow = 5, 3
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow), sharex=True, sharey=True)
for g, ax in enumerate(axes.ravel()):
    m = lab_bhc == g
    ax.scatter(U[~m, 0], U[~m, 1], s=1.5, c=MUTED, lw=0, alpha=0.5, rasterized=True)
    ax.scatter(U[m, 0], U[m, 1], s=2.5, c=S1, lw=0, alpha=0.75, rasterized=True)
    ax.set_title(f"B{g} · n={m.sum():,}\n{top_dx(lab_bhc, g)[:34]}", fontsize=8, color=INK, loc="left")
    ax.tick_params(labelsize=7)
fig.suptitle("그림 6. BHC ClinicalBERT(마스킹) UMAP — k-means k=15 클러스터별 위치", fontsize=11, x=0.01, ha="left", color=INK)
fig.supxlabel("UMAP 1", color=INK2, fontsize=9)
fig.supylabel("UMAP 2", color=INK2, fontsize=9)
fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
fig.savefig(FIG / "fig6_bhc_umap_clusters.png", dpi=150)
plt.close(fig)

# ---------------- 그림 7: 같은 UMAP 좌표에 diagnose 클러스터 ----------------
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow), sharex=True, sharey=True)
for g, ax in enumerate(axes.ravel()):
    m = lab_dx == g
    ax.scatter(U[~m, 0], U[~m, 1], s=1.5, c=MUTED, lw=0, alpha=0.5, rasterized=True)
    ax.scatter(U[m, 0], U[m, 1], s=2.5, c="#eb6834", lw=0, alpha=0.75, rasterized=True)
    ax.set_title(f"C{g} · n={m.sum():,}\n{top_dx(lab_dx, g)[:34]}", fontsize=8, color=INK, loc="left")
    ax.tick_params(labelsize=7)
fig.suptitle("그림 7. 같은 UMAP 좌표(BHC 임베딩)에 diagnose C1_id k=15 클러스터를 얹은 것 — 뭉치면 텍스트가 코드 정보를 담는다",
             fontsize=11, x=0.01, ha="left", color=INK)
fig.supxlabel("UMAP 1", color=INK2, fontsize=9)
fig.supylabel("UMAP 2", color=INK2, fontsize=9)
fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
fig.savefig(FIG / "fig7_bhc_umap_dxoverlay.png", dpi=150)
plt.close(fig)

# ---------------- 그림 8: 교차표 히트맵 (행 정규화) ----------------
ct = pd.crosstab(pd.Series(lab_bhc, name="BHC"), pd.Series(lab_dx, name="DX"))
ctn = ct.div(ct.sum(1), axis=0)
# 행/열을 대각 우세하게 정렬
order_r = ctn.max(1).sort_values(ascending=False).index
ctn = ctn.loc[order_r]
order_c = ctn.idxmax(1).drop_duplicates().tolist() + [c for c in ctn.columns if c not in ctn.idxmax(1).values]
seen = []
for c in order_c:
    if c not in seen:
        seen.append(c)
ctn = ctn[seen]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(ctn.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ctn.columns)), [f"C{c}" for c in ctn.columns], fontsize=8)
ax.set_yticks(range(len(ctn.index)), [f"B{r}" for r in ctn.index], fontsize=8)
for i in range(ctn.shape[0]):
    for j in range(ctn.shape[1]):
        v = ctn.iloc[i, j]
        if v >= 0.10:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 0.55 else INK)
ax.set_xlabel("diagnose 클러스터 (C1_id k=15)")
ax.set_ylabel("BHC 임베딩 클러스터 (CB masked k=15)")
ax.set_title("그림 8. BHC 클러스터 × diagnose 클러스터 — 행 정규화 비율", loc="left", fontsize=11, color=INK)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("행 내 비율", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig8_bhc_dx_heatmap.png", dpi=150)
plt.close(fig)

print("saved fig6, fig7, fig8")
