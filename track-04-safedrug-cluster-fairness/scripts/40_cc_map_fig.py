"""fig32 — 주 증상 군집 24개를 2차원 지도로.

앞서 SVD 2축 산점도를 넣었다가 뺐다. 50차원에서 나눈 것을 선형 2축에 눌러 담으면
겹쳐 보이는 것이 당연한데, 그림만 보면 "군집이 안 갈라졌다"로 읽히기 때문이다.

UMAP 을 방문 12,804건에 그대로 돌리면 군집이 산산조각 난다. CC 는 중앙 3단어로 짧아
같은 문자열이 대량 중복되고(고유 7,028개 / 방문 12,804), 같은 문자열은 같은 임베딩이라
(좌표 편차 3e-4) 조밀한 동점 덩어리가 생긴다. n_neighbors=15 에서는 그 덩어리마다
끊긴 성분이 되어 화면이 색종이 조각처럼 된다.

그래서 두 가지를 바꿨다.
  1) 고유 문자열 단위로 투영한다. 점 하나 = CC 표현 하나, 점 크기 = 그 표현의 방문 수.
  2) n_neighbors 를 늘려(30) 국소 덩어리보다 전체 배치가 보이게 한다.

이 그림은 분리도의 '증거'가 아니다. UMAP 은 뭉쳐 보이게 만드는 성질이 있다. 군집화를
실제로 한 공간(SVD 50차원)에서 잰 실루엣을 제목에 같이 적는다 — 그 숫자가 근거고
그림은 배치를 보여줄 뿐이다.

산출: out/figs/fig32_cc_map.png, out/40_cc_map_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"

SURF, INK, INK2, FAINT = "#fcfcfb", "#0b0b0b", "#52514e", "#a9a8a3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "text.color": INK, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})
SEED = 0
CLEAN = 60          # 38 의 fig29 와 같은 기준: 한 개념이 이만큼이면 이름을 붙인다
NN, MD = 30, 0.25   # 동점 덩어리로 성분이 끊기지 않을 만큼 넓게

cl = json.load(open(OUT / "37_cc_cluster_meta.json", encoding="utf-8"))
asg = pd.read_csv(OUT / "cc_cluster_assignments.csv")
z = np.load(OUT / "37_cc_svd.npz")
asg["cc_demog_stripped"] = asg["cc_demog_stripped"].fillna("")
s = asg[asg.stratum == "CC"].reset_index(drop=True)
s["t"] = s["cc_demog_stripped"].str.lower()
K = int(cl["B"]["k"])
lab = s["clusterCC_bert"].to_numpy().astype(int)
Z = z["Z_cb"]
assert len(Z) == len(s), (len(Z), len(s))

# 군집 이름 — 38_cc_report.py 의 CONCEPTS 를 그대로 읽어 온다. 두 벌로 두면 어긋난다.
src = (ROOT / "scripts" / "38_cc_report.py").read_text(encoding="utf-8")
CONCEPTS = eval(src.split("CONCEPTS = ")[1].split("\n]")[0] + "\n]")
CM = pd.DataFrame({nm: s.t.str.contains(rx, regex=True, na=False).to_numpy()
                   for nm, rx in CONCEPTS})
H = CM.groupby(lab).mean() * 100
name = {c: (H.loc[c].idxmax() if H.loc[c].max() >= CLEAN else None) for c in H.index}
size = pd.Series(lab).value_counts()
# 개념 이름만 쓰면 C5 와 C10 이 둘 다 "호흡곤란"이라 그림이 고장난 것처럼 보인다.
# 실제로는 동의어가 갈린 것이므로(table48) 대표 표현을 같이 적어 그 사실이 보이게 한다.
top_cc = s.groupby(lab)["t"].agg(lambda v: v.value_counts().index[0])

sil = round(float(silhouette_score(Z, lab, sample_size=5000, random_state=SEED)), 4)

# 고유 문자열 단위로 접는다. 같은 문자열은 좌표도 군집도 같으므로 첫 행을 대표로 쓴다.
first = s.reset_index().groupby("t", sort=False)["index"].first()
cnt = s["t"].value_counts()
ui = first.to_numpy()
ucnt = cnt.reindex(first.index).to_numpy()
ulab = lab[ui]
U = umap.UMAP(n_neighbors=NN, min_dist=MD, metric="cosine",
              random_state=SEED).fit_transform(Z[ui])

fig, ax = plt.subplots(figsize=(11.6, 8.8))
named = [c for c in sorted(H.index) if name[c]]
other = [c for c in sorted(H.index) if not name[c]]
cmap = plt.get_cmap("tab20")
col = {c: cmap(i % 20) for i, c in enumerate(named)}
# 점 크기 = 그 표현의 방문 수. 고유 문자열로 접으면 "chest pain 370건"이 점 하나가 되므로
# 크기 폭을 넓게 준다. 안 그러면 순도 높은 큰 군집이 화면에서 사라진다.
ms = 2.0 + 30.0 * np.sqrt(ucnt / ucnt.max())
for c in other:
    m = ulab == c
    ax.scatter(U[m, 0], U[m, 1], s=ms[m], c="#dcdbd7", lw=0, alpha=0.6, zorder=2)
for c in named:
    m = ulab == c
    ax.scatter(U[m, 0], U[m, 1], s=ms[m], color=col[c], lw=0, alpha=0.72, zorder=3)

# 라벨은 그 군집에서 가장 흔한 표현의 위치에 건다. 중앙값은 흩어진 군집에서 빈 곳을 가리킨다.
gap = (U[:, 1].max() - U[:, 1].min()) * 0.043
anchor = []
for c in named:
    m = np.where(ulab == c)[0]
    j = m[np.argmax(ucnt[m])]
    anchor.append((U[j, 0], U[j, 1], c))
prev = -1e9
for cx, cy, c in sorted(anchor, key=lambda t: t[1]):
    ty = max(cy, prev + gap)
    prev = ty
    ax.plot([cx], [cy], marker="X", ms=9, color=col[c], mec=SURF, mew=1.3, zorder=7)
    ax.annotate(f"{name[c]} · {top_cc[c][:26]}  C{c} ({size[c]:,})", (cx, cy),
                xytext=(cx + (U[:, 0].max() - U[:, 0].min()) * 0.05, ty),
                fontsize=8.2, color=INK, zorder=8, va="center", ha="left",
                bbox=dict(fc=SURF, ec="#e6e5e1", lw=0.7, pad=2.0, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.7,
                                shrinkA=0, shrinkB=5))
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title(f"fig32 · 주 증상 군집 {K}개의 2차원 지도 (ClinicalBERT, UMAP "
             f"n_neighbors={NN})\n"
             f"점 하나 = CC 표현 하나(고유 {len(ui):,}개), 점 크기 = 그 표현의 방문 수. "
             f"색과 이름은 한 주소견이 {CLEAN}% 이상인 군집 {len(named)}개, "
             f"회색은 나머지 {len(other)}개다.\n"
             f"이름 뒤는 그 군집의 최빈 표현이다 — 같은 주소견이 표현별로 갈린 군집이 "
             f"보인다(호흡곤란: shortness of breath / dyspnea).\n"
             f"군집화는 SVD 50차원에서 했고 실루엣은 그 공간에서 {sil} 다. UMAP 은 보기 위한 "
             f"투영이므로 덩어리로 보이는 것 자체를 분리도의 증거로 쓰면 안 된다.",
             fontsize=10, color=INK2, loc="left", pad=11)
fig.tight_layout()
fig.savefig(FIGS / "fig32_cc_map.png", dpi=170)
plt.close(fig)

json.dump({"fig": "fig32_cc_map.png", "k": K, "n_visits": int(len(s)),
           "n_unique_cc": int(len(ui)), "seed": SEED,
           "umap": {"n_neighbors": NN, "min_dist": MD, "metric": "cosine",
                    "unit": "unique CC string"},
           "silhouette_svd50": sil, "clean_threshold_pct": CLEAN,
           "named_clusters": {int(c): name[c] for c in named},
           "unnamed_clusters": [int(c) for c in other]},
          open(OUT / "40_cc_map_meta.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"[=] fig32_cc_map.png | 고유 표현 {len(ui):,} · 이름 붙은 군집 {len(named)}/{K} "
      f"· 실루엣(SVD50) {sil}")
