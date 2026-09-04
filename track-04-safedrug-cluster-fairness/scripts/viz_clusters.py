"""§트랙 비교 그림 — 공간 2개 × 라벨 2개.

이전 판은 UMAP 을 **한 번만** 계산해서(가중치 long 임베딩) Track1/Track2 두 그림에
같은 좌표를 쓰고 색만 바꿔 그렸다. 그래놓고 리포트에는
  - Track1: "기존의 모든 진단을 평균 낸 Baseline 임베딩 공간입니다"  (아니다)
  - Track2: "새로 추출한 가중치 임베딩으로 UMAP을 다시 그렸기 때문입니다"  (안 그렸다)
라고 적혀 있었다. 두 문장 다 사실이 아니고, 게다가 Track2 의 색(primary_cluster)은
**concise** 로 군집한 결과인데 좌표는 **long** 에서 뽑은 것이었다.

그래서 "가중치를 주니 뭉게구름처럼 갈라졌다" 는 결론이 성립하지 않는다. 모양은 좌표계가
정하고 색은 라벨이 정하는데, 둘을 섞어 놓으면 어느 쪽 공로인지 알 수 없다.

이 판은 2×2 로 그린다. 행 = 좌표계(기본 / 가중치), 열 = 라벨(CCS 주진단 / 가중치 k-means).
같은 라벨을 두 좌표계에 얹어 보면 "구름"이 라벨이 아니라 좌표계의 성질임이 바로 보인다.

산출: out/figs/fig42_space_vs_label.png (+ 바탕화면 사본)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
DESK = Path.home() / "Desktop"
NN, MD, SEED = 50, 0.5, 42

df = pd.read_pickle(OUT / "40_dxtext.pkl")
HADM = df["HADM_ID"].to_numpy()

SPACES = [("기본 임베딩 (진단목록 통째, long)", "emb_dxtext_long.npz"),
          ("가중치 임베딩 (V_main×5 + V_sub, concise)", "emb_dxtext_weight_concise.npz")]

U = {}
for title, f in SPACES:
    z = np.load(OUT / f)
    assert (z["HADM_ID"] == HADM).all(), f"{f} 방문 순서 불일치"
    print(f"[.] UMAP: {title}", flush=True)
    U[title] = umap.UMAP(n_components=2, n_neighbors=NN, min_dist=MD,
                         metric="cosine", random_state=SEED).fit_transform(z["E"])

# ── 라벨 두 벌 ──────────────────────────────────────────────────────────────
ccs = pd.read_csv(OUT / "ccs_assignments.csv", usecols=["HADM_ID", "group"])
ccs = ccs.set_index("HADM_ID").reindex(HADM).reset_index()
wt = pd.read_csv(OUT / "dxtext_weight_cluster_assignments.csv",
                 usecols=["HADM_ID", "primary_cluster"])
wt = wt.set_index("HADM_ID").reindex(HADM).reset_index()

ccs_lab = ccs["group"].to_numpy()
wt_lab = wt["primary_cluster"].to_numpy()

# CCS: 큰 범주만 이름표를 단다(작은 것 + 기타는 회색)
ccs_big = [c for c, n in pd.Series(ccs_lab).value_counts().items()
           if n >= 200 and not c.startswith(("기타", "미지정"))]
pal_ccs = dict(zip(ccs_big, sns.color_palette("husl", len(ccs_big))))
pal_wt = dict(enumerate(sns.color_palette("husl", int(wt_lab.max()) + 1)))
GREY = (0.62, 0.62, 0.66)

LABELS = [
    ("CCS 주진단 범주 (의학 표준)", ccs_lab, pal_ccs, True),
    (f"가중치 k-means k={int(wt_lab.max()) + 1} (알고리즘)", wt_lab, pal_wt, False),
]

fig, axes = plt.subplots(2, 2, figsize=(20, 17), facecolor="#1a1a2e")
for r, (sp_title, _) in enumerate(SPACES):
    xy = U[sp_title]
    for c, (lb_title, lab, pal, is_ccs) in enumerate(LABELS):
        ax = axes[r, c]
        ax.set_facecolor("#1a1a2e")
        keys = ccs_big if is_ccs else sorted(pal)
        rest = ~np.isin(lab, keys)
        if rest.any():
            ax.scatter(xy[rest, 0], xy[rest, 1], c=[GREY], s=5, alpha=0.35,
                       linewidths=0, rasterized=True)
        for kkey in keys:
            m = lab == kkey
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], c=[pal[kkey]], s=6, alpha=0.65,
                       linewidths=0, rasterized=True)
            cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
            ax.text(cx, cy, str(kkey), fontsize=7.5, fontweight="bold",
                    color="white", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc=pal[kkey],
                              ec="white", alpha=0.85, linewidth=0.9))
        ax.set_title(f"{lb_title}\n좌표계: {sp_title}", fontsize=13,
                     color="white", pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#444")

fig.suptitle("모양은 좌표계가, 색은 라벨이 정한다 — 같은 라벨을 두 임베딩에 얹어 본 것",
             fontsize=19, color="white", y=0.985)
fig.text(0.5, 0.012,
         "아래 행이 '뭉게구름'처럼 갈라지는 것은 라벨 때문이 아니라 가중치 임베딩의 "
         "96.2%가 주진단 한 문장(고유 1,221종)으로만 결정되기 때문이다.",
         ha="center", fontsize=12, color="#ffd479")
plt.tight_layout(rect=(0, 0.028, 1, 0.972))
FIGS.mkdir(parents=True, exist_ok=True)
for p in (FIGS / "fig42_space_vs_label.png", DESK / "fig42_space_vs_label.png"):
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("[+] out/figs/fig42_space_vs_label.png (+ 바탕화면)")
