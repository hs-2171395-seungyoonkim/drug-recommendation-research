"""§두 개의 '최적'을 나란히 뽑는다 — 영역분할 품질 최적 vs 성능격차 최적.

목적이 둘로 갈렸다.
  (가) 영역분할이 잘 된 군집  = 기하학적으로 잘 뭉치고 잘 떨어진 분할
  (나) 성능 격차가 드러나는 군집 = 53번에서 뽑은, 추천 성능이 군집별로 갈리는 분할
두 기준이 같은 구성을 고르지 않는다. 그 상충 자체를 보여주는 것이 이 스크립트의 요점이다.

(가)의 채점은 지표 하나로 하지 않는다. 실루엣·CH 는 k 가 커지면 기계적으로 떨어지고
DB 는 그렇지 않아 서로 어긋나기 때문이다. 42개 구성에 대해 네 지표의 순위를 매겨
평균 순위를 쓴다 — 실루엣↑ · CH↑ · DB↓ · 시드간 ARI↑ (재현되지 않는 분할은 잘 나뉜
것이 아니다).

k 에 제약을 걸지 않으면 어느 지표든 k=2 를 고른다. 그래서 세 층위로 보고한다:
  무제약 / k>=10 / 최소군집>=100(규칙B와 동일).

산출: out/table77_partition_quality.csv, out/54_best_meta.json,
      out/figs/fig41_two_optima.png, out/figs/fig42_best_vs_gap_umap.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
VARIANTS = ["short", "long", "concise"]

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
RED, BLUE, PURPLE, GREY = "#c0392b", "#2471a3", "#7d3c98", "#8a8985"
COL = {"short": BLUE, "long": RED, "concise": PURPLE}
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
})

q = pd.read_csv(OUT / "table62_dxtext_ksweep.csv", encoding="utf-8-sig")
gap = pd.read_csv(OUT / "table75_gap_ksweep.csv", encoding="utf-8-sig")
labs = np.load(OUT / "42_dxtext_labels.npz")
proj = np.load(OUT / "51_dxtext_proj_allvar.npz")

q["구성"] = q["변형"] + " k=" + q["k"].astype(str)
g = gap[gap["그룹"] == "dxtext"][["구성", "상수_격차", "상수_가중SD", "copyprev_격차",
                                  "약물수상관_상수", "test최소군집", "실질k"]]
t = q.merge(g, on="구성", how="left")

# ---------------- (가) 분할 품질 평균 순위 ----------------
t["r_sil"] = t["실루엣"].rank(ascending=False)
t["r_ch"] = t["CH"].rank(ascending=False)
t["r_db"] = t["DB"].rank(ascending=True)          # 낮을수록 좋다
t["r_ari"] = t["ARI_시드간"].rank(ascending=False)
t["평균순위"] = t[["r_sil", "r_ch", "r_db", "r_ari"]].mean(axis=1).round(2)
t["격차순위"] = t["상수_격차"].rank(ascending=False)

TIERS = {
    "무제약": t,
    "k>=10": t[t["k"] >= 10],
    "최소군집>=100": t[t["최소군집"] >= 100],
}
picks = {}
for nm, sub in TIERS.items():
    best_q = sub.sort_values("평균순위").iloc[0]
    best_g = sub.sort_values("상수_격차", ascending=False).iloc[0]
    picks[nm] = {"분할최적": best_q["구성"], "분할최적_평균순위": float(best_q["평균순위"]),
                 "분할최적_실루엣": float(best_q["실루엣"]),
                 "분할최적_격차": (float(best_q["상수_격차"])
                               if pd.notna(best_q["상수_격차"]) else None),
                 "격차최적": best_g["구성"], "격차최적_격차": float(best_g["상수_격차"]),
                 "격차최적_실루엣": float(best_g["실루엣"]),
                 "격차최적_약물수상관": float(best_g["약물수상관_상수"])}
    print(f"[=] {nm:14s} 분할최적 {best_q['구성']:14s}(평균순위 {best_q['평균순위']:.2f}, "
          f"실루엣 {best_q['실루엣']:.4f})  |  격차최적 {best_g['구성']:14s}"
          f"(격차 {best_g['상수_격차']:.4f})", flush=True)

cols = ["구성", "변형", "k", "실루엣", "CH", "DB", "ARI_시드간", "최소군집",
        "평균순위", "상수_격차", "상수_가중SD", "약물수상관_상수", "test최소군집", "격차순위"]
t.sort_values("평균순위")[cols].to_csv(OUT / "table77_partition_quality.csv",
                                       index=False, encoding="utf-8-sig")

# 두 기준의 상관 — 상충인가 무관인가
rho = t[["평균순위", "격차순위"]].corr(method="spearman").iloc[0, 1]
rho10 = (t[t["k"] >= 10][["평균순위", "격차순위"]]
         .corr(method="spearman").iloc[0, 1])
print(f"[=] 분할품질 순위 vs 격차 순위 Spearman ρ = {rho:.3f} (전체), {rho10:.3f} (k>=10)",
      flush=True)

# ---------------- fig41 : 두 목표의 상충 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2))

ax = axes[0]
for v in VARIANTS:
    s = t[t["변형"] == v].sort_values("k")
    for col, ls, mk in [("실루엣", "-", "o"), ("DB", "--", "s")]:
        z = (s[col] - s[col].min()) / (s[col].max() - s[col].min())
        if col == "DB":
            z = 1 - z                                   # 높을수록 좋게 뒤집는다
        ax.plot(s["k"], z, ls, marker=mk, ms=3.2, lw=1.3, color=COL[v],
                alpha=1.0 if col == "실루엣" else 0.45)
ax.set_xlabel("k"); ax.set_ylabel("정규화 점수 (↑ 좋음)")
ax.set_title("분할 품질 — 실선 실루엣, 파선 DB(반전)", fontsize=10.5)
ax.text(0.97, 0.62, "실루엣: k=2 최대\nDB: 큰 k 선호\n→ 지표끼리 어긋난다",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)

ax = axes[1]
for v in VARIANTS:
    s = t[t["변형"] == v]
    ax.scatter(s["실루엣"], s["상수_격차"], s=26, color=COL[v], label=v, alpha=0.85,
               linewidths=0)
bq = t.sort_values("평균순위").iloc[0]
bg = t.sort_values("상수_격차", ascending=False).iloc[0]
for r, txt, dy in [(bq, f"분할최적\n{bq['구성']}", 8), (bg, f"격차최적\n{bg['구성']}", -22)]:
    ax.annotate(txt, (r["실루엣"], r["상수_격차"]),
                textcoords="offset points", xytext=(6, dy), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
ax.set_xlabel("실루엣 (분할 품질)"); ax.set_ylabel("군집 간 Jaccard 범위 (성능 격차)")
ax.set_title(f"두 기준은 상충한다  (Spearman ρ = {rho:.2f})", fontsize=10.5)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
for v in VARIANTS:
    s = t[t["변형"] == v]
    ax.scatter(s["ARI_시드간"], s["상수_격차"], s=26, color=COL[v], label=v, alpha=0.85,
               linewidths=0)
ax.axvline(0.7, color=GREY, ls=":", lw=1.1)
ax.text(0.7, ax.get_ylim()[1], " 재현 하한 0.7", fontsize=8, color=GREY, va="top")
ax.set_xlabel("시드 간 ARI (재현성)"); ax.set_ylabel("군집 간 Jaccard 범위")
ax.set_title("격차가 큰 구성일수록 재현이 안 된다", fontsize=10.5)
ax.legend(fontsize=8, frameon=False)

for ax in axes:
    for sp in ax.spines.values():
        sp.set_color("#d9d8d4")
fig.suptitle("영역분할 최적 vs 성능격차 최적 — 같은 42개 구성", fontsize=11.5)
fig.tight_layout()
fig.savefig(FIGS / "fig41_two_optima.png", dpi=185, bbox_inches="tight")
plt.close(fig)
print("[+] fig41_two_optima.png", flush=True)

# ---------------- fig42 : 두 선택을 같은 좌표에 ----------------
PAL = ([plt.get_cmap("tab10")(i) for i in range(10)]
       + [plt.get_cmap("tab20b")(i) for i in range(20)]
       + [plt.get_cmap("tab20")(i) for i in range(1, 20, 2)]
       + [plt.get_cmap("tab20c")(i) for i in range(0, 20, 4)])

SHOW = [("분할 품질 최적 (k>=10)", picks["k>=10"]["분할최적"]),
        ("분할 품질 최적 (최소군집>=100)", picks["최소군집>=100"]["분할최적"]),
        ("성능 격차 최적", picks["무제약"]["격차최적"])]
fig, axes = plt.subplots(1, len(SHOW), figsize=(5.1 * len(SHOW), 5.3))
for ax, (ttl, name) in zip(axes, SHOW):
    v, k = name.split(" k=")
    k = int(k)
    lab = labs[f"{v}_k{k}"]
    co = proj[f"{v}_umap2"]
    ax.scatter(co[:, 0], co[:, 1], s=1.9, alpha=0.62, linewidths=0,
               c=[PAL[i % len(PAL)] for i in lab])
    r = t[t["구성"] == name].iloc[0]
    ax.set_title(f"{ttl}\n{name}  ·  실루엣 {r['실루엣']:.4f}  ·  격차 {r['상수_격차']:.4f}",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d9d8d4")
fig.suptitle("같은 UMAP 좌표계, 다른 기준으로 고른 분할", fontsize=11.5)
fig.tight_layout()
fig.savefig(FIGS / "fig42_best_vs_gap_umap.png", dpi=185, bbox_inches="tight")
plt.close(fig)
print("[+] fig42_best_vs_gap_umap.png", flush=True)

(OUT / "54_best_meta.json").write_text(json.dumps({
    "quality_score": "실루엣↑·CH↑·DB↓·시드간ARI↑ 네 지표의 평균 순위 (42구성)",
    "tiers": picks,
    "spearman_quality_vs_gap": {"전체": round(float(rho), 3), "k>=10": round(float(rho10), 3)},
    "note": "k 제약이 없으면 어느 지표든 k=2 를 고른다. 세 층위를 모두 보고한다.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] table77, 54_best_meta.json", flush=True)
