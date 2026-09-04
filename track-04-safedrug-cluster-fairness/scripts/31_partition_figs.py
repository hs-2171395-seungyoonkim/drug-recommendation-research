"""fig22~25 — D1/D2 분할 자체를 눈으로 보는 그림 네 장 (성능이 아니라 구성·분리도).

fig16~18 이 성능(Jaccard)이고 이 넷은 그 분할이 무엇으로 채워져 있고 진단 공간에서
얼마나 뭉치는가를 본다. 새로 계산하는 것은 SVD 2차원 좌표뿐이고 나머지는 기존 산출물에서 읽는다.

  fig22  D1 18개 chapter 구성 막대 (방문 수 순 · test≥30 통과 여부 색 · 대표 주진단)
  fig23  chapter x 그 chapter 의 SEQ_NUM=1 상위 5개 진단 히트맵 (fig18 은 동반진단 기준)
  fig24  진단 multi-hot SVD 2차원 산점도 — D1 색칠 / C1_id k=15 색칠 두 패널
  fig25  D1 순환기 4,211 방문이 D2 에서 8블록으로 갈라지는 구조 (생키)

라벨(분할 이름)은 한글, 진단명은 D_ICD_DIAGNOSES 의 SHORT_TITLE 원문 그대로 쓴다.
test≥30 미달 라벨은 회색 + 낮은 alpha 로 흐리게 처리한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
SEED, MIN_TEST, TOPN = 0, 30, 5

# ---------------------------------------------------------------- 입력
df = pd.read_pickle(OUT / "20_icd_features.pkl")
part = pd.read_csv(OUT / "icd_partition_assignments.csv", dtype={"icd9_seq1": str})
power = pd.read_csv(OUT / "table13_icd_power.csv")
cmp16 = pd.read_csv(OUT / "table16_icd_compare.csv")
dd = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype=str)
assert (df["HADM_ID"].to_numpy() == part["HADM_ID"].to_numpy()).all(), "행 순서 불일치"

NAME = {}
for c, s, l in zip(dd["ICD9_CODE"].fillna(""), dd["SHORT_TITLE"].fillna(""), dd["LONG_TITLE"].fillna("")):
    NAME[c.strip()] = (s.strip() or l.strip())


def dxname(code, width=None):
    """SEQ_NUM=1 코드 -> 이름. 사전에 없으면 코드만 돌려준다(예: 7775)."""
    nm = NAME.get(str(code).strip(), "")
    nm = f"{code} {nm}".strip() if nm else f"{code} (명칭 없음)"
    return nm if width is None or len(nm) <= width else nm[:width - 1] + "…"


D1P = power[power["분할"] == "D1"].sort_values("전체방문", ascending=False).reset_index(drop=True)
D2P = power[power["분할"] == "D2"].set_index("라벨")
PASS = dict(zip(D1P["라벨"], D1P["ge30"]))
SIL = dict(zip(cmp16["분할"], cmp16["실루엣(공통공간)"]))

# C1_id k=15 (23_icd_evaluate.py 와 같은 방식으로 붙인다)
a = pd.read_pickle(OUT / "cluster_assignments.pkl")
sub = a[(a["partition"] == "C1_id") & (a["k"] == 15)]
C1 = df["HADM_ID"].map(dict(zip(sub["HADM_ID"], sub["cluster_id"]))).to_numpy(int)
t2 = pd.read_csv(OUT / "table2_cluster_performance.csv")
t2c = t2[(t2["partition"] == "C1_id") & (t2["k"] == 15)]
C1TOP = {int(r["cluster"]): str(r["top3_dx"]).split("|")[0].strip() for _, r in t2c.iterrows()}

# ---------------------------------------------------------------- 스타일 (24_icd_report.py 와 동일)
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
META = {}

# ---------------------------------------------------------------- 공통 — chapter 별 주진단 상위
SEQ1 = part["icd9_seq1"].astype(str).str.strip().to_numpy()


def top_seq1(mask, n=TOPN):
    vc = pd.Series(SEQ1[mask]).value_counts()
    tot = int(mask.sum())
    return [(c, int(v), 100.0 * v / tot) for c, v in vc.head(n).items()]


D1LAB = part["d1_label"].to_numpy()
TOPDX = {ch: top_seq1(D1LAB == ch) for ch in D1P["라벨"]}

# ================================================================ fig22 — chapter 구성 막대
fig, ax = plt.subplots(figsize=(11.6, 0.46 * len(D1P) + 2.2))
y = np.arange(len(D1P))[::-1]
for yy, (_, r) in zip(y, D1P.iterrows()):
    ok = bool(r["ge30"])
    base, dark = (S1, "#1a4f92") if ok else (MUTED, FAINT)
    ax.barh(yy, r["전체방문"], color=base, height=0.66, zorder=3, alpha=1.0 if ok else 0.55)
    ax.barh(yy, r["test방문"], color=dark, height=0.66, zorder=4, alpha=1.0 if ok else 0.7)
    t = TOPDX[r["라벨"]][0]
    ax.text(r["전체방문"] + 60, yy, f"test {r['test방문']:,}   대표 {dxname(t[0], 30)}  {t[2]:.0f}%",
            va="center", ha="left", fontsize=8, color=INK2 if ok else FAINT, zorder=5)
    ax.text(-60, yy, f"{r['라벨']}  {r['전체방문']:,}", va="center", ha="right",
            fontsize=8.5, color=INK if ok else FAINT, zorder=5)
ax.set_yticks([])
ax.set_xlim(-1250, 5400)
ax.set_xticks(range(0, 5001, 1000))                    # 왼쪽 음수 구간은 라벨 자리라 눈금을 두지 않는다
ax.set_ylim(-0.8, len(D1P) - 0.2)
ax.set_xlabel("방문 수 (연한 막대 = 전체 14,541 중, 진한 막대 = test 2,880 중)")
ax.set_title(f"fig22 · D1 (주 진단 ICD chapter) 18개 라벨의 구성 — 방문 수 순\n"
             f"파랑 = test ≥{MIN_TEST} 통과 {int(D1P['ge30'].sum())}개 · 회색 = 미달 {int((~D1P['ge30']).sum())}개"
             f"(전체 방문의 {100 * D1P.loc[~D1P['ge30'], '전체방문'].sum() / D1P['전체방문'].sum():.1f}%, "
             f"test 방문의 {100 * D1P.loc[~D1P['ge30'], 'test방문'].sum() / D1P['test방문'].sum():.1f}%)",
             fontsize=10, color=INK2, loc="left")
ax.grid(axis="x", lw=0.6, zorder=0)
fig.tight_layout()
fig.savefig(FIGS / "fig22_D1_chapter_composition.png", dpi=170)
plt.close(fig)
META["fig22"] = {"n_labels": len(D1P), "n_pass": int(D1P["ge30"].sum()),
                 "dropped_visit_pct": round(100 * float(D1P.loc[~D1P["ge30"], "전체방문"].sum()) / 14541, 2),
                 "대표진단": {r["라벨"]: dxname(TOPDX[r["라벨"]][0][0]) for _, r in D1P.iterrows()}}
print("[.] fig22", flush=True)

# ================================================================ fig23 — 주 진단 히트맵
order = list(D1P["라벨"])
M = np.full((len(order), TOPN), np.nan)
TXT = [["" for _ in range(TOPN)] for _ in order]
for i, ch in enumerate(order):
    for j, (c, v, p) in enumerate(TOPDX[ch]):
        M[i, j] = p
        TXT[i][j] = f"{dxname(c, 30)}\n{p:.1f}%  (n={v:,})"

fig, ax = plt.subplots(figsize=(13.2, 0.74 * len(order) + 2.4))
vmax = float(np.nanpercentile(M, 92))
cmap = plt.get_cmap("YlGnBu").copy()
cmap.set_bad(SURF)                                     # 상위 5개가 안 되는 chapter(임신·출산)의 빈 칸
im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
ax.set_xticks(range(TOPN))
ax.set_xticklabels([f"{j + 1}위" for j in range(TOPN)], fontsize=9)
ax.set_yticks(range(len(order)))
lab = []
for ch in order:
    r = D1P[D1P["라벨"] == ch].iloc[0]
    cum = sum(p for _, _, p in TOPDX[ch])
    lab.append(f"{ch}\nn={r['전체방문']:,} · 상위5 합 {cum:.0f}%")
ax.set_yticklabels(lab, fontsize=8.2)
for tick, ch in zip(ax.get_yticklabels(), order):
    tick.set_color(INK if PASS[ch] else FAINT)
for i, ch in enumerate(order):
    if not PASS[ch]:                                   # 미달 라벨은 칸 자체를 흐리게 덮는다
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), TOPN, 1, color=SURF, alpha=0.62, lw=0, zorder=3))
    for j in range(TOPN):
        if not TXT[i][j]:
            continue
        ax.text(j, i, TXT[i][j], ha="center", va="center", fontsize=6.4, zorder=4,
                color=("white" if M[i, j] > 0.62 * vmax else INK) if PASS[ch] else INK2)
ax.set_title("fig23 · D1 chapter x 그 chapter 안의 주 진단(SEQ_NUM=1) 상위 5개 · 값 = chapter 내 방문 비율%\n"
             "fig18 은 동반진단 보유율이고 이것은 '무슨 병으로 입원했는가' 기준이다 · "
             f"흐린 행 = test <{MIN_TEST} 미달 라벨",
             fontsize=10, color=INK2, loc="left")
fig.colorbar(im, ax=ax, shrink=0.55, label="chapter 내 비율 %", extend="max")
fig.tight_layout()
fig.savefig(FIGS / "fig23_D1_seq1_heatmap.png", dpi=170)
plt.close(fig)
META["fig23"] = {ch: {"상위5 합%": round(sum(p for _, _, p in TOPDX[ch]), 1),
                      "상위5": [f"{dxname(c)} {p:.1f}%" for c, _, p in TOPDX[ch]]} for ch in order}
print("[.] fig23", flush=True)

# ================================================================ fig24 — SVD 2차원 분리도
X = sparse.load_npz(OUT / "20_icd_multihot.npz").tocsr()
svd2 = TruncatedSVD(n_components=2, random_state=SEED)
P2 = svd2.fit_transform(normalize(X, norm="l2", axis=1))
EVR = svd2.explained_variance_ratio_
del X

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(P2))                       # 그리는 순서 편향 제거

pass_ch = [c for c in order if PASS[c]]
pal = plt.get_cmap("tab20").colors
CH_COL = {c: pal[i % 20] for i, c in enumerate(pass_ch)}
GREY = (0.79, 0.78, 0.76)
col_d1 = np.array([CH_COL.get(c, GREY) for c in D1LAB])
is_pass = np.array([PASS[c] for c in D1LAB])


def centroids(labels, keys):
    return np.array([[P2[labels == k, 0].mean(), P2[labels == k, 1].mean()] for k in keys])


def mean_pair_dist(C):
    dm = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    return float(dm[np.triu_indices(len(C), 1)].mean())


def box_of(C, pad=0.012):
    return (C[:, 0].min() - pad, C[:, 0].max() + pad, C[:, 1].min() - pad, C[:, 1].max() + pad)


def label_column(ax, xy, texts, colors, x_text, gap):
    """라벨을 한 열에 세로로 밀어내 겹치지 않게 놓고 중심까지 지시선을 긋는다."""
    o = list(np.argsort([-p[1] for p in xy]))
    ys = [xy[i][1] for i in o]
    for a in range(1, len(ys)):
        ys[a] = min(ys[a], ys[a - 1] - gap)
    shift = float(np.mean([xy[i][1] for i in o])) - float(np.mean(ys))
    for rank, i in enumerate(o):
        ax.annotate(texts[i], xy=tuple(xy[i]), xytext=(x_text, ys[rank] + shift), fontsize=6.6,
                    va="center", ha="left", color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=colors[i], lw=0.9, alpha=0.93),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=colors[i], shrinkA=0, shrinkB=1))


CEN_D1 = centroids(D1LAB, pass_ch)
CEN_C1 = centroids(C1, list(range(15)))
BOX = {"D1": box_of(CEN_D1), "C1": box_of(CEN_C1)}
MPD = {"D1": mean_pair_dist(CEN_D1), "C1": mean_pair_dist(CEN_C1)}

fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.6), sharex=True, sharey=True)

ax = axes[0]
ax.scatter(P2[perm][~is_pass[perm], 0], P2[perm][~is_pass[perm], 1], s=2.2,
           c=col_d1[perm][~is_pass[perm]], alpha=0.30, lw=0, zorder=2)
ax.scatter(P2[perm][is_pass[perm], 0], P2[perm][is_pass[perm], 1], s=2.2,
           c=col_d1[perm][is_pass[perm]], alpha=0.55, lw=0, zorder=3)
ax.scatter(CEN_D1[:, 0], CEN_D1[:, 1], s=95, marker="X", c=[CH_COL[c] for c in pass_ch],
           edgecolors="white", linewidths=1.1, zorder=6)
ax.set_title(f"D1 — 주 진단 ICD chapter (통과 11개) · 실루엣 {SIL['D1 (ICD chapter)']:+.4f} · "
             f"중심 간 평균 거리 {MPD['D1']:.3f}", fontsize=10, color=INK2, loc="left")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=CH_COL[c], label=c) for c in pass_ch] +
                  [Line2D([], [], marker="o", ls="", ms=5, color=GREY, label=f"test <{MIN_TEST} 미달 7개"),
                   Line2D([], [], marker="X", ls="", ms=7, color=INK2, label="X = 라벨 중심")],
          frameon=False, fontsize=7.4, loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8)

ax = axes[1]
pal15 = plt.get_cmap("tab20").colors
ax.scatter(P2[perm, 0], P2[perm, 1], s=2.2, c=[pal15[i % 20] for i in C1[perm]], alpha=0.55, lw=0, zorder=3)
ax.scatter(CEN_C1[:, 0], CEN_C1[:, 1], s=95, marker="X", c=[pal15[i % 20] for i in range(15)],
           edgecolors="white", linewidths=1.1, zorder=6)
ax.set_title(f"C1_id k=15 — 동반질환 프로파일 군집 · 실루엣 {SIL['C1_id k=15 (동반질환 군집)']:+.4f} · "
             f"중심 간 평균 거리 {MPD['C1']:.3f}", fontsize=10, color=INK2, loc="left")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=pal15[i % 20],
                          label=f"C{i} {C1TOP.get(i, '')[:24]}") for i in range(15)],
          frameon=False, fontsize=7.0, loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8)

# 중심만 확대한 삽입 패널 — 중심이 겹쳐 본 그림에 이름을 못 다는 것 자체가 왼쪽의 결과다
for ax, C, keys, cols, tag in [(axes[0], CEN_D1, pass_ch, [CH_COL[c] for c in pass_ch], "D1"),
                               (axes[1], CEN_C1, [f"C{i}" for i in range(15)],
                                [pal15[i % 20] for i in range(15)], "C1")]:
    x0, x1, y0, y1 = BOX[tag]
    axins = ax.inset_axes([0.50, 0.02, 0.48, 0.40], facecolor=SURF)
    axins.scatter(P2[perm, 0], P2[perm, 1], s=1.4,
                  c=(col_d1[perm] if tag == "D1" else [pal15[i % 20] for i in C1[perm]]),
                  alpha=0.16, lw=0, zorder=1)
    axins.scatter(C[:, 0], C[:, 1], s=60, marker="X", c=cols, edgecolors="white", linewidths=1.0, zorder=6)
    label_column(axins, C, keys, cols, x_text=x1 - (x1 - x0) * 0.30, gap=(y1 - y0) / (len(C) + 1.4))
    axins.set_xlim(x0, x1 + (x1 - x0) * 0.42)
    axins.set_ylim(y0 - (y1 - y0) * 0.17, y1 + (y1 - y0) * 0.17)
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_visible(True)
        s.set_color(INK2)
    axins.text(0.02, 0.975, f"중심 확대 · 가로 {x1 - x0:.3f} x 세로 {y1 - y0:.3f}", transform=axins.transAxes,
               va="top", ha="left", fontsize=7.4, color=INK2, zorder=9)
    ax.indicate_inset_zoom(axins, edgecolor=INK2, lw=0.9, alpha=0.8)

for ax in axes:
    ax.set_xlabel(f"SVD 1축 (설명분산 {100 * EVR[0]:.1f}%)")
    ax.grid(lw=0.5, zorder=0)
axes[0].set_ylabel(f"SVD 2축 (설명분산 {100 * EVR[1]:.1f}%)")
fig.suptitle(f"fig24 · 같은 좌표(진단 multi-hot 4,514 → L2 → TruncatedSVD)에 14,541 방문을 찍고 분할만 바꿔 색칠 · "
             f"1~2축 설명분산 합 {100 * EVR.sum():.1f}%\n"
             f"chapter 는 이 공간에서 뭉치지 않는다(실루엣 {SIL['D1 (ICD chapter)']:+.4f}) — 색이 서로 관통하고 11개 중심이 "
             f"{BOX['D1'][1] - BOX['D1'][0]:.3f} x {BOX['D1'][3] - BOX['D1'][2]:.3f} 상자 안에 모두 들어간다"
             f"(C1_id 는 {BOX['C1'][1] - BOX['C1'][0]:.3f} x {BOX['C1'][3] - BOX['C1'][2]:.3f}).\n"
             f"실루엣은 보고서와 같은 100차원 공간 값이고 그림은 그 중 2축만 보여준다.",
             fontsize=10, color=INK2, x=0.012, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(FIGS / "fig24_svd_separation.png", dpi=170)
plt.close(fig)
META["fig24"] = {"evr1": round(float(EVR[0]), 4), "evr2": round(float(EVR[1]), 4),
                 "evr_sum": round(float(EVR.sum()), 4),
                 "silhouette_D1": SIL["D1 (ICD chapter)"], "silhouette_C1_id": SIL["C1_id k=15 (동반질환 군집)"],
                 "중심간 평균거리(2D)": {"D1": round(MPD["D1"], 4), "C1_id": round(MPD["C1"], 4)},
                 "중심 상자": {k: [round(float(v), 4) for v in b] for k, b in BOX.items()}}
print("[.] fig24", flush=True)

# ================================================================ fig25 — 순환기 분해 생키
cv = D1LAB == "순환기"
blocks = (pd.Series(part.loc[cv, "d2_label"]).value_counts().rename_axis("라벨").reset_index(name="방문"))
blocks["test"] = [int(D2P.loc[b, "test방문"]) for b in blocks["라벨"]]
blocks["ok"] = [bool(D2P.loc[b, "ge30"]) for b in blocks["라벨"]]
blocks["대표"] = [top_seq1((part["d2_label"] == b).to_numpy())[0] for b in blocks["라벨"]]
TOTAL = int(cv.sum())
assert blocks["방문"].sum() == TOTAL

GAP = TOTAL * 0.018
H = TOTAL + GAP * (len(blocks) - 1)
pal8 = ["#2a78d6", "#2e7d5b", "#c0392b", "#7b5ea7", "#d98324", "#3f8fa0", "#b0567f", "#6b7a3a"]


def ribbon(ax, x0, x1, y0, y1, h0, h1, color, alpha):
    t = np.linspace(0, 1, 120)
    s = (1 - np.cos(np.pi * t)) / 2
    x = x0 + (x1 - x0) * t
    ax.fill_between(x, y0 + (y1 - y0) * s, y0 + h0 + (y1 + h1 - y0 - h0) * s,
                    color=color, alpha=alpha, lw=0, zorder=2)


fig, ax = plt.subplots(figsize=(12.4, 6.6))
yl, yr, tops = 0.0, 0.0, []
for i, r in blocks.iterrows():
    col = pal8[i % len(pal8)] if r["ok"] else MUTED
    al = 0.85 if r["ok"] else 0.45
    ax.add_patch(plt.Rectangle((0.54, yr), 0.07, r["방문"], color=col, alpha=al, lw=0, zorder=4))
    ribbon(ax, 0.14, 0.54, yl, yr, r["방문"], r["방문"], col, al * 0.42)
    tops.append(yr)
    yl += r["방문"]
    yr += r["방문"] + GAP

mid = [t + v / 2 for t, v in zip(tops, blocks["방문"])]   # 작은 블록 셋은 라벨이 겹치므로 밀어낸다
ly, MINGAP = list(mid), H * 0.075
for a in range(1, len(ly)):
    ly[a] = max(ly[a], ly[a - 1] + MINGAP)
over = ly[-1] - (H - MINGAP * 0.5)
if over > 0:
    ly = [y - over for y in ly]
for i, r in blocks.iterrows():
    col = pal8[i % len(pal8)] if r["ok"] else MUTED
    c, v, p = r["대표"]
    if abs(ly[i] - mid[i]) > 1:
        ax.plot([0.615, 0.645], [mid[i], ly[i]], lw=0.8, color=col, alpha=0.8, zorder=3)
    ax.text(0.655, ly[i],
            f"{r['라벨']}   {r['방문']:,}방문 (test {r['test']})"
            f"{'' if r['ok'] else f'  ← test <{MIN_TEST} 미달'}\n"
            f"대표 주진단 {dxname(c, 34)}  {p:.0f}% (n={v})",
            va="center", ha="left", fontsize=8.2, color=INK if r["ok"] else FAINT, zorder=5)
ax.add_patch(plt.Rectangle((0.02, 0.0), 0.12, TOTAL, color="#1a4f92", alpha=0.9, lw=0, zorder=4))
ax.text(0.08, TOTAL / 2, f"D1 순환기\n{TOTAL:,}방문\n(test {int(D1P[D1P['라벨'] == '순환기']['test방문'].iloc[0]):,})",
        va="center", ha="center", fontsize=9, color="white", zorder=6)
ax.set_xlim(0, 1.42)
ax.set_ylim(H * 1.01, -H * 0.01)
ax.axis("off")
n_ok = int(blocks["ok"].sum())
ax.set_title(f"fig25 · D1 순환기 {TOTAL:,}방문(전체의 {100 * TOTAL / 14541:.1f}%)이 D2 에서 8블록으로 갈라지는 구조 · "
             f"블록 높이 = 방문 수\ntest ≥{MIN_TEST} 통과 {n_ok}개 / 미달 {len(blocks) - n_ok}개"
             f"(미달 블록 합 {int(blocks.loc[~blocks['ok'], '방문'].sum()):,}방문)",
             fontsize=10, color=INK2, loc="left")
fig.tight_layout()
fig.savefig(FIGS / "fig25_circulatory_split.png", dpi=170)
plt.close(fig)
META["fig25"] = {"순환기 방문": TOTAL, "블록": [
    {"라벨": r["라벨"], "방문": int(r["방문"]), "test": int(r["test"]), "ge30": bool(r["ok"]),
     "대표 주진단": f"{dxname(r['대표'][0])} {r['대표'][2]:.1f}%"} for _, r in blocks.iterrows()]}
print("[.] fig25", flush=True)

json.dump(META, open(OUT / "31_partition_figs_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("[.] 그림 4장 + 31_partition_figs_meta.json 저장", flush=True)
