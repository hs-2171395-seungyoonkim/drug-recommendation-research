"""fig31 — 진단코드로 나눈 분할이 무엇으로 정의되는가, 한 장.

fig20 은 패널이 4개다. 그중 셋(COMORB-elix, CHR-persist, CHR-idf)은 동반질환을 지워 본
사후 실험이라 "진단으로 나누면 만성질환 축이 된다"는 이야기와는 무관하다. 교수님께 보낼
첨부에서 그 셋이 질문만 만든다. 그래서 사전등록 주 분할인 C1_id k=15 패널만 떼어 낸다.

계산은 30_chronic_report.py 의 fig20 첫 패널과 같다 — table34 의 클러스터별 상위 lift
진단 2개를 모아 원 진단 리스트(제거 전) 기준 유병률을 센다. 색·크기만 다르다.

산출: out/figs/fig31_dx_axis.png
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

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
RED, BLUE, GREY = "#c0392b", "#2471a3", "#8a8985"
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
})

PAN = "C1_id k=15 (기준)"
t32 = pd.read_csv(OUT / "table32_chronic_axis.csv")
t34 = pd.read_csv(OUT / "table34_chronic_toplift.csv")
asg = pd.read_csv(OUT / "chronic_partition_assignments.csv")
feat = pd.read_pickle(OUT / "20_icd_features.pkl")
FULL = [set(str(c).strip() for c in l) for l in feat["icd_l"]]

tl = t34[t34["분할"] == PAN]
top2 = tl.sort_values(["cluster", "lift_diff"], ascending=[True, False]).groupby("cluster").head(2)
codes = list(dict.fromkeys(top2["code"].astype(str)))
labels = asg["c1_id_label"].to_numpy()
cls = sorted(c for c in set(labels) if c >= 0)
M = np.zeros((len(cls), len(codes)))
for i, g in enumerate(cls):
    idx = np.where(labels == g)[0]
    for j, code in enumerate(codes):
        M[i, j] = sum(code in FULL[x] for x in idx) / len(idx)

nm = dict(zip(tl["code"].astype(str), tl["name"]))
# table34 에 "(사전에 없음)" 으로 남은 코드가 있다. MIMIC 의 D_ICD_DIAGNOSES 는 ICD-9-CM
# 최종판이라 개정 때 하위분류된 상위코드가 빠져 있는데, DIAGNOSES_ICD 의 기록은 개정 전
# 판으로 코딩돼 있다. 전체 진단 기록의 2.50%(143종 16,291건)가 여기 해당하고 그중 142종은
# 사전에 더 긴 하위코드가 존재한다. 코드 자체는 그대로 쓰였으니 군집화에는 영향이 없고
# 이름만 비는 것이다. 이 그림에 나오는 것은 0414 하나뿐이라 손으로 적는다.
RETIRED = {"0414": "E coli infection*"}   # 하위 04141~04149 로 분리된 상위코드
nm.update({c: RETIRED[c] for c in RETIRED if c in codes})
cc = dict(zip(tl["code"].astype(str), tl["cci"]))
acute = float(t32.set_index("분할").loc[PAN, "특징 진단 중 CCI 급성%"])
n_chr = sum(cc.get(c) == "만성" for c in codes)

fig, ax = plt.subplots(figsize=(12.2, 6.4))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(len(cls)):
    for j in range(len(codes)):
        if M[i, j] >= 0.35:
            ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=6.8, color=SURF if M[i, j] >= 0.7 else INK)
ax.set_xticks(range(len(codes)))
ax.set_xticklabels([nm.get(c, c)[:24] for c in codes], rotation=68, ha="right", fontsize=7.6)
for lab, c in zip(ax.get_xticklabels(), codes):
    lab.set_color({"급성": RED, "만성": BLUE}.get(cc.get(c), GREY))
ax.set_yticks(range(len(cls)))
ax.set_yticklabels([f"C{g}" for g in cls], fontsize=8.4)
ax.set_xticks(np.arange(len(codes) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(cls) + 1) - 0.5, minor=True)
ax.grid(which="minor", color=SURF, lw=1.2)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012).set_label("클러스터 내 유병률", fontsize=8)
ax.set_title(f"fig31 · 진단코드로 나눈 분할은 무엇으로 정의되는가 — 사전등록 주 분할 "
             f"C1_id k=15 (동반질환 프로파일 k-means, 14,541 방문)\n"
             f"클러스터를 특징짓는 진단 {len(codes)}개 중 {n_chr}개가 만성(파랑)이다. "
             f"AHRQ CCI 기준 급성 비율 {acute:.0f}%.\n"
             f"칸 = 그 클러스터 방문 중 해당 진단을 가진 비율. 35% 이상만 숫자를 적었다. "
             f"빨강 = 급성, 파랑 = 만성, 회색 = 미분류.\n"
             f"* 0414 는 개정 때 하위분류된 ICD-9 상위코드라 MIMIC 사전에 이름이 없다. "
             f"코드는 그대로 집계되었고 이름만 손으로 적었다.",
             fontsize=10, color=INK2, loc="left", pad=11)
fig.tight_layout()
fig.savefig(FIGS / "fig31_dx_axis.png", dpi=170)
plt.close(fig)
print(f"[=] fig31_dx_axis.png | 특징 진단 {len(codes)}개 중 만성 {n_chr}개 · 급성 {acute:.0f}%")
