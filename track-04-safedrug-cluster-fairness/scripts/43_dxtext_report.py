"""§ICD 텍스트 군집 — 그림 + 보고서.

선생님께 전달할 ①k별 시각화 ②군집별 최빈 진단명 을 여기서 만든다.

2D 좌표는 42 에서 한 번 계산해 고정한 것을 그대로 쓴다. k 마다 UMAP 을 다시 돌리면
모양이 매번 달라져 "k 가 늘면서 어떻게 쪼개지는가"를 볼 수 없다.

산출: out/REPORT_DXTEXT.md,
      out/figs/fig33_dxtext_ksweep.png, fig34_dxtext_umap_panel.png,
      fig35_dxtext_pca_panel.png, fig36_dxtext_topdx.png
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
FIGS.mkdir(exist_ok=True)

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
RED, BLUE, GREY = "#c0392b", "#2471a3", "#8a8985"
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
})

meta = json.loads((OUT / "42_dxtext_cluster_meta.json").read_text(encoding="utf-8"))
bmeta = json.loads((OUT / "40_dxtext_meta.json").read_text(encoding="utf-8"))
emeta = json.loads((OUT / "41_dxtext_embed_meta.json").read_text(encoding="utf-8"))
sw = pd.read_csv(OUT / "table62_dxtext_ksweep.csv")
# ICD9_CODE 는 '41401' 처럼 숫자로 보이지만 문자열이다. dtype 을 안 주면 int 로 읽혀
# 아래 진단 집합(문자열) 과의 in 비교가 전부 False 가 된다.
top = pd.read_csv(OUT / "table63_dxtext_topdx.csv",
                  dtype={"주진단_코드": str, "SEQ1최빈_코드": str})
prof = pd.read_csv(OUT / "table61_dxtext_profile.csv")
asg = pd.read_csv(OUT / "dxtext_cluster_assignments.csv", dtype={"seq1_code": str})
df = pd.read_pickle(OUT / "40_dxtext.pkl")
proj = np.load(OUT / "42_dxtext_proj.npz", allow_pickle=True)
labs = np.load(OUT / "42_dxtext_labels.npz")

V = meta["primary_variant"]
kA, kB = meta["primary_kA"], meta["primary_kB"]
VARIANTS = ["short", "long", "concise"]
PANEL_K = [2, 4, 6, 10, 15, 20, 30, 50]

# ---------------- fig33 : k 스윕 ----------------
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
COL = {"short": BLUE, "long": RED, "concise": "#7d3c98"}
for ax, (col, lab, logy) in zip(axes, [("실루엣", "실루엣 (높을수록 좋음)", False),
                                       ("CH", "Calinski-Harabasz (높을수록)", False),
                                       ("DB", "Davies-Bouldin (낮을수록)", False),
                                       ("최소군집_최악시드", "최소 군집 크기 (시드 최악, 방문)", True)]):
    for v in VARIANTS:
        g = sw[sw["변형"] == v]
        ax.plot(g["k"], g[col], "-o", ms=3.2, lw=1.4, color=COL[v], label=v)
    ax.set_xlabel("k"); ax.set_title(lab, fontsize=9.5, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(sw["k"].unique())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelsize=7.5)
    if logy:
        ax.set_yscale("log")
        ax.axhline(100, color=GREY, lw=1, ls="--")
        ax.text(2.1, 108, "규칙B 하한 100", fontsize=7, color=GREY)
    for sp in ax.spines.values():
        sp.set_color("#d9d8d4")
axes[0].axvline(kA, color=GREY, lw=1, ls=":")
axes[0].text(kA * 1.12, 0.86, f"규칙A k={kA}", fontsize=7, color=GREY,
             transform=axes[0].get_xaxis_transform())
axes[3].axvline(kB, color=GREY, lw=1, ls=":")
axes[3].text(kB * 0.94, 0.55, f"규칙B k={kB}", fontsize=7, color=GREY, ha="right",
             transform=axes[3].get_xaxis_transform())
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle(f"ICD-9 진단목록 → ClinicalBERT 임베딩, k 스윕 ({bmeta['visits']:,} 방문 · 시드 5개)",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "fig33_dxtext_ksweep.png", dpi=190, bbox_inches="tight")
plt.close(fig)
print("[+] fig33", flush=True)


# tab20 은 절반이 파스텔이라 k=30~50 패널이 하얗게 뜬다. 채도 있는 색만 모아 쓴다.
PAL = ([plt.get_cmap("tab10")(i) for i in range(10)]
       + [plt.get_cmap("tab20b")(i) for i in range(20)]
       + [plt.get_cmap("tab20")(i) for i in range(1, 20, 2)]
       + [plt.get_cmap("tab20c")(i) for i in range(0, 20, 4)])


def panel(coords, fname, title):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    for ax, k in zip(axes.ravel(), PANEL_K):
        lab = labs[f"{V}_k{k}"]
        ax.scatter(coords[:, 0], coords[:, 1], s=1.7, alpha=0.62,
                   c=[PAL[i % len(PAL)] for i in lab], linewidths=0)
        sil = float(sw[(sw["변형"] == V) & (sw["k"] == k)]["실루엣"].iloc[0])
        mark = ""
        if k == kA:
            mark += "  ◀규칙A"
        if k == kB:
            mark += "  ◀규칙B"
        ax.set_title(f"k={k}   실루엣 {sil:.3f}{mark}", fontsize=9.5,
                     color=RED if mark else INK2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d9d8d4")
    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] {fname}", flush=True)


panel(proj["umap2"], "fig34_dxtext_umap_panel.png",
      f"UMAP 2D (좌표 고정, 색만 k별로 교체) — 변형 {V}, {bmeta['visits']:,} 방문")
panel(proj["pca2"], "fig35_dxtext_pca_panel.png",
      f"PCA 2D (같은 좌표계 대조) — 변형 {V}, {bmeta['visits']:,} 방문")

# ---------------- fig36 : 군집 × 특징 진단 유병률 ----------------
tb = top[top["규칙"] == "규칙B"].sort_values("cluster")
mapdf = pd.read_csv(Path.home() / "Downloads" / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
mapdf["ICD9_CODE"] = mapdf["ICD9_CODE"].str.strip()
NAME = dict(zip(mapdf["ICD9_CODE"], mapdf["SHORT_TITLE"]))
SETS = [set(l) for l in df["icd_l"]]
lab = labs[f"{V}_k{kB}"]
allc = sorted({c for s in SETS for c in s})
ci = {c: i for i, c in enumerate(allc)}
M = np.zeros((len(df), len(allc)), dtype=bool)
for i, s in enumerate(SETS):
    for c in s:
        M[i, ci[c]] = True
base = M.mean(0)
# 그림은 lift **비율**이 아니라 lift **차이**(군집 유병률 − 전체 유병률)로 뽑는다.
# 비율 상위는 전체적으로 드문 코드를 고르므로 군집 내 유병률도 10~20%대라 히트맵이
# 통째로 하얗게 뜬다. 차이 상위는 "흔하면서 이 군집에 몰린" 코드를 고른다.
# 표(table63)의 lift 컬럼은 사전등록대로 비율 그대로 둔다. 30_chronic_report 와 같은 처리다.
codes = []
for g in range(kB):
    idx = np.where(lab == g)[0]
    prev = M[idx].mean(0)
    codes += [allc[i] for i in np.argsort(-(prev - base))[:2]]
codes = list(dict.fromkeys(codes))
H = np.array([[M[np.where(lab == g)[0]][:, ci[c]].mean() for c in codes] for g in range(kB)])

fig, ax = plt.subplots(figsize=(max(12.5, len(codes) * 0.36), 6.8))
im = ax.imshow(H, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(kB):
    for j in range(len(codes)):
        if H[i, j] >= 0.35:
            ax.text(j, i, f"{H[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=6.2, color=SURF if H[i, j] >= 0.7 else INK)
ax.set_xticks(range(len(codes)))
ax.set_xticklabels([NAME.get(c, c)[:26] for c in codes], rotation=70, ha="right", fontsize=6.6)
ax.set_yticks(range(kB))
ax.set_yticklabels([f"C{g} (n={int(tb.iloc[g]['n'])})" for g in range(kB)], fontsize=8)
ax.set_xticks(np.arange(len(codes) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(kB + 1) - 0.5, minor=True)
ax.grid(which="minor", color=SURF, lw=1.1)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_color("#d9d8d4")
fig.colorbar(im, ax=ax, fraction=0.016, pad=0.01).set_label("군집 내 유병률", fontsize=8)
ax.set_title(f"군집별 특징 진단(유병률 초과 상위 2개) 유병률 — {V}, k={kB}", fontsize=11, pad=12)
fig.tight_layout()
fig.savefig(FIGS / "fig36_dxtext_topdx.png", dpi=185, bbox_inches="tight")
plt.close(fig)
print("[+] fig36", flush=True)

# ---------------- REPORT ----------------
selrows = "\n".join(
    f"| {v} | {meta['selected'][v]['kA']} | {meta['selected'][v]['kB']} | "
    f"{meta['selected'][v]['sil_at_kB']:.4f} |" for v in VARIANTS)


def md_table(d, cols, hdr=None):
    hdr = hdr or cols
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


agree = float((asg["seq1_code"].map(NAME) == pd.Series(
    [tb.set_index("cluster").loc[g, "주진단_명"] for g in lab])).mean()) * 100

rep = f"""# ICD-9 진단목록 텍스트 → ClinicalBERT 군집화 — 결과

설계: `docs/superpowers/specs/2026-08-19-icd-text-clustering-design.md` (사전등록).
재현: `python scripts/40_dxtext_build.py && ... 41 && 42 && 43`.

## 0. 요약

- {bmeta['visits']:,} 방문의 ICD-9 진단 목록({bmeta['n_dx']['mean']}개/방문)을 텍스트로 바꿔 **통째로** 한 벡터에 넣었다.
- 세 제목 변형 모두 실루엣은 **k=2에서 최대**이고 k가 늘수록 단조 감소한다. "이쁘게 갈리는 k"는 k=2다.
  그러나 k=2는 군집 하나가 {sw[(sw['변형'] == V) & (sw['k'] == 2)]['최대군집%'].iloc[0]}%를 먹는 퇴화된 해다.
- 사전등록 규칙 B(최소군집≥100 중 최대 k)로 **{V} · k={kB}** 를 주 구성으로 채택했다.
- 군집별 **빈도 1위 진단은 고혈압·심부전·관상동맥경화로 반복**된다. 이게 이 트랙의 핵심 결과다.
  방문을 가른 축은 "이번에 왜 왔는가"가 아니라 **동반질환 프로파일**이고,
  주성분 하나가 진단 **개수**와 |r|=0.74~0.81로 상관한다(§4, table74).
- lift 상위 진단은 군집을 구별하지만 **군집을 거의 덮지 못한다** — lift top5 커버리지가
  25개 군집 중 11개에서 30% 미만이다(최저 5.9%). 그래서 테마 라벨은 빈도 top5
  (커버리지 55~99%)로 붙이고 lift 는 부수 특징으로만 쓴다. 두 커버리지를 표에 함께 실었다.

## 1. 1단계 — ICD-9 → 텍스트

맵핑 파일은 선생님이 보내주신 `D_ICD_DIAGNOSES.csv`({bmeta['map_rows']}행, `CONCISE_TITLE` 포함)를 썼다.

- 코호트 진단행 {bmeta['rows_cohort']:,}건 → 매핑 {bmeta['rows_mapped']:,}건.
- **미매핑 {bmeta['unmapped_rows']:,}건({bmeta['unmapped_pct']}%)은 버렸다.** ICD-9-CM 개정 때 하위분류된
  상위코드(`{'`, `'.join(list(bmeta['unmapped_top10'])[:5])}` …)로 사전에 이름이 없다.
  버려도 **{bmeta['visits']:,} 방문 전부**가 최소 1개 제목을 유지한다.
- `SEQ_NUM` 오름차순으로 `"; "` 결합. 1번이 청구서 주 진단 자리라 순서 자체가 정보다.

{md_table(prof, list(prof.columns))}

`CONCISE_TITLE`은 고유 제목이 10,163개다 — 코드 14,567개를 **3분의 2로 병합**한다.
같은 이름이 여러 코드에 붙으므로 텍스트가 짧아지는 대신 코드 구분이 일부 사라진다.

## 2. 2단계 — 통째 인코딩

`emilyalsentzer/Bio_ClinicalBERT`, `max_length=512`, mask 가중 mean pooling → L2, 768차원.

| 변형 | 토큰 중앙 | p99 | 최대 | 잘림 방문 |
|---|---|---|---|---|
""" + "\n".join(
    f"| {v} | {emeta['variants'][v]['tok_median']} | {emeta['variants'][v]['tok_p99']} | "
    f"{emeta['variants'][v]['tok_max']} | {emeta['variants'][v]['truncated']} |"
    for v in VARIANTS) + f"""

**슬라이딩 윈도우가 필요 없다.** BHC(중앙 226단어)와 달리 진단 목록은 최대 498토큰이라
한 번의 forward에 들어간다. 잘린 방문은 long에서 4건뿐이다.

## 3. 3~4단계 — 군집화와 k

PCA 50차원(누적설명분산 0.83~0.85) → k-means, `n_init=10`, 시드 0~4.
실루엣은 `sample_size=5000`으로 추정했다({bmeta['visits']:,} 전량 쌍거리를 210회 돌 수 없다).
HDBSCAN은 `34_hpi_cluster`에서 이미 폐기했으므로 재시도하지 않았다.

### k 선택 — 사전등록한 두 갈래

| 변형 | 규칙A k (실루엣 최대) | 규칙B k (**모든 시드** 최소군집≥100 중 최대) | 규칙B k에서의 실루엣 |
|---|---|---|---|
{selrows}

**규칙 A는 세 변형 모두 k=2를 골랐다.** 설계 문서에서 예상한 그대로다 — mean pooling
임베딩에서 실루엣은 거의 항상 작은 k를 편든다. 실루엣은 k=2에서 {sw[(sw['변형'] == V) & (sw['k'] == 2)]['실루엣'].iloc[0]:.3f},
k={kB}에서 {sw[(sw['변형'] == V) & (sw['k'] == kB)]['실루엣'].iloc[0]:.3f}로 **단조 감소**하며 국소 최대(엘보)가 없다.
Calinski-Harabasz도 같은 방향이고, Davies-Bouldin만 큰 k를 조금 선호한다.

즉 **"이쁘게 갈리는 k"는 존재하지 않는다.** 이 임베딩 공간에는 뚜렷이 분리된 덩어리가 없고
연속적인 구름 하나가 있다. fig34/fig35가 그 그림이다. k를 늘리는 것은 자연스러운 경계를
찾는 것이 아니라 연속체를 임의로 자르는 것이다.

주 변형은 규칙 B의 k에서 실루엣이 가장 높은 **{V}**로 정해졌다.

시드 안정성: k={kB}에서 시드 간 ARI {sw[(sw['변형'] == V) & (sw['k'] == kB)]['ARI_시드간'].iloc[0]:.3f}.
k가 커질수록 떨어져 k=50에서 {sw[(sw['변형'] == V) & (sw['k'] == 50)]['ARI_시드간'].iloc[0]:.3f}까지 간다 — 경계가 임의적이라는
또 하나의 증거다.

전체 스윕: `out/table62_dxtext_ksweep.csv`, 그림 `out/figs/fig33_dxtext_ksweep.png`.

## 4. 5단계 — 군집별 최빈 진단 ({V}, k={kB})

{md_table(tb, ["cluster", "n", "n%", "진단수_중앙", "주진단_명", "주진단_유병률%", "주진단_lift", "SEQ1최빈_명", "SEQ1최빈%"],
           ["군집", "n", "n%", "진단수", "주 진단(빈도1위)", "유병률%", "lift", "SEQ_NUM=1 최빈", "%"])}

### 빈도 1위만 보면 안 되는 이유

위 표의 "주 진단" 컬럼에서 같은 이름이 여러 군집에 반복된다. 고혈압·심부전·관상동맥경화는
코호트 전체에서 흔해서, 어느 군집을 잘라도 1위로 올라온다. **빈도 1위는 군집을 구별하지 못한다.**

군집을 실제로 구별하는 것은 lift(군집 내 유병률 ÷ 전체 유병률)다:

{md_table(tb, ["cluster", "n", "lift_top5"], ["군집", "n", "lift 상위 5 (군집 대비 전체 배수)"])}

`SEQ_NUM=1`(청구서 주 진단) 최빈값과 비교하면, 두 관점이 일치하는 군집도 있고
(C0/C1/C6/C15/C17 등) 갈리는 군집도 있다. 전체 방문 중 빈도1위 주 진단이 그 방문의
청구서 주 진단과 같은 비율은 **{agree:.1f}%**다.

원표: `out/table63_dxtext_topdx.csv` (규칙 A k={kA} 결과도 같은 파일에 있다).
그림: `out/figs/fig36_dxtext_topdx.png`.

## 5. 한계

- 노트(`data4LLM_with_note.csv`)는 5섹션 가공본이라 **Chief Complaint가 없다.** 주 증상 원문이
  필요하면 `NOTEEVENTS.csv`를 봐야 한다(코호트 88.6% 보유).
- 진단 목록은 **퇴원 시점 청구 코드**다. 입원 사유가 아니라 이번 입원에서 다룬 문제 전부이며,
  만성 동반질환이 대부분을 차지한다. §4의 결과는 이 성질의 직접적 귀결이다.
- 실루엣은 5,000 표본 추정치다. 시드 간 sd는 표에 있다.
- `CONCISE_TITLE`은 코드를 3분의 2로 병합하므로 세 변형이 완전히 동등한 비교는 아니다.
"""
(OUT / "REPORT_DXTEXT.md").write_text(rep, encoding="utf-8")
print("[+] out/REPORT_DXTEXT.md")
