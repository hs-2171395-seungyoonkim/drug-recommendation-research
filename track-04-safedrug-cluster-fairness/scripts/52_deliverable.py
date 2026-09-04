"""§전달물 2종을 한 문서로 묶는다.

선생님이 요청하신 것은 둘이다.
  1. 클러스터 개수에 따른 임베딩 시각화 그림
  2. 클러스터 별 가장 많이 등장하는 진단 명

둘 다 이미 out/ 에 있으나 표·그림·보고서에 흩어져 있다. 여기서는 **새로 계산하지
않고** 기존 산출물을 읽어 하나의 인수 문서로 정리한다. 숫자를 다시 만들지 않는 것이
요점이다 — 재계산하면 보고서마다 값이 갈릴 수 있다.

산출: out/DELIVERABLE.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
VARIANTS = ["short", "long", "concise"]

bmeta = json.loads((OUT / "40_dxtext_meta.json").read_text(encoding="utf-8"))
emeta = json.loads((OUT / "41_dxtext_embed_meta.json").read_text(encoding="utf-8"))
cmeta = json.loads((OUT / "42_dxtext_cluster_meta.json").read_text(encoding="utf-8"))
vmeta = json.loads((OUT / "51_dxtext_variants_meta.json").read_text(encoding="utf-8"))

prof = pd.read_csv(OUT / "table61_dxtext_profile.csv", encoding="utf-8-sig")
sw = pd.read_csv(OUT / "table62_dxtext_ksweep.csv", encoding="utf-8-sig")
cmp_ = pd.read_csv(OUT / "table73_dxtext_variant_compare.csv", encoding="utf-8-sig")
topall = pd.read_csv(OUT / "table72_dxtext_topdx_allvar.csv", encoding="utf-8-sig",
                     dtype={"주진단_코드": str, "SEQ1최빈_코드": str})
SEL = cmeta["selected"]
PRIM = cmeta["primary_variant"]
kB = cmeta["primary_kB"]


def md(df, cols=None, fmt=None):
    """DataFrame -> 마크다운 표."""
    d = df[cols] if cols else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = []
    for r in d.itertuples(index=False):
        cells = []
        for c, v in zip(d.columns, r):
            if fmt and c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


L = []
A = L.append

A(f"""# ICD 진단 텍스트 군집화 — 전달물

방문 **{bmeta['visits']:,}건** · 진단 {bmeta['rows_cohort']:,}행 · 방문당 진단 중앙 {bmeta['n_dx']['median']}개(최대 {bmeta['n_dx']['max']}개)
맵핑 `D_ICD_DIAGNOSES.csv` {bmeta['map_rows']:,}행 · 코드→제목 변환 실패 {bmeta['unmapped_pct']}%
인코더 `{emeta['model']}` · {emeta['pooling']} · maxlen {emeta['maxlen']}
군집화 PCA{cmeta['pca_dim']} → k-means (시드 {len(cmeta['seeds'])}개, 실루엣 중앙값 시드 채택)

**세 제목 변형(SHORT_TITLE / LONG_TITLE / CONCISE_TITLE)을 모두 수행했습니다.**
주 분석은 사전 규칙에 따라 `{PRIM}` · k={kB} 이고, 나머지 둘은 민감도 분석입니다.

| 변형 | 방문 | 문자 중앙 | 단어 중앙 | 고유 제목 수 | 토큰 중앙 | 잘린 방문 |
|---|---|---|---|---|---|---|""")
for _, r in prof.iterrows():
    v = r["변형"]
    A(f"| {v} | {r['방문']:,} | {r['문자 중앙']} | {r['단어 중앙']} | {r['고유 제목 수']:,} | "
      f"{emeta['variants'][v]['tok_median']} | {emeta['variants'][v]['truncated']} |")

A(f"""
> `CONCISE_TITLE`은 고유 제목이 {prof[prof['변형'] == 'concise']['고유 제목 수'].iloc[0]:,}개로,
> 코드 {bmeta['map_rows']:,}개를 3분의 2로 병합합니다. 세 변형이 완전히 동등한 비교는 아닙니다.

---

## 1. 클러스터 개수에 따른 임베딩 시각화

| 그림 | 내용 |
|---|---|
| `out/figs/fig33_dxtext_ksweep.png` | k에 따른 실루엣·CH·DB·최소군집 곡선 — **세 변형 겹쳐 그림** |
| `out/figs/fig34_dxtext_umap_panel.png` | **UMAP 2D, k=2/4/6/10/15/20/30/50 8패널** (변형 long) |
| `out/figs/fig35_dxtext_pca_panel.png` | 같은 8패널을 PCA 2D 좌표계로 (대조군, 변형 long) |
| `out/figs/fig38_dxtext_umap_panel_short.png` | UMAP 8패널 (변형 short) |
| `out/figs/fig39_dxtext_umap_panel_concise.png` | UMAP 8패널 (변형 concise) |

패널 안에서 **2D 좌표는 고정하고 색만 k별로 바꿉니다.** k마다 UMAP을 다시 돌리면 모양이
매번 달라져 "k가 늘면서 어떻게 쪼개지는가"를 볼 수 없기 때문입니다. UMAP은 시각화
전용이고 군집화는 PCA{cmeta['pca_dim']} 공간에서 했습니다(`random_state=0` 고정).

### 이쁘게 갈리는 k

| 변형 | 규칙A: 실루엣 최대 k | 규칙B: 최소군집≥100 중 최대 k | 규칙B k에서 실루엣 |
|---|---|---|---|""")
for v in VARIANTS:
    s = SEL[v]
    A(f"| {v} | **{s['kA']}** | {s['kB']} | {s['sil_at_kB']} |")

A(f"""
**세 변형 모두 실루엣이 k=2에서 최대입니다.** 이후 전반적으로 감소하며, long은 k=4~5,
concise는 k=5~6에서 소폭 반등이 있으나 k=2 수준에는 크게 못 미칩니다. 그림에서도
덩어리가 여럿 보이는 게 아니라 연속적인 구름 하나가 있고, k를 늘리는 것은 자연 경계를
찾는 게 아니라 그 구름을 잘라 나가는 모양입니다. 따라서 "이쁘게 갈리는 k"를 실루엣으로
고르면 k=2가 되지만, k=2에서는 군집별 주 진단이 무의미해집니다(양쪽 다 고혈압·심부전).

그래서 k를 두 갈래로 미리 정해두고 둘 다 보고합니다 — 규칙A(이쁨)와 규칙B(해상도).
아래 §2는 규칙B({PRIM} k={kB})를 씁니다.

전체 스윕 수치: `out/table62_dxtext_ksweep.csv` (변형 3 × k 14 = 42행)

### 변형에 따라 분할이 얼마나 달라지나

| 비교 | ARI |
|---|---|""")
for kk, vv in vmeta["cross_variant_ARI_at_kB"].items():
    A(f"| {kk.replace('ARI(', '').replace(')', '')} | {vv} |")

A("""
같은 진단 목록을 제목만 바꿔 인코딩했는데 분할 일치도가 0.15~0.20에 그칩니다.
**제목 변형 선택이 결과를 크게 좌우합니다** — 세 가지를 모두 해보라는 지시가 옳았습니다.

### 외적 기준 비교 (주진단 축을 얼마나 되찾나)

| 규칙 | 변형 | k | 최대군집% | 최소군집 | ARI | NMI | chapter순도% | 코드순도% |
|---|---|---|---|---|---|---|---|---|""")
for _, r in cmp_.iterrows():
    A(f"| {r['규칙']} | {r['변형']} | {r['k']} | {r['최대군집%']} | {r['최소군집']} | "
      f"{r['ARI_vs_chapter']} | {r['NMI_vs_chapter']} | {r['chapter순도%']} | {r['코드순도%']} |")

cB = cmp_[cmp_["규칙"] == "규칙B"].set_index("변형")
best_chap = cB["chapter순도%"].idxmax()
best_code = cB["코드순도%"].idxmax()
seed_ari = {v: sw[(sw["변형"] == v) & (sw["k"] == SEL[v]["kB"])]["ARI_시드간"].iloc[0]
            for v in VARIANTS}
A(f"""
기준선(무작위 분할): chapter 순도 **29.0%**, 코드 순도 **5.5%**.

- **{best_chap} k={SEL[best_chap]['kB']}** 이 ARI({cB.loc[best_chap, 'ARI_vs_chapter']}) ·
  NMI({cB.loc[best_chap, 'NMI_vs_chapter']}) · chapter순도({cB.loc[best_chap, 'chapter순도%']}%)에서 1위입니다.
- **{best_code} k={SEL[best_code]['kB']}** 은 코드순도 {cB.loc[best_code, '코드순도%']}%로 1위입니다
  (기준선의 약 {cB.loc[best_code, '코드순도%'] / 5.5:.1f}배).
- 다만 **어느 변형도 chapter 순도가 무작위 기준선(29.0%)을 뚜렷이 넘지 못합니다.**
  1위인 {best_chap} 가 {cB.loc[best_chap, 'chapter순도%']}%, 주 분석인 {PRIM} 은
  {cB.loc[PRIM, 'chapter순도%']}%로 기준선과 사실상 같습니다.
- 시드 간 ARI 는 k={kB} 에서 short {seed_ari['short']} · long {seed_ari['long']} ·
  concise {seed_ari['concise']} 입니다. 시드만 바꿔도 분할의 상당 부분이 달라집니다.
- 주 변형을 사전 규칙으로 고른 결과가 {PRIM} 인데, chapter 축을 중시하면 {best_chap} 가 낫습니다.
  두 표를 모두 넣어두었으니 어느 축을 볼지에 따라 고르시면 됩니다.

---

## 2. 클러스터 별 가장 많이 등장하는 진단 명

집계 재료는 **원본 ICD 코드 집합**이므로 제목 변형과 무관합니다. 변형이 바꾸는 것은
어떤 방문끼리 묶이느냐(라벨)뿐입니다. 진단명 표기는 세 표 모두 `SHORT_TITLE`로 통일했습니다.

- 유병률 = 그 군집 방문 중 해당 코드를 가진 비율
- lift = 유병률 ÷ 전체 유병률 (1.0이면 전체와 같음 = 그 군집의 특징이 아님)
- SEQ1최빈 = 그 군집에서 `SEQ_NUM=1`(공식 주진단)로 가장 흔한 코드
""")

for v in VARIANTS:
    k = SEL[v]["kB"]
    g = (topall[(topall["변형"] == v) & (topall["규칙"] == "규칙B")]
         .sort_values("n", ascending=False))
    star = " ★ 주 분석" if v == PRIM else ""
    A(f"\n### 변형 `{v}` · k={k}{star}\n")
    A("| 군집 | n | n% | 진단수 중앙 | 최빈 진단 | 유병률% | lift | SEQ1최빈 | SEQ1% |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        A(f"| {r['cluster']} | {r['n']:,} | {r['n%']} | {r['진단수_중앙']} | "
          f"**{r['주진단_명']}** | {r['주진단_유병률%']} | {r['주진단_lift']} | "
          f"{r['SEQ1최빈_명']} | {r['SEQ1최빈%']} |")

A(f"""
전체 표(빈도 top5·lift top5 포함): `out/table72_dxtext_topdx_allvar.csv`
그림: `out/figs/fig36_dxtext_topdx.png` (군집 × 특징 진단 유병률 히트맵)

### 읽을 때 주의할 점

**큰 군집의 최빈 진단은 그 군집의 특징이 아닙니다.** 세 표 모두에서 가장 큰 군집들의
최빈 진단은 `Hypertension NOS` / `CHF NOS`인데 lift가 1.0 안팎입니다. 코호트 전체에서
흔한 동반질환이라 어느 군집에서나 1위로 올라올 뿐입니다.

반대로 **작은 군집일수록 임상적으로 뚜렷한 이름이 붙습니다** — lift가 10~27배까지
올라갑니다(비파열 뇌동맥류, 알코올성 간경변, MRSA 폐렴, 당뇨병성 신경병증 등).

즉 "최빈 진단 = 주 진단"이라는 가정은 큰 군집에서 깨집니다.
""")

A("""---

## 부록 — 재현

```
python scripts/40_dxtext_build.py        # ICD9 -> 텍스트 (세 변형)
python scripts/41_dxtext_embed.py        # ClinicalBERT 인코딩 (세 변형)
python scripts/42_dxtext_cluster.py      # PCA50 -> k-means, k 스윕
python scripts/43_dxtext_report.py       # fig33~36, 보고서
python scripts/51_dxtext_variant_tables.py   # short/concise 표 + fig38/39
python scripts/52_deliverable.py         # 이 문서
```

상세 보고서: `out/REPORT_DXTEXT.md` (트랙 간 비교 §6 포함)
""")

(OUT / "DELIVERABLE.md").write_text("\n".join(L), encoding="utf-8")
print(f"[+] out/DELIVERABLE.md ({sum(len(x) for x in L):,}자)")
