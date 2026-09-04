"""§ICD-텍스트 트랙 vs 이전 군집화 트랙 정면 비교.

같은 14,541 방문을 서로 다른 표현으로 군집화한 결과가 지금까지 5개 쌓였다.
"지금이 나은가"를 답하려면 표현마다 다른 공간에서 잰 실루엣을 나열해선 안 된다
(공간이 다르면 비교가 안 된다). 그래서 **공간과 무관한 외적 기준**을 주로 쓴다:

  1. ARI/NMI vs D1 — 주진단 chapter(ICD-9 장) 를 얼마나 되찾는가.
  2. 주진단 순도 — 군집 내 최빈 주진단 chapter / 최빈 seq1 코드 점유율의 가중 중앙값.
     무작위 분할이면 전체 최빈 chapter 비율(순환기 ~29%) 근처에 머문다.
  3. 크기 균형 — 최대 군집 %, 최소 군집.
  4. 시드 안정성 ARI — 각 트랙 meta 에 기록된 값(재계산 안 함).
실루엣은 자기 공간에서만 계산하고 **트랙 간 비교 금지** 라고 표에 못박는다.

CC 트랙은 CC 가 있는 12,804 방문만 군집화했다. 외적 지표는 각 트랙이 실제로 라벨을
붙인 방문에서만 계산하고, 대상 방문 수를 표에 같이 적는다.

산출: out/table66_track_compare.csv, out/45_track_compare_meta.json,
      out/figs/fig37_track_compare.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIG = OUT / "figs"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SIL_SAMPLE = 5000

# ---- 기준축: 주진단(SEQ_NUM=1) 코드와 chapter ----
d1 = pd.read_csv(OUT / "icd_partition_assignments.csv", dtype={"icd9_seq1": str})
base = d1[["HADM_ID", "icd9_seq1", "chapter", "d1_label"]].copy()
ORDER = base["HADM_ID"].to_numpy()

# ---- 각 트랙의 라벨 ----
T = {}


def add(name, repr_, hadm, lab, k, space=None, seed_ari=None):
    T[name] = dict(repr=repr_, hadm=np.asarray(hadm), lab=np.asarray(lab),
                   k=k, space=space, seed_ari=seed_ari)


ca = pd.read_pickle(OUT / "cluster_assignments.pkl")
emb = np.load(OUT / "embeddings.npz")
for k in (15, 20):
    g = ca[(ca["partition"] == "C1_id") & (ca["k"] == k)]
    add(f"C1_id k={k}", "ICD 멀티핫 → SVD100", g["HADM_ID"], g["cluster_id"], k,
        space=emb["Z_id"])

bhc = pd.read_csv(OUT / "cluster_assignments_bhc.csv")
for p, lbl, sa in [("BHC_CB_masked", "BHC 노트 ClinicalBERT", 0.561),
                   ("BHC_TFIDF_masked", "BHC 노트 TF-IDF", 0.755)]:
    g = bhc[(bhc["partition"] == p) & (bhc["k"] == 20)]
    add(f"{p} k=20", lbl, g["HADM_ID"], g["cluster_id"], 20, seed_ari=sa)

hpi = pd.read_csv(OUT / "hpi_cluster_assignments.csv")
hz = np.load(OUT / "34_hpi_svd.npz")
assert (hz["HADM_ID"] == hpi["HADM_ID"].to_numpy()).all()
add("HPI TF-IDF k=24", "HPI 급성절 TF-IDF → SVD50", hpi["HADM_ID"], hpi["clusterA"], 24,
    space=hz["Z_tf"])
add("HPI ClinicalBERT k=24", "HPI 원문 ClinicalBERT → SVD50", hpi["HADM_ID"],
    hpi["clusterB"], 24, space=hz["Z_cb"])

cc = pd.read_csv(OUT / "cc_cluster_assignments.csv")
cz = np.load(OUT / "37_cc_svd.npz")
ccm = cc[cc["HADM_ID"].isin(cz["HADM_ID"])].set_index("HADM_ID").loc[cz["HADM_ID"]]
add("CC TF-IDF k=40", "주증상 TF-IDF → SVD50", cz["HADM_ID"], ccm["clusterA"], 40,
    space=cz["Z_tf"])
add("CC ClinicalBERT k=24", "주증상 ClinicalBERT → SVD50", cz["HADM_ID"],
    ccm["clusterCC_bert"], 24, space=cz["Z_cb"])

dx = pd.read_csv(OUT / "dxtext_cluster_assignments.csv")
from sklearn.decomposition import PCA  # noqa: E402
for v, col, k, sa in [("short", "short_kB25", 25, None),
                      ("long", "long_kB20", 20, 0.738),
                      ("concise", "concise_kB20", 20, None)]:
    z = np.load(OUT / f"emb_dxtext_{v}.npz")
    sp = PCA(n_components=50, random_state=0).fit_transform(z["E"]).astype(np.float32)
    add(f"★ ICD텍스트 {v} k={k}", f"ICD 텍스트({v}) ClinicalBERT → PCA50",
        dx["HADM_ID"], dx[col], k, space=sp, seed_ari=sa)

# ---- 지표 ----
CHAP_BASE = base["chapter"].value_counts(normalize=True).iloc[0]
rows = []
for name, t in T.items():
    b = base.set_index("HADM_ID").reindex(t["hadm"])
    lab = t["lab"]
    ok = b["chapter"].notna().to_numpy()
    ch, c1 = b["chapter"].to_numpy()[ok], b["icd9_seq1"].to_numpy()[ok]
    L = lab[ok]
    cnt = pd.Series(L).value_counts()

    df = pd.DataFrame({"L": L, "ch": ch, "c1": c1})
    pur_ch = df.groupby("L")["ch"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    pur_c1 = df.groupby("L")["c1"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    w = cnt.reindex(pur_ch.index).to_numpy()

    def wmed(v):                       # 방문 가중 중앙값
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        return float(v[o][np.searchsorted(cw, 0.5)])

    sil = np.nan
    if t["space"] is not None:
        sil = float(silhouette_score(t["space"], lab, sample_size=SIL_SAMPLE,
                                     random_state=0))
    rows.append({
        "트랙": name, "표현": t["repr"], "k": t["k"], "대상방문": int(len(L)),
        "최대군집%": round(float(cnt.max()) / len(L) * 100, 1),
        "최소군집": int(cnt.min()),
        "ARI_vs_주진단chapter": round(adjusted_rand_score(ch, L), 4),
        "NMI_vs_주진단chapter": round(normalized_mutual_info_score(ch, L), 4),
        "주진단chapter순도%": round(wmed(pur_ch.to_numpy()) * 100, 1),
        "주진단코드순도%": round(wmed(pur_c1.to_numpy()) * 100, 1),
        "실루엣_자기공간": (round(sil, 4) if sil == sil else ""),
        "시드ARI": (t["seed_ari"] if t["seed_ari"] else ""),
    })
r = pd.DataFrame(rows)
r.to_csv(OUT / "table66_track_compare.csv", index=False, encoding="utf-8-sig")
print(r.drop(columns=["표현"]).to_string(index=False))

# ---- 새 트랙이 기존 ICD 멀티핫과 같은 분할인가 ----
def pair_ari(a, b):
    """reindex 로 생긴 결측을 빼고 겹치는 방문에서만 잰다."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return round(adjusted_rand_score(a[m], b[m]), 4), int(m.sum())


gid = ca[(ca["partition"] == "C1_id") & (ca["k"] == 20)].set_index("HADM_ID")
cross = {}
for v, col in [("short", "short_kB25"), ("long", "long_kB20"), ("concise", "concise_kB20")]:
    a = dx.set_index("HADM_ID")[col]
    j = gid["cluster_id"].reindex(a.index)
    cross[f"ARI({v} vs C1_id k=20)"] = pair_ari(j, a)
hj = hpi.set_index("HADM_ID")["clusterB"].reindex(dx["HADM_ID"])
cc_b = ccm["clusterCC_bert"]
cc_b.index = cz["HADM_ID"]
cross["ARI(long vs HPI ClinicalBERT k=24)"] = pair_ari(hj, dx["long_kB20"])
cross["ARI(long vs CC ClinicalBERT k=24)"] = pair_ari(
    cc_b.reindex(dx["HADM_ID"]), dx["long_kB20"])
bj = bhc[(bhc["partition"] == "BHC_CB_masked") & (bhc["k"] == 20)] \
    .set_index("HADM_ID")["cluster_id"].reindex(dx["HADM_ID"])
cross["ARI(long vs BHC ClinicalBERT k=20)"] = pair_ari(bj, dx["long_kB20"])
print(json.dumps(cross, ensure_ascii=False, indent=1))

# ---- 그림 ----
r2 = r.sort_values("주진단chapter순도%")
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
col = ["#c0392b" if s.startswith("★") else "#7f8c8d" for s in r2["트랙"]]
ax[0].barh(r2["트랙"], r2["주진단chapter순도%"], color=col)
ax[0].axvline(CHAP_BASE * 100, ls="--", c="k", lw=1)
ax[0].text(CHAP_BASE * 100, -0.8, f" 무작위 기준선 {CHAP_BASE * 100:.0f}%", fontsize=9)
ax[0].set_xlabel("군집 내 최빈 주진단 chapter 점유율 (가중 중앙값, %)")
ax[0].set_title("주진단 축을 얼마나 되찾는가")
r3 = r.sort_values("ARI_vs_주진단chapter")
col3 = ["#c0392b" if s.startswith("★") else "#7f8c8d" for s in r3["트랙"]]
ax[1].barh(r3["트랙"], r3["ARI_vs_주진단chapter"], color=col3)
ax[1].set_xlabel("ARI vs 주진단 chapter")
ax[1].set_title("주진단 chapter 와의 일치도 (ARI)")
for a in ax:
    a.grid(axis="x", alpha=0.3)
fig.suptitle("트랙 비교 — ★ 가 이번 ICD 텍스트 트랙", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "fig37_track_compare.png", dpi=160)

# ---- REPORT 에 §6 삽입 (한계는 §7 로 밀린다) ----
def md(d, cols, hdr):
    o = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for _, x in d.iterrows():
        o.append("| " + " | ".join(str(x[c]) for c in cols) + " |")
    return chr(10).join(o)


SEQ1_BASE = base["icd9_seq1"].value_counts(normalize=True).iloc[0] * 100
rr = r.sort_values("ARI_vs_주진단chapter", ascending=False)
sec = f"""## 6. 이전 트랙과의 비교

같은 14,541 방문에 대해 이 프로젝트에서 지금까지 만든 군집화는 5종이다. 표현이 다르면
공간이 달라 실루엣을 나란히 놓을 수 없으므로, **공간과 무관한 외적 기준**으로 비교한다:
군집이 주진단(SEQ_NUM=1) 축을 얼마나 되찾는가.

무작위 분할의 기준선 — 주진단 chapter 최빈 비율 **{CHAP_BASE * 100:.1f}%**(순환기),
주진단 코드 최빈 비율 **{SEQ1_BASE:.1f}%**(패혈증 0389, 고유 코드 1,474개).

{md(rr, ["트랙", "표현", "k", "대상방문", "ARI_vs_주진단chapter", "NMI_vs_주진단chapter",
         "주진단chapter순도%", "주진단코드순도%", "최대군집%", "최소군집", "실루엣_자기공간", "시드ARI"],
    ["트랙", "표현", "k", "방문", "ARI", "NMI", "chapter순도%", "코드순도%",
     "최대%", "최소", "실루엣(자기공간)", "시드ARI"])}

실루엣 열은 각 트랙 **자기 공간**에서 잰 값이다. 공간이 다르므로 트랙 간 비교에 쓰면 안 된다.
시드ARI 는 각 트랙 meta 에 기록돼 있는 것만 옮겼다.

그림: `out/figs/fig37_track_compare.png`.

### 읽는 법

- **어떤 방법도 주진단 축을 되찾지 못한다.** ARI 가 전부 0.019~0.060 구간이다. 이건 이번
  트랙의 실패가 아니라 이 데이터에서 반복 확인된 사실이다(§4, 28_chronic_cluster 도 0.056).
- **ICD 텍스트 concise 가 ARI·NMI 에서 전 트랙 1위**({rr.iloc[0]['ARI_vs_주진단chapter']} / {rr.iloc[0]['NMI_vs_주진단chapter']}) 지만
  2위 C1_id k=15(0.0565)와의 차이는 유의미하다고 주장할 만한 폭이 아니다.
- **주진단 코드 순도는 ICD 텍스트 long 이 10.1% 로 1위**다(기준선 {SEQ1_BASE:.1f}%, 2위 BHC 8.2%).
  같은 정확 코드를 공유하는 방문을 가장 잘 모은다.
- **노트 기반 트랙(BHC)은 기준선 아래**다(chapter 순도 27.0 / 24.5% < {CHAP_BASE * 100:.1f}%).
  퇴원요약 서술은 주진단 축과 거의 무관하게 흩어진다.
- **시드 안정성은 이번 트랙이 낫다**: ICD텍스트 long k=20 의 시드 간 ARI 0.738 vs
  BHC ClinicalBERT k=20 의 0.561. HPI 트랙은 UMAP 비재현성 때문에 k-means 로 되돌아가야 했고
  (34_hpi_cluster method_history), 이번 트랙은 그런 폴백이 없었다.

### 같은 ICD 코드인데 표현을 바꾸면 다른 분할이 나온다

| 비교 | ARI | 겹치는 방문 |
|---|---|---|
{chr(10).join(f"| {k} | {v[0]} | {v[1]:,} |" for k, v in cross.items())}

C1_id(멀티핫)와 ICD텍스트(BERT)는 **입력이 같은 진단 코드인데도 ARI 0.06~0.11** 이다.
표현을 바꾼 것이 사실상 다른 분할을 만든다는 뜻이고, 어느 쪽이 옳은지는 외적 지표가
비슷해서 이 데이터만으로 가릴 수 없다. 노트 기반 트랙과는 ARI 0.01~0.03 으로 거의 무관하다.

### 결론

이번 트랙이 이전보다 **명확히 나은 점**은 (1) 시드 안정성, (2) 주진단 코드 순도(long),
(3) 파이프라인이 단순하고 완전히 재현된다는 것(노트 파싱·섹션 추출 없음, 잘림 0건,
인코딩 30초, HDBSCAN/UMAP 재현성 문제 없음)이다.
**동급인 점**은 주진단 chapter 회복력이다 — concise 가 ARI 1위지만 2위와의 차이를
주장할 폭이 아니다.
**뒤처지는 점**도 있다. 주 분석으로 고른 long k=20 의 chapter 순도는 34.0% 로,
C1_id k=20(38.2%)·HPI ClinicalBERT k=24(40.7%)·concise(39.9%) 보다 낮다.
long 을 고른 근거는 실루엣이었는데 그 기준이 chapter 순도와 어긋난 것이다. chapter 축을
중시한다면 concise 로 주 분석을 바꾸는 편이 낫다(§4 표는 long 기준으로 만들어져 있다).
어느 쪽이든 "주진단 축으로 안 갈린다"는 결론은 이번이 다섯 번째 재확인이다.

"""
# 재실행해도 §6 이 중복되지 않게, 이미 삽입돼 있으면 그 구간을 갈아끼운다.
rep = (OUT / "REPORT_DXTEXT.md").read_text(encoding="utf-8")
if "## 6. 이전 트랙과의 비교" in rep:
    a, b = rep.index("## 6. 이전 트랙과의 비교"), rep.index("## 7. 한계")
    rep = rep[:a] + sec + rep[b:]
else:
    h = rep.index("## 6. 한계")
    rep = rep[:h] + sec + rep[h:].replace("## 6. 한계", "## 7. 한계", 1)
(OUT / "REPORT_DXTEXT.md").write_text(rep, encoding="utf-8")

(OUT / "45_track_compare_meta.json").write_text(json.dumps({
    "chapter_baseline_pct": round(float(CHAP_BASE) * 100, 2),
    "silhouette_note": "각 트랙 자기 공간에서 계산. 공간이 다르므로 트랙 간 비교 불가.",
    "cross_ari": cross,
    "rows": rows,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] out/table66_track_compare.csv, figs/fig37_track_compare.png")
