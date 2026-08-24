"""§가중치 결합(V_main×5 + V_sub×1) 트랙 사후 진단.

받은 지적 5개를 코드·데이터로 직접 확인한다. 새 분석을 만드는 게 아니라
"왜 이렇게 나왔나"를 숫자로 못 박는 것이 목적이다. 확인 항목:

  (1) 결합 임베딩에서 부진단이 코사인 유사도에 기여하는 비율.
      L2 정규화한 두 블록을 [w·E_main, E_sub] 로 붙이면
      cos = (w^2·cos_main + cos_sub) / (w^2 + 1).  w=5 이면 부진단은 1/26 = 3.8%.
      해석적으로 유도되지만 실제 배열로도 재서 확인한다.
  (2) E_main 의 서로 다른 행 수. 주진단 텍스트는 사전 항목이라 방문 수보다 훨씬 적다.
      이 수가 작으면 UMAP 은 환자가 아니라 ICD 사전을 그린 것이 된다.
  (3) w 스윕: w=0(부진단만) ~ w=inf(주진단만) 까지 실루엣과 분할이 어떻게 변하는지.
      ×5 가 실제로 한 일이 "주진단만 쓰기"와 얼마나 다른지 ARI 로 잰다.
  (4) 실제 출하된 k=30 분할의 군집별 top-1 주진단 순도·고유 진단 수·chapter 순도.
  (5) Track1(chapter) 과 Track2(가중치 k=30) 의 AMI/ARI, 그리고 Track2 의 chapter 순도.
      두 트랙이 독립이 아니면 "트랙 간 교차검증"이라는 말을 쓸 수 없다.

  (6) 실루엣 비교 가능성: 같은 특징공간에서 chapter 라벨 / k-means 라벨 /
      "SEQ1 최빈 29코드 + 기타" 사전 조회 분할의 실루엣을 나란히 잰다.

산출: out/table78_weight_wsweep.csv, out/table79_weight_cluster_audit.csv,
      out/56_weight_diagnose.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
VARIANT = "concise"      # 42w 가 고른 주 변형
K = 30                   # 42w 규칙B 의 k
PCA_DIM, SEEDS, SIL_SAMPLE = 50, [0, 1, 2, 3, 4], 5000
W_GRID = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
W_MAIN_SHIPPED = 5.0

df = pd.read_pickle(OUT / "40w_weight.pkl")
z = np.load(OUT / f"emb_dxtext_weight_{VARIANT}.npz")
E, HADM = z["E"], z["HADM_ID"]
assert (HADM == df["HADM_ID"].to_numpy()).all(), "방문 순서 불일치"
N, D2 = E.shape
D = D2 // 2
print(f"[.] {VARIANT}: 방문 {N}, 결합차원 {D2} (블록 {D}+{D})", flush=True)

# ── (1) 블록 분리 & 코사인 기여 비율 ────────────────────────────────────────
# 41w 는 E_main, E_sub 를 각각 단위벡터로 만든 뒤 [w·E_main, E_sub] 를 다시 정규화했다.
# 따라서 앞블록/뒷블록을 각각 다시 정규화하면 원래 E_main, E_sub 를 정확히 복원한다.
A, B = E[:, :D].astype(np.float64), E[:, D:].astype(np.float64)
nA, nB = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
E_main, E_sub = A / nA[:, None].clip(1e-12), B / nB[:, None].clip(1e-12)
ratio = float(np.median(nA / nB))
print(f"[.] 블록 노름비 median(|main|/|sub|) = {ratio:.4f}  (설계값 {W_MAIN_SHIPPED})",
      flush=True)

rng = np.random.default_rng(0)
ii, jj = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
ok = ii != jj
ii, jj = ii[ok], jj[ok]
cos_cat = (E[ii] * E[jj]).sum(1)
cos_m = (E_main[ii] * E_main[jj]).sum(1)
cos_s = (E_sub[ii] * E_sub[jj]).sum(1)
w2 = W_MAIN_SHIPPED ** 2
pred = (w2 * cos_m + cos_s) / (w2 + 1)
max_err = float(np.abs(cos_cat - pred).max())
sub_share = 1.0 / (w2 + 1)
print(f"[=] cos = (w²·cos_main + cos_sub)/(w²+1) 확인, 최대오차 {max_err:.2e}", flush=True)
print(f"[=] 부진단이 유사도에 기여하는 비율 = 1/{w2 + 1:.0f} = {sub_share * 100:.1f}%",
      flush=True)

# ── (2) 서로 다른 행 수 ─────────────────────────────────────────────────────
uniq_main_rows = int(len(np.unique(np.round(E_main, 5), axis=0)))
uniq_sub_rows = int(len(np.unique(np.round(E_sub, 5), axis=0)))
uniq_main_txt = int(df[f"{VARIANT}_main"].nunique())
uniq_sub_txt = int(df[f"{VARIANT}_sub"].nunique())
uniq_seq1 = int(df["seq1_code"].nunique())
print(f"[=] E_main 고유행 {uniq_main_rows} (텍스트 {uniq_main_txt}종) / "
      f"E_sub 고유행 {uniq_sub_rows} (텍스트 {uniq_sub_txt}종) / 방문 {N}", flush=True)

# ── (3) w 스윕 ──────────────────────────────────────────────────────────────
def build(w):
    if w == "main":
        return E_main.copy()
    Z = np.concatenate([E_main * w, E_sub], axis=1)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)


def cluster(X, k=K):
    """42w 와 같은 절차: PCA50 -> k-means 5시드 -> 실루엣 중앙값 시드 채택."""
    P = PCA(n_components=PCA_DIM, random_state=0).fit_transform(X).astype(np.float32)
    sils, labs = [], []
    for s in SEEDS:
        lab = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(P)
        sils.append(silhouette_score(P, lab, sample_size=SIL_SAMPLE, random_state=0))
        labs.append(lab)
    med = int(np.argsort(sils)[len(sils) // 2])
    return P, labs[med], float(np.mean(sils)), float(np.std(sils)), labs


sweep, LABS, PROJ = [], {}, {}
for w in W_GRID + ["main"]:
    X = build(w)
    P, lab, sil, sd, _ = cluster(X)
    LABS[w], PROJ[w] = lab, P
    cnt = np.bincount(lab, minlength=K)
    tag = "주진단만(w=∞)" if w == "main" else f"w={w:g}"
    sweep.append({"구성": tag, "w": (np.inf if w == "main" else w),
                  "부진단_기여%": (0.0 if w == "main" else
                                round(100.0 / (w ** 2 + 1), 1)),
                  "실루엣": round(sil, 4), "실루엣sd": round(sd, 4),
                  "최소군집": int(cnt.min()), "최대군집%": round(cnt.max() / N * 100, 2)})
    print(f"    {tag:14s} 실루엣={sil:.4f}  최소군집={cnt.min():4d}", flush=True)

for r in sweep:
    key = "main" if np.isinf(r["w"]) else r["w"]
    r["ARI_vs_주진단만"] = round(adjusted_rand_score(LABS[key], LABS["main"]), 4)
    r["ARI_vs_출하판(w=5)"] = round(adjusted_rand_score(LABS[key], LABS[W_MAIN_SHIPPED]), 4)
ws = pd.DataFrame(sweep)
ws.to_csv(OUT / "table78_weight_wsweep.csv", index=False, encoding="utf-8-sig")

# ── (6) 같은 공간에서 세 분할의 실루엣 ──────────────────────────────────────
P5 = PROJ[W_MAIN_SHIPPED]
chap = pd.read_csv(OUT / "dxtext_chapter_assignments.csv", usecols=["HADM_ID", "chapter"])
chap = chap.set_index("HADM_ID").reindex(df["HADM_ID"]).reset_index()
assert chap["chapter"].notna().all()
chap_lab = pd.factorize(chap["chapter"])[0]

s1 = df["seq1_code"].fillna("__NA__")
top29 = list(s1.value_counts().head(K - 1).index)
naive_lab = pd.factorize(s1.where(s1.isin(top29), "__기타__"))[0]

asg = pd.read_csv(OUT / "dxtext_weight_cluster_assignments.csv",
                  usecols=["HADM_ID", "primary_cluster"])
asg = asg.set_index("HADM_ID").reindex(df["HADM_ID"]).reset_index()
ship_lab = asg["primary_cluster"].to_numpy()

sil_cmp = {}
for nm, lab in [("Track1_chapter(18군)", chap_lab),
                ("Track2_kmeans_k30(출하판)", ship_lab),
                (f"SEQ1최빈{K - 1}코드+기타(사전조회)", naive_lab),
                ("주진단만_kmeans_k30", LABS["main"])]:
    sil_cmp[nm] = round(float(silhouette_score(P5, lab, sample_size=SIL_SAMPLE,
                                               random_state=0)), 4)
    print(f"    [w=5 공간] {nm:32s} 실루엣={sil_cmp[nm]:.4f}", flush=True)

# ── (4) 출하된 k=30 분할 군집별 감사 ────────────────────────────────────────
d = pd.read_csv(ROOT / "DIAGNOSES_ICD.csv", dtype={"ICD9_CODE": str},
                usecols=["HADM_ID", "SEQ_NUM", "ICD9_CODE"])
d = d[d["HADM_ID"].isin(set(df["HADM_ID"]))].dropna(subset=["ICD9_CODE"])
d["ICD9_CODE"] = d["ICD9_CODE"].str.strip()
mapdf = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
mapdf["ICD9_CODE"] = mapdf["ICD9_CODE"].str.strip()
NAME = dict(zip(mapdf["ICD9_CODE"], mapdf["SHORT_TITLE"]))

aud = pd.DataFrame({"HADM_ID": df["HADM_ID"], "cluster": ship_lab,
                    "seq1": df["seq1_code"], "chapter": chap["chapter"]})
rows = []
for c, g in aud.groupby("cluster"):
    vc = g["seq1"].value_counts()
    ch = g["chapter"].value_counts()
    top1 = vc.index[0] if len(vc) else ""
    rows.append({"cluster": int(c), "n": len(g),
                 "top1_SEQ1코드": top1, "top1_SEQ1명": NAME.get(top1, "?"),
                 "top1_순도%_군집n기준": round(float(vc.iloc[0]) / len(g) * 100, 1),
                 "고유_SEQ1코드수": int(g["seq1"].nunique()),
                 "chapter최빈": ch.index[0],
                 "chapter순도%": round(float(ch.iloc[0]) / len(g) * 100, 1)})
au = pd.DataFrame(rows).sort_values("cluster")
au.to_csv(OUT / "table79_weight_cluster_audit.csv", index=False, encoding="utf-8-sig")
pur_w = float((au["top1_순도%_군집n기준"] * au["n"]).sum() / au["n"].sum())
chp_w = float((au["chapter순도%"] * au["n"]).sum() / au["n"].sum())
print(f"[=] 출하 k=30: top-1 주진단 순도 가중평균 {pur_w:.1f}% | "
      f"chapter 순도 가중평균 {chp_w:.1f}%", flush=True)
print(au.to_string(index=False), flush=True)

# ── (5) 두 트랙의 독립성 ────────────────────────────────────────────────────
ami = float(adjusted_mutual_info_score(chap_lab, ship_lab))
ari = float(adjusted_rand_score(chap_lab, ship_lab))
print(f"[=] Track1 vs Track2: AMI={ami:.4f} ARI={ari:.4f}", flush=True)

meta = {
    "변형": VARIANT, "k": K, "방문": int(N), "출하_가중치": W_MAIN_SHIPPED,
    "코사인_분해": {"식": "cos = (w²·cos_main + cos_sub)/(w²+1)",
                "최대오차": max_err,
                "부진단_기여%": round(sub_share * 100, 2),
                "블록노름비_median": round(ratio, 4)},
    "고유행": {"E_main": uniq_main_rows, "E_sub": uniq_sub_rows,
             "main텍스트종": uniq_main_txt, "sub텍스트종": uniq_sub_txt,
             "SEQ1코드종": uniq_seq1, "방문": int(N)},
    "w스윕": ws.to_dict("records"),
    "같은공간_실루엣": sil_cmp,
    "출하분할": {"top1_주진단순도_가중평균%": round(pur_w, 1),
              "chapter순도_가중평균%": round(chp_w, 1),
              "단일코드군집": au.loc[au["고유_SEQ1코드수"] <= 2, "cluster"].tolist(),
              "고유코드_최대": int(au["고유_SEQ1코드수"].max())},
    "트랙독립성": {"AMI": round(ami, 4), "ARI": round(ari, 4)},
    "_note": "사후 진단이다. 사전 등록한 가설이 아니고, 새 결론을 만들지 않는다.",
}
(OUT / "56_weight_diagnose.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] out/table78_weight_wsweep.csv, table79_weight_cluster_audit.csv, "
      "56_weight_diagnose.json")
