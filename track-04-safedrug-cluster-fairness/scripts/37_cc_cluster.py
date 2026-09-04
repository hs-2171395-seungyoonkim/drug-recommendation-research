"""§주증상 군집화 — Chief Complaint 텍스트를 임베딩하고 군집화한다.

34_hpi_cluster.py 와 같은 결정적 구성(TruncatedSVD -> k-means)을 쓴다. UMAP/HDBSCAN 은
재현되지 않아 이미 폐기했다(34 의 method_history 참조).

CC 는 3단어라 33_hpi_embed.py 를 따로 부를 이유가 없어 임베딩까지 여기서 한다.

층화 (설계 판단):
  CC 가 있는 방문만 군집화한다. 없는 방문(약 12%)에 HPI 급성 절을 채워 넣으면 3단어와
  11단어 텍스트가 한 판에 섞여 길이가 축이 된다 — REPORT_BHC.md 의 η²(길이)=0.2484 실패다.
  대신 별도 층으로 남기고 HPI 트랙 라벨(clusterA)을 컬럼으로 붙여 둔다. 전 방문이 추적된다.

두 표현을 겨루게 한다:
  A  TF-IDF        축약어를 문자 그대로 본다. sob 와 shortness of breath 가 갈릴 수 있다.
  B  ClinicalBERT  동의어를 붙일 수 있다. CC 처럼 짧고 깨끗한 텍스트가 임베딩에 유리하다.

k 와 시드는 규칙으로 정한다(아래 상수 주석). 1차 실행의 시드 규칙이 퇴화한 해를 골랐고,
그 경위는 37_cc_cluster_meta.json 의 method_history 에 남긴다.

산출: out/cc_cluster_assignments.csv, out/table46_cc_clusters.csv,
      out/table47_cc_ksweep.csv, out/37_cc_svd.npz, out/37_cc_cluster_meta.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score as ari
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN, BATCH = 64, 256          # CC 는 중앙 3단어. 64 토큰이면 p99 도 들어간다.
SVD_DIM = 50
K_GRID = [15, 20, 24, 30, 40, 50, 60]
SEEDS = [0, 1, 2, 3, 4]
MIN_CLUSTER = 100                # 군집별 성능 추정이 가능한 최소 규모

# k 선택 규칙 (사전 고정): 최소 군집 >= MIN_CLUSTER 를 지키는 k 중 가장 큰 것.
# 트랙마다 따로 적용한다. 세분화될수록 주소견이 섞이지 않지만, 군집이 너무 작으면
# 군집별 SafeDrug 성능을 못 잰다. 두 요구의 경계가 이 규칙이다.
#
# 시드 선택 규칙 수정 (2026-08-13, 1차 실행 후):
#   1차에는 "시드 5개 중 최소 군집이 가장 큰 것"을 썼다. 잘못된 규칙이다. k=24 TF-IDF 에서
#   시드별 순도(군집 내 최빈 CC 점유율)가 6.6~33.5% 로 흔들렸는데, 이 규칙은 하필 6.6% 를
#   골랐다(최소 군집만 크고 내용은 뒤섞인 해). n_init=50 으로 올려도 편차가 남았다.
#   -> k-means 가 실제로 최적화하는 inertia 최소로 바꾼다. 같은 조건에서 33.5% 를 고른다.
#   HDBSCAN 사전등록 규칙이 k=2 로 퇴화했던 것과 같은 종류의 실수였다.

cc = pd.read_pickle(OUT / "36_cc_text.pkl")
hpi = pd.read_csv(OUT / "hpi_cluster_assignments.csv")[["HADM_ID", "clusterA", "route"]]
feat = pd.read_pickle(OUT / "20_icd_features.pkl")[["HADM_ID", "split", "n_icd"]]
df = cc.merge(hpi, on="HADM_ID", how="left").merge(feat, on="HADM_ID", how="left")

sub = df[df.has_cc].reset_index(drop=True)
texts = sub["cc_demog_stripped"].tolist()
print(f"[.] 전체 {len(df):,} | CC 있음 {len(sub):,} ({len(sub) / len(df) * 100:.1f}%) "
      f"| 층외(CC 없음) {int((~df.has_cc).sum()):,}", flush=True)


def purity(lab, k):
    """군집별 최빈 CC 점유율의 중앙값. 군집이 하나의 주소견을 뜻하는지 본다."""
    s = pd.Series(texts).str.lower()
    return round(float(s.groupby(lab).agg(
        lambda x: x.value_counts().iloc[0] / len(x)).median() * 100), 1)


def kmeans_best(Z, k, tag):
    """시드 5개 중 inertia 최소. k-means 가 실제로 최적화하는 값이다."""
    best = None
    for s in SEEDS:
        km = KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z)
        if best is None or km.inertia_ < best[0].inertia_:
            best = (km, s)
    km, s = best
    lab = km.labels_
    c = np.bincount(lab, minlength=k)
    info = {"k": k, "seed": s, "inertia": round(float(km.inertia_), 2),
            "largest_pct": round(float(c.max() / len(lab)) * 100, 1),
            "median_size": int(np.median(c)), "min_size": int(c.min()),
            "top_cc_share_median_pct": purity(lab, k)}
    print(f"[.] {tag} k={k} seed={s} 최대 {info['largest_pct']}% 최소 {info['min_size']} "
          f"순도 {info['top_cc_share_median_pct']}%", flush=True)
    return lab, info


def pick_k(sweep, tag):
    """최소 군집 >= MIN_CLUSTER 인 k 중 가장 큰 것. 없으면 가장 작은 k."""
    ok = [r for r in sweep if r["min_size"] >= MIN_CLUSTER]
    k = max(r["k"] for r in ok) if ok else min(r["k"] for r in sweep)
    print(f"[=] {tag} k={k} 선택 (최소군집>={MIN_CLUSTER} 만족: "
          f"{[r['k'] for r in ok] or '없음'})", flush=True)
    return k


# ---------------------------------------------------------------- A: TF-IDF
vec = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2),
                      strip_accents="unicode", lowercase=True,
                      token_pattern=r"(?u)\b[A-Za-z][A-Za-z'\-/]+\b")
X = vec.fit_transform(texts)
svd = TruncatedSVD(SVD_DIM, random_state=0)
Z_tf = svd.fit_transform(X)
Z_tf /= np.linalg.norm(Z_tf, axis=1, keepdims=True).clip(min=1e-9)
ev = round(float(svd.explained_variance_ratio_.sum()) * 100, 1)
print(f"[.] TF-IDF {X.shape[0]}x{X.shape[1]} | SVD{SVD_DIM} 설명분산 {ev}%", flush=True)

sweep_A, keep_A = [], {}
for k in K_GRID:
    keep_A[k], i = kmeans_best(Z_tf, k, "sweep A")
    sweep_A.append(i | {"track": "A"})
K_A = pick_k(sweep_A, "A TF-IDF")
lab_A = keep_A[K_A]
info_A = next(r for r in sweep_A if r["k"] == K_A) | {
    "repr": "TF-IDF", "reducer": f"TruncatedSVD({SVD_DIM})", "algo": "k-means",
    "svd_explained_variance_pct": ev, "vocab": int(X.shape[1])}

# ---------------------------------------------------------------- B: ClinicalBERT
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()
if dev == "cuda":
    model = model.half()

t0 = time.time()
E = np.zeros((len(texts), 768), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], max_length=MAXLEN, truncation=True,
                  padding=True, return_tensors="pt")
        ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        h = model(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1)
        E[i:i + BATCH] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
E /= np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)
print(f"[.] ClinicalBERT {len(texts)}건 {time.time() - t0:.1f}초 ({dev})", flush=True)

Z_cb = TruncatedSVD(SVD_DIM, random_state=0).fit_transform(E)
Z_cb /= np.linalg.norm(Z_cb, axis=1, keepdims=True).clip(min=1e-9)

sweep_B, keep_B = [], {}
for k in K_GRID:
    keep_B[k], i = kmeans_best(Z_cb, k, "sweep B")
    sweep_B.append(i | {"track": "B"})
K_B = pick_k(sweep_B, "B ClinicalBERT")
lab_B = keep_B[K_B]
info_B = next(r for r in sweep_B if r["k"] == K_B) | {
    "repr": "ClinicalBERT", "reducer": f"TruncatedSVD({SVD_DIM})", "algo": "k-means"}

pd.DataFrame(sweep_A + sweep_B).to_csv(OUT / "table47_cc_ksweep.csv",
                                       index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 비교
hpi_lab = sub["clusterA"].to_numpy()
ok = ~pd.isna(hpi_lab)
cmp_ = {
    "ARI_A_vs_B": round(float(ari(lab_A, lab_B)), 4),
    "ARI_A_vs_HPI": round(float(ari(lab_A[ok], hpi_lab[ok].astype(int))), 4),
    "ARI_B_vs_HPI": round(float(ari(lab_B[ok], hpi_lab[ok].astype(int))), 4),
}
print(f"[.] ARI(A,B)={cmp_['ARI_A_vs_B']} | ARI(A,HPI)={cmp_['ARI_A_vs_HPI']} "
      f"| ARI(B,HPI)={cmp_['ARI_B_vs_HPI']}", flush=True)

sub["clusterCC"], sub["clusterCC_bert"] = lab_A, lab_B
asg = df.merge(sub[["HADM_ID", "clusterCC", "clusterCC_bert"]], on="HADM_ID", how="left")
asg["stratum"] = np.where(asg.has_cc, "CC", "no_CC")
asg[["SUBJECT_ID", "HADM_ID", "split", "stratum", "cc_demog_stripped", "n_word",
     "n_icd", "route", "clusterA", "clusterCC", "clusterCC_bert"]].to_csv(
    OUT / "cc_cluster_assignments.csv", index=False, encoding="utf-8-sig")

g = sub.groupby("clusterCC")
tab = pd.DataFrame({
    "cluster": g.size().index, "n": g.size().values,
    "pct": (g.size() / len(sub) * 100).round(1).values,
    "words_median": g["n_word"].median().values,
    "n_icd_median": g["n_icd"].median().values,
    "top_cc": g["cc_demog_stripped"].agg(lambda s: s.str.lower().value_counts().index[0]).values,
    "top_cc_share": g["cc_demog_stripped"].agg(
        lambda s: round(s.str.lower().value_counts().iloc[0] / len(s) * 100, 1)).values,
})
tab.to_csv(OUT / "table46_cc_clusters.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(OUT / "37_cc_svd.npz", Z_tf=Z_tf, Z_cb=Z_cb,
                    HADM_ID=sub["HADM_ID"].to_numpy())

meta = {"n_visits": len(df), "n_clustered": len(sub),
        "stratum_no_cc": int((~df.has_cc).sum()),
        "stratification_reason": "3단어 CC 와 11단어 HPI 급성 절을 섞으면 길이가 축이 된다"
                                 " (REPORT_BHC.md η²(길이)=0.2484)",
        "k_grid": K_GRID, "seeds": SEEDS, "A": info_A, "B": info_B, "compare": cmp_,
        "k_rule": f"최소 군집 >= {MIN_CLUSTER} 를 지키는 k 중 가장 큰 것 (트랙별로 적용)",
        "seed_rule": "시드 5개 중 inertia 최소",
        "method_history": [
            {"when": "1차 실행", "rule": "시드 5개 중 최소 군집이 가장 큰 것",
             "result": "k=24 TF-IDF 에서 seed=4 를 골랐다. 순도(군집 내 최빈 CC 점유율) "
                       "중앙 6.6% 로, 시드 8개 중 최악이었다(6.6~33.5%). "
                       "최소 군집만 크고 내용은 뒤섞인 해다. n_init=50 으로도 안 낫는다.",
             "superseded_by": "inertia 최소"},
            {"when": "설계", "rule": "k=24 고정 (HPI 트랙과 맞춤)",
             "result": "잔여 군집 C9 가 20.4% 를 삼켰다. 비교 편의로 k 를 고정할 이유가 없다.",
             "superseded_by": f"최소 군집 >= {MIN_CLUSTER} 중 최대 k"}],
        "reproducible": True}
with open(OUT / "37_cc_cluster_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"\n[=] 최빈 CC 점유율 중앙 {tab.top_cc_share.median():.1f}% "
      f"(HPI 트랙과 비교할 지표)")
print(tab.sort_values("n", ascending=False).to_string(index=False))
