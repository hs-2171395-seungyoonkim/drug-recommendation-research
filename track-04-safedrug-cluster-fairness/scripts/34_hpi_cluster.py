"""§4 HPI 군집 — A(급성 절) / B(원문 + 잔차화).

A 는 텍스트를 손보는 접근이고 B 는 임베딩에서 방향을 빼는 접근이다. REPORT_CHRONIC.md 의
교훈("Elixhauser 를 지워도 만성 축이 사라지지 않고 옆으로 옮겨 앉았다")이 B 에서 재현될
가능성이 있고, 그것 자체가 기록할 값이다.

--- 방법 선택 이력 (전부 기록한다) ---

1차 사전등록 규칙("노이즈<=50% 중 min_cluster_size 최대")은 잘못 적은 것이었다.
min_cluster_size 를 키우면 k 가 줄어드는 것이 당연하므로 사실상 k 를 최소화하는 규칙이었고,
UMAP+HDBSCAN 에서 ClinicalBERT mcs=200/ms=15 -> k=2, 최대 군집 73.1% 라는 퇴화된 해를 골랐다.

수정 규칙(사용자 승인): 노이즈<=50% & 최대군집<=25% & k>=2 중 노이즈 최소.

그 뒤 **UMAP 이 재현되지 않는다는 것을 확인했다.** random_state=0, n_jobs=1 에서도 같은 입력에
대해 실행마다 k=9/11/12/13, 노이즈 43.3~47.7% 로 흔들린다(umap-learn 0.5.12). 선택 규칙 전체가
실행마다 달라지는 숫자 위에 서게 된다.

결정적 축소(TruncatedSVD, 두 번 실행 라벨 100% 일치)로 바꾸면 HDBSCAN 이 무너진다 — 균형 잡힌
설정은 노이즈 50~63%, 아니면 한 군집이 93~97% 를 먹는다. 즉 HDBSCAN 은 UMAP 의 다양체 구조에
의존하는데 그 UMAP 을 신뢰할 수 없다.

따라서 승인된 폴백 규칙 3("자격 통과 조합이 없으면 k-means 로 전환하고 전환 사실을 남긴다")을
발동한다. **결과가 나빠서가 아니라 방법이 재현되지 않아서다.** 전환 후에는 UMAP 을 쓰지 않는다.

  A : TF-IDF -> TruncatedSVD(50) -> L2 -> k-means(k=24)      전 방문 배정, 노이즈 범주 없음
  B : 원문 ClinicalBERT -> 잡음변수 잔차화 -> L2 -> k-means(k=24)

k=24 는 사용자가 승인한 값이다. k 스윕 표(table43)를 함께 남겨 다른 k 로 바꾸는 것은 한 줄이면 되게 한다.

이 스크립트는 판정축(ARI vs D1/C1_id, CCI 급성 비중, η², 순열)을 만들지 않는다 — 이번 범위 밖이다.
A/B 비교는 교차표와 ARI 하나뿐이고, 우열이 아니라 "같은 것을 보는가"만 본다.

산출: out/hpi_cluster_assignments.csv, out/table41_hpi_clusters.csv, out/table42_hpi_AB.csv,
      out/table43_hpi_ksweep.csv, out/34_hpi_svd.npz, out/34_hpi_cluster_meta.json
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

SVD_DIM = 50
K = 24
K_GRID = [15, 20, 24, 30, 40]
SEEDS = [0, 1, 2, 3, 4]

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SW = set("""the of a an with and who was were is are be been to for in on at by or as from
this that these those there here he she his her him they them their it its we our you your
not no but also then than had has have having do does did
patient patients pt pts mr mrs ms dr year years old yo y/o male female man woman gentleman lady
history hx admitted admission admit hospital ed er icu status post s p per h/o who's""".split())


def content(t):
    return [w.lower() for w in WORD.findall(t) if len(w) >= 2 and w.lower() not in SW]


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-9)


def shape_of(lab):
    s = np.sort(np.bincount(lab))[::-1]
    return {"k": int(len(s)), "largest_pct": round(float(s[0]) / len(lab) * 100, 1),
            "second_pct": round(float(s[1]) / len(lab) * 100, 1),
            "median_size": int(np.median(s)), "min_size": int(s[-1])}


def kmeans_best(Z, k):
    best = None
    for sd in SEEDS:
        km = KMeans(n_clusters=k, n_init=10, random_state=sd).fit(Z)
        if best is None or km.inertia_ < best[0]:
            best = (km.inertia_, sd, km.labels_)
    return best


# ---------------------------------------------------------------- 입력
txt = pd.read_pickle(OUT / "32_hpi_text.pkl").reset_index(drop=True)
za = np.load(OUT / "emb_hpi_acute.npz", allow_pickle=True)
zr = np.load(OUT / "emb_hpi_raw.npz", allow_pickle=True)
assert (za["HADM_ID"] == txt["HADM_ID"].to_numpy()).all()
assert (zr["HADM_ID"] == txt["HADM_ID"].to_numpy()).all()
E_ac, E_rw = za["E"], zr["E"]

feat = pd.read_pickle(OUT / "20_icd_features.pkl")[["HADM_ID", "n_icd", "split"]]
txt = txt.merge(feat, on="HADM_ID", how="left")
print(f"[.] 방문 {len(txt)} | n_icd 결측 {int(txt.n_icd.isna().sum())}", flush=True)

# ---------------------------------------------------------------- A: TF-IDF -> SVD -> k-means
X = sp.load_npz(OUT / "33_hpi_tfidf.npz")
svd = TruncatedSVD(n_components=SVD_DIM, random_state=0)
Z_tf = l2(svd.fit_transform(X))
print(f"[.] A: TF-IDF {X.shape} -> SVD{SVD_DIM} "
      f"(설명분산 {svd.explained_variance_ratio_.sum() * 100:.1f}%)", flush=True)

ksweep = []
for k in K_GRID:
    lab = kmeans_best(Z_tf, k)[2]
    ksweep.append({"k": k, **shape_of(lab)})
    print(f"    k={k:3d} -> 최대 {ksweep[-1]['largest_pct']:5.1f}%  중앙 {ksweep[-1]['median_size']:5d}  "
          f"최소 {ksweep[-1]['min_size']:4d}", flush=True)

inertiaA, seedA, labA = kmeans_best(Z_tf, K)
dA = shape_of(labA)
print(f"[.] A 채택: k={K}, seed {seedA} | 최대 {dA['largest_pct']}% 최소 {dA['min_size']}", flush=True)

# 같은 k 로 ClinicalBERT 급성 절 (표현 대조)
Z_cb = l2(TruncatedSVD(n_components=SVD_DIM, random_state=0).fit_transform(E_ac))
labO = kmeans_best(Z_cb, K)[2]
dO = shape_of(labO)
print(f"[.] 표현 대조(ClinicalBERT 급성 절, k={K}): 최대 {dO['largest_pct']}% "
      f"최소 {dO['min_size']}", flush=True)

# ---------------------------------------------------------------- B: 원문 + 잔차화 -> k-means
n_tok = txt["n_tok_raw"].to_numpy(float)
n_before = txt["hpi_nochron"].map(lambda t: len(content(t))).to_numpy(float)
overlap_n = n_before * txt["pmh_removed_frac"].to_numpy(float)
n_icd = txt["n_icd"].fillna(txt["n_icd"].median()).to_numpy(float)

Z = np.column_stack([np.ones(len(txt)), n_tok, overlap_n, n_icd])
beta, *_ = np.linalg.lstsq(Z, E_rw, rcond=None)
R = E_rw - Z @ beta
r2 = 1 - (R ** 2).sum() / ((E_rw - E_rw.mean(0)) ** 2).sum()
R = l2(R)
print(f"[.] B: 잔차화 — [토큰수, PMH겹침수, 진단수] 가 설명한 분산 {r2 * 100:.2f}%", flush=True)

inertiaB, seedB, labB = kmeans_best(R, K)
dB = shape_of(labB)
print(f"[.] B 채택: k={K}, seed {seedB} | 최대 {dB['largest_pct']}% 최소 {dB['min_size']}", flush=True)

# ---------------------------------------------------------------- 산출
txt["clusterA"] = labA
txt["clusterA_other"] = labO
txt["clusterB"] = labB
cols = ["SUBJECT_ID", "HADM_ID", "split", "route", "pmh_removed_frac", "no_acute",
        "n_content_acute", "clusterA", "clusterA_other", "clusterB"]
txt[cols].to_csv(OUT / "hpi_cluster_assignments.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(ksweep).to_csv(OUT / "table43_hpi_ksweep.csv", index=False, encoding="utf-8-sig")

rows = []
for c in range(K):
    m = labA == c
    rows.append({"cluster": int(c), "n": int(m.sum()), "pct": round(float(m.mean()) * 100, 1),
                 "no_acute_pct": round(float(txt.loc[m, "no_acute"].mean()) * 100, 1),
                 "route_full_pct": round(float((txt.loc[m, "route"] == "full").mean()) * 100, 1),
                 "words_median": float(txt.loc[m, "n_content_acute"].median()),
                 "n_icd_median": float(txt.loc[m, "n_icd"].median())})
pd.DataFrame(rows).to_csv(OUT / "table41_hpi_clusters.csv", index=False, encoding="utf-8-sig")

pd.crosstab(pd.Series(labA, name="A"), pd.Series(labB, name="B")).to_csv(
    OUT / "table42_hpi_AB.csv", encoding="utf-8-sig")

meta = {
    "method_history": {
        "prereg_rule": "노이즈<=50% 중 min_cluster_size 최대 — 잘못 적은 규칙",
        "prereg_result": {"repr": "ClinicalBERT", "reducer": "UMAP", "algo": "HDBSCAN",
                          "min_cluster_size": 200, "min_samples": 15,
                          "n_clusters": 2, "noise_pct": 25.0, "largest_pct": 73.1},
        "amended_rule": "노이즈<=50% & 최대군집<=25% & k>=2 중 노이즈 최소 (사용자 승인)",
        "umap_reproducibility": {
            "reproducible": False,
            "note": "random_state=0, n_jobs=1 에서도 같은 입력에 실행마다 다른 결과",
            "observed_k": [9, 11, 12, 13], "observed_noise_pct": [43.3, 47.1, 47.3, 47.7],
            "umap_version": "0.5.12"},
        "hdbscan_on_deterministic_svd": {
            "note": "SVD 는 재현되지만(라벨 100% 일치) HDBSCAN 이 무너짐",
            "balanced_configs_noise_pct": [53.0, 54.6, 57.1, 57.2, 58.5, 63.5],
            "low_noise_configs_largest_pct": [66.9, 69.9, 92.7, 93.8, 96.7, 97.1]},
        "final": "승인된 폴백 규칙 3 발동 -> k-means. 결과가 아니라 재현성 때문",
    },
    "A": {"repr": "TF-IDF", "reducer": f"TruncatedSVD({SVD_DIM})", "algo": "k-means",
          "k": K, "seed": int(seedA), "k_approved_by_user": True,
          "svd_explained_variance_pct": round(float(svd.explained_variance_ratio_.sum()) * 100, 1),
          **dA},
    "A_other_repr": {"repr": "ClinicalBERT(급성 절)", "k": K, **dO},
    "B": {"repr": "ClinicalBERT(원문)", "algo": "k-means", "k": K, "seed": int(seedB),
          "nuisance": ["n_tok_raw", "pmh_overlap_n", "n_icd"],
          "variance_explained_by_nuisance_pct": round(float(r2) * 100, 2), **dB},
    "compare": {"ARI_A_vs_B": round(float(adjusted_rand_score(labA, labB)), 4),
                "ARI_A_vs_other_repr": round(float(adjusted_rand_score(labA, labO)), 4)},
    "reproducible": True,
}
with open(OUT / "34_hpi_cluster_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
np.savez_compressed(OUT / "34_hpi_svd.npz", Z_tf=Z_tf, Z_cb=Z_cb,
                    HADM_ID=txt["HADM_ID"].to_numpy())
print("\n" + json.dumps(meta["compare"], ensure_ascii=False, indent=2))
