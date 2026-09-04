"""
§2 벡터화 (id / name 두 버전) → §3 분할 P1/C1/C2 → §4 무엇으로 갈리는지 → §5 약물 개수 교란.

§8: 분석 함수는 전부 '벡터 행렬 X'를 인자로 받는다. 노트 임베딩이 오면 build_partitions(X, ...)에
그대로 넣으면 된다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
SEED = 0
KS = list(range(5, 31))

df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
n = len(df)

# ---------------------------------------------------------------- §2 벡터화
def multihot(list_col, vocab=None):
    items = sorted({v for l in list_col for v in l}) if vocab is None else vocab
    idx = {v: j for j, v in enumerate(items)}
    X = np.zeros((len(list_col), len(items)), dtype=np.float32)
    for i, l in enumerate(list_col):
        for v in l:
            X[i, idx[v]] = 1.0
    return X, items


X_id_raw, vocab_id = multihot(df["diag_id_l"])
X_nm_raw, vocab_nm = multihot(df["diagnose_l"])

# id -> 대표 이름 (id->name 은 1:1)
id2name = {}
for names, ids in zip(df["diagnose_l"], df["diag_id_l"]):
    for nm, i in zip(names, ids):
        id2name[i] = nm

n_diag = df["diag_id_l"].map(len).to_numpy(float)
n_drug = df["drug_id_l"].map(len).to_numpy(float)


def embed(X_raw, n_comp=100, seed=SEED):
    """L2 정규화 → SVD → 다시 L2 정규화. 군집화는 이 공간에서."""
    Xn = normalize(X_raw, norm="l2", axis=1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Z = svd.fit_transform(Xn)
    return normalize(Z, norm="l2", axis=1), svd


Z_id, svd_id = embed(X_id_raw)
Z_nm, svd_nm = embed(X_nm_raw)

vect_info = {
    "id": {"dim": X_id_raw.shape[1], "svd_explained_var": round(float(svd_id.explained_variance_ratio_.sum()), 4)},
    "name": {"dim": X_nm_raw.shape[1], "svd_explained_var": round(float(svd_nm.explained_variance_ratio_.sum()), 4)},
    "density_id": round(float(X_id_raw.mean()), 6),
    "density_name": round(float(X_nm_raw.mean()), 6),
}

# ---------------------------------------------------------------- §3 분할
def p1_partition(min_visits):
    """첫 진단(diag_id[0]) 기준. 방문 >= min_visits 인 라벨은 유지, 나머지는 '기타'(-1)."""
    first = df["diag_id_l"].map(lambda l: l[0]).to_numpy()
    vc = pd.Series(first).value_counts()
    keep = set(vc[vc >= min_visits].index)
    return np.array([f if f in keep else -1 for f in first]), keep


def kmeans_labels(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def ward_tree(Z):
    return linkage(Z, method="ward")


print("[.] Ward linkage (id) ...", flush=True)
Zt_id = ward_tree(Z_id)
print("[.] Ward linkage (name) ...", flush=True)
Zt_nm = ward_tree(Z_nm)

partitions = {}  # name -> {k: labels}
partitions["P1_ge100"] = {0: p1_partition(100)[0]}
partitions["P1_ge30"] = {0: p1_partition(30)[0]}
for tag, Z, Zt in [("id", Z_id, Zt_id), ("name", Z_nm, Zt_nm)]:
    partitions[f"C1_{tag}"] = {k: kmeans_labels(Z, k) for k in KS}
    partitions[f"C2_{tag}"] = {k: fcluster(Zt, t=k, criterion="maxclust") - 1 for k in KS}
    print(f"[.] clustered {tag}", flush=True)

# ---------------------------------------------------------------- §4/§5 지표
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    m = len(x)
    return float((2 * np.arange(1, m + 1) - m - 1).dot(x) / (m * x.sum()))


def eta2(labels, y):
    """범주 라벨이 연속변수 y의 분산을 얼마나 설명하는가 (0~1)."""
    y = np.asarray(y, float)
    grand = y.mean()
    ssb = sum(((y[labels == g].mean() - grand) ** 2) * (labels == g).sum() for g in np.unique(labels))
    sst = ((y - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


def kruskal_p(labels, y):
    groups = [y[labels == g] for g in np.unique(labels) if (labels == g).sum() > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*groups)
    return float(h), float(p)


def describe_partition(labels, Z, name, k):
    sizes = pd.Series(labels).value_counts()
    subj = df.assign(_l=labels).groupby("_l")["SUBJECT_ID"].nunique()
    other = int((labels == -1).sum())
    row = {
        "partition": name,
        "k": k if k else int(len(sizes)),
        "n_clusters": int(len(sizes)),
        "pct_other": round(other / n * 100, 1),
        "coverage_pct": round((n - other) / n * 100, 1),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "median_size": float(sizes.median()),
        "gini_size": round(gini(sizes.values), 3),
        "min_subjects": int(subj.min()),
        "eta2_diag_count": round(eta2(labels, n_diag), 4),
        "eta2_drug_count": round(eta2(labels, n_drug), 4),
    }
    h, p = kruskal_p(labels, n_diag)
    row["kruskal_H_diag"], row["kruskal_p_diag"] = round(h, 1), p
    h, p = kruskal_p(labels, n_drug)
    row["kruskal_H_drug"], row["kruskal_p_drug"] = round(h, 1), p
    if Z is not None and len(np.unique(labels)) > 1:
        row["silhouette"] = round(float(silhouette_score(Z, labels, sample_size=5000, random_state=SEED)), 4)
        row["calinski_harabasz"] = round(float(calinski_harabasz_score(Z, labels)), 1)
    else:
        row["silhouette"] = np.nan
        row["calinski_harabasz"] = np.nan
    return row


ZMAP = {"id": Z_id, "name": Z_nm}
rows = []
for pname, kmap in partitions.items():
    tag = pname.split("_")[-1]
    Z = ZMAP.get(tag)  # P1 은 None -> 실루엣 계산 안 함(벡터공간 무관 분할)
    for k, labels in kmap.items():
        rows.append(describe_partition(labels, Z if Z is not None else None, pname, k))
        print(f"    {pname} k={k} done", flush=True)

tab1 = pd.DataFrame(rows)
tab1.to_csv(OUT / "table1_partition_metrics.csv", index=False)

# P1 은 두 벡터공간 각각에서 실루엣을 따로 계산해 참고용으로 붙임
p1_sil = {}
for pn in ["P1_ge100", "P1_ge30"]:
    lab = partitions[pn][0]
    p1_sil[pn] = {
        t: round(float(silhouette_score(ZMAP[t], lab, sample_size=5000, random_state=SEED)), 4)
        for t in ["id", "name"]
    }

# ---------------------------------------------------------------- id vs name 일치도
agree = {}
for algo in ["C1", "C2"]:
    agree[algo] = {
        str(k): round(float(adjusted_rand_score(partitions[f"{algo}_id"][k], partitions[f"{algo}_name"][k])), 4)
        for k in KS
    }
# 알고리즘 간 일치도(같은 벡터화)
agree_algo = {
    t: {str(k): round(float(adjusted_rand_score(partitions[f"C1_{t}"][k], partitions[f"C2_{t}"][k])), 4) for k in KS}
    for t in ["id", "name"]
}

# ---------------------------------------------------------------- 안정성 (seed 5개, k-means)
stab = {}
for t in ["id", "name"]:
    for k in [10, 15, 20, 25]:
        labs = [kmeans_labels(ZMAP[t], k, seed=s) for s in range(5)]
        aris = [
            adjusted_rand_score(labs[a], labs[b]) for a in range(5) for b in range(a + 1, 5)
        ]
        stab[f"C1_{t}_k{k}"] = {"mean_pairwise_ARI": round(float(np.mean(aris)), 3), "min": round(float(np.min(aris)), 3)}

# ---------------------------------------------------------------- §4 클러스터 프로파일
def profile(labels, topn=5):
    out = []
    for g in sorted(set(labels)):
        m = labels == g
        cnt = {}
        for l in df.loc[m, "diagnose_l"]:
            for d in set(l):
                cnt[d] = cnt.get(d, 0) + 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:topn]
        out.append(
            {
                "cluster": int(g),
                "visits": int(m.sum()),
                "subjects": int(df.loc[m, "SUBJECT_ID"].nunique()),
                "mean_n_diag": round(float(n_diag[m].mean()), 1),
                "mean_n_drug": round(float(n_drug[m].mean()), 1),
                "top_diagnoses": [f"{d} ({c/m.sum()*100:.0f}%)" for d, c in top],
            }
        )
    return out


profiles = {}
for pn in ["P1_ge100", "P1_ge30"]:
    profiles[f"{pn}"] = profile(partitions[pn][0])
for pn in ["C1_id", "C2_id", "C1_name", "C2_name"]:
    for k in [10, 15, 20, 25]:
        profiles[f"{pn}_k{k}"] = profile(partitions[pn][k])

# ---------------------------------------------------------------- 저장
np.savez_compressed(OUT / "embeddings.npz", Z_id=Z_id, Z_nm=Z_nm)
recs = []
for pname, kmap in partitions.items():
    for k, labels in kmap.items():
        recs.append(
            pd.DataFrame(
                {
                    "SUBJECT_ID": df["SUBJECT_ID"],
                    "HADM_ID": df["HADM_ID"],
                    "partition": pname,
                    "k": k,
                    "cluster_id": labels,
                }
            )
        )
assign = pd.concat(recs, ignore_index=True)
assign.to_csv(OUT / "cluster_assignments.csv", index=False)
assign.to_pickle(OUT / "cluster_assignments.pkl")

meta = {
    "vectorization": vect_info,
    "diag_drug_count_corr": {
        "pearson_r": round(float(stats.pearsonr(n_diag, n_drug).statistic), 4),
        "spearman_rho": round(float(stats.spearmanr(n_diag, n_drug).statistic), 4),
    },
    "p1_silhouette_reference": p1_sil,
    "ari_id_vs_name": agree,
    "ari_kmeans_vs_ward": agree_algo,
    "kmeans_seed_stability": stab,
    "cluster_profiles": profiles,
}
with open(OUT / "11_cluster_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

print(tab1.to_string(index=False))
print("\nsaved: table1_partition_metrics.csv, cluster_assignments.csv, 11_cluster_meta.json")
