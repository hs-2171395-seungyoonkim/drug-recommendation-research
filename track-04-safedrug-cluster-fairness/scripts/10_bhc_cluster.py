"""
§3 군집화 + §4 판정. 표현 4종: CB(masked/clean), TFIDF(masked/clean).
- k-means k=5~30 스윕: 실루엣, η²(진단수/약물수)
- 판정 A: 길이 더미 분할과의 ARI
- 판정 C: diagnose C1_id k=15 와의 ARI/NMI
- 판정 D: P1_ge100 (첫 진단) 와의 ARI/NMI — 전체 & 기타 제외
- 민감도: masked vs clean, CB vs TFIDF
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
KS = list(range(5, 31))
CAND_KS = [10, 15, 20, 25]

bhc = pd.read_pickle(OUT / "bhc_prep.pkl").reset_index(drop=True)
n = len(bhc)

# ---------------- 표현 4종 ----------------
reps = {}
for t in ["masked", "clean"]:
    z = np.load(OUT / f"emb_bhc_{t}.npz")
    assert (z["HADM_ID"] == bhc["HADM_ID"].to_numpy()).all()
    reps[f"CB_{t}"] = z["E"]

for t in ["masked", "clean"]:
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
    X = tf.fit_transform(bhc[f"bhc_{t}"])
    svd = TruncatedSVD(n_components=100, random_state=SEED)
    Z = normalize(svd.fit_transform(X))
    reps[f"TFIDF_{t}"] = Z.astype(np.float32)
    print(f"TFIDF_{t}: vocab={X.shape[1]}, svd_var={svd.explained_variance_ratio_.sum():.3f}", flush=True)

# ---------------- 기준 파티션 ----------------
assign = pd.read_pickle(OUT / "cluster_assignments.pkl")


def ref_labels(pn, k):
    s = assign[(assign.partition == pn) & (assign.k == k)]
    return bhc["HADM_ID"].map(dict(zip(s.HADM_ID, s.cluster_id))).to_numpy()


ref_C1 = ref_labels("C1_id", 15)
ref_P1 = ref_labels("P1_ge100", 0)
p1_mask = ref_P1 != -1  # 기타 제외용 (36.5%)

n_diag = bhc["n_diag"].to_numpy(float)
n_drug = bhc["n_drug"].to_numpy(float)
wlen = bhc["bhc_clean"].str.split().map(len).to_numpy(float)


def len_dummy(k):
    """BHC 단어수 분위수로만 나눈 더미 분할."""
    q = np.quantile(wlen, np.linspace(0, 1, k + 1))
    q[0], q[-1] = -np.inf, np.inf
    return np.digitize(wlen, q[1:-1])


def eta2(labels, y):
    grand = y.mean()
    ssb = sum(((y[labels == g].mean() - grand) ** 2) * (labels == g).sum() for g in np.unique(labels))
    return float(ssb / ((y - grand) ** 2).sum())


# ---------------- 스윕 ----------------
km_cache = {}


def kml(rep, k, seed=SEED):
    key = (rep, k, seed)
    if key not in km_cache:
        km_cache[key] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(reps[rep])
    return km_cache[key]


rows = []
for rep in reps:
    for k in KS:
        lab = kml(rep, k)
        sizes = pd.Series(lab).value_counts()
        rows.append({
            "rep": rep, "k": k,
            "min_size": int(sizes.min()), "max_size": int(sizes.max()),
            "silhouette": round(float(silhouette_score(reps[rep], lab, sample_size=5000, random_state=SEED)), 4),
            "eta2_diag_count": round(eta2(lab, n_diag), 4),
            "eta2_drug_count": round(eta2(lab, n_drug), 4),
            "eta2_notelen": round(eta2(lab, wlen), 4),
            "ARI_lendummy": round(float(ari(lab, len_dummy(k))), 4),
            "ARI_C1id15": round(float(ari(lab, ref_C1)), 4),
            "NMI_C1id15": round(float(nmi(lab, ref_C1)), 4),
            "ARI_P1_full": round(float(ari(lab, ref_P1)), 4),
            "ARI_P1_labeled_only": round(float(ari(lab[p1_mask], ref_P1[p1_mask])), 4),
            "NMI_P1_labeled_only": round(float(nmi(lab[p1_mask], ref_P1[p1_mask])), 4),
        })
        print(f"{rep} k={k} done", flush=True)
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "tableA_bhc_sweep.csv", index=False)

# ---------------- 민감도 ----------------
sens = {}
for k in CAND_KS:
    sens[str(k)] = {
        "ARI_CBmasked_vs_CBclean": round(float(ari(kml("CB_masked", k), kml("CB_clean", k))), 4),
        "ARI_CBmasked_vs_TFIDFmasked": round(float(ari(kml("CB_masked", k), kml("TFIDF_masked", k))), 4),
        "ARI_TFIDFmasked_vs_TFIDFclean": round(float(ari(kml("TFIDF_masked", k), kml("TFIDF_clean", k))), 4),
    }

# seed 안정성 (CB_masked, TFIDF_masked)
stab = {}
for rep in ["CB_masked", "TFIDF_masked"]:
    for k in CAND_KS:
        labs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(reps[rep]) for s in range(5)]
        aris = [ari(labs[a], labs[b]) for a in range(5) for b in range(a + 1, 5)]
        stab[f"{rep}_k{k}"] = {"mean": round(float(np.mean(aris)), 3), "min": round(float(np.min(aris)), 3)}

# ---------------- 표 B: CB_masked k=15 프로파일 ----------------
parsed = pd.read_pickle(OUT / "parsed.pkl")[["HADM_ID", "diagnose_l"]]
bhc2 = bhc.merge(parsed, on="HADM_ID")


def profile(labels, topn=3):
    out = []
    for g in sorted(set(labels)):
        m = labels == g
        cnt = {}
        for l in bhc2.loc[m, "diagnose_l"]:
            for d in set(l):
                cnt[d] = cnt.get(d, 0) + 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:topn]
        out.append({
            "cluster": int(g), "visits": int(m.sum()),
            "mean_n_diag": round(float(n_diag[m].mean()), 1),
            "mean_n_drug": round(float(n_drug[m].mean()), 1),
            "mean_words": int(wlen[m].mean()),
            "top_dx": " | ".join(f"{d} {c/m.sum()*100:.0f}%" for d, c in top),
        })
    return out


profiles = {f"CB_masked_k{k}": profile(kml("CB_masked", k)) for k in [15]}
profiles["TFIDF_masked_k15"] = profile(kml("TFIDF_masked", 15))

# ---------------- 배정 저장 (§8 형식) ----------------
recs = []
for rep in reps:
    for k in CAND_KS:
        recs.append(pd.DataFrame({
            "SUBJECT_ID": bhc["SUBJECT_ID"], "HADM_ID": bhc["HADM_ID"],
            "partition": f"BHC_{rep}", "k": k, "cluster_id": kml(rep, k),
        }))
pd.concat(recs, ignore_index=True).to_csv(OUT / "cluster_assignments_bhc.csv", index=False)

with open(OUT / "15_bhc_cluster_meta.json", "w", encoding="utf-8") as f:
    json.dump({"sensitivity_ari": sens, "seed_stability": stab, "profiles": profiles},
              f, ensure_ascii=False, indent=2)

print(sweep[sweep.k.isin([5, 10, 15, 20, 25, 30])].to_string(index=False))
print(json.dumps(sens, indent=1))
