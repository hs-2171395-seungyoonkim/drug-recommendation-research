"""트랙 A — 전체 4,514 ICD 코드 어휘 군집화 vs 필터본(1,454) C1_id k=15.

판정 기준은 seed 노이즈다. C1_id k=15 의 seed 간 평균 ARI(0.599)보다 두 버전 사이 ARI 가
낮으면 어휘 필터가 구조를 실제로 바꾼 것이고, 비슷하거나 높으면 seed 흔들림 수준이다.

파이프라인/지표 함수는 03_cluster.py, 04_power_baseline.py, 14_permutation.py 에서 그대로 가져왔다.
산출: out/table6_trackA_compare.csv, out/table7_trackA_clusters.csv,
      out/21_trackA_meta.json, out/trackA_assignments.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
K = 15
N_PERM = 10000
MIN_TEST_VISITS = 30

df = pd.read_pickle(OUT / "20_icd_features.pkl")          # parsed.pkl 원본 순서
n = len(df)
n_diag = df["diag_id_l"].map(len).to_numpy(float)          # 03 과 동일 정의(협업자 어휘 기준)
n_drug = df["n_drugs"].to_numpy(float)
n_icd = df["n_icd"].to_numpy(float)

# ---------------------------------------------------------------- 벡터화 (03 과 동일)
def multihot(list_col, vocab=None):
    items = sorted({v for l in list_col for v in l}) if vocab is None else vocab
    idx = {v: j for j, v in enumerate(items)}
    X = np.zeros((len(list_col), len(items)), dtype=np.float32)
    for i, l in enumerate(list_col):
        for v in l:
            X[i, idx[v]] = 1.0
    return X, items


def embed(X_raw, n_comp=100, seed=SEED):
    Xn = normalize(X_raw, norm="l2", axis=1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Z = svd.fit_transform(Xn)
    return normalize(Z, norm="l2", axis=1), svd


def kmeans_labels(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


X_full = np.asarray(sparse.load_npz(OUT / "20_icd_multihot.npz").todense(), dtype=np.float32)
X_filt, vocab_filt = multihot(df["diag_id_l"])
print(f"[.] full {X_full.shape}  filtered {X_filt.shape}", flush=True)

Z_full, svd_full = embed(X_full)
Z_filt, svd_filt = embed(X_filt)
del X_full

lab_full = {s: kmeans_labels(Z_full, K, seed=s) for s in range(5)}
lab_filt = {s: kmeans_labels(Z_filt, K, seed=s) for s in range(5)}
print("[.] kmeans done", flush=True)

# ---------------------------------------------------------------- 재현성 확인: 필터본 seed0 == 저장된 C1_id k=15
a = pd.read_pickle(OUT / "cluster_assignments.pkl")
sub = a[(a["partition"] == "C1_id") & (a["k"] == 15)]
stored = df["HADM_ID"].map(dict(zip(sub["HADM_ID"], sub["cluster_id"]))).to_numpy()
repro_ari = float(adjusted_rand_score(stored, lab_filt[0]))

# ---------------------------------------------------------------- ARI 비교
def pair_aris(labs):
    return [adjusted_rand_score(labs[i], labs[j]) for i in range(5) for j in range(i + 1, 5)]


within_full = pair_aris(lab_full)
within_filt = pair_aris(lab_filt)
cross = [adjusted_rand_score(lab_filt[i], lab_full[j]) for i in range(5) for j in range(5)]
cross_same_seed = [adjusted_rand_score(lab_filt[s], lab_full[s]) for s in range(5)]

ari_cmp = {
    "within_filtered_mean": round(float(np.mean(within_filt)), 3),
    "within_filtered_min": round(float(np.min(within_filt)), 3),
    "within_full_mean": round(float(np.mean(within_full)), 3),
    "within_full_min": round(float(np.min(within_full)), 3),
    "cross_mean_25pairs": round(float(np.mean(cross)), 3),
    "cross_min": round(float(np.min(cross)), 3),
    "cross_max": round(float(np.max(cross)), 3),
    "cross_same_seed": [round(float(x), 3) for x in cross_same_seed],
    "ari_stored_C1id15_vs_full_seed0": round(float(adjusted_rand_score(stored, lab_full[0])), 3),
    "nmi_stored_C1id15_vs_full_seed0": round(float(normalized_mutual_info_score(stored, lab_full[0])), 3),
    "reproduction_ari_filtered_seed0_vs_stored": round(repro_ari, 4),
    "seed_noise_reference_from_11_cluster_meta": 0.599,
}
print(json.dumps(ari_cmp, ensure_ascii=False, indent=2), flush=True)

# ---------------------------------------------------------------- 03 의 지표 함수 (그대로)
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    m = len(x)
    return float((2 * np.arange(1, m + 1) - m - 1).dot(x) / (m * x.sum()))


def eta2(labels, y):
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
        "partition": name, "k": k if k else int(len(sizes)), "n_clusters": int(len(sizes)),
        "pct_other": round(other / n * 100, 1), "coverage_pct": round((n - other) / n * 100, 1),
        "min_size": int(sizes.min()), "max_size": int(sizes.max()), "median_size": float(sizes.median()),
        "gini_size": round(gini(sizes.values), 3), "min_subjects": int(subj.min()),
        "eta2_diag_count": round(eta2(labels, n_diag), 4), "eta2_drug_count": round(eta2(labels, n_drug), 4),
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
    row["eta2_icd_count"] = round(eta2(labels, n_icd), 4)
    return row


desc = pd.DataFrame([
    describe_partition(lab_filt[0], Z_filt, "C1_id k=15 (필터본 1454)", K),
    describe_partition(lab_full[0], Z_full, "C1_icdfull k=15 (전체 4514)", K),
])

# ---------------------------------------------------------------- 04 의 평가 함수 (그대로)
test = df[df["split"] == "test"].reset_index(drop=True)
split_arr = df["split"].to_numpy()


def boot_ci(sub, col, n_boot=1000, seed=SEED):
    r = np.random.default_rng(seed)
    subs = sub["SUBJECT_ID"].unique()
    by = {s: g[col].to_numpy() for s, g in sub.groupby("SUBJECT_ID")}
    means = []
    for _ in range(n_boot):
        pick = r.choice(subs, size=len(subs), replace=True)
        vals = np.concatenate([by[s] for s in pick])
        vals = vals[~np.isnan(vals)]
        if len(vals):
            means.append(vals.mean())
    if not means:
        return (np.nan, np.nan)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def evaluate_partition(partition, k, labels, n_boot=1000):
    te = test.copy()
    te["cluster"] = labels[split_arr == "test"]
    rows = []
    for g, sub in te.groupby("cluster"):
        cnt = {}
        for l in sub["diagnose_l"]:
            for d in set(l):
                cnt[d] = cnt.get(d, 0) + 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:3]
        lo1, hi1 = boot_ci(sub, "jac_const", n_boot)
        sp = sub[sub["jac_prev"].notna()]
        lo2, hi2 = boot_ci(sp, "jac_prev", n_boot) if len(sp) > 1 else (np.nan, np.nan)
        rows.append({
            "partition": partition, "k": k, "cluster": int(g), "test_visits": int(len(sub)),
            "test_subjects": int(sub["SUBJECT_ID"].nunique()),
            "mean_n_diag": round(float(sub["n_icd"].mean()), 1),
            "mean_n_drug": round(float(sub["n_drugs"].mean()), 1),
            "mean_rarity_seq1": round(float(sub["rarity_seq1"].mean()), 3),
            "top3_dx": " | ".join(f"{d} {c/len(sub)*100:.0f}%" for d, c in top),
            "jac_const": round(float(sub["jac_const"].mean()), 4), "jac_const_lo": lo1, "jac_const_hi": hi1,
            "n_prev": int(len(sp)),
            "jac_prev": round(float(sp["jac_prev"].mean()), 4) if len(sp) else np.nan,
            "jac_prev_lo": lo2, "jac_prev_hi": hi2,
        })
    return pd.DataFrame(rows)


tabA = evaluate_partition("C1_icdfull", K, lab_full[0])
tabA.to_csv(OUT / "table7_trackA_clusters.csv", index=False)
print(tabA.drop(columns=["top3_dx"]).to_string(index=False), flush=True)

# ---------------------------------------------------------------- 14 의 순열검정 (그대로)
def gap_stats(cluster_of_visit, n_clusters, jc, jp_mask, jp_vals):
    cnt = np.bincount(cluster_of_visit, minlength=n_clusters).astype(float)
    tot = np.bincount(cluster_of_visit, weights=jc, minlength=n_clusters)
    ok = cnt >= MIN_TEST_VISITS
    cl_p = cluster_of_visit[jp_mask]
    cnt_p = np.bincount(cl_p, minlength=n_clusters).astype(float)
    tot_p = np.bincount(cl_p, weights=jp_vals, minlength=n_clusters)
    ok_p = ok & (cnt_p > 0)

    def _rng_wsd(mask, tot_, cnt_):
        if mask.sum() < 2:
            return np.nan, np.nan
        m = tot_[mask] / cnt_[mask]
        w = cnt_[mask]
        mbar = np.sum(w * m) / np.sum(w)
        wsd = np.sqrt(np.sum(w * (m - mbar) ** 2) / np.sum(w))
        return float(m.max() - m.min()), float(wsd)

    r_c, s_c = _rng_wsd(ok, tot, cnt)
    r_p, s_p = _rng_wsd(ok_p, tot_p, cnt_p)
    return r_c, s_c, r_p, s_p, int(ok.sum()), int(ok_p.sum())


def largest_remainder(props, total):
    raw = props * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    return base


def run_partition(name, labels, n_perm=N_PERM, seed=SEED):
    d = df[["SUBJECT_ID", "HADM_ID", "split", "jac_const", "jac_prev"]].copy()
    d["cl"] = labels
    uniq = np.sort(d["cl"].unique())
    d["cl"] = d["cl"].map({c: i for i, c in enumerate(uniq)}).astype(int)
    Kn = len(uniq)
    pats = np.sort(d["SUBJECT_ID"].unique())
    d["pat"] = d["SUBJECT_ID"].map({p: i for i, p in enumerate(pats)}).astype(int)
    n_pat = len(pats)

    te = d[d["split"] == "test"]
    jc = te["jac_const"].to_numpy(float)
    jp = te["jac_prev"].to_numpy(float)
    jp_mask = ~np.isnan(jp)
    jp_vals = jp[jp_mask]
    pat_of_test_visit = te["pat"].to_numpy()
    obs = gap_stats(te["cl"].to_numpy(), Kn, jc, jp_mask, jp_vals)

    pat_per_cl = d.groupby("cl")["SUBJECT_ID"].nunique().reindex(range(Kn), fill_value=0).to_numpy()
    target = largest_remainder(pat_per_cl / pat_per_cl.sum(), n_pat)
    rng = np.random.default_rng(seed)
    null = np.full((n_perm, 4), np.nan)
    for b in range(n_perm):
        perm = rng.permutation(n_pat)
        cl_of_pat = np.empty(n_pat, int)
        s = 0
        for c, sz in enumerate(target):
            cl_of_pat[perm[s:s + sz]] = c
            s += sz
        null[b] = gap_stats(cl_of_pat[pat_of_test_visit], Kn, jc, jp_mask, jp_vals)[:4]
    meta = {
        "partition": name, "n_clusters": Kn, "universe_visits": int(len(d)), "universe_subjects": n_pat,
        "test_visits": int(len(te)), "test_visits_with_prev": int(jp_mask.sum()),
        "subjects_spanning_multiple_clusters": int((d.groupby("SUBJECT_ID")["cl"].nunique() > 1).sum()),
        "n_clusters_ge30_observed": obs[4], "n_clusters_ge30_prev_observed": obs[5],
    }
    return obs[:4], null, meta


STATS = [("const", "range", 0), ("const", "wsd", 1), ("prev", "range", 2), ("prev", "wsd", 3)]
obs, null, pmeta = run_partition("C1_icdfull k=15", lab_full[0])
perm_rows = []
for pred, stat, j in STATS:
    o = obs[j]
    nd = null[:, j]
    nd = nd[~np.isnan(nd)]
    p = (1 + int((nd >= o).sum())) / (1 + len(nd))
    perm_rows.append({
        "partition": "C1_icdfull k=15", "predictor": {"const": "상수", "prev": "copy-prev"}[pred],
        "statistic": {"range": "범위(max-min)", "wsd": "가중 SD"}[stat],
        "observed": round(float(o), 4), "null_mean": round(float(nd.mean()), 4),
        "null_sd": round(float(nd.std(ddof=1)), 4), "null_p95": round(float(np.percentile(nd, 95)), 4),
        "p_value": round(p, 5), "n_perm_valid": len(nd),
        "z_vs_null": round(float((o - nd.mean()) / nd.std(ddof=1)), 2),
    })
perm = pd.DataFrame(perm_rows)
perm.to_csv(OUT / "table8_trackA_perm.csv", index=False)
print(perm.to_string(index=False), flush=True)

# ---------------------------------------------------------------- 비교표 (§A-5)
prev_perm = pd.read_csv(OUT / "table5_perm_10000.csv")
pf = prev_perm[prev_perm["partition"] == "diagnose C1_id k=15"]


def gap_of(t):
    big = t[t["test_visits"] >= MIN_TEST_VISITS]
    return {
        "n_clusters_ge30_test": len(big),
        "jac_const_spread": round(float(big["jac_const"].max() - big["jac_const"].min()), 4),
        "jac_prev_spread": round(float(big["jac_prev"].max() - big["jac_prev"].min()), 4),
        "corr_jacconst_vs_ndrug": round(float(big["jac_const"].corr(big["mean_n_drug"])), 3),
        "corr_jacprev_vs_ndrug": round(float(big["jac_prev"].corr(big["mean_n_drug"])), 3),
    }


tab2 = pd.read_csv(OUT / "table2_cluster_performance.csv")
filt_tab = tab2[(tab2["partition"] == "C1_id") & (tab2["k"] == 15)]
g_filt, g_full = gap_of(filt_tab), gap_of(tabA)
d_filt, d_full = desc.iloc[0], desc.iloc[1]

cmp_rows = [
    {"버전": "필터본 C1_id k=15 (어휘 1,454)", **{
        "커버리지%": d_filt["coverage_pct"], "클러스터 수": d_filt["n_clusters"], "최소 크기": d_filt["min_size"],
        "gini_size": d_filt["gini_size"], "silhouette": d_filt["silhouette"],
        "test 클러스터(>=30)": g_filt["n_clusters_ge30_test"], "상수 격차": g_filt["jac_const_spread"],
        "copy-prev 격차": g_filt["jac_prev_spread"],
        "순열 p(상수·범위)": float(pf[pf["statistic"] == "범위(max-min)"].iloc[0]["p_value"]) if len(pf) else np.nan,
        "순열 p(상수·가중SD)": float(pf[pf["statistic"] == "가중 SD"].iloc[0]["p_value"]) if len(pf) else np.nan,
        "약물수 상관(상수)": g_filt["corr_jacconst_vs_ndrug"], "eta2_drug": d_filt["eta2_drug_count"]}},
    {"버전": "전체본 C1_icdfull k=15 (어휘 4,514)", **{
        "커버리지%": d_full["coverage_pct"], "클러스터 수": d_full["n_clusters"], "최소 크기": d_full["min_size"],
        "gini_size": d_full["gini_size"], "silhouette": d_full["silhouette"],
        "test 클러스터(>=30)": g_full["n_clusters_ge30_test"], "상수 격차": g_full["jac_const_spread"],
        "copy-prev 격차": g_full["jac_prev_spread"],
        "순열 p(상수·범위)": perm_rows[0]["p_value"], "순열 p(상수·가중SD)": perm_rows[1]["p_value"],
        "약물수 상관(상수)": g_full["corr_jacconst_vs_ndrug"], "eta2_drug": d_full["eta2_drug_count"]}},
    {"버전": "ICD chapter 분할 D1", "커버리지%": None, "클러스터 수": None, "최소 크기": None, "gini_size": None,
     "silhouette": None, "test 클러스터(>=30)": None, "상수 격차": None, "copy-prev 격차": None,
     "순열 p(상수·범위)": None, "순열 p(상수·가중SD)": None, "약물수 상관(상수)": None, "eta2_drug": None},
]
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(OUT / "table6_trackA_compare.csv", index=False)
desc.to_csv(OUT / "table6b_trackA_describe.csv", index=False)

pd.DataFrame({"SUBJECT_ID": df["SUBJECT_ID"], "HADM_ID": df["HADM_ID"],
              "partition": "C1_icdfull", "k": K, "cluster_id": lab_full[0]}).to_csv(
    OUT / "trackA_assignments.csv", index=False)

meta = {
    "vectorization": {"full_dim": 4514, "filtered_dim": int(X_filt.shape[1]),
                      "svd_explained_var_full": round(float(svd_full.explained_variance_ratio_.sum()), 4),
                      "svd_explained_var_filtered": round(float(svd_filt.explained_variance_ratio_.sum()), 4)},
    "ari": ari_cmp,
    "describe": desc.to_dict("records"),
    "gap": {"filtered": g_filt, "full": g_full},
    "permutation": {"n_perm": N_PERM, "meta": pmeta, "table": perm_rows},
}
with open(OUT / "21_trackA_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
print("\n" + cmp.to_string(index=False))
print("\nsaved: table6_trackA_compare.csv, table6b_trackA_describe.csv, table7_trackA_clusters.csv, "
      "table8_trackA_perm.csv, trackA_assignments.csv, 21_trackA_meta.json")
