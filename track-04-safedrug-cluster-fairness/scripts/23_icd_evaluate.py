"""§3~§5 — D1/D2/D2b 평가·순열검정·비교표·C1/C10 교차표.

지표 함수는 03_cluster.py / 04_power_baseline.py / 14_permutation.py 에서 그대로 가져왔다
(19_trackA_fullvocab.py 와 동일한 사본). 새로 짠 것은 다음 세 가지뿐이다:
  1) run_partition 에 keep 마스크 인자 추가 — P1_ge100 의 기타(-1) 63.5% 를 순열 우주에서 빼기 위해.
  2) sil_common() — 여섯 분할의 실루엣을 같은 공간(전체 어휘 SVD)에서 재는 비교용 열.
  3) C1/C10 교차표.

산출: out/table14_icd_clusters.csv, table15_icd_perm.csv, table16_icd_compare.csv,
      table17_c1c10_crosstab.csv, out/24_icd_eval_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
N_PERM = 10000
MIN_TEST_VISITS = 30

df = pd.read_pickle(OUT / "20_icd_features.pkl")
n = len(df)
n_diag = df["diag_id_l"].map(len).to_numpy(float)
n_drug = df["n_drugs"].to_numpy(float)
n_icd = df["n_icd"].to_numpy(float)

part = pd.read_csv(OUT / "icd_partition_assignments.csv")
assert (part["HADM_ID"].to_numpy() == df["HADM_ID"].to_numpy()).all(), "행 순서 불일치"

# ---------------------------------------------------------------- 라벨 준비
def encode(series):
    """문자열 라벨 → 0..K-1 정수. 기존 함수들이 정수 라벨을 전제한다."""
    names = sorted(pd.unique(series.astype(str)))
    m = {v: i for i, v in enumerate(names)}
    return series.astype(str).map(m).to_numpy(int), names


LAB, NAMES = {}, {}
for tag, col in [("D1", "d1_label"), ("D2", "d2_label"), ("D2b", "d2b_label")]:
    LAB[tag], NAMES[tag] = encode(part[col])

a = pd.read_pickle(OUT / "cluster_assignments.pkl")
sub = a[(a["partition"] == "C1_id") & (a["k"] == 15)]
LAB["C1_id k=15"] = df["HADM_ID"].map(dict(zip(sub["HADM_ID"], sub["cluster_id"]))).to_numpy(int)
subp = a[a["partition"] == "P1_ge100"]
LAB["P1_ge100"] = df["HADM_ID"].map(dict(zip(subp["HADM_ID"], subp["cluster_id"]))).to_numpy(int)
trA = pd.read_csv(OUT / "trackA_assignments.csv")
LAB["C1_icdfull k=15"] = df["HADM_ID"].map(dict(zip(trA["HADM_ID"], trA["cluster_id"]))).to_numpy(int)

# ---------------------------------------------------------------- 임베딩 (19 와 동일)
def embed(X_raw, n_comp=100, seed=SEED):
    Xn = normalize(X_raw, norm="l2", axis=1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Z = svd.fit_transform(Xn)
    return normalize(Z, norm="l2", axis=1), svd


X_full = np.asarray(sparse.load_npz(OUT / "20_icd_multihot.npz").todense(), dtype=np.float32)
Z_full, _ = embed(X_full)
del X_full
print("[.] 임베딩 완료", flush=True)


def sil_common(labels):
    """여섯 분할 공통 비교용 실루엣 — 전체 어휘 SVD 공간, 기타(-1) 제외."""
    m = labels != -1
    if len(np.unique(labels[m])) < 2:
        return np.nan
    return round(float(silhouette_score(Z_full[m], labels[m], sample_size=5000, random_state=SEED)), 4)


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


# ---------------------------------------------------------------- 14 의 순열검정 (통계량은 그대로, keep 인자만 추가)
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


def run_partition(name, labels, n_perm=N_PERM, seed=SEED, keep=None):
    keep = np.ones(n, bool) if keep is None else keep
    d = df.loc[keep, ["SUBJECT_ID", "HADM_ID", "split", "jac_const", "jac_prev"]].copy()
    d["cl"] = labels[keep]
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


# ---------------------------------------------------------------- §3 실행
desc_rows, eval_tabs, perm_rows, perm_meta = [], [], [], {}
STATS = [("const", "range", 0), ("const", "wsd", 1), ("prev", "range", 2), ("prev", "wsd", 3)]

for tag in ["D1", "D2", "D2b"]:
    lab = LAB[tag]
    r = describe_partition(lab, Z_full, tag, 0)
    r["sil_common"] = sil_common(lab)
    desc_rows.append(r)
    t = evaluate_partition(tag, 0, lab)
    t["라벨"] = [NAMES[tag][c] for c in t["cluster"]]
    eval_tabs.append(t)
    obs, null, pm = run_partition(tag, lab)
    perm_meta[tag] = pm
    np.savez_compressed(OUT / f"perm_null_icd_{tag}.npz", null=null)
    for pred, stat, j in STATS:
        o, nd = obs[j], null[:, j][~np.isnan(null[:, j])]
        perm_rows.append({
            "분할": tag, "예측기": {"const": "상수", "prev": "copy-prev"}[pred],
            "통계량": {"range": "범위(max-min)", "wsd": "가중 SD"}[stat],
            "observed": round(float(o), 4), "null_mean": round(float(nd.mean()), 4),
            "null_sd": round(float(nd.std(ddof=1)), 4), "null_p95": round(float(np.percentile(nd, 95)), 4),
            "p_value": round((1 + int((nd >= o).sum())) / (1 + len(nd)), 5), "n_perm_valid": int(len(nd)),
            "z_vs_null": round(float((o - nd.mean()) / nd.std(ddof=1)), 2),
        })
    print(f"[.] {tag} 완료 (라벨 {pm['n_clusters']}, test>=30 {pm['n_clusters_ge30_observed']})", flush=True)

# P1_ge100 은 기타(-1) 63.5% 를 우주에서 빼고 돌린다 — 커버된 방문들만의 분할로 취급
keep_p1 = LAB["P1_ge100"] != -1
obs_p1, null_p1, pm_p1 = run_partition("P1_ge100", LAB["P1_ge100"], keep=keep_p1)
perm_meta["P1_ge100"] = pm_p1
np.savez_compressed(OUT / "perm_null_icd_P1_ge100.npz", null=null_p1)
for pred, stat, j in STATS:
    o, nd = obs_p1[j], null_p1[:, j][~np.isnan(null_p1[:, j])]
    perm_rows.append({
        "분할": "P1_ge100 (기타 제외)", "예측기": {"const": "상수", "prev": "copy-prev"}[pred],
        "통계량": {"range": "범위(max-min)", "wsd": "가중 SD"}[stat],
        "observed": round(float(o), 4), "null_mean": round(float(nd.mean()), 4),
        "null_sd": round(float(nd.std(ddof=1)), 4), "null_p95": round(float(np.percentile(nd, 95)), 4),
        "p_value": round((1 + int((nd >= o).sum())) / (1 + len(nd)), 5), "n_perm_valid": int(len(nd)),
        "z_vs_null": round(float((o - nd.mean()) / nd.std(ddof=1)), 2),
    })
print("[.] P1_ge100 완료", flush=True)

clusters = pd.concat(eval_tabs, ignore_index=True)
clusters.to_csv(OUT / "table14_icd_clusters.csv", index=False)
perm = pd.DataFrame(perm_rows)
perm.to_csv(OUT / "table15_icd_perm.csv", index=False)
desc = pd.DataFrame(desc_rows)

# ---------------------------------------------------------------- §4 비교표
tab2 = pd.read_csv(OUT / "table2_cluster_performance.csv")
tab7 = pd.read_csv(OUT / "table7_trackA_clusters.csv")
tab1 = pd.read_csv(OUT / "table1_partition_metrics.csv")
tab6b = pd.read_csv(OUT / "table6b_trackA_describe.csv")
perm10k = pd.read_csv(OUT / "table5_perm_10000.csv")
permA = pd.read_csv(OUT / "table8_trackA_perm.csv")


def gap_of(t):
    big = t[t["test_visits"] >= MIN_TEST_VISITS]
    return {
        "test_labels_ge30": int(len(big)),
        "const_gap": round(float(big["jac_const"].max() - big["jac_const"].min()), 4),
        "prev_gap": round(float(big["jac_prev"].max() - big["jac_prev"].min()), 4),
        "corr_const_ndrug": round(float(big["jac_const"].corr(big["mean_n_drug"])), 3),
        "bottom": big.nsmallest(3, "jac_const"),
        "top": big.nlargest(3, "jac_const"),
    }


PERF = {
    "D1": clusters[clusters["partition"] == "D1"], "D2": clusters[clusters["partition"] == "D2"],
    "D2b": clusters[clusters["partition"] == "D2b"],
    "C1_id k=15": tab2[(tab2["partition"] == "C1_id") & (tab2["k"] == 15)],
    "C1_icdfull k=15": tab7,
    "P1_ge100": tab2[(tab2["partition"] == "P1_ge100")],
}
GAPS = {k: gap_of(v) for k, v in PERF.items()}


def pget(table, partition, stat):
    r = table[(table["partition"] == partition) & (table["statistic"] == stat) & (table["predictor"] == "상수")]
    return float(r.iloc[0]["p_value"]) if len(r) else np.nan


def pget_new(tag, stat):
    r = perm[(perm["분할"] == tag) & (perm["통계량"] == stat) & (perm["예측기"] == "상수")]
    return float(r.iloc[0]["p_value"]) if len(r) else np.nan


cov = {"D1": 100.0, "D2": 100.0, "D2b": 100.0, "C1_id k=15": 100.0, "C1_icdfull k=15": 100.0,
       "P1_ge100": float(tab1[(tab1["partition"] == "P1_ge100")].iloc[0]["coverage_pct"])}
nlab = {t: int(desc[desc["partition"] == t].iloc[0]["n_clusters"]) for t in ["D1", "D2", "D2b"]}
nlab.update({"C1_id k=15": 15, "C1_icdfull k=15": 15,
             "P1_ge100": int(tab1[(tab1["partition"] == "P1_ge100")].iloc[0]["n_clusters"])})
minsz = {t: int(desc[desc["partition"] == t].iloc[0]["min_size"]) for t in ["D1", "D2", "D2b"]}
minsz.update({
    "C1_id k=15": int(tab1[(tab1["partition"] == "C1_id") & (tab1["k"] == 15)].iloc[0]["min_size"]),
    "C1_icdfull k=15": int(tab6b.iloc[1]["min_size"]),
    "P1_ge100": int(tab1[(tab1["partition"] == "P1_ge100")].iloc[0]["min_size"])})
eta_drug = {t: float(desc[desc["partition"] == t].iloc[0]["eta2_drug_count"]) for t in ["D1", "D2", "D2b"]}
eta_drug.update({
    "C1_id k=15": float(tab1[(tab1["partition"] == "C1_id") & (tab1["k"] == 15)].iloc[0]["eta2_drug_count"]),
    "C1_icdfull k=15": float(tab6b.iloc[1]["eta2_drug_count"]),
    "P1_ge100": float(tab1[(tab1["partition"] == "P1_ge100")].iloc[0]["eta2_drug_count"])})
sil = {t: float(desc[desc["partition"] == t].iloc[0]["sil_common"]) for t in ["D1", "D2", "D2b"]}
sil.update({k: sil_common(LAB[k]) for k in ["C1_id k=15", "C1_icdfull k=15", "P1_ge100"]})
pr = {t: (pget_new(t, "범위(max-min)"), pget_new(t, "가중 SD")) for t in ["D1", "D2", "D2b"]}
pr["P1_ge100"] = (pget_new("P1_ge100 (기타 제외)", "범위(max-min)"), pget_new("P1_ge100 (기타 제외)", "가중 SD"))
pr["C1_id k=15"] = (pget(perm10k, "diagnose C1_id k=15", "범위(max-min)"),
                    pget(perm10k, "diagnose C1_id k=15", "가중 SD"))
pr["C1_icdfull k=15"] = (pget(permA, "C1_icdfull k=15", "범위(max-min)"), pget(permA, "C1_icdfull k=15", "가중 SD"))

ORDER = ["D1", "D2", "D2b", "C1_id k=15", "C1_icdfull k=15", "P1_ge100"]
LABEL_KO = {"D1": "D1 (ICD chapter)", "D2": "D2 (D1 + 순환기 8블록)", "D2b": "D2b (D2 + 420–429 3자리)",
            "C1_id k=15": "C1_id k=15 (동반질환 군집)", "C1_icdfull k=15": "C1_icdfull k=15 (전체 어휘 군집)",
            "P1_ge100": "P1_ge100 (주진단명 ≥100)"}
cmp_rows = []
for t in ORDER:
    g = GAPS[t]
    cmp_rows.append({
        "분할": LABEL_KO[t], "커버리지%": cov[t], "라벨 수": nlab[t], "test≥30 라벨 수": g["test_labels_ge30"],
        "최소 크기": minsz[t], "실루엣(공통공간)": sil[t], "상수 격차": g["const_gap"],
        "순열 p(상수·범위)": pr[t][0], "순열 p(상수·가중SD)": pr[t][1],
        "copy-prev 격차": g["prev_gap"], "약물수 상관(상수)": g["corr_const_ndrug"],
        "η²(약물수)": eta_drug[t],
    })
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(OUT / "table16_icd_compare.csv", index=False)

# ---------------------------------------------------------------- §5-5 C1/C10 교차표
c1 = LAB["C1_id k=15"]
ct_rows = []
for cl in [1, 10]:
    m = c1 == cl
    for tag in ["D1", "D2"]:
        vc = pd.Series([NAMES[tag][x] for x in LAB[tag][m]]).value_counts()
        for lab_name, cnt in vc.items():
            share_in_label = cnt / int((np.array([NAMES[tag][x] for x in LAB[tag]]) == lab_name).sum()) * 100
            ct_rows.append({"C1_id 클러스터": cl, "분할": tag, "라벨": lab_name, "방문 수": int(cnt),
                            "클러스터 내 비중%": round(cnt / m.sum() * 100, 1),
                            "그 라벨 안에서 이 클러스터가 차지하는 비중%": round(share_in_label, 1)})
ct = pd.DataFrame(ct_rows)
ct.to_csv(OUT / "table17_c1c10_crosstab.csv", index=False)

# 바닥 라벨 확인 (§5-5)
bottom = {}
for t in ["D1", "D2"]:
    big = PERF[t][PERF[t]["test_visits"] >= MIN_TEST_VISITS].sort_values("jac_const")
    bottom[t] = big[["라벨", "test_visits", "jac_const", "jac_const_lo", "jac_const_hi", "mean_n_drug"]].to_dict("records")

meta = {
    "min_test_visits": MIN_TEST_VISITS, "n_perm": N_PERM,
    "describe": desc.to_dict("records"),
    "permutation_meta": perm_meta,
    "permutation_table": perm_rows,
    "compare": cmp_rows,
    "gaps": {k: {kk: vv for kk, vv in v.items() if kk not in ("bottom", "top")} for k, v in GAPS.items()},
    "bottom_labels": bottom,
    "c1_c10_crosstab": ct_rows,
    "label_names": NAMES,
}
with open(OUT / "24_icd_eval_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

pd.set_option("display.width", 250)
print("\n===== §4 비교표 =====")
print(cmp.to_string(index=False))
print("\n===== 순열검정 (상수 예측기만) =====")
print(perm[perm["예측기"] == "상수"].to_string(index=False))
print("\n===== D1 라벨별 성능 (test>=30, 낮은 순) =====")
print(PERF["D1"][PERF["D1"]["test_visits"] >= 30].sort_values("jac_const")[
    ["라벨", "test_visits", "jac_const", "jac_const_lo", "jac_const_hi", "jac_prev", "mean_n_drug", "mean_rarity_seq1"]
].to_string(index=False))
print("\n===== D2 라벨별 성능 (test>=30, 낮은 순) =====")
print(PERF["D2"][PERF["D2"]["test_visits"] >= 30].sort_values("jac_const")[
    ["라벨", "test_visits", "jac_const", "jac_const_lo", "jac_const_hi", "jac_prev", "mean_n_drug", "mean_rarity_seq1"]
].to_string(index=False))
print("\n===== C1/C10 교차표 (상위 6행씩) =====")
for cl in [1, 10]:
    print(ct[(ct["C1_id 클러스터"] == cl) & (ct["분할"] == "D1")].head(6).to_string(index=False))
print("\nsaved: table14_icd_clusters.csv, table15_icd_perm.csv, table16_icd_compare.csv, "
      "table17_c1c10_crosstab.csv, 24_icd_eval_meta.json")
