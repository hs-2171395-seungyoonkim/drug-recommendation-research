"""§3 동반질환 제거 후 군집화 + §4 판정 — 축이 실제로 옮겨갔는가.

파이프라인은 C1_id 와 완전히 동일하다: multi-hot → L2 → TruncatedSVD(100, seed 0) → L2
→ KMeans(k, n_init=10, seed 0). 지표/평가 함수는 19_trackA_fullvocab.py 에서 그대로 가져왔다.

어휘 컬럼은 모든 정의에서 4,514 로 고정한다(제거된 코드는 0 열이 되거나, 주 진단 보존분만 남는다).
그래야 SVD 차원이 같은 조건에서 비교된다.

주 버전 4개는 주 진단 보존. 비보존 2개는 k=15 민감도로만 돌린다.

산출: out/28_chronic_labels.pkl, out/chronic_partition_assignments.csv,
      out/table29_chronic_describe.csv, out/table30_chronic_ksweep.csv,
      out/table31_chronic_seed.csv, out/table32_chronic_axis.csv,
      out/table33_chronic_clusters.csv, out/table34_chronic_toplift.csv,
      out/28_chronic_cluster_meta.json
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
MIN_TEST_VISITS = 30
KSWEEP = list(range(5, 31))
TOP_LIFT = 5
MIN_PREV_LIFT = 0.05        # lift 상위 진단 후보의 최소 클러스터 내 유병률

# ---------------------------------------------------------------- 입력
df = pd.read_pickle(OUT / "20_icd_features.pkl")
n = len(df)
n_diag = df["diag_id_l"].map(len).to_numpy(float)
n_drug = df["n_drugs"].to_numpy(float)
n_icd = df["n_icd"].to_numpy(float)

CH = pd.read_pickle(OUT / "27_chronic_lists.pkl")
VOCAB = CH["vocab"]
IDF = CH["idf"]
TAU_MAIN, TAU_SCALE = CH["tau_main"], CH["tau_scale"]
SETS = json.load(open(OUT / "27_chronic_sets.json", encoding="utf-8"))
ELIX_VOCAB = set(SETS["elix"]["removed_vocab"])
CCI_CHRONIC = SETS["cci"]["chronic"]      # 코드 → 만성 여부 (AHRQ CCI 2015, 외부 표준)

dic = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
SHORT = dict(zip(dic["ICD9_CODE"].str.strip(), dic["SHORT_TITLE"].astype(str)))

MAIN = ["COMORB-elix", f"CHR-persist(τ={TAU_MAIN})", f"CHR-persist(τ={TAU_SCALE}, 규모정합)", "CHR-idf"]
SENS = ["COMORB-elix (주진단 비보존)", f"CHR-persist(τ={TAU_MAIN}, 주진단 비보존)"]
SHORTNAME = {MAIN[0]: "COMORB-elix", MAIN[1]: f"CHR-persist{TAU_MAIN}", MAIN[2]: f"CHR-persist{TAU_SCALE}",
             MAIN[3]: "CHR-idf", SENS[0]: "COMORB-elix-rmseq1", SENS[1]: f"CHR-persist{TAU_MAIN}-rmseq1"}

# 기존 분할 (비교 기준)
a = pd.read_pickle(OUT / "cluster_assignments.pkl")
sub = a[(a["partition"] == "C1_id") & (a["k"] == 15)]
C1 = df["HADM_ID"].map(dict(zip(sub["HADM_ID"], sub["cluster_id"]))).to_numpy()
ip = pd.read_csv(OUT / "icd_partition_assignments.csv")
D1_NAME = df["HADM_ID"].map(dict(zip(ip["HADM_ID"], ip["d1_label"])))
D1 = pd.factorize(D1_NAME)[0]
ta = pd.read_csv(OUT / "trackA_assignments.csv")
ta = ta[(ta["partition"] == "C1_icdfull") & (ta["k"] == 15)]
CFULL = df["HADM_ID"].map(dict(zip(ta["HADM_ID"], ta["cluster_id"]))).to_numpy()

# ---------------------------------------------------------------- 벡터화 (19 와 동일)
def multihot(list_col, vocab=None):
    items = sorted({v for l in list_col for v in l}) if vocab is None else vocab
    idx = {v: j for j, v in enumerate(items)}
    X = np.zeros((len(list_col), len(items)), dtype=np.float32)
    for i, l in enumerate(list_col):
        for v in l:
            if v in idx:
                X[i, idx[v]] = 1.0
    return X, items


def embed(X_raw, n_comp=100, seed=SEED):
    Xn = normalize(X_raw, norm="l2", axis=1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Z = svd.fit_transform(Xn)
    return normalize(Z, norm="l2", axis=1), svd


def kmeans_labels(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


Z_OF, ZERO_OF, NLEFT_OF = {}, {}, {}
for name in MAIN + SENS:
    lists = CH["lists"][name]
    X, _ = multihot(lists, VOCAB)
    if name == "CHR-idf":
        X = X * IDF.astype(np.float32)
    zero = np.array([len(l) == 0 for l in lists])
    Z, svd = embed(X)
    Z_OF[name], ZERO_OF[name] = Z, zero
    NLEFT_OF[name] = CH["n_left"][name]
    print(f"[.] {name:38s} X{X.shape} 남은진단0={zero.sum():3d} SVD설명분산={svd.explained_variance_ratio_.sum():.3f}",
          flush=True)
    del X

# 남은 진단이 0 개인 방문 처리 방침 — 군집에서 빼지 않고 그대로 배정하되, 라벨 -1 로 따로 표시한다.
# (주 버전 4개는 0 개 방문이 없으므로 실제로는 비보존 민감도에서만 발생한다.)
LAB = {}
for name in MAIN:
    for k in KSWEEP:
        LAB[(name, k, SEED)] = kmeans_labels(Z_OF[name], k)
    for s in range(1, 5):
        LAB[(name, K, s)] = kmeans_labels(Z_OF[name], K, seed=s)
for name in SENS:
    LAB[(name, K, SEED)] = kmeans_labels(Z_OF[name], K)
print("[.] kmeans 완료", flush=True)


def labels_of(name, k=K, seed=SEED):
    """남은 진단이 0 개인 방문은 -1 로 뺀다."""
    l = LAB[(name, k, seed)].copy()
    l[ZERO_OF[name]] = -1
    return l


# ---------------------------------------------------------------- 19 의 지표 함수 (그대로)
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


# ---------------------------------------------------------------- 19 의 평가 함수 (그대로)
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


# ---------------------------------------------------------------- §4-1 축 판정
# 원 진단 리스트(제거 전) 기준으로 클러스터를 특징짓는 진단을 뽑는다.
# 제거 후 리스트로 뽑으면 제거된 코드가 못 나오는 게 당연해서 순환논증이 된다.
FULL_SETS = [set(l) for l in df["icd_l"].map(lambda l: [str(c).strip() for c in l])]
BASE_PREV = pd.Series({c: sum(c in s for s in FULL_SETS) / n for c in VOCAB})


def top_lift(labels, topn=TOP_LIFT):
    """클러스터별 (클러스터 내 유병률 − 전체 유병률) 상위 진단."""
    rows = []
    for g in sorted(set(labels)):
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            continue
        cnt = {}
        for i in idx:
            for c in FULL_SETS[i]:
                cnt[c] = cnt.get(c, 0) + 1
        prev = pd.Series({c: v / len(idx) for c, v in cnt.items()})
        prev = prev[prev >= MIN_PREV_LIFT]
        if prev.empty:
            continue
        lift = (prev - BASE_PREV.reindex(prev.index).fillna(0)).sort_values(ascending=False)
        for c in lift.head(topn).index:
            rows.append({"cluster": int(g), "code": c, "name": SHORT.get(c, "(사전에 없음)"),
                         "prev_in_cluster": round(float(prev[c]), 3),
                         "prev_overall": round(float(BASE_PREV.get(c, 0)), 3),
                         "lift_diff": round(float(lift[c]), 3),
                         "is_elix": bool(c in ELIX_VOCAB),
                         "cci": {True: "만성", False: "급성"}.get(CCI_CHRONIC.get(c), "미분류")})
    return pd.DataFrame(rows)


def axis_row(name, labels, ref_name=None):
    tl = top_lift(labels)
    elix_share = float(tl["is_elix"].mean() * 100)
    per_cl = tl.groupby("cluster")["is_elix"].mean()
    known = tl[tl["cci"] != "미분류"]
    acute_share = float((known["cci"] == "급성").mean() * 100) if len(known) else np.nan
    per_cl_ac = known.assign(ac=known["cci"] == "급성").groupby("cluster")["ac"].mean()
    return {
        "분할": name,
        "클러스터 수": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "특징 진단 중 CCI 급성%": round(acute_share, 1),
        "특징 진단이 급성 과반인 클러스터": int((per_cl_ac >= 0.5).sum()),
        "특징 진단 중 Elixhauser 동반질환%": round(elix_share, 1),
        "동반질환이 특징 진단 과반인 클러스터": int((per_cl >= 0.5).sum()),
        "동반질환 특징 진단 0개 클러스터": int((per_cl == 0).sum()),
        "ARI vs D1(주진단 chapter)": round(float(adjusted_rand_score(D1, labels)), 3),
        "NMI vs D1": round(float(normalized_mutual_info_score(D1, labels)), 3),
        "ARI vs C1_id k=15": round(float(adjusted_rand_score(C1, labels)), 3),
        "NMI vs C1_id k=15": round(float(normalized_mutual_info_score(C1, labels)), 3),
        "ARI vs C1_icdfull k=15": round(float(adjusted_rand_score(CFULL, labels)), 3),
        "η²(진단수)": round(eta2(labels, n_icd), 4),
        "η²(약물수)": round(eta2(labels, n_drug), 4),
    }, tl


axis_rows, lift_tabs = [], []
for name, labels in ([("C1_id k=15 (기준)", C1), ("C1_icdfull k=15 (기준)", CFULL), ("D1 주진단 chapter (기준)", D1)]
                     + [(SHORTNAME[nm], labels_of(nm)) for nm in MAIN + SENS]):
    r, tl = axis_row(name, np.asarray(labels))
    axis_rows.append(r)
    tl.insert(0, "분할", name)
    lift_tabs.append(tl)
tab_axis = pd.DataFrame(axis_rows)
tab_lift = pd.concat(lift_tabs, ignore_index=True)

# ---------------------------------------------------------------- §3 안정성 / k 스윕
sweep_rows = []
for name in MAIN:
    for k in KSWEEP:
        l = labels_of(name, k)
        sizes = pd.Series(l[l >= 0]).value_counts()
        te_sizes = pd.Series(l[(split_arr == "test") & (l >= 0)]).value_counts()
        sweep_rows.append({
            "분할": SHORTNAME[name], "k": k, "n_clusters": int(l[l >= 0].max() + 1),
            "최소 클러스터": int(sizes.min()), "최대 클러스터": int(sizes.max()),
            "test≥30 클러스터": int((te_sizes >= MIN_TEST_VISITS).sum()),
            "silhouette": round(float(silhouette_score(Z_OF[name], LAB[(name, k, SEED)],
                                                       sample_size=5000, random_state=SEED)), 4),
            "η²(약물수)": round(eta2(l, n_drug), 4),
            "ARI vs C1_id15": round(float(adjusted_rand_score(C1, l)), 3),
        })
tab_sweep = pd.DataFrame(sweep_rows)

seed_rows = []
for name in MAIN:
    labs = {s: labels_of(name, K, s) for s in range(5)}
    pair = [adjusted_rand_score(labs[i], labs[j]) for i in range(5) for j in range(i + 1, 5)]
    seed_rows.append({"분할": SHORTNAME[name], "seed 쌍별 ARI 평균": round(float(np.mean(pair)), 3),
                      "최소": round(float(np.min(pair)), 3), "최대": round(float(np.max(pair)), 3),
                      "C1_id k=15 seed 노이즈 기준": 0.599})
tab_seed = pd.DataFrame(seed_rows)

# ---------------------------------------------------------------- §3 describe / §5 evaluate
desc_rows, clus_tabs = [], []
for name in MAIN + SENS:
    l = labels_of(name)
    row = describe_partition(l, Z_OF[name], SHORTNAME[name], K)
    # 남은 진단 0개 방문은 SVD 공간의 원점에 몰려 모든 점에서 거리 1.0 이 되므로 실루엣을 끌어내린다.
    # 구조 비교용으로 -1 을 뺀 값을 따로 둔다(주 버전 4개는 -1 이 없어 두 값이 같다).
    m = l >= 0
    row["silhouette_excl_other"] = round(float(silhouette_score(Z_OF[name][m], l[m], sample_size=5000,
                                                               random_state=SEED)), 4)
    desc_rows.append(row)
    clus_tabs.append(evaluate_partition(SHORTNAME[name], K, l))
tab_desc = pd.DataFrame(desc_rows)
tab_clus = pd.concat(clus_tabs, ignore_index=True)

# ---------------------------------------------------------------- §4-2 주 진단 보존 여부 민감도
sens_rows = []
for keep, drop in [("COMORB-elix", "COMORB-elix (주진단 비보존)"),
                   (f"CHR-persist(τ={TAU_MAIN})", f"CHR-persist(τ={TAU_MAIN}, 주진단 비보존)")]:
    lk, ld = labels_of(keep), labels_of(drop)
    both = (lk >= 0) & (ld >= 0)
    sens_rows.append({
        "정의": SHORTNAME[keep],
        "주진단 제거 방문%": round(float(np.isin(df["icd_seq1"].astype(str).str.strip(),
                                          list(ELIX_VOCAB)).mean() * 100), 2) if "elix" in keep.lower() else None,
        "두 버전 ARI": round(float(adjusted_rand_score(lk[both], ld[both])), 3),
        "두 버전 NMI": round(float(normalized_mutual_info_score(lk[both], ld[both])), 3),
        "ARI vs D1 (보존)": round(float(adjusted_rand_score(D1, lk)), 3),
        "ARI vs D1 (비보존)": round(float(adjusted_rand_score(D1[ld >= 0], ld[ld >= 0])), 3),
        "ARI vs C1_id (보존)": round(float(adjusted_rand_score(C1, lk)), 3),
        "ARI vs C1_id (비보존)": round(float(adjusted_rand_score(C1[ld >= 0], ld[ld >= 0])), 3),
        "특징진단 CCI 급성% (보존)": tab_axis.set_index("분할").loc[SHORTNAME[keep], "특징 진단 중 CCI 급성%"],
        "특징진단 CCI 급성% (비보존)": tab_axis.set_index("분할").loc[SHORTNAME[drop], "특징 진단 중 CCI 급성%"],
    })
tab_sens = pd.DataFrame(sens_rows)

# ---------------------------------------------------------------- 산출
asg = df[["SUBJECT_ID", "HADM_ID", "split"]].copy()
asg["comorb_elix_label"] = labels_of(MAIN[0])
asg["chr_persist_label"] = labels_of(MAIN[1])
asg["chr_persist_scale_label"] = labels_of(MAIN[2])
asg["chr_idf_label"] = labels_of(MAIN[3])
asg["comorb_elix_rmseq1_label"] = labels_of(SENS[0])
asg["chr_persist_rmseq1_label"] = labels_of(SENS[1])
asg["n_remaining_dx"] = NLEFT_OF[MAIN[0]]           # 주 정의(COMORB-elix) 기준
asg["n_remaining_persist"] = NLEFT_OF[MAIN[1]]
asg["n_remaining_persist_scale"] = NLEFT_OF[MAIN[2]]
asg["c1_id_label"] = C1
asg["d1_label"] = D1
asg.to_csv(OUT / "chronic_partition_assignments.csv", index=False)

pd.to_pickle({"labels": {SHORTNAME[nm]: labels_of(nm) for nm in MAIN + SENS},
              "sweep": {(SHORTNAME[nm], k): labels_of(nm, k) for nm in MAIN for k in KSWEEP},
              "Z_shapes": {SHORTNAME[nm]: Z_OF[nm].shape for nm in MAIN + SENS},
              "main": [SHORTNAME[nm] for nm in MAIN], "sens": [SHORTNAME[nm] for nm in SENS]},
             OUT / "28_chronic_labels.pkl")

tab_desc.to_csv(OUT / "table29_chronic_describe.csv", index=False)
tab_sweep.to_csv(OUT / "table30_chronic_ksweep.csv", index=False)
tab_seed.to_csv(OUT / "table31_chronic_seed.csv", index=False)
tab_axis.to_csv(OUT / "table32_chronic_axis.csv", index=False)
tab_clus.to_csv(OUT / "table33_chronic_clusters.csv", index=False)
tab_lift.to_csv(OUT / "table34_chronic_toplift.csv", index=False)
tab_sens.to_csv(OUT / "table35_chronic_seq1_sens.csv", index=False)

json.dump({
    "pipeline": "multihot(4514 고정) → L2 → TruncatedSVD(100, seed 0) → L2 → KMeans(k, n_init=10, seed 0)",
    "identical_to": "C1_id k=15",
    "zero_remaining_policy": "라벨 -1 (군집에서 제외, 파일에는 유지)",
    "main_partitions": [SHORTNAME[nm] for nm in MAIN], "sensitivity": [SHORTNAME[nm] for nm in SENS],
    "k_sweep": KSWEEP, "top_lift_n": TOP_LIFT, "min_prev_for_lift": MIN_PREV_LIFT,
    "axis": tab_axis.to_dict("records"), "seed_stability": tab_seed.to_dict("records"),
    "seq1_sensitivity": tab_sens.to_dict("records"),
}, open(OUT / "28_chronic_cluster_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 화면 보고
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
print("\n===== §4 축 판정")
print(tab_axis.to_string(index=False))
print("\n===== §4-2 주 진단 보존 여부 민감도")
print(tab_sens.to_string(index=False))
print("\n===== §3 seed 안정성")
print(tab_seed.to_string(index=False))
print("\n===== §3 describe")
print(tab_desc.to_string(index=False))
print("\n===== §4-1 클러스터 특징 진단 (COMORB-elix, lift 상위)")
print(tab_lift[tab_lift["분할"] == "COMORB-elix"].to_string(index=False))
print("\n===== §4-1 클러스터 특징 진단 (C1_id 기준)")
print(tab_lift[tab_lift["분할"] == "C1_id k=15 (기준)"].to_string(index=False))
print("\n===== k 스윕 (요약)")
print(tab_sweep[tab_sweep["k"].isin([5, 10, 15, 20, 25, 30])].to_string(index=False))
