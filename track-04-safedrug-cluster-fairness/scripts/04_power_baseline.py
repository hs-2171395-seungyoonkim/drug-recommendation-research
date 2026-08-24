"""
§6 검정력 (k 상한 역산) + §7 자명 baseline 의 클러스터별 성능.

§8: evaluate_partition(labels, ...) 은 라벨 벡터만 받는다. 노트 임베딩 군집이 오면 그대로 넣으면 된다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
rng = np.random.default_rng(SEED)

df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
assign = pd.read_pickle(OUT / "cluster_assignments.pkl")
df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"])
df["drug_set"] = df["drug_id_l"].map(set)

# ---------------------------------------------------------------- 환자 단위 8:2 분할
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
is_test = np.zeros(len(df), bool)
is_test[te_idx] = True
df["split"] = np.where(is_test, "test", "train")

split_info = {
    "train_visits": int((~is_test).sum()),
    "test_visits": int(is_test.sum()),
    "train_subjects": int(df.loc[~is_test, "SUBJECT_ID"].nunique()),
    "test_subjects": int(df.loc[is_test, "SUBJECT_ID"].nunique()),
    "leakage_subjects_in_both": int(
        len(set(df.loc[~is_test, "SUBJECT_ID"]) & set(df.loc[is_test, "SUBJECT_ID"]))
    ),
}

# ---------------------------------------------------------------- §7 두 개의 자명 예측기
train = df[~is_test]
K_CONST = int(round(train["drug_id_l"].map(len).mean()))
freq = pd.Series([d for s in train["drug_id_l"] for d in s]).value_counts()
const_set = set(freq.head(K_CONST).index)

df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
df["visit_order"] = df.groupby("SUBJECT_ID").cumcount()
df["prev_set"] = df.groupby("SUBJECT_ID")["drug_set"].shift(1)


def jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


df["jac_const"] = [jaccard(const_set, t) for t in df["drug_set"]]
df["jac_prev"] = [
    jaccard(p, t) if isinstance(p, set) else np.nan for p, t in zip(df["prev_set"], df["drug_set"])
]

test = df[df["split"] == "test"].reset_index(drop=True)
baseline_info = {
    "K_const": K_CONST,
    "const_drugs_top10": [str(x) for x in freq.head(10).index.tolist()],
    "test_visits": int(len(test)),
    "test_visits_with_prev": int(test["jac_prev"].notna().sum()),
    "overall_jac_const": round(float(test["jac_const"].mean()), 4),
    "overall_jac_prev": round(float(test["jac_prev"].mean()), 4),
    "sd_jac_const": round(float(test["jac_const"].std()), 4),
    "sd_jac_prev": round(float(test["jac_prev"].std()), 4),
}

# 클러스터 배정을 test 에 붙이기 위한 룩업
assign_idx = assign.set_index(["partition", "k"])


def labels_for(partition, k):
    sub = assign[(assign["partition"] == partition) & (assign["k"] == k)]
    m = dict(zip(sub["HADM_ID"], sub["cluster_id"]))
    return df["HADM_ID"].map(m).to_numpy()


# ---------------------------------------------------------------- §6 검정력
Z_ALPHA, Z_POWER = 1.959964, 0.8416212  # α=.05 양측, power=.80
FACTOR = Z_ALPHA + Z_POWER


def mdd(s, n1, n2):
    """두 독립 평균 비교에서 검출 가능한 최소 격차."""
    if n1 < 2 or n2 < 2:
        return np.inf
    return FACTOR * s * np.sqrt(1 / n1 + 1 / n2)


def power_row(partition, k, labels):
    te = labels[df["split"].to_numpy() == "test"]
    sizes = pd.Series(te).value_counts()
    s = float(test["jac_const"].std())
    N = len(te)
    n_min = int(sizes.min())
    mdd_min_vs_rest = mdd(s, n_min, N - n_min)
    # 모든 클러스터가 '나머지 전체'와 비교 가능한지
    all_mdd = [mdd(s, int(v), N - int(v)) for v in sizes.values]
    return {
        "partition": partition,
        "k": k,
        "n_clusters_in_test": int(len(sizes)),
        "test_min_cluster": n_min,
        "test_median_cluster": float(sizes.median()),
        "test_max_cluster": int(sizes.max()),
        "n_clusters_lt_30": int((sizes < 30).sum()),
        "n_clusters_lt_50": int((sizes < 50).sum()),
        "se_smallest": round(s / np.sqrt(n_min), 4),
        "mdd_smallest_vs_rest": round(mdd_min_vs_rest, 4),
        "worst_mdd": round(max(all_mdd), 4),
    }


KS = list(range(5, 31))
power_rows = []
for pn in ["P1_ge100", "P1_ge30"]:
    power_rows.append(power_row(pn, 0, labels_for(pn, 0)))
for pn in ["C1_id", "C2_id", "C1_name", "C2_name"]:
    for k in KS:
        power_rows.append(power_row(pn, k, labels_for(pn, k)))
power = pd.DataFrame(power_rows)
power.to_csv(OUT / "table3_power.csv", index=False)

# 임계값별 최대 k
kmax = {}
for pn in ["C1_id", "C2_id", "C1_name", "C2_name"]:
    sub = power[power["partition"] == pn].sort_values("k")
    for thr in [0.03, 0.05, 0.10]:
        ok = sub[sub["worst_mdd"] <= thr]["k"]
        kmax.setdefault(f"MDD<={thr}", {})[pn] = int(ok.max()) if len(ok) else None

# ---------------------------------------------------------------- §7 클러스터별 성능 + 부트스트랩
def boot_ci(sub, col, n_boot=1000, seed=SEED):
    """환자 단위 리샘플 95% CI."""
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
    te["cluster"] = labels[df["split"].to_numpy() == "test"]
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
        rows.append(
            {
                "partition": partition,
                "k": k,
                "cluster": int(g),
                "test_visits": int(len(sub)),
                "test_subjects": int(sub["SUBJECT_ID"].nunique()),
                "mean_n_diag": round(float(sub["diag_id_l"].map(len).mean()), 1),
                "mean_n_drug": round(float(sub["drug_id_l"].map(len).mean()), 1),
                "top3_dx": " | ".join(f"{d} {c/len(sub)*100:.0f}%" for d, c in top),
                "jac_const": round(float(sub["jac_const"].mean()), 4),
                "jac_const_lo": lo1,
                "jac_const_hi": hi1,
                "n_prev": int(len(sp)),
                "jac_prev": round(float(sp["jac_prev"].mean()), 4) if len(sp) else np.nan,
                "jac_prev_lo": lo2,
                "jac_prev_hi": hi2,
            }
        )
    return pd.DataFrame(rows)


CAND = {"P1_ge100": [0], "P1_ge30": [0], "C1_id": [10, 15, 20, 25], "C2_id": [10, 15, 20, 25],
        "C1_name": [10, 15, 20, 25], "C2_name": [10, 15, 20, 25]}
tab2 = pd.concat(
    [evaluate_partition(pn, k, labels_for(pn, k)) for pn, ks in CAND.items() for k in ks],
    ignore_index=True,
)
tab2.to_csv(OUT / "table2_cluster_performance.csv", index=False)

# 클러스터 간 격차 요약
gap = []
for (pn, k), g in tab2.groupby(["partition", "k"]):
    big = g[g["test_visits"] >= 30]
    gap.append(
        {
            "partition": pn,
            "k": k,
            "n_clusters": len(g),
            "n_clusters_ge30_test": len(big),
            "jac_const_min": round(float(big["jac_const"].min()), 4) if len(big) else np.nan,
            "jac_const_max": round(float(big["jac_const"].max()), 4) if len(big) else np.nan,
            "jac_const_spread": round(float(big["jac_const"].max() - big["jac_const"].min()), 4) if len(big) else np.nan,
            "jac_prev_min": round(float(big["jac_prev"].min()), 4) if len(big) else np.nan,
            "jac_prev_max": round(float(big["jac_prev"].max()), 4) if len(big) else np.nan,
            "jac_prev_spread": round(float(big["jac_prev"].max() - big["jac_prev"].min()), 4) if len(big) else np.nan,
            "corr_jacconst_vs_ndrug": round(float(big["jac_const"].corr(big["mean_n_drug"])), 3) if len(big) > 2 else np.nan,
            "corr_jacprev_vs_ndrug": round(float(big["jac_prev"].corr(big["mean_n_drug"])), 3) if len(big) > 2 else np.nan,
        }
    )
gap = pd.DataFrame(gap)
gap.to_csv(OUT / "table4_gap_summary.csv", index=False)

with open(OUT / "12_power_meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {"split": split_info, "baseline": baseline_info, "kmax_by_threshold": kmax},
        f, ensure_ascii=False, indent=2, default=str,
    )

print(json.dumps({"split": split_info, "baseline": baseline_info, "kmax": kmax}, ensure_ascii=False, indent=2))
print()
print(gap.to_string(index=False))
