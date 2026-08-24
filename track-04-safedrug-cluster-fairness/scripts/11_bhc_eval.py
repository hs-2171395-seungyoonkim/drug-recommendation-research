"""
§5 성능 연결: CB_masked k=15 클러스터에 자명 baseline 성능 (04와 동일 절차/분할).
+ UMAP 좌표 계산 (그림 전용) — out/umap_bhc_masked.npz 캐시.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0

# ---------- 04 와 동일한 per-visit jaccard (전체 14,541 기준 → HADM_ID 조인) ----------
df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
is_test = np.zeros(len(df), bool)
is_test[te_idx] = True
df["split"] = np.where(is_test, "test", "train")
df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"])
df["drug_set"] = df["drug_id_l"].map(set)

train = df[~is_test]
K_CONST = int(round(train["drug_id_l"].map(len).mean()))
freq = pd.Series([d for s in train["drug_id_l"] for d in s]).value_counts()
const_set = set(freq.head(K_CONST).index)

df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
df["prev_set"] = df.groupby("SUBJECT_ID")["drug_set"].shift(1)


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else np.nan


df["jac_const"] = [jac(const_set, t) for t in df["drug_set"]]
df["jac_prev"] = [jac(p, t) if isinstance(p, set) else np.nan for p, t in zip(df["prev_set"], df["drug_set"])]

# ---------- BHC 클러스터 라벨 조인 ----------
bassign = pd.read_csv(OUT / "cluster_assignments_bhc.csv")
sub = bassign[(bassign.partition == "BHC_CB_masked") & (bassign.k == 15)]
df["cluster"] = df["HADM_ID"].map(dict(zip(sub.HADM_ID, sub.cluster_id)))

test = df[(df["split"] == "test") & df["cluster"].notna()].copy()
test["cluster"] = test["cluster"].astype(int)


def boot_ci(sub, col, n_boot=1000, seed=SEED):
    r = np.random.default_rng(seed)
    subs = sub["SUBJECT_ID"].unique()
    by = {s: g[col].to_numpy() for s, g in sub.groupby("SUBJECT_ID")}
    means = []
    for _ in range(n_boot):
        vals = np.concatenate([by[s] for s in r.choice(subs, size=len(subs), replace=True)])
        vals = vals[~np.isnan(vals)]
        if len(vals):
            means.append(vals.mean())
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)) if means else (np.nan, np.nan)


rows = []
for g, s in test.groupby("cluster"):
    cnt = {}
    for l in s["diagnose_l"]:
        for d in set(l):
            cnt[d] = cnt.get(d, 0) + 1
    top = sorted(cnt.items(), key=lambda x: -x[1])[:3]
    lo1, hi1 = boot_ci(s, "jac_const")
    sp = s[s["jac_prev"].notna()]
    lo2, hi2 = boot_ci(sp, "jac_prev") if len(sp) > 1 else (np.nan, np.nan)
    rows.append({
        "cluster": int(g), "test_visits": len(s), "test_subjects": s["SUBJECT_ID"].nunique(),
        "mean_n_diag": round(float(s["diag_id_l"].map(len).mean()), 1),
        "mean_n_drug": round(float(s["drug_id_l"].map(len).mean()), 1),
        "top3_dx": " | ".join(f"{d} {c/len(s)*100:.0f}%" for d, c in top),
        "jac_const": round(float(s["jac_const"].mean()), 4), "jac_const_lo": lo1, "jac_const_hi": hi1,
        "n_prev": len(sp), "jac_prev": round(float(sp["jac_prev"].mean()), 4) if len(sp) else np.nan,
        "jac_prev_lo": lo2, "jac_prev_hi": hi2,
    })
perf = pd.DataFrame(rows).sort_values("jac_const", ascending=False)
perf.to_csv(OUT / "tableB_bhc_cluster_performance.csv", index=False)

big = perf[perf.test_visits >= 30]
summary = {
    "partition": "BHC_CB_masked k=15",
    "n_clusters_ge30_test": len(big),
    "jac_const_spread": round(float(big["jac_const"].max() - big["jac_const"].min()), 4),
    "jac_prev_spread": round(float(big["jac_prev"].max() - big["jac_prev"].min()), 4),
    "corr_jacconst_vs_ndrug": round(float(big["jac_const"].corr(big["mean_n_drug"])), 3),
    "corr_jacprev_vs_ndrug": round(float(big["jac_prev"].corr(big["mean_n_drug"])), 3),
    "diagnose_C1_id_k15_reference": {"jac_const_spread": 0.085, "jac_prev_spread": 0.059},
}
with open(OUT / "16_bhc_perf_meta.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(perf.to_string(index=False))
print(json.dumps(summary, ensure_ascii=False, indent=2))

# ---------- UMAP (그림 전용, 캐시) ----------
cache = OUT / "umap_bhc_masked.npz"
if not cache.exists():
    import umap

    E = np.load(OUT / "emb_bhc_masked.npz")["E"]
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(E)
    np.savez_compressed(cache, U=U)
    print("umap cached")
