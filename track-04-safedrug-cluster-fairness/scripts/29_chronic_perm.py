"""§5 평가 — 환자블록 순열검정. 14_permutation.py / 19_trackA_fullvocab.py 의 함수를 그대로 쓴다.

주 분할 4개 + 주진단 비보존 민감도 2개, 각 10,000 회. 클러스터 크기(환자 수)는 largest_remainder 로 보존.
단측 p = (1 + #{null >= obs}) / (1 + n).

산출: out/table36_chronic_perm.csv, out/table37_chronic_gap.csv,
      out/perm_null_chronic_*.npz, out/29_chronic_perm_meta.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
N_PERM = 10000
MIN_TEST_VISITS = 30

df = pd.read_pickle(OUT / "20_icd_features.pkl")
L = pd.read_pickle(OUT / "28_chronic_labels.pkl")
NAMES = L["main"] + L["sens"]


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
rows, metas = [], []
for name in NAMES:
    labels = L["labels"][name]
    obs, null, pmeta = run_partition(name, labels)
    metas.append(pmeta)
    np.savez_compressed(OUT / f"perm_null_chronic_{re.sub(r'[^A-Za-z0-9]', '_', name)}.npz", null=null)
    for pred, stat, j in STATS:
        o, nd = obs[j], null[:, j]
        nd = nd[~np.isnan(nd)]
        p = (1 + int((nd >= o).sum())) / (1 + len(nd))
        rows.append({
            "partition": name, "predictor": {"const": "상수", "prev": "copy-prev"}[pred],
            "statistic": {"range": "범위(max-min)", "wsd": "가중 SD"}[stat],
            "observed": round(float(o), 4), "null_mean": round(float(nd.mean()), 4),
            "null_sd": round(float(nd.std(ddof=1)), 4), "null_p95": round(float(np.percentile(nd, 95)), 4),
            "p_value": round(p, 5), "n_perm_valid": len(nd),
            "z_vs_null": round(float((o - nd.mean()) / nd.std(ddof=1)), 2),
        })
    print(f"[.] {name} 완료", flush=True)

perm = pd.DataFrame(rows)
perm.to_csv(OUT / "table36_chronic_perm.csv", index=False)

# ---------------------------------------------------------------- 격차 요약 (19 의 gap_of 와 동일 정의)
clus = pd.read_csv(OUT / "table33_chronic_clusters.csv")
gap_rows = []
for name in NAMES:
    t = clus[clus["partition"] == name]
    big = t[t["test_visits"] >= MIN_TEST_VISITS]
    bp = big[big["jac_prev"].notna()]
    gap_rows.append({
        "분할": name, "test≥30 클러스터": len(big),
        "jac_const 최소": round(float(big["jac_const"].min()), 4),
        "jac_const 최대": round(float(big["jac_const"].max()), 4),
        "jac_const 격차": round(float(big["jac_const"].max() - big["jac_const"].min()), 4),
        "jac_prev 격차": round(float(bp["jac_prev"].max() - bp["jac_prev"].min()), 4),
        "약물수-성능 상관(Pearson)": round(float(np.corrcoef(big["mean_n_drug"], big["jac_const"])[0, 1]), 3),
        "약물수-성능 상관(Spearman)": round(float(pd.Series(big["mean_n_drug"]).corr(
            pd.Series(big["jac_const"]), method="spearman")), 3),
    })
gap = pd.DataFrame(gap_rows)
gap.to_csv(OUT / "table37_chronic_gap.csv", index=False)

json.dump({"n_perm": N_PERM, "min_test_visits": MIN_TEST_VISITS, "seed": SEED,
           "partitions": metas, "perm": perm.to_dict("records"), "gap": gap.to_dict("records")},
          open(OUT / "29_chronic_perm_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

pd.set_option("display.width", 250, "display.max_columns", 30)
print("\n===== §5 순열검정")
print(perm.to_string(index=False))
print("\n===== §5 격차 요약")
print(gap.to_string(index=False))
