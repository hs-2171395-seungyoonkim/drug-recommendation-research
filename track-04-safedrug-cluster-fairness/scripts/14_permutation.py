"""
§9 클러스터별 성능 격차 순열검정.

귀무: "클러스터 라벨은 성능과 무관하다". 환자 단위로 라벨을 섞되 클러스터 크기 분포를
보존하고, 관측과 동일한 필터(test 방문 >=30)를 귀무 표본에도 적용한다.

대상 2개 파티션 x 예측기 2개(상수/copy-prev) x 통계 2개(범위/가중SD) = 8개 p값.
산출: out/table5_perm.csv, out/17_perm_meta.json, out/perm_null.npz, figs/fig_perm_*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
SEED = 0
MIN_TEST_VISITS = 30
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

# ================================================================= 04 와 동일한 전처리
df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"])
df["drug_set"] = df["drug_id_l"].map(set)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
is_test = np.zeros(len(df), bool)
is_test[te_idx] = True
df["split"] = np.where(is_test, "test", "train")

train = df[~is_test]
K_CONST = int(round(train["drug_id_l"].map(len).mean()))
freq = pd.Series([d for s in train["drug_id_l"] for d in s]).value_counts()
const_set = set(freq.head(K_CONST).index)

df = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
df["prev_set"] = df.groupby("SUBJECT_ID")["drug_set"].shift(1)


def jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


df["jac_const"] = [jaccard(const_set, t) for t in df["drug_set"]]
df["jac_prev"] = [
    jaccard(p, t) if isinstance(p, set) else np.nan for p, t in zip(df["prev_set"], df["drug_set"])
]

# ================================================================= 격차 통계
def gap_stats(cluster_of_visit, n_clusters, jc, jp_mask, jp_vals):
    """관측/귀무 공통. 반환: (range_const, wsd_const, range_prev, wsd_prev, n_ok, n_ok_prev)

    필터는 관측 절차와 동일하게 '클러스터의 test 방문 수 >= 30'. copy-prev 평균은 그
    클러스터 안에서 직전 방문이 있는 방문만으로 계산한다(관측과 동일).
    """
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


# ================================================================= 파티션별 순열
def largest_remainder(props, total):
    """비율 -> 합이 total 인 정수 배분."""
    raw = props * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    return base


def run_partition(name, label_map, n_perm=N_PERM, seed=SEED):
    """label_map: HADM_ID -> cluster_id (해당 파티션의 우주에 속한 방문만)."""
    d = df.copy()
    d["cluster"] = d["HADM_ID"].map(label_map)
    d = d[d["cluster"].notna()].copy()
    d["cluster"] = d["cluster"].astype(int)

    # 클러스터 라벨 0..K-1 로 재색인
    uniq = np.sort(d["cluster"].unique())
    remap = {c: i for i, c in enumerate(uniq)}
    d["cl"] = d["cluster"].map(remap).astype(int)
    K = len(uniq)

    # 환자 색인 (우주 전체 = train+test)
    pats = np.sort(d["SUBJECT_ID"].unique())
    pat_ix = {p: i for i, p in enumerate(pats)}
    d["pat"] = d["SUBJECT_ID"].map(pat_ix).astype(int)
    n_pat = len(pats)

    te = d[d["split"] == "test"]
    jc = te["jac_const"].to_numpy(float)
    jp = te["jac_prev"].to_numpy(float)
    jp_mask = ~np.isnan(jp)
    jp_vals = jp[jp_mask]
    pat_of_test_visit = te["pat"].to_numpy()
    cl_obs_test = te["cl"].to_numpy()

    obs = gap_stats(cl_obs_test, K, jc, jp_mask, jp_vals)

    # --- 크기 분포: 관측 파티션의 클러스터별 '환자 수' (우주 전체 기준)
    pat_per_cl = d.groupby("cl")["SUBJECT_ID"].nunique().reindex(range(K), fill_value=0).to_numpy()
    n_multi = int((d.groupby("SUBJECT_ID")["cl"].nunique() > 1).sum())
    props = pat_per_cl / pat_per_cl.sum()
    target = largest_remainder(props, n_pat)

    rng = np.random.default_rng(seed)
    null = np.full((n_perm, 4), np.nan)
    n_ok_hist = np.zeros((n_perm, 2), int)
    for b in range(n_perm):
        perm = rng.permutation(n_pat)
        cl_of_pat = np.empty(n_pat, int)
        s = 0
        for c, sz in enumerate(target):
            cl_of_pat[perm[s:s + sz]] = c
            s += sz
        cl_test = cl_of_pat[pat_of_test_visit]
        r_c, s_c, r_p, s_p, nok, nokp = gap_stats(cl_test, K, jc, jp_mask, jp_vals)
        null[b] = (r_c, s_c, r_p, s_p)
        n_ok_hist[b] = (nok, nokp)

    meta = {
        "partition": name,
        "n_clusters": K,
        "universe_visits": int(len(d)),
        "universe_subjects": n_pat,
        "test_visits": int(len(te)),
        "test_visits_with_prev": int(jp_mask.sum()),
        "subjects_spanning_multiple_clusters": n_multi,
        "observed_cluster_patient_counts": pat_per_cl.tolist(),
        "null_cluster_patient_targets": target.tolist(),
        "n_clusters_ge30_observed": obs[4],
        "n_clusters_ge30_prev_observed": obs[5],
        "n_clusters_ge30_null_mean": round(float(n_ok_hist[:, 0].mean()), 2),
        "n_clusters_ge30_null_min": int(n_ok_hist[:, 0].min()),
        "n_clusters_ge30_prev_null_mean": round(float(n_ok_hist[:, 1].mean()), 2),
        "n_clusters_ge30_prev_null_min": int(n_ok_hist[:, 1].min()),
    }
    return obs[:4], null, meta


# ---- 파티션 1: diagnose C1_id k=15
a = pd.read_pickle(OUT / "cluster_assignments.pkl")
sub = a[(a["partition"] == "C1_id") & (a["k"] == 15)]
map_dx = dict(zip(sub["HADM_ID"], sub["cluster_id"]))

# ---- 파티션 2: BHC CB_masked k=15
b = pd.read_csv(OUT / "cluster_assignments_bhc.csv")
sub_b = b[(b["partition"] == "BHC_CB_masked") & (b["k"] == 15)]
map_bhc = dict(zip(sub_b["HADM_ID"], sub_b["cluster_id"]))

PARTS = [("diagnose C1_id k=15", map_dx), ("BHC CB_masked k=15", map_bhc)]
STATS = [("const", "range"), ("const", "wsd"), ("prev", "range"), ("prev", "wsd")]
COL = {("const", "range"): 0, ("const", "wsd"): 1, ("prev", "range"): 2, ("prev", "wsd"): 3}

rows, metas, nulls = [], [], {}
for name, m in PARTS:
    obs, null, meta = run_partition(name, m)
    metas.append(meta)
    nulls[name] = null
    for pred, stat in STATS:
        j = COL[(pred, stat)]
        o = obs[j]
        nd = null[:, j]
        nd = nd[~np.isnan(nd)]
        p = (1 + int((nd >= o).sum())) / (1 + len(nd))
        rows.append({
            "partition": name,
            "predictor": {"const": "상수", "prev": "copy-prev"}[pred],
            "statistic": {"range": "범위(max-min)", "wsd": "가중 SD"}[stat],
            "observed": round(float(o), 4),
            "null_mean": round(float(nd.mean()), 4),
            "null_sd": round(float(nd.std(ddof=1)), 4),
            "null_p95": round(float(np.percentile(nd, 95)), 4),
            "null_max": round(float(nd.max()), 4),
            "p_value": round(p, 5),
            "n_perm_valid": len(nd),
            "z_vs_null": round(float((o - nd.mean()) / nd.std(ddof=1)), 2),
        })

tab = pd.DataFrame(rows)
tab.to_csv(OUT / f"table5_perm_{N_PERM}.csv", index=False)
np.savez_compressed(
    OUT / f"perm_null_{N_PERM}.npz",
    **{f"null_{i}": nulls[n] for i, (n, _) in enumerate(PARTS)},
    partitions=np.array([n for n, _ in PARTS]),
)
with open(OUT / f"17_perm_meta_{N_PERM}.json", "w", encoding="utf-8") as f:
    json.dump({"n_perm": N_PERM, "seed": SEED, "min_test_visits": MIN_TEST_VISITS,
               "partitions": metas, "table": rows}, f, ensure_ascii=False, indent=2)

pd.set_option("display.width", 250)
print(json.dumps(metas, ensure_ascii=False, indent=2))
print()
print(tab.to_string(index=False))

# ================================================================= 그림 (파티션 x 예측기 = 4장)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, RED, MUTED = "#2a78d6", "#c0392b", "#c9c8c3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
})

FIGNAME = {
    ("diagnose C1_id k=15", "const"): "fig9_perm_dx_const",
    ("diagnose C1_id k=15", "prev"): "fig10_perm_dx_prev",
    ("BHC CB_masked k=15", "const"): "fig11_perm_bhc_const",
    ("BHC CB_masked k=15", "prev"): "fig12_perm_bhc_prev",
}
PRED_KO = {"const": "상수 예측기", "prev": "copy-prev 예측기"}

for name, _ in PARTS:
    null = nulls[name]
    obs_row = {(r["predictor"], r["statistic"]): r for r in rows if r["partition"] == name}
    for pred in ["const", "prev"]:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        for ax, stat in zip(axes, ["range", "wsd"]):
            j = COL[(pred, stat)]
            nd = null[:, j]
            nd = nd[~np.isnan(nd)]
            r = obs_row[({"const": "상수", "prev": "copy-prev"}[pred],
                         {"range": "범위(max-min)", "wsd": "가중 SD"}[stat])]
            o, p = r["observed"], r["p_value"]
            ax.hist(nd, bins=45, color=MUTED, edgecolor="none")
            ax.axvline(np.percentile(nd, 95), color=INK2, ls=":", lw=1.1, label="귀무 95백분위")
            ax.axvline(o, color=RED, lw=1.8, label=f"관측 {o:.4f}")
            ax.set_xlim(min(nd.min(), o) * 0.95, max(nd.max(), o) * 1.05)
            ax.set_xlabel({"range": "범위 (max-min)", "wsd": "클러스터 간 가중 SD"}[stat])
            ax.set_ylabel("순열 표본 수")
            floor = 1 / (N_PERM + 1)
            ax.set_title(f"p <= {floor:.4f} (귀무가 관측을 넘은 횟수 0)" if p <= 1.5 * floor else f"p = {p:.4f}",
                         fontsize=9, color=INK2)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{name} · {PRED_KO[pred]} · 환자단위 순열 {N_PERM:,}회", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(FIGS / f"{FIGNAME[(name, pred)]}.png", dpi=170)
        plt.close(fig)
print("\nfigs written:", ", ".join(sorted(v for v in FIGNAME.values())))
