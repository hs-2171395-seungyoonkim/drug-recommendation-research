"""§9 부록: 순열 설계 타당성 점검.

(a) 방문단위 셔플 vs 환자단위 셔플 — 귀무분포 폭 비교 (환자 내 상관 반영 확인)
(b) iid 근사 대비 귀무 폭 (클러스터 크기만으로 기대되는 범위와 비교)
결과: out/18_perm_check.json
"""
import json

import numpy as np
import pandas as pd

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

spec = importlib.util.spec_from_file_location("p14", ROOT / "scripts" / "14_permutation.py")
# 14 를 import 하면 전체 재실행이므로, 필요한 조각만 복제한다.
from sklearn.model_selection import GroupShuffleSplit

SEED, MIN_TEST = 0, 30
df = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"])
df["drug_set"] = df["drug_id_l"].map(set)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
_, te_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
is_test = np.zeros(len(df), bool)
is_test[te_idx] = True
df["split"] = np.where(is_test, "test", "train")
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

a = pd.read_pickle(OUT / "cluster_assignments.pkl")
s = a[(a["partition"] == "C1_id") & (a["k"] == 15)]
df["cl"] = df["HADM_ID"].map(dict(zip(s["HADM_ID"], s["cluster_id"])))
d = df[df["cl"].notna()].copy()
d["cl"] = d["cl"].astype(int)
pats = np.sort(d["SUBJECT_ID"].unique())
pat_ix = {p: i for i, p in enumerate(pats)}
d["pat"] = d["SUBJECT_ID"].map(pat_ix).astype(int)
K, n_pat = 15, len(pats)

te = d[d["split"] == "test"]
jc = te["jac_const"].to_numpy(float)
jp = te["jac_prev"].to_numpy(float)
mask = ~np.isnan(jp)
jpv = jp[mask]
pat_v = te["pat"].to_numpy()
n_test = len(te)


def stats(cl_test):
    cnt = np.bincount(cl_test, minlength=K).astype(float)
    tot = np.bincount(cl_test, weights=jc, minlength=K)
    ok = cnt >= MIN_TEST
    clp = cl_test[mask]
    cp = np.bincount(clp, minlength=K).astype(float)
    tp = np.bincount(clp, weights=jpv, minlength=K)
    okp = ok & (cp > 0)
    out = []
    for msk, t_, c_ in [(ok, tot, cnt), (okp, tp, cp)]:
        if msk.sum() < 2:
            out += [np.nan, np.nan]
            continue
        m = t_[msk] / c_[msk]
        w = c_[msk]
        mb = np.sum(w * m) / np.sum(w)
        out += [float(m.max() - m.min()), float(np.sqrt(np.sum(w * (m - mb) ** 2) / np.sum(w)))]
    return out


pat_per_cl = d.groupby("cl")["SUBJECT_ID"].nunique().reindex(range(K), fill_value=0).to_numpy()
raw = pat_per_cl / pat_per_cl.sum() * n_pat
target = np.floor(raw).astype(int)
rem = n_pat - target.sum()
if rem > 0:
    target[np.argsort(-(raw - target))[:rem]] += 1
vis_per_cl = np.bincount(te["cl"].to_numpy(), minlength=K)

NB = 2000
rng = np.random.default_rng(SEED)
by_pat, by_visit = [], []
for _ in range(NB):
    perm = rng.permutation(n_pat)
    cp_ = np.empty(n_pat, int)
    st = 0
    for c, sz in enumerate(target):
        cp_[perm[st:st + sz]] = c
        st += sz
    by_pat.append(stats(cp_[pat_v]))
    # 방문 단위: test 방문을 관측 클러스터 크기대로 무작위 배분
    vperm = rng.permutation(n_test)
    cv = np.empty(n_test, int)
    st = 0
    for c, sz in enumerate(vis_per_cl):
        cv[vperm[st:st + sz]] = c
        st += sz
    by_visit.append(stats(cv))

by_pat = np.array(by_pat)
by_visit = np.array(by_visit)
NAMES = ["const_range", "const_wsd", "prev_range", "prev_wsd"]
obs = stats(te["cl"].to_numpy())

# iid 근사: 동일 크기 클러스터 가정 시 기대 범위 (E[range of K normals] ~ 3.472*SE for K=15)
sd_c, sd_p = float(te["jac_const"].std()), float(te.loc[mask, "jac_prev"].std())
n_c, n_p = n_test / K, int(mask.sum()) / K
iid = {"const_range_iid_approx": round(3.472 * sd_c / np.sqrt(n_c), 4),
       "prev_range_iid_approx": round(3.472 * sd_p / np.sqrt(n_p), 4)}

rep = {"partition": "diagnose C1_id k=15", "n_perm": NB,
       "subjects_spanning_multiple_observed_clusters": int((d.groupby("SUBJECT_ID")["cl"].nunique() > 1).sum()),
       "n_subjects": n_pat, "iid_approx": iid, "by_stat": {}}
for i, nm in enumerate(NAMES):
    pv_pat = (1 + int((by_pat[:, i] >= obs[i]).sum())) / (1 + NB)
    pv_vis = (1 + int((by_visit[:, i] >= obs[i]).sum())) / (1 + NB)
    rep["by_stat"][nm] = {
        "observed": round(obs[i], 4),
        "patient_shuffle_null_mean": round(float(by_pat[:, i].mean()), 4),
        "patient_shuffle_null_sd": round(float(by_pat[:, i].std(ddof=1)), 4),
        "patient_shuffle_p": round(pv_pat, 5),
        "visit_shuffle_null_mean": round(float(by_visit[:, i].mean()), 4),
        "visit_shuffle_null_sd": round(float(by_visit[:, i].std(ddof=1)), 4),
        "visit_shuffle_p": round(pv_vis, 5),
        "null_sd_ratio_patient_over_visit": round(float(by_pat[:, i].std(ddof=1) / by_visit[:, i].std(ddof=1)), 2),
    }

with open(OUT / "18_perm_check.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(json.dumps(rep, ensure_ascii=False, indent=2))
