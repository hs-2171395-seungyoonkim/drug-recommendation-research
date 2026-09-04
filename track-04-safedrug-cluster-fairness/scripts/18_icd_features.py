"""트랙 A/B 공용 입력 생성 — DIAGNOSES_ICD 전체 어휘 multi-hot + 방문별 희귀도 지표.

원칙:
  - 진단은 전부 DIAGNOSES_ICD 에서 가져온다. parsed.pkl 은 처방(정답 집합)과 코호트 정의에만 쓴다.
  - 희귀도(유병률)는 **train split 방문에서만** 추정한다. test 로 추정하면 누수다.
  - 분할은 04_power_baseline.py 와 동일: GroupShuffleSplit(test_size=0.2, random_state=0), SUBJECT_ID 기준.

산출:
  out/20_icd_features.pkl   방문별 특징 (df0 = parsed.pkl 원본 순서)
  out/20_icd_multihot.npz   14,541 x V 희소 multi-hot (df0 행 순서)
  out/20_icd_vocab.json     코드 어휘 + train 유병률
  out/20_icd_features_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
LOG = np.log10  # "−log" 는 전부 log10 으로 통일

# ---------------------------------------------------------------- 코호트 + 분할 (04 와 동일)
df0 = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
df0["ADMITTIME"] = pd.to_datetime(df0["ADMITTIME"])
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(df0, groups=df0["SUBJECT_ID"]))
is_test = np.zeros(len(df0), bool)
is_test[te_idx] = True
df0["split"] = np.where(is_test, "test", "train")

# ---------------------------------------------------------------- ICD 조인
d = pd.read_csv(ROOT / "DIAGNOSES_ICD.csv", dtype={"ICD9_CODE": str})
d["ICD9_CODE"] = d["ICD9_CODE"].str.strip()
d = d[d["HADM_ID"].isin(set(df0["HADM_ID"]))].dropna(subset=["SEQ_NUM", "ICD9_CODE"])
d["SEQ_NUM"] = d["SEQ_NUM"].astype(int)
d = d.sort_values(["HADM_ID", "SEQ_NUM"])

icd_lists = d.groupby("HADM_ID")["ICD9_CODE"].apply(list)
seq1 = d[d["SEQ_NUM"] == 1].drop_duplicates("HADM_ID").set_index("HADM_ID")["ICD9_CODE"]

df0["icd_l"] = df0["HADM_ID"].map(icd_lists)
df0["has_icd"] = df0["icd_l"].map(lambda x: isinstance(x, list) and len(x) > 0)
df0["icd_l"] = df0["icd_l"].map(lambda x: x if isinstance(x, list) else [])
df0["icd_seq1"] = df0["HADM_ID"].map(seq1)
df0["n_icd"] = df0["icd_l"].map(len)
df0["n_ours"] = df0["diag_id_l"].map(len)
df0["n_dropped"] = (df0["n_icd"] - df0["n_ours"]).clip(lower=0)
df0["frac_dropped"] = np.where(df0["n_icd"] > 0, df0["n_dropped"] / df0["n_icd"].replace(0, np.nan), np.nan)

# ---------------------------------------------------------------- 전체 어휘 multi-hot (희소)
vocab = sorted({c for l in df0["icd_l"] for c in l})
vidx = {c: j for j, c in enumerate(vocab)}
rows, cols = [], []
for i, l in enumerate(df0["icd_l"]):
    for c in set(l):
        rows.append(i)
        cols.append(vidx[c])
X = sparse.csr_matrix(
    (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(df0), len(vocab))
)

# ---------------------------------------------------------------- train 유병률 (누수 차단)
tr_mask = (df0["split"] == "train").to_numpy()
n_train = int(tr_mask.sum())
cnt_train = np.asarray(X[tr_mask].sum(axis=0)).ravel()          # train 방문 중 해당 코드 포함 수
prev = (cnt_train + 0.5) / (n_train + 1.0)                       # 가법 평활 (train 미출현 코드 대비)
prev_of = dict(zip(vocab, prev))
FLOOR = 0.5 / (n_train + 1.0)


def _prevs(codes):
    return np.array([prev_of.get(c, FLOOR) for c in codes], float)


rar_mean, rar_meanlog, rar_minprev = [], [], []
for l in df0["icd_l"]:
    if not l:
        rar_mean.append(np.nan); rar_meanlog.append(np.nan); rar_minprev.append(np.nan); continue
    p = _prevs(l)
    rar_mean.append(float(-LOG(p.mean())))       # 명세: "유병률 평균에 −log"
    rar_meanlog.append(float(np.mean(-LOG(p))))  # 강건성: −log 의 평균
    rar_minprev.append(float(p.min()))

df0["rarity_mean"] = rar_mean
df0["rarity_meanlog"] = rar_meanlog
df0["rarity_min_prev"] = rar_minprev
df0["prev_seq1"] = df0["icd_seq1"].map(lambda c: prev_of.get(c, np.nan) if isinstance(c, str) else np.nan)
# 회귀용 스케일: 클수록 희귀. 계수 부호가 음수면 "희귀할수록 못 맞춘다"(주장 지지).
df0["rarity_seq1"] = -LOG(df0["prev_seq1"])
df0["rarity_minprev_log"] = -LOG(df0["rarity_min_prev"])

# ---------------------------------------------------------------- 정답 집합 + 두 자명 예측기 (04 와 동일)
df0["drug_set"] = df0["drug_id_l"].map(set)
train = df0[~is_test]
K_CONST = int(round(train["drug_id_l"].map(len).mean()))
freq = pd.Series([x for s in train["drug_id_l"] for x in s]).value_counts()
const_set = set(freq.head(K_CONST).index)

s = df0.sort_values(["SUBJECT_ID", "ADMITTIME"]).copy()
s["prev_set"] = s.groupby("SUBJECT_ID")["drug_set"].shift(1)


def jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


s["jac_const"] = [jaccard(const_set, t) for t in s["drug_set"]]
s["jac_prev"] = [jaccard(p, t) if isinstance(p, set) else np.nan for p, t in zip(s["prev_set"], s["drug_set"])]
df0["jac_const"] = df0["HADM_ID"].map(dict(zip(s["HADM_ID"], s["jac_const"])))
df0["jac_prev"] = df0["HADM_ID"].map(dict(zip(s["HADM_ID"], s["jac_prev"])))
df0["n_drugs"] = df0["drug_id_l"].map(len)

keep = ["SUBJECT_ID", "HADM_ID", "GENDER", "AGE", "ADMITTIME", "split", "has_icd", "icd_l", "icd_seq1",
        "n_icd", "n_ours", "n_dropped", "frac_dropped", "rarity_mean", "rarity_meanlog", "rarity_min_prev",
        "rarity_minprev_log", "prev_seq1", "rarity_seq1", "jac_const", "jac_prev", "n_drugs",
        "diag_id_l", "diagnose_l", "drug_id_l"]
df0[keep].to_pickle(OUT / "20_icd_features.pkl")
sparse.save_npz(OUT / "20_icd_multihot.npz", X)
with open(OUT / "20_icd_vocab.json", "w", encoding="utf-8") as f:
    json.dump({"vocab": vocab, "train_prevalence": [round(float(p), 8) for p in prev],
               "n_train_visits": n_train, "smoothing": "(count+0.5)/(n_train+1)"}, f, ensure_ascii=False)

te = df0[is_test]
meta = {
    "split": {"train_visits": n_train, "test_visits": int(is_test.sum()),
              "train_subjects": int(df0.loc[~is_test, "SUBJECT_ID"].nunique()),
              "test_subjects": int(df0.loc[is_test, "SUBJECT_ID"].nunique()),
              "leakage_subjects_in_both": int(len(set(df0.loc[~is_test, "SUBJECT_ID"]) & set(df0.loc[is_test, "SUBJECT_ID"])))},
    "icd": {"visits_total": int(len(df0)), "visits_with_icd": int(df0["has_icd"].sum()),
            "visits_without_icd": int((~df0["has_icd"]).sum()),
            "visits_with_seq1": int(df0["icd_seq1"].notna().sum()),
            "vocab_full": len(vocab), "vocab_filtered_ids": int(len({i for l in df0["diag_id_l"] for i in l})),
            "nnz": int(X.nnz), "density": round(float(X.nnz / (X.shape[0] * X.shape[1])), 6)},
    "dropped": {"mean_n_icd": round(float(df0["n_icd"].mean()), 2),
                "mean_n_ours": round(float(df0["n_ours"].mean()), 2),
                "mean_n_dropped": round(float(df0["n_dropped"].mean()), 2),
                "mean_frac_dropped": round(float(df0["frac_dropped"].mean()), 4),
                "pct_visits_lost_ge1": round(float((df0["n_dropped"] > 0).mean()) * 100, 2)},
    "rarity_seq1_neglog10": {"mean": round(float(df0["rarity_seq1"].mean()), 4),
                             "sd": round(float(df0["rarity_seq1"].std()), 4),
                             "min": round(float(df0["rarity_seq1"].min()), 4),
                             "max": round(float(df0["rarity_seq1"].max()), 4),
                             "n_missing": int(df0["rarity_seq1"].isna().sum())},
    "prev_seq1": {"median": round(float(df0["prev_seq1"].median()), 6),
                  "p05": round(float(df0["prev_seq1"].quantile(0.05)), 6),
                  "p95": round(float(df0["prev_seq1"].quantile(0.95)), 6)},
    "baseline": {"K_const": K_CONST, "test_jac_const": round(float(te["jac_const"].mean()), 4),
                 "test_jac_prev": round(float(te["jac_prev"].mean()), 4),
                 "test_visits_with_prev": int(te["jac_prev"].notna().sum())},
}
with open(OUT / "20_icd_features_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2))
