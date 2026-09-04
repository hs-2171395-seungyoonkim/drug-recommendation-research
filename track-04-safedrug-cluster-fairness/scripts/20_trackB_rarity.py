"""트랙 B — 희귀도 연속변수. 클러스터를 만들지 않고 방문 단위로 사슬 3고리를 검증한다.

핵심 규율:
  - 희귀도(유병률)는 train split 방문에서만 추정한 값이다 (18_icd_features.py 에서 생성).
  - rarity_seq1 = -log10(SEQ_NUM=1 코드의 train 유병률). 값이 클수록 희귀.
    따라서 계수가 **음수면** "주 진단이 희귀할수록 전역 추천이 안 맞는다"(주장 지지).
  - n_drugs 통제는 필수. 상수 예측기 Jaccard 는 정답 집합 크기에 기계적으로 딸려간다.
  - 표준오차는 전부 환자 단위 클러스터 로버스트.

산출: out/table10_trackB_reg.csv, table10b_trackB_fullmodel.csv, table11_trackB_quintile.csv,
      table12_trackB_control.csv, table12b_trackB_placebo.csv,
      out/22_trackB_meta.json, out/perm_rarity_null.npz
      (사슬 고리 1·2 의 상관값은 22_trackB_meta.json 의 link1_*/link2_* 에 있다)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SEED = 0
N_PERM = 1000

d = pd.read_pickle(OUT / "20_icd_features.pkl")
d["n_diagnoses"] = d["n_icd"].astype(float)
d["age"] = d["AGE"].astype(float)
d["male"] = (d["GENDER"] == "male").astype(float)
d["n_drugs"] = d["n_drugs"].astype(float)

te = d[d["split"] == "test"].reset_index(drop=True)
te_p = te[te["jac_prev"].notna()].reset_index(drop=True)
RAR_SD = float(te["rarity_seq1"].std())
AGE_SD = float(te["age"].std())

# ================================================================= B-2 고리 1
# 희귀할수록 어휘에서 더 탈락하는가: rarity_mean vs frac_dropped
m1 = d[d["frac_dropped"].notna() & d["rarity_mean"].notna()]
link1 = {
    "n_visits": int(len(m1)),
    "pearson_r": round(float(stats.pearsonr(m1["rarity_mean"], m1["frac_dropped"]).statistic), 4),
    "pearson_p": float(stats.pearsonr(m1["rarity_mean"], m1["frac_dropped"]).pvalue),
    "spearman_rho": round(float(stats.spearmanr(m1["rarity_mean"], m1["frac_dropped"]).statistic), 4),
    "spearman_p": float(stats.spearmanr(m1["rarity_mean"], m1["frac_dropped"]).pvalue),
    "rarity_meanlog_pearson_r": round(float(stats.pearsonr(m1["rarity_meanlog"], m1["frac_dropped"]).statistic), 4),
    "rarity_seq1_vs_fracdropped_pearson_r": round(float(stats.pearsonr(m1["rarity_seq1"], m1["frac_dropped"]).statistic), 4),
}

# 보조: 코드 수준 — 협업자 어휘가 표현할 수 있는 ICD 코드 집합 vs 표현 못 하는 집합의 유병률 비교
# (진단 개수가 정확히 일치하는 방문에서 diag_id -> ICD 다수결 사전을 만들고, 그 상(image)을 '표현 가능'으로 본다)
pairs = defaultdict(Counter)
for ids, icds in zip(d["diag_id_l"], d["icd_l"]):
    if len(ids) == len(icds):
        for i, c in zip(ids, icds):
            pairs[i][c] += 1
representable = {c.most_common(1)[0][0] for c in pairs.values()}
voc = json.load(open(OUT / "20_icd_vocab.json", encoding="utf-8"))
prev_of = dict(zip(voc["vocab"], voc["train_prevalence"]))
kept = np.array([prev_of[c] for c in voc["vocab"] if c in representable])
lost = np.array([prev_of[c] for c in voc["vocab"] if c not in representable])
link1_code = {
    "n_codes_total": len(voc["vocab"]),
    "n_codes_representable": int(len(kept)),
    "n_codes_dropped": int(len(lost)),
    "median_train_prevalence_kept": round(float(np.median(kept)), 6),
    "median_train_prevalence_dropped": round(float(np.median(lost)), 6),
    "mean_neglog10_prev_kept": round(float(np.mean(-np.log10(kept))), 3),
    "mean_neglog10_prev_dropped": round(float(np.mean(-np.log10(lost))), 3),
    "mannwhitney_p": float(stats.mannwhitneyu(kept, lost).pvalue),
}

# ================================================================= B-2 고리 2 (주 가설)
link2 = {
    "n_test_visits": int(len(te)),
    "pearson_r": round(float(stats.pearsonr(te["rarity_seq1"], te["jac_const"]).statistic), 4),
    "pearson_p": float(stats.pearsonr(te["rarity_seq1"], te["jac_const"]).pvalue),
    "spearman_rho": round(float(stats.spearmanr(te["rarity_seq1"], te["jac_const"]).statistic), 4),
    "spearman_p": float(stats.spearmanr(te["rarity_seq1"], te["jac_const"]).pvalue),
    "pearson_r_vs_jac_prev": round(float(stats.pearsonr(te_p["rarity_seq1"], te_p["jac_prev"]).statistic), 4),
    "pearson_p_vs_jac_prev": float(stats.pearsonr(te_p["rarity_seq1"], te_p["jac_prev"]).pvalue),
    "rarity_seq1_vs_n_drugs_r": round(float(stats.pearsonr(te["rarity_seq1"], te["n_drugs"]).statistic), 4),
    "n_drugs_vs_jac_const_r": round(float(stats.pearsonr(te["n_drugs"], te["jac_const"]).statistic), 4),
}

# ================================================================= B-3 회귀 (환자 단위 클러스터 로버스트)
def fit(data, formula, label, dv, focus="rarity_seq1", scale=None):
    m = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["SUBJECT_ID"]}
    )
    b = float(m.params[focus])
    se = float(m.bse[focus])
    ci = m.conf_int().loc[focus]
    sc = scale if scale is not None else 1.0
    return {
        "모형": label, "종속변수": dv, "초점변수": focus, "n": int(m.nobs),
        "환자수": int(data["SUBJECT_ID"].nunique()),
        "계수": round(b, 5), "로버스트SE": round(se, 5),
        "t": round(float(m.tvalues[focus]), 2), "p": float(m.pvalues[focus]),
        "CI저": round(float(ci[0]), 5), "CI고": round(float(ci[1]), 5),
        "계수_1SD당": round(b * sc, 5), "SD당_CI저": round(float(ci[0]) * sc, 5),
        "SD당_CI고": round(float(ci[1]) * sc, 5), "R2": round(float(m.rsquared), 4),
    }, m


F1 = "{dv} ~ rarity_seq1"
F2 = "{dv} ~ rarity_seq1 + n_drugs"
F3 = "{dv} ~ rarity_seq1 + n_drugs + n_diagnoses + frac_dropped + age + male"
F_L3 = "{dv} ~ rarity_seq1 + frac_dropped"          # 고리 3 전용: 탈락률만 통제
F_L3b = "{dv} ~ rarity_seq1 + n_drugs + frac_dropped"

reg_rows, models = [], {}
for dv, data in [("상수 top-23 Jaccard", te), ("copy-prev Jaccard", te_p)]:
    col = "jac_const" if data is te else "jac_prev"
    for lbl, F in [("1단계 단변량", F1), ("2단계 +약물수", F2), ("3단계 전체통제", F3),
                   ("고리3 +탈락률만", F_L3), ("고리3 +약물수+탈락률", F_L3b)]:
        r, m = fit(data, F.format(dv=col), lbl, dv, scale=RAR_SD)
        reg_rows.append(r)
        models[(dv, lbl)] = m
reg = pd.DataFrame(reg_rows)
reg.to_csv(OUT / "table10_trackB_reg.csv", index=False)

# 전체통제 모형의 모든 계수 (교란이 어디로 갔는지 보기 위해)
full_terms = []
for dv, data in [("상수 top-23 Jaccard", te), ("copy-prev Jaccard", te_p)]:
    m = models[(dv, "3단계 전체통제")]
    for t in m.params.index:
        ci = m.conf_int().loc[t]
        full_terms.append({"종속변수": dv, "항": t, "계수": round(float(m.params[t]), 5),
                           "로버스트SE": round(float(m.bse[t]), 5), "t": round(float(m.tvalues[t]), 2),
                           "p": float(m.pvalues[t]), "CI저": round(float(ci[0]), 5), "CI고": round(float(ci[1]), 5)})
pd.DataFrame(full_terms).to_csv(OUT / "table10b_trackB_fullmodel.csv", index=False)

# ================================================================= 오분위 표
def boot_ci(sub, col, n_boot=1000, seed=SEED):
    r = np.random.default_rng(seed)
    subs = sub["SUBJECT_ID"].unique()
    by = {s: g[col].to_numpy() for s, g in sub.groupby("SUBJECT_ID")}
    means = []
    for _ in range(n_boot):
        pick = r.choice(subs, size=len(subs), replace=True)
        v = np.concatenate([by[s] for s in pick])
        v = v[~np.isnan(v)]
        if len(v):
            means.append(v.mean())
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


te = te.copy()
te["quintile"] = pd.qcut(te["rarity_seq1"], 5, labels=["Q1 최다빈", "Q2", "Q3", "Q4", "Q5 최희귀"])
q_rows = []
for q, sub in te.groupby("quintile", observed=True):
    lo, hi = boot_ci(sub, "jac_const")
    sp = sub[sub["jac_prev"].notna()]
    lo2, hi2 = boot_ci(sp, "jac_prev")
    q_rows.append({
        "오분위": str(q), "n": len(sub), "환자수": sub["SUBJECT_ID"].nunique(),
        "rarity_seq1 평균": round(float(sub["rarity_seq1"].mean()), 3),
        "유병률 중앙값": round(float(sub["prev_seq1"].median()), 5),
        "상수 Jaccard": round(float(sub["jac_const"].mean()), 4), "CI저": lo, "CI고": hi,
        "copy-prev Jaccard": round(float(sp["jac_prev"].mean()), 4), "prev CI저": lo2, "prev CI고": hi2,
        "평균 약물수": round(float(sub["n_drugs"].mean()), 1),
        "평균 진단수": round(float(sub["n_diagnoses"].mean()), 1),
        "평균 탈락률": round(float(sub["frac_dropped"].mean()), 4),
    })
quint = pd.DataFrame(q_rows)
quint.to_csv(OUT / "table11_trackB_quintile.csv", index=False)

# ================================================================= B-4 대조 1: 희귀도 환자단위 순열
def patient_block_shuffle(rar, groups, rng):
    """환자 블록 통째로 재배정. 길이가 다르면 기증자 블록을 순환시켜 채운다(길이·환자내 상관 보존)."""
    order = rng.permutation(len(groups))
    out = np.empty(len(rar))
    for i, g in enumerate(groups):
        donor = groups[order[i]]
        v = rar[donor]
        out[g] = v[np.arange(len(g)) % len(v)]
    return out


perm_res = {}
null_store = {}
rng = np.random.default_rng(SEED)
for dv, data in [("상수 top-23 Jaccard", te), ("copy-prev Jaccard", te_p)]:
    col = "jac_const" if dv.startswith("상수") else "jac_prev"
    groups = [np.asarray(ix) for ix in data.reset_index(drop=True).groupby("SUBJECT_ID").indices.values()]
    rar0 = data["rarity_seq1"].to_numpy(float)
    work = data.copy().reset_index(drop=True)
    for lbl, F in [("1단계 단변량", F1), ("3단계 전체통제", F3)]:
        obs = float(models[(dv, lbl)].params["rarity_seq1"])
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            work["rarity_seq1"] = patient_block_shuffle(rar0, groups, rng)
            null[b] = float(smf.ols(F.format(dv=col), data=work).fit().params["rarity_seq1"])
        p_two = (1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + N_PERM)
        p_one = (1 + int((null <= obs).sum())) / (1 + N_PERM)
        perm_res[f"{dv} | {lbl}"] = {
            "observed_coef": round(obs, 5), "null_mean": round(float(null.mean()), 5),
            "null_sd": round(float(null.std(ddof=1)), 5),
            "null_p2.5": round(float(np.percentile(null, 2.5)), 5),
            "null_p97.5": round(float(np.percentile(null, 97.5)), 5),
            "p_two_sided": round(p_two, 5), "p_one_sided_negative": round(p_one, 5),
            "z_vs_null": round(float((obs - null.mean()) / null.std(ddof=1)), 2), "n_perm": N_PERM,
        }
        null_store[f"{dv}|{lbl}"] = null
    work["rarity_seq1"] = rar0

# ================================================================= B-4 대조 2: 역방향 위약 (AGE)
FP1 = "{dv} ~ age"
FP2 = "{dv} ~ age + n_drugs"
FP3 = "{dv} ~ age + n_drugs + n_diagnoses + frac_dropped + male"
plac_rows = []
for dv, data in [("상수 top-23 Jaccard", te), ("copy-prev Jaccard", te_p)]:
    col = "jac_const" if dv.startswith("상수") else "jac_prev"
    for lbl, F in [("1단계 단변량", FP1), ("2단계 +약물수", FP2), ("3단계 전체통제", FP3)]:
        r, _ = fit(data, F.format(dv=col), lbl, dv, focus="age", scale=AGE_SD)
        plac_rows.append(r)
plac = pd.DataFrame(plac_rows)

ctrl = pd.concat([
    pd.DataFrame([{"대조": "희귀도 환자단위 순열", "모형": k, **v} for k, v in perm_res.items()]),
], ignore_index=True)
ctrl.to_csv(OUT / "table12_trackB_control.csv", index=False)
plac.to_csv(OUT / "table12b_trackB_placebo.csv", index=False)
np.savez_compressed(OUT / "perm_rarity_null.npz", **{k.replace("|", "__"): v for k, v in null_store.items()})

# ================================================================= 강건성: 다른 희귀도 지표
robust = {}
for var in ["rarity_mean", "rarity_meanlog", "rarity_minprev_log"]:
    m = smf.ols(f"jac_const ~ {var} + n_drugs + n_diagnoses + frac_dropped + age + male", data=te).fit(
        cov_type="cluster", cov_kwds={"groups": te["SUBJECT_ID"]})
    robust[var] = {"coef": round(float(m.params[var]), 5), "se": round(float(m.bse[var]), 5),
                   "p": float(m.pvalues[var]),
                   "coef_per_SD": round(float(m.params[var]) * float(te[var].std()), 5)}

meta = {
    "sample": {"test_visits": int(len(te)), "test_subjects": int(te["SUBJECT_ID"].nunique()),
               "test_visits_with_prev": int(len(te_p)), "rarity_seq1_sd": round(RAR_SD, 4),
               "age_sd": round(AGE_SD, 4),
               "note": "rarity_seq1 = -log10(train 유병률). 클수록 희귀. 음수 계수 = 주장 지지."},
    "link1_rarity_vs_dropped": link1,
    "link1_code_level": link1_code,
    "link2_rarity_vs_jaccard": link2,
    "regression": reg.to_dict("records"),
    "quintile": quint.to_dict("records"),
    "permutation": perm_res,
    "placebo_age": plac.to_dict("records"),
    "robustness_other_rarity_metrics": robust,
}
with open(OUT / "22_trackB_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

pd.set_option("display.width", 250)
print("=== 고리 1: rarity_mean vs frac_dropped ===")
print(json.dumps(link1, ensure_ascii=False, indent=2))
print(json.dumps(link1_code, ensure_ascii=False, indent=2))
print("\n=== 고리 2: rarity_seq1 vs Jaccard ===")
print(json.dumps(link2, ensure_ascii=False, indent=2))
print("\n=== B-3 회귀 계수표 (환자 단위 클러스터 로버스트 SE) ===")
print(reg.to_string(index=False))
print("\n=== 전체통제 모형 전체 계수 ===")
print(pd.DataFrame(full_terms).to_string(index=False))
print("\n=== 오분위 표 ===")
print(quint.to_string(index=False))
print("\n=== 순열 대조 ===")
print(json.dumps(perm_res, ensure_ascii=False, indent=2))
print("\n=== 위약(AGE) ===")
print(plac.to_string(index=False))
print("\n=== 다른 희귀도 지표 ===")
print(json.dumps(robust, ensure_ascii=False, indent=2))
