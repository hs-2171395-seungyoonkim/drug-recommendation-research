"""§1 동반질환 정의 3종 + §2 제거 전 점검.

정의 (주 버전은 모두 **주 진단 보존** — SEQ_NUM=1 코드는 목록에 해당해도 제거하지 않는다):
  COMORB-elix : Elixhauser 동반질환(Quan et al. 2005 ICD-9-CM 매핑) 해당 코드 제거
  CHR-persist : train split 다방문 환자에서 재출현률 >= tau 인 코드 제거 (tau 민감도 동반)
  CHR-idf     : 제거 없음. train IDF 가중만 곱함

이름 주의 — Elixhauser 는 '만성' 목록이 아니라 **동반질환** 목록이고 급성 상태(응고장애,
수분·전해질 장애, 실혈성 빈혈, 체중감소)를 포함한다. 그래서 CHR-elix 가 아니라 COMORB-elix 다.
지속성 기반인 CHR-persist 와 결과가 갈리면 이 차이가 첫 번째 설명 후보다.

구현 근거 — Quan et al. 2005 의 ICD-9-CM 코드 매핑만 쓴다. AHRQ Elixhauser SAS 소프트웨어가
쓰는 **DRG 기반 배제(입원 사유에 해당하는 동반질환을 DRG 로 걸러내기)는 사용하지 않는다.**
주 진단 보존이 그 자리를 대신한다(AHRQ 가 DRG 로 하는 일을 SEQ_NUM=1 로 근사).

매핑은 손으로 만들지 않는다 — comorbidipy 0.8.0 의 codemaps/mapping.py 를 파일 경로로 직접
로드한다(패키지 __init__ 이 polars 를 import 하므로 패키지 import 는 하지 않는다).

산출: out/27_chronic_sets.json, out/27_chronic_lists.pkl, out/table21~26_chronic_*.csv,
      out/27_chronic_recurrence.csv, out/27_chronic_prep_meta.json
"""
import csv
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

MIN_PAT = 10                                  # 재출현률을 추정할 최소 환자 수
TAU_GRID = [round(0.05 * i, 2) for i in range(1, 15)]   # 0.05 ~ 0.70
TAUS_REPORT = [0.3, 0.5, 0.7]                 # 임계 민감도 보고용
TAU_MAIN = 0.5

# Elixhauser 31 카테고리 중 급성으로 의심한 것들 — 최종 판정은 손이 아니라 AHRQ CCI 가 한다.
# 아래는 한글 표기와 '사전 예상'일 뿐이고, 표의 급성/만성 열은 CCI 건수가중 만성비율로 계산된다.
ACUTE_CATS = {"coag": "응고장애", "fed": "수분·전해질 장애", "blane": "실혈성 빈혈", "wloss": "체중감소"}
BORDERLINE_CATS = {"pcd": "폐순환장애(폐색전 등 급성 사건 포함)"}
CCI_ACUTE_CUT = 50.0     # CCI 만성 건수가중% 가 이 값 미만이면 급성 우세 카테고리로 본다

# ---------------------------------------------------------------- Elixhauser 매핑 (외부 표준)
mp = Path(sys.prefix) / "Lib" / "site-packages" / "comorbidipy" / "codemaps" / "mapping.py"
if not mp.exists():
    sys.exit(f"[X] comorbidipy 매핑 파일 없음: {mp}\n    pip install --no-deps comorbidipy")
spec = importlib.util.spec_from_file_location("cmap", mp)
cmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmap)
ELIX = cmap.mapping["elixhauser_icd9_quan"]
ELIX_PREFIXES = {p: cat for cat, codes in ELIX.items() for p in codes}
N_PREFIX_LISTED = sum(len(v) for v in ELIX.values())
print(f"[.] Elixhauser(Quan 2005 ICD-9-CM, DRG 배제 미사용) {len(ELIX)} 카테고리 / "
      f"{N_PREFIX_LISTED} 코드(고유 {len(ELIX_PREFIXES)})", flush=True)

# ------------------------------------------- AHRQ 만성상태지표 CCI 2015 (급성/만성 판정의 외부 표준)
# 손으로 만성 목록을 만들지 않기 위해 쓴다. icd-mappings 0.6.2 의 data_files/cci2015.csv 를 직접 읽는다.
# 원출처: https://www.hcup-us.ahrq.gov/toolssoftware/chronic/chronic.jsp  (1 = 만성, 0 = 급성)
cci_path = Path(sys.prefix) / "Lib" / "site-packages" / "icdmappings" / "data_files" / "cci2015.csv"
CCI = {}
if cci_path.exists():
    with open(cci_path, encoding="utf-8") as f:
        r = csv.reader(f, quotechar="'")
        next(r)
        for row in r:
            if len(row) >= 3:
                CCI[row[0].strip().strip("'").strip()] = row[2].strip().strip("'")
    print(f"[.] AHRQ CCI 2015 {len(CCI):,} 코드 로드", flush=True)
else:
    print("[!] CCI 파일 없음 — 급성/만성 외부 판정 생략", flush=True)

is_chronic = lambda c: (CCI.get(c) == "1") if c in CCI else None

# ---------------------------------------------------------------- 입력
d = pd.read_pickle(OUT / "20_icd_features.pkl")
vocab = json.load(open(OUT / "20_icd_vocab.json", encoding="utf-8"))
VOCAB = vocab["vocab"] if isinstance(vocab, dict) and "vocab" in vocab else vocab
if isinstance(VOCAB, dict):
    VOCAB = list(VOCAB)
VOCAB = [str(c).strip() for c in VOCAB]
VOCAB_SET = set(VOCAB)

d = d.copy()
d["icd_l"] = d["icd_l"].map(lambda l: [str(c).strip() for c in l])
d["icd_seq1"] = d["icd_seq1"].astype(str).str.strip()
is_train = (d["split"] == "train").to_numpy()
SEQ1 = d["icd_seq1"].to_numpy()

dic = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
SHORT = dict(zip(dic["ICD9_CODE"].str.strip(), dic["SHORT_TITLE"].astype(str)))
name_of = lambda c: SHORT.get(c, "(사전에 없음)")

# ---------------------------------------------------------------- COMORB-elix
def elix_cat(code):
    """가장 긴 프리픽스 매칭. Quan 매핑은 접두 매칭이 규약이다."""
    for L in (5, 4, 3):
        if code[:L] in ELIX_PREFIXES:
            return ELIX_PREFIXES[code[:L]]
    return None


elix_cat_of = {c: elix_cat(c) for c in VOCAB}
REM_ELIX = {c for c, cat in elix_cat_of.items() if cat is not None}
cat_hits = pd.Series([cat for cat in elix_cat_of.values() if cat]).value_counts()

# ---------------------------------------------------------------- CHR-persist (train 전용 추정)
tr = d[is_train]
vis_per_pat = tr.groupby("SUBJECT_ID")["HADM_ID"].size()
multi = set(vis_per_pat[vis_per_pat >= 2].index)
tr_multi = tr[tr["SUBJECT_ID"].isin(multi)]

pat_counts = {}
for sid, codes in zip(tr_multi["SUBJECT_ID"].to_numpy(), tr_multi["icd_l"]):
    for c in set(codes):
        pat_counts.setdefault(c, {})
        pat_counts[c][sid] = pat_counts[c].get(sid, 0) + 1

rec_rows = []
for c in VOCAB:
    pc = pat_counts.get(c, {})
    n_pat = len(pc)
    n_rep = sum(1 for v in pc.values() if v >= 2)
    rec_rows.append({"code": c, "n_pat_train_multi": n_pat, "n_pat_repeat": n_rep,
                     "recur": (n_rep / n_pat) if n_pat >= MIN_PAT else np.nan})
rec = pd.DataFrame(rec_rows).set_index("code")
REM_PERSIST = {t: set(rec.index[(rec["recur"] >= t).fillna(False)]) for t in TAU_GRID}

# ---------------------------------------------------------------- CHR-idf (제거 없음)
cnt_tr = {}
for codes in tr["icd_l"]:
    for c in set(codes):
        cnt_tr[c] = cnt_tr.get(c, 0) + 1
df_train = pd.Series({c: cnt_tr.get(c, 0) for c in VOCAB}, dtype="int64")
N_TR = int(is_train.sum())
IDF = np.log((1 + N_TR) / (1 + df_train.to_numpy())) + 1.0   # sklearn 규약

# ---------------------------------------------------------------- 방문별 남은 진단 리스트
N_INSTANCES = int(d["icd_l"].map(lambda l: len(set(l))).sum())


def apply_removal(rem, keep_seq1=True):
    """방문별 남은 코드 리스트. keep_seq1 이면 주 진단은 목록에 있어도 남긴다."""
    rem = rem & VOCAB_SET
    out = []
    for codes, s1 in zip(d["icd_l"], SEQ1):
        cs = set(codes) & VOCAB_SET
        out.append(sorted(c for c in cs if (c not in rem) or (keep_seq1 and c == s1)))
    return out


def summarize(name, rem, keep_seq1, lists=None):
    lists = lists if lists is not None else apply_removal(rem, keep_seq1)
    n_left = np.array([len(l) for l in lists])
    rem_v = rem & VOCAB_SET
    seq1_removed = np.array([(s in rem_v) and not keep_seq1 for s in SEQ1])
    q = np.percentile(n_left, [0, 25, 50, 75, 100])
    left_vocab = set().union(*lists) if lists else set()
    return {
        "정의": name, "주 진단 보존": "○" if keep_seq1 else "×",
        "제거 코드 수": len(rem_v), "남은 어휘": len(left_vocab),
        "제거된 진단 건수 비율%": round(float((1 - n_left.sum() / N_INSTANCES) * 100), 1),
        "방문당 남은 진단 평균": round(float(n_left.mean()), 2),
        "min": int(q[0]), "Q1": int(q[1]), "중앙": int(q[2]), "Q3": int(q[3]), "max": int(q[4]),
        "남은 진단 0개 방문": int((n_left == 0).sum()),
        "0개 비율%": round(float((n_left == 0).mean() * 100), 2),
        "SEQ_NUM=1 제거 방문%": round(float(seq1_removed.mean() * 100), 2),
    }, n_left, lists


# 주 진단 보존 기준 elix 의 제거 규모에 맞는 tau 를 고른다
elix_row, elix_left, elix_lists = summarize("COMORB-elix", REM_ELIX, True)
ELIX_SHARE = elix_row["제거된 진단 건수 비율%"]
tau_scan = []
for t in TAU_GRID:
    r, _, _ = summarize(f"τ={t}", REM_PERSIST[t], True)
    tau_scan.append({"τ": t, "제거 코드 수": r["제거 코드 수"], "제거된 진단 건수 비율%": r["제거된 진단 건수 비율%"],
                     "0개 비율%": r["0개 비율%"], "elix 와의 규모 차": round(abs(r["제거된 진단 건수 비율%"] - ELIX_SHARE), 2)})
tau_scan = pd.DataFrame(tau_scan)
TAU_SCALE = float(tau_scan.loc[tau_scan["elix 와의 규모 차"].idxmin(), "τ"])
print(f"[.] 주 진단 보존 후 elix 제거 비율 {ELIX_SHARE}% → 규모 정합 τ = {TAU_SCALE}", flush=True)

# ---------------------------------------------------------------- §2 표
DEFS = [
    ("COMORB-elix", REM_ELIX, True),
    (f"CHR-persist(τ={TAU_MAIN})", REM_PERSIST[TAU_MAIN], True),
    (f"CHR-persist(τ={TAU_SCALE}, 규모정합)", REM_PERSIST[TAU_SCALE], True),
    ("CHR-idf", set(), True),
    ("COMORB-elix (주진단 비보존)", REM_ELIX, False),
    (f"CHR-persist(τ={TAU_MAIN}, 주진단 비보존)", REM_PERSIST[TAU_MAIN], False),
]
for t in TAUS_REPORT:
    if t not in (TAU_MAIN, TAU_SCALE):
        DEFS.append((f"CHR-persist(τ={t})", REM_PERSIST[t], True))

rows, dist_rows, LISTS, LEFT = [], [], {}, {}
for name, rem, ks in DEFS:
    r, n_left, lists = summarize(name, rem, ks)
    rows.append(r)
    LISTS[name], LEFT[name] = lists, n_left
    for lab, mask in [("전체", np.ones(len(d), bool)), ("train", is_train), ("test", ~is_train)]:
        dist_rows.append({"정의": name, "split": lab, "방문 수": int(mask.sum()),
                          "남은 진단 평균": round(float(n_left[mask].mean()), 2),
                          "0개 방문": int((n_left[mask] == 0).sum()),
                          "0개 비율%": round(float((n_left[mask] == 0).mean() * 100), 2)})
tab21, tab22 = pd.DataFrame(rows), pd.DataFrame(dist_rows)

# 정의 간 제거 집합 겹침 (코드 집합 기준 — 주 진단 보존은 방문별 예외라 집합은 동일)
ov = []
sets = {"COMORB-elix": REM_ELIX, f"CHR-persist(τ={TAU_MAIN})": REM_PERSIST[TAU_MAIN],
        f"CHR-persist(τ={TAU_SCALE})": REM_PERSIST[TAU_SCALE]}
for t in TAUS_REPORT:
    if t not in (TAU_MAIN, TAU_SCALE):
        sets[f"CHR-persist(τ={t})"] = REM_PERSIST[t]
for a, b in combinations(sets, 2):
    A, B = sets[a] & VOCAB_SET, sets[b] & VOCAB_SET
    ov.append({"A": a, "B": b, "|A|": len(A), "|B|": len(B), "교집합": len(A & B),
               "Jaccard": round(len(A & B) / max(len(A | B), 1), 3),
               "교집합이 차지한 진단 건수%": round(float(sum(df_train.get(c, 0) for c in A & B) / df_train.sum() * 100), 1)})
tab_ov = pd.DataFrame(ov)

# Elixhauser 31 카테고리를 AHRQ CCI 로 급성/만성 판정 — '만성 제거'가 아니라는 외부 근거
ELIX_TOT = max(sum(df_train.get(c, 0) for c in REM_ELIX), 1)
acu_rows = []
for cat in sorted(ELIX):
    codes = [c for c, cc in elix_cat_of.items() if cc == cat]
    if not codes:
        continue
    inst = int(sum(df_train.get(c, 0) for c in codes))
    chr_flags = [is_chronic(c) for c in codes]
    known = [f for f in chr_flags if f is not None]
    # 건수 가중 만성 비율 — 어휘가 아니라 실제로 붙은 진단 기준
    w_chr = sum(df_train.get(c, 0) for c, f in zip(codes, chr_flags) if f)
    w_known = sum(df_train.get(c, 0) for c, f in zip(codes, chr_flags) if f is not None)
    top = sorted(codes, key=lambda c: -df_train.get(c, 0))[:3]
    acu_rows.append({
        "카테고리": cat, "뜻": ACUTE_CATS.get(cat, BORDERLINE_CATS.get(cat, "")),
        "우리 표기": "급성" if cat in ACUTE_CATS else "경계" if cat in BORDERLINE_CATS else "만성",
        "어휘 수": len(codes), "train 진단 건수": inst,
        "elix 제거 건수 중 비중%": round(float(inst / ELIX_TOT * 100), 1),
        "CCI 만성 코드%": round(float(np.mean(known) * 100), 1) if known else np.nan,
        "CCI 만성 건수가중%": round(float(w_chr / w_known * 100), 1) if w_known else np.nan,
        "대표 코드": ", ".join(f"{c} {name_of(c)}" for c in top),
    })
tab_acute = pd.DataFrame(acu_rows).sort_values("CCI 만성 건수가중%")
tab_acute.insert(3, "CCI 판정", np.where(tab_acute["CCI 만성 건수가중%"] < CCI_ACUTE_CUT, "급성 우세", "만성 우세"))
ACUTE_BY_CCI = tab_acute.loc[tab_acute["CCI 판정"] == "급성 우세", "카테고리"].tolist()

# 제거 대상 전체에서 CCI 급성이 차지하는 비중
elix_acute_inst = sum(df_train.get(c, 0) for c in REM_ELIX if is_chronic(c) is False)
elix_known_inst = sum(df_train.get(c, 0) for c in REM_ELIX if is_chronic(c) is not None)
pers_acute_inst = sum(df_train.get(c, 0) for c in REM_PERSIST[TAU_MAIN] if is_chronic(c) is False)
pers_known_inst = sum(df_train.get(c, 0) for c in REM_PERSIST[TAU_MAIN] if is_chronic(c) is not None)
CCI_SUMMARY = {
    "COMORB-elix 제거분 중 CCI 급성 건수%": round(float(elix_acute_inst / max(elix_known_inst, 1) * 100), 1),
    "COMORB-elix 제거 어휘 중 CCI 급성 코드%": round(float(np.mean([is_chronic(c) is False for c in REM_ELIX
                                                             if is_chronic(c) is not None]) * 100), 1),
    f"CHR-persist(τ={TAU_MAIN}) 제거분 중 CCI 급성 건수%": round(float(pers_acute_inst / max(pers_known_inst, 1) * 100), 1),
    "어휘 전체 중 CCI 만성 코드%": round(float(np.mean([is_chronic(c) for c in VOCAB if is_chronic(c) is not None]) * 100), 1),
    "CCI 미분류 어휘": int(sum(1 for c in VOCAB if is_chronic(c) is None)),
    "급성 우세 Elixhauser 카테고리(CCI)": ACUTE_BY_CCI,
    "사전 예상했던 급성 카테고리": list(ACUTE_CATS),
}
print("\n[.] CCI 요약:", json.dumps(CCI_SUMMARY, ensure_ascii=False), flush=True)

# 제거된 코드 상위
top_rows = []
for name, rem in [("COMORB-elix", REM_ELIX),
                  (f"CHR-persist(τ={TAU_MAIN})", REM_PERSIST[TAU_MAIN]),
                  (f"CHR-persist(τ={TAU_SCALE}, 규모정합)", REM_PERSIST[TAU_SCALE])]:
    rem = rem & VOCAB_SET
    s = pd.Series({c: int(df_train.get(c, 0)) for c in rem}).sort_values(ascending=False)
    for c in s.head(10).index:
        cat = elix_cat_of.get(c)
        top_rows.append({"정의": name, "코드": c, "명칭": name_of(c),
                         "train 방문 비율%": round(float(s[c] / N_TR * 100), 1),
                         "Elixhauser 카테고리": cat or "—",
                         "카테고리 성격": ("급성" if cat in ACUTE_CATS else "경계" if cat in BORDERLINE_CATS
                                     else "만성" if cat else "—"),
                         "재출현률": None if np.isnan(rec.loc[c, "recur"]) else round(float(rec.loc[c, "recur"]), 3)})
tab23 = pd.DataFrame(top_rows)

# 두 정의가 엇갈린 코드
only_e = sorted(REM_ELIX - REM_PERSIST[TAU_MAIN], key=lambda c: -df_train.get(c, 0))[:8]
only_p = sorted(REM_PERSIST[TAU_MAIN] - REM_ELIX, key=lambda c: -df_train.get(c, 0))[:8]
tab_diff = pd.DataFrame(
    [{"쪽": "Elixhauser만 제거", "코드": c, "명칭": name_of(c),
      "train 방문%": round(float(df_train.get(c, 0) / N_TR * 100), 1),
      "카테고리": elix_cat_of.get(c),
      "재출현률": None if np.isnan(rec.loc[c, "recur"]) else round(float(rec.loc[c, "recur"]), 3)} for c in only_e]
    + [{"쪽": "persist만 제거", "코드": c, "명칭": name_of(c),
        "train 방문%": round(float(df_train.get(c, 0) / N_TR * 100), 1), "카테고리": "—",
        "재출현률": None if np.isnan(rec.loc[c, "recur"]) else round(float(rec.loc[c, "recur"]), 3)} for c in only_p])

# 주 진단 보존이 되살린 방문 — 비보존이었다면 지워졌을 주 진단
seq1_saved = d[d["icd_seq1"].map(lambda c: c in REM_ELIX)]
seq1_top = seq1_saved["icd_seq1"].value_counts().head(10)
tab_seq1 = pd.DataFrame([{"코드": c, "명칭": name_of(c), "방문 수": int(n),
                          "비중%": round(float(n / len(seq1_saved) * 100), 1),
                          "Elixhauser 카테고리": elix_cat_of.get(c)} for c, n in seq1_top.items()])

# ---------------------------------------------------------------- 저장
tab21.to_csv(OUT / "table21_chronic_removal.csv", index=False)
tab22.to_csv(OUT / "table22_chronic_remaining.csv", index=False)
tab23.to_csv(OUT / "table23_chronic_removed_top.csv", index=False)
tab_ov.to_csv(OUT / "table24_chronic_overlap.csv", index=False)
tab_diff.to_csv(OUT / "table25_chronic_defdiff.csv", index=False)
tab_seq1.to_csv(OUT / "table26_chronic_seq1_kept.csv", index=False)
tab_acute.to_csv(OUT / "table27_chronic_acute_cats.csv", index=False)
tau_scan.to_csv(OUT / "table28_chronic_tau_scan.csv", index=False)
rec.reset_index().to_csv(OUT / "27_chronic_recurrence.csv", index=False)

pd.to_pickle({"lists": LISTS, "n_left": LEFT, "vocab": VOCAB, "idf": IDF,
              "tau_main": TAU_MAIN, "tau_scale": TAU_SCALE}, OUT / "27_chronic_lists.pkl")

json.dump({
    "elix": {"source": "comorbidipy 0.8.0 codemaps/mapping.py :: elixhauser_icd9_quan",
             "reference": "Quan H, et al. Med Care 2005 — ICD-9-CM coding algorithm",
             "drg_exclusion_used": False,
             "primary_dx_preserved_in_main_version": True,
             "n_categories": len(ELIX), "n_prefixes_listed": N_PREFIX_LISTED,
             "n_prefixes_unique": len(ELIX_PREFIXES),
             "acute_categories": ACUTE_CATS, "borderline_categories": BORDERLINE_CATS,
             "removed_vocab": sorted(REM_ELIX),
             "category_hits": {k: int(v) for k, v in cat_hits.items()}},
    "persist": {"min_patients": MIN_PAT, "tau_grid": TAU_GRID, "tau_main": TAU_MAIN,
                "tau_scale_matched": TAU_SCALE,
                "n_train_subjects": int(tr["SUBJECT_ID"].nunique()),
                "n_train_multi_visit_subjects": len(multi),
                "n_codes_estimable": int(rec["recur"].notna().sum()),
                "removed_vocab": {str(t): sorted(REM_PERSIST[t]) for t in TAUS_REPORT + [TAU_SCALE]}},
    "idf": {"n_train_visits": N_TR, "formula": "log((1+N)/(1+df))+1 (train only)",
            "idf": {c: round(float(v), 4) for c, v in zip(VOCAB, IDF)}},
    "cci": {"source": "icd-mappings 0.6.2 data_files/cci2015.csv",
            "reference": "AHRQ Chronic Condition Indicator (ICD-9-CM), hcup-us.ahrq.gov/toolssoftware/chronic",
            "n_codes": len(CCI), "summary": CCI_SUMMARY,
            "chronic": {c: bool(is_chronic(c)) for c in VOCAB if is_chronic(c) is not None}},
    "vocab_size": len(VOCAB),
}, open(OUT / "27_chronic_sets.json", "w", encoding="utf-8"), ensure_ascii=False)

json.dump({
    "naming": "CHR-elix → COMORB-elix (Elixhauser 는 만성 목록이 아니라 동반질환 목록)",
    "mapping": "Quan et al. 2005 ICD-9-CM, DRG 기반 배제 미사용",
    "primary_dx_preserved": True,
    "vocab_size": len(VOCAB), "n_visits": int(len(d)),
    "n_diagnosis_instances": N_INSTANCES,
    "n_train_visits": N_TR, "n_test_visits": int((~is_train).sum()),
    "n_train_multi_visit_subjects": len(multi),
    "n_codes_estimable_recurrence": int(rec["recur"].notna().sum()),
    "elix_n_removed_codes": len(REM_ELIX),
    "elix_removed_share_pct_keepseq1": ELIX_SHARE,
    "elix_removed_share_pct_rmseq1": [r for r in rows if r["정의"] == "COMORB-elix (주진단 비보존)"][0]["제거된 진단 건수 비율%"],
    "seq1_in_elix_pct": round(float(np.isin(SEQ1, list(REM_ELIX)).mean() * 100), 2),
    "seq1_in_elix_n": int(np.isin(SEQ1, list(REM_ELIX)).sum()),
    "tau_main": TAU_MAIN, "tau_scale_matched": TAU_SCALE,
    "zero_remaining_pct": {r["정의"]: r["0개 비율%"] for r in rows},
}, open(OUT / "27_chronic_prep_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 화면 보고
pd.set_option("display.width", 220, "display.max_columns", 40)
print("\n===== §2-1 제거 규모와 남은 진단 (주 버전은 주 진단 보존)")
print(tab21.to_string(index=False))
print("\n===== §2-2 τ 스캔 (주 진단 보존 기준, elix 규모와의 정합)")
print(tau_scan.to_string(index=False))
print("\n===== §2-3 정의 간 제거 집합 겹침")
print(tab_ov.to_string(index=False))
print("\n===== §2-4 Elixhauser 31 카테고리의 급성/만성 (AHRQ CCI 2015 판정)")
print(tab_acute.to_string(index=False))
print("\n===== §2-5 제거된 코드 상위")
print(tab23.to_string(index=False))
print("\n===== §2-6 두 정의가 엇갈린 코드")
print(tab_diff.to_string(index=False))
print(f"\n===== §2-7 주 진단 보존이 되살린 방문 {len(seq1_saved):,}건의 주 진단 상위")
print(tab_seq1.to_string(index=False))
print("\n===== split 별 0개 방문")
print(tab22.to_string(index=False))
