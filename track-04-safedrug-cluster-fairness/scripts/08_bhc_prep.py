"""
§1 BHC 전처리: 추출 → 요약실패 제외 → [DEID] 정규화 → 약물 마스킹(+leakage 실측).
산출: out/bhc_prep.pkl (원본/마스킹본 두 벌), out/13_bhc_prep.json (통계).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

sec = pd.read_pickle(OUT / "note_sections.pkl").reset_index(drop=True)
parsed = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)
df = sec[["SUBJECT_ID", "HADM_ID", "sec_brief_hospital_course"]].merge(
    parsed[["SUBJECT_ID", "HADM_ID", "drug_name_l", "diag_id_l"]], on=["SUBJECT_ID", "HADM_ID"]
)
df = df.rename(columns={"sec_brief_hospital_course": "bhc_raw"})
assert len(df) == 14541

rep = {"n_input": int(len(df))}

# ---------------------------------------------------------------- 1-2. 요약 실패 제외
FAIL_PATTERNS = {
    "no_input_provided": r"(?i)\bno input (was )?provided\b",
    "please_provide": r"(?i)\bplease provide\b",
    "based_on_the_input": r"(?i)\bbased on the (input|provided)\b",
    "i_will_extract": r"(?i)\bI will extract\b",
    "empty_or_too_short": None,  # 아래 별도
}
excl = {}
mask_keep = pd.Series(True, index=df.index)
for name, pat in FAIL_PATTERNS.items():
    if pat is None:
        continue
    m = df["bhc_raw"].str.contains(pat, regex=True)
    excl[name] = int((m & mask_keep).sum())
    mask_keep &= ~m
too_short = df["bhc_raw"].str.split().map(len) < 10
excl["empty_or_too_short_lt10w"] = int((too_short & mask_keep).sum())
mask_keep &= ~too_short
rep["excluded_by_reason"] = excl
rep["n_excluded_total"] = int((~mask_keep).sum())
rep["n_kept"] = int(mask_keep.sum())

df = df[mask_keep].reset_index(drop=True)

# ---------------------------------------------------------------- 3. [DEID]
DEID_RE = re.compile(r"\[\*\*[^\]]*?\*\*\]")
n_deid_notes = int(df["bhc_raw"].str.contains(DEID_RE).sum())
n_deid_spans = int(df["bhc_raw"].str.count(DEID_RE).sum())
df["bhc_clean"] = df["bhc_raw"].str.replace(DEID_RE, "[DEID]", regex=True)
# 드물게 짝 잃은 형태
LOOSE_RE = re.compile(r"\[\s*\*\*|\*\*\s*\]|\[Known lastname[^\]]*\]", re.I)
n_loose = int(df["bhc_clean"].str.count(LOOSE_RE).sum())
df["bhc_clean"] = df["bhc_clean"].str.replace(LOOSE_RE, "[DEID]", regex=True)
rep["deid"] = {"notes_with_deid": n_deid_notes, "pct": round(n_deid_notes / len(df) * 100, 1),
               "spans_replaced": n_deid_spans, "loose_spans": n_loose}

# ---------------------------------------------------------------- 4. 약물 마스킹 + leakage 실측
vocab = sorted({d for l in df["drug_name_l"] for d in l})
assert len(vocab) == 151

# 브랜드/약어 → 성분 (성분이 어휘에 있는 것만 채택)
BRAND2ING = {
    "Lasix": "Furosemide", "Coumadin": "Warfarin", "Tylenol": "Acetaminophen",
    "Lopressor": "Metoprolol", "Toprol": "Metoprolol", "Protonix": "Pantoprazole",
    "Prevacid": "Lansoprazole", "Pepcid": "Famotidine", "Zofran": "Ondansetron",
    "Ativan": "Lorazepam", "Versed": "Midazolam", "Dilaudid": "Hydromorphone",
    "Levaquin": "Levofloxacin", "Flagyl": "Metronidazole", "Cipro": "Ciprofloxacin",
    "Lipitor": "Atorvastatin", "Aldactone": "Spironolactone", "Reglan": "Metoclopramide",
    "Dulcolax": "Bisacodyl", "Albuterol": "Salbutamol", "Atrovent": "Ipratropium",
    "Aspirin": "Acetylsalicylic acid", "ASA": "Acetylsalicylic acid",
    "Zosyn": "Tazobactam", "Vanco": "Vancomycin", "Percocet": "Oxycodone",
    "Oxycontin": "Oxycodone", "NTG": "Nitroglycerin", "Neo-Synephrine": "Phenylephrine",
    "Haldol": "Haloperidol", "Colace": "Docusate", "Zantac": "Ranitidine",
    "Prilosec": "Omeprazole", "Plavix": "Clopidogrel", "Cardizem": "Diltiazem",
    "Norvasc": "Amlodipine", "Zocor": "Simvastatin", "Glucophage": "Metformin",
    "Lanoxin": "Digoxin", "Nexium": "Esomeprazole", "Keppra": "Levetiracetam",
    "Dilantin": "Phenytoin", "Solu-Medrol": "Methylprednisolone", "Decadron": "Dexamethasone",
    "Amp": None,  # 모호 — 제외
}
brand2ing = {b: g for b, g in BRAND2ING.items() if g in set(vocab)}
rep["brand_dict"] = {"candidates": len(BRAND2ING), "adopted_ingredient_in_vocab": len(brand2ing),
                     "adopted": sorted(brand2ing)}

# 클래스 표현 사전 (마스킹 전용 — 어떤 성분인지 특정 불가, 한계로 기록)
CLASS_TERMS = [
    r"beta[- ]?blockers?", r"beta[- ]?blockade", r"ace[- ]?inhibitors?", r"arb", r"statins?",
    r"diuretics?", r"diures\w+", r"pressors?", r"vasopressors?", r"inotropes?",
    r"antibiotics?", r"abx", r"broad[- ]spectrum", r"anticoagulat\w+", r"anticoagulants?",
    r"antiplatelets?", r"steroids?", r"sedati\w+", r"analgesi\w+", r"opioids?", r"narcotics?",
    r"benzodiazepines?", r"ppi", r"proton[- ]pump inhibitors?", r"bronchodilators?",
    r"calcium[- ]channel blockers?", r"antiemetics?", r"laxatives?", r"antihypertensives?",
    r"antifungals?", r"antivirals?", r"immunosuppress\w+", r"chemotherapy", r"insulin",
    r"heparin (drip|gtt)",
]
rep["class_terms_n"] = len(CLASS_TERMS)

# 성분/브랜드별 개별 정규식 (언급 탐지용) — 단어 경계, 복수형 허용
def rx(term):
    return re.compile(r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"s?(?![A-Za-z])", re.I)

ing_rx = {g: rx(g) for g in vocab}
brand_rx = {b: rx(b) for b in brand2ing}

# 방문별 언급 집합 (strict = 성분만 / extended = +브랜드)
bhc = df["bhc_clean"].tolist()
presc = [set(l) for l in df["drug_name_l"]]
mention_strict, mention_ext = [], []
for t in bhc:
    ms = {g for g, r in ing_rx.items() if r.search(t)}
    me = ms | {brand2ing[b] for b, r in brand_rx.items() if r.search(t)}
    mention_strict.append(ms)
    mention_ext.append(me)

def rp(mentions):
    rec, prec, ninter = [], [], []
    for m, p in zip(mentions, presc):
        inter = m & p
        rec.append(len(inter) / len(p) if p else np.nan)
        prec.append(len(inter) / len(m) if m else np.nan)
        ninter.append(len(inter))
    return np.array(rec, float), np.array(prec, float), np.array(ninter, float)

for tag, men in [("strict_ingredient_only", mention_strict), ("extended_plus_brands", mention_ext)]:
    rec, prec, ninter = rp(men)
    nm = np.array([len(m) for m in men], float)
    rep[f"leakage_{tag}"] = {
        "pct_notes_with_any_mention": round(float((nm > 0).mean()) * 100, 1),
        "mentions_per_visit_mean": round(float(nm.mean()), 2),
        "mentions_per_visit_median": float(np.median(nm)),
        "recall_mean": round(float(np.nanmean(rec)), 4),
        "recall_median": round(float(np.nanmedian(rec)), 4),
        "recall_q1": round(float(np.nanquantile(rec, .25)), 4),
        "recall_q3": round(float(np.nanquantile(rec, .75)), 4),
        "recall_p90": round(float(np.nanquantile(rec, .90)), 4),
        "pct_recall_ge_50": round(float(np.nanmean(rec >= .5)) * 100, 1),
        "precision_mean": round(float(np.nanmean(prec)), 4),
        "precision_median": round(float(np.nanmedian(prec)), 4),
    }

# ---------------------------------------------------------------- 마스킹 (긴 것 먼저)
all_drug_terms = sorted(list(vocab) + list(brand2ing), key=len, reverse=True)
drug_big_rx = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(t).replace(r"\ ", r"[\s-]+") for t in all_drug_terms) + r")s?(?![A-Za-z])",
    re.I,
)
class_big_rx = re.compile(r"(?<![A-Za-z])(?:" + "|".join(CLASS_TERMS) + r")(?![A-Za-z])", re.I)

n_drug_spans = int(df["bhc_clean"].str.count(drug_big_rx).sum())
df["bhc_masked"] = df["bhc_clean"].str.replace(drug_big_rx, "[DRUG]", regex=True)
n_class_spans = int(df["bhc_masked"].str.count(class_big_rx).sum())
df["bhc_masked"] = df["bhc_masked"].str.replace(class_big_rx, "[DRUG]", regex=True)

tok_clean = df["bhc_clean"].str.split().map(len)
tok_masked_drug = df["bhc_masked"].str.count(re.escape("[DRUG]"))
rep["masking"] = {
    "drug_spans_replaced": n_drug_spans,
    "class_spans_replaced": n_class_spans,
    "pct_notes_with_any_drug_token": round(float((tok_masked_drug > 0).mean()) * 100, 1),
    "drug_tokens_per_note_mean": round(float(tok_masked_drug.mean()), 2),
    "pct_tokens_masked_mean": round(float((tok_masked_drug / tok_clean).mean()) * 100, 2),
    "limitation": "클래스 표현 사전은 불완전하다(용량 표현, 신규 브랜드, 오탈자 미포함). "
                  "브랜드 사전은 성분이 151개 어휘에 있는 것만 채택.",
}

df["n_drug"] = df["drug_name_l"].map(len)
df["n_diag"] = df["diag_id_l"].map(len)
df["recall_ext"], df["precision_ext"], _ = rp(mention_ext)
df[["SUBJECT_ID", "HADM_ID", "bhc_raw", "bhc_clean", "bhc_masked",
    "n_drug", "n_diag", "recall_ext", "precision_ext"]].to_pickle(OUT / "bhc_prep.pkl")

with open(OUT / "13_bhc_prep.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(json.dumps(rep, ensure_ascii=False, indent=2))
