"""
§2 데이터 파악: 행/환자/방문 수, 노트 중복·결측, 필드 길이, 진단 커버리지.
결과는 out/02_profile.json 과 stdout 리포트로 남긴다.
"""
import ast
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data4LLM_with_note.csv"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

LIST_COLS = ["diagnose", "procedure", "drug_name", "diag_id", "pro_id", "drug_id"]


def parse_list(x):
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        v = ast.literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else [v]
    except (ValueError, SyntaxError):
        return []


# ---------- 로드 ----------
df = pd.read_csv(SRC, dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"})

import sys

SMOKE = "--smoke" in sys.argv
if SMOKE:  # §8: 환자 200명 스모크
    keep_ids = df["SUBJECT_ID"].drop_duplicates().head(200)
    df = df[df["SUBJECT_ID"].isin(keep_ids)].reset_index(drop=True)
    OUT = OUT / "smoke"
    OUT.mkdir(exist_ok=True)

report = {"source": str(SRC), "smoke": SMOKE, "columns": list(df.columns)}

for c in LIST_COLS:
    df[c + "_l"] = df[c].map(parse_list)

report["shape"] = {"rows": int(len(df)), "cols": int(df.shape[1] - len(LIST_COLS))}
report["unique"] = {
    "subject_id": int(df["SUBJECT_ID"].nunique()),
    "hadm_id": int(df["HADM_ID"].nunique()),
}

# ---------- 방문당 노트 개수 / 중복 ----------
per_hadm = df.groupby("HADM_ID").size()
report["notes_per_hadm"] = {
    "distribution": {str(k): int(v) for k, v in per_hadm.value_counts().sort_index().items()},
    "max": int(per_hadm.max()),
}
per_subj = df.groupby("SUBJECT_ID")["HADM_ID"].nunique()
report["hadm_per_subject"] = {
    "mean": float(per_subj.mean()),
    "median": float(per_subj.median()),
    "max": int(per_subj.max()),
    "distribution_head": {str(k): int(v) for k, v in per_subj.value_counts().sort_index().head(12).items()},
}

note = df["NOTE"].fillna("")
report["note_quality"] = {
    "missing_or_empty": int((note.str.strip() == "").sum()),
    "exact_duplicate_text_rows": int(note.duplicated(keep=False).sum()),
    "n_unique_note_texts": int(note.nunique()),
    "duplicate_row_key_subject_hadm": int(df.duplicated(["SUBJECT_ID", "HADM_ID"]).sum()),
}

# ---------- 결측률 ----------
miss = {}
for c in df.columns:
    if c.endswith("_l"):
        continue
    s = df[c]
    empty = s.isna()
    if s.dtype == object:
        empty = empty | (s.astype(str).str.strip().isin(["", "[]", "nan", "None"]))
    miss[c] = round(float(empty.mean()) * 100, 3)
report["missing_pct"] = miss

# ---------- 리스트 컬럼 통계 ----------
listing = {}
for c in ["diagnose", "procedure", "drug_name"]:
    n = df[c + "_l"].map(len)
    listing[c] = {
        "pct_rows_empty": round(float((n == 0).mean()) * 100, 3),
        "mean_per_visit": round(float(n.mean()), 2),
        "median_per_visit": float(n.median()),
        "p95": float(n.quantile(0.95)),
        "max": int(n.max()),
        "n_unique_values": int(len(set().union(*df[c + "_l"]))) if len(df) else 0,
    }
report["list_columns"] = listing

# id 리스트와 name 리스트 길이 일치 여부
for name, idc in [("diagnose", "diag_id"), ("procedure", "pro_id"), ("drug_name", "drug_id")]:
    mism = (df[name + "_l"].map(len) != df[idc + "_l"].map(len)).sum()
    report.setdefault("id_length_mismatch", {})[f"{name}_vs_{idc}"] = int(mism)

# ---------- NOTE 섹션 파싱 ----------
HEADER_RE = re.compile(r"(?m)^\s*([A-Z][A-Za-z][A-Za-z /'\-]{2,60}?):")
hdr_counter = Counter()
for t in note.sample(min(5000, len(note)), random_state=0):
    hdr_counter.update({h.strip().lower() for h in HEADER_RE.findall(t)})
report["note_header_candidates_top30"] = [
    {"header": h, "pct_of_sampled_notes": round(c / min(5000, len(note)) * 100, 2)}
    for h, c in hdr_counter.most_common(30)
]

CANON = [
    "history of present illness",
    "past medical history",
    "allergies",
    "medications on admission",
    "brief hospital course",
]
SPLIT_RE = re.compile(
    r"(?mi)^\s*(" + "|".join(re.escape(h) for h in CANON) + r")\s*:", re.MULTILINE
)


def split_sections(text):
    parts = SPLIT_RE.split(text)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        body = parts[i + 1].strip().rstrip(",").strip()
        out[key] = (out.get(key, "") + " " + body).strip()
    out["_preamble"] = parts[0].strip() if parts else ""
    return out


sections = note.map(split_sections)
for h in CANON:
    df["sec_" + h.replace(" ", "_")] = sections.map(lambda d, h=h: d.get(h, ""))
df["sec_unparsed_preamble"] = sections.map(lambda d: d.get("_preamble", ""))

sec_cols = ["sec_" + h.replace(" ", "_") for h in CANON]

# ---------- 필드별 길이 (단어 수, BERT 토큰 근사 = 단어수 * 1.35) ----------
def wc(s):
    return s.str.split().map(len)


lens = {}
for c in ["NOTE"] + sec_cols:
    s = df[c].fillna("") if c != "NOTE" else note
    w = wc(s)
    lens[c] = {
        "pct_present": round(float((w > 0).mean()) * 100, 2),
        "words_mean": round(float(w.mean()), 1),
        "words_median": float(w.median()),
        "words_p90": float(w.quantile(0.90)),
        "words_p99": float(w.quantile(0.99)),
        "words_max": int(w.max()),
        "approx_tokens_p90": round(float(w.quantile(0.90)) * 1.35, 0),
        "pct_over_512_tokens_approx": round(float((w * 1.35 > 512).mean()) * 100, 2),
    }
report["field_lengths"] = lens

# 섹션 파싱 실패(=5개 섹션 하나도 못 찾음) 비율
none_found = (df[sec_cols].apply(lambda col: col.str.len() == 0).all(axis=1))
report["note_section_parsing"] = {
    "pct_notes_with_no_canonical_section": round(float(none_found.mean()) * 100, 2),
    "pct_notes_with_all_5_sections": round(
        float(df[sec_cols].apply(lambda col: col.str.len() > 0).all(axis=1).mean()) * 100, 2
    ),
    "pct_with_nonempty_preamble": round(
        float((df["sec_unparsed_preamble"].str.len() > 0).mean()) * 100, 2
    ),
}

# ---------- LLM 생성 아티팩트 탐지 ----------
ARTIFACT_PATTERNS = {
    "based_on_the_input": r"(?i)based on the (input|provided)",
    "i_will_extract": r"(?i)I will extract",
    "here_is_the_list": r"(?i)here (is|are) the (list|extracted)",
    "input3_marker": r"(?i)\binput\s*\d\b",
    "cannot_or_no_info": r"(?i)(no (drug|information|medication)s? (were |are )?(mentioned|found|listed)|not (specified|provided|mentioned))",
    "deid_placeholder": r"\[\*\*.*?\*\*\]",
}
art = {}
for k, pat in ARTIFACT_PATTERNS.items():
    art[k] = {"pct_of_notes": round(float(note.str.contains(pat, regex=True).mean()) * 100, 2)}
    for c in sec_cols:
        art[k][c] = round(float(df[c].str.contains(pat, regex=True).mean()) * 100, 2)
report["llm_artifacts_pct"] = art

# ---------- 진단 커버리지 ----------
has_note = note.str.strip() != ""
has_bhc = df["sec_brief_hospital_course"].str.len() > 0
has_diag = df["diagnose_l"].map(len) > 0
has_drug = df["drug_name_l"].map(len) > 0

def cov(mask, label):
    return {
        "visits_n": int(mask.sum()),
        "visits_pct": round(float(mask.mean()) * 100, 2),
        "subjects_n": int(df.loc[mask, "SUBJECT_ID"].nunique()),
        "subjects_pct": round(df.loc[mask, "SUBJECT_ID"].nunique() / df["SUBJECT_ID"].nunique() * 100, 2),
        "label": label,
    }

report["coverage"] = {
    "note_nonempty": cov(has_note, "NOTE 비어있지 않음"),
    "bhc_present": cov(has_bhc, "Brief hospital course 섹션 존재"),
    "diagnose_present": cov(has_diag, "diagnose 리스트 비어있지 않음"),
    "drug_present": cov(has_drug, "drug_name 리스트 비어있지 않음"),
    "note_and_diag": cov(has_note & has_diag, "노트+진단 동시 존재"),
    "bhc_and_diag_and_drug": cov(has_bhc & has_diag & has_drug, "BHC+진단+처방 동시 존재 (분석 가능 집합)"),
}

# ---------- 진단 라벨 후보 살펴보기 ----------
first_diag = df["diagnose_l"].map(lambda l: l[0] if l else None)
dc = Counter([d for l in df["diagnose_l"] for d in l])
report["diagnosis_vocab"] = {
    "n_unique_diagnosis_strings": len(dc),
    "top30_overall": dc.most_common(30),
    "top30_as_first_listed": Counter([d for d in first_diag if d]).most_common(30),
    "singleton_share_pct": round(sum(1 for v in dc.values() if v == 1) / max(len(dc), 1) * 100, 2),
    "looks_like_icd_code_rows": int(
        df["diagnose"].fillna("").str.contains(r"'\s*[A-Z]?\d{3}\.?\d*\s*'", regex=True).mean() * 100
    ),
}

# 약물 어휘
drc = Counter([d for l in df["drug_name_l"] for d in l])
report["drug_vocab"] = {
    "n_unique_drug_strings": len(drc),
    "top40": drc.most_common(40),
}

# ---------- 저장 ----------
with open(OUT / "02_profile.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

keep = df[["SUBJECT_ID", "HADM_ID"] + sec_cols + ["sec_unparsed_preamble"]].copy()
keep["diagnose_l"] = df["diagnose_l"]
keep["drug_name_l"] = df["drug_name_l"]
keep["procedure_l"] = df["procedure_l"]
keep.to_pickle(OUT / "note_sections.pkl")

print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])
