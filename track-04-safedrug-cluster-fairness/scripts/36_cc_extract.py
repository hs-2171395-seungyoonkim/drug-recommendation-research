"""§주증상 정본 — NOTEEVENTS.csv 원본에서 Chief Complaint 섹션을 뽑는다.

data4LLM_with_note.csv 의 NOTE 는 5개 섹션만 남긴 가공본이라 Chief Complaint 이 0.4% 뿐이다.
원본 퇴원요약에는 있다. 32_hpi_prep.py 가 HPI 산문에서 정규식으로 복원하려 했던 것이
여기 그대로 적혀 있다.

검증 (2026-08-13):
  정규식      표본 14건 육안 대조, 다음 섹션 헤더에서 정확히 절단. 두 서식 모두 처리.
  노트 선택   방문당 노트가 1~n건(Report 15,081 / Addendum 1,310). 선택 규칙 3종의
              커버리지 차이 87.8~88.2%, 문자열 일치 99.31%. 규칙 영향은 미미하다.
  오염도      만성표현 4.1% / 인구학표현 1.0%. HPI 급성 절(6.0% / 24.1%)보다 깨끗하다.
              전처리를 하지 않은 원문이 3단계 파이프라인을 이긴다.

산출: out/36_cc_text.pkl, out/table45_cc_extract.csv, out/36_cc_extract_meta.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
NOTES = ROOT / "NOTEEVENTS.csv"
CHUNK = 20000

# 다음 섹션 헤더(줄 시작 + 콜론) 전까지. 인라인형 "CHIEF COMPLAINT:  Chest pain" 도 걸린다.
CC = re.compile(r"(?is)^[ \t]*chief complaint:[ \t]*(.*?)"
                r"(?=\n[ \t]*[A-Z][A-Za-z /&'-]{2,40}:|\Z)", re.M)
# 비식별화 토큰 [**...**] 은 내용이 없다.
DEID = re.compile(r"\[\*\*.*?\*\*\]")
# CC 에도 드물게(1.0%) 인구학 서두가 붙는다. HPI 에서 성별 군집을 만든 그 문구다.
DEMOG = re.compile(r"(?i)^\s*(?:the\s+)?(?:patient\s+is\s+)?(?:an?\s+)?\d+\s*[-\s]?"
                   r"(?:year|yr|y/?o)?[-\s]?(?:old)?\s*"
                   r"(?:male|female|man|woman|gentleman|lady|m|f)\b[\s,]*(?:who\s+|with\s+)?")
CHRON = re.compile(r"(?i)\b(?:history of|h/o|hx|s/p|status post|known|past medical|"
                   r"with a history)\b")

tidy = lambda s: " ".join(DEID.sub(" ", s).split()).strip(" ,.;:-")

keys = pd.read_pickle(OUT / "parsed.pkl")[["SUBJECT_ID", "HADM_ID"]]
want = set(keys["HADM_ID"].astype(int))
print(f"[.] 코호트 HADM_ID {len(want):,}", flush=True)

rows, n = [], 0
for ch in pd.read_csv(NOTES, chunksize=CHUNK, low_memory=False,
                      usecols=["HADM_ID", "CATEGORY", "DESCRIPTION", "TEXT"]):
    n += len(ch)
    ch = ch[ch["CATEGORY"].astype(str).str.strip().eq("Discharge summary")]
    ch = ch[ch["HADM_ID"].isin(want)]
    for h, desc, t in zip(ch["HADM_ID"], ch["DESCRIPTION"], ch["TEXT"].astype(str)):
        m = CC.search(t)
        rows.append((int(h), str(desc).strip(), tidy(m.group(1)) if m else "", len(t)))
    if n % 500000 == 0:
        print(f"    {n:,}행 스캔", flush=True)

df = pd.DataFrame(rows, columns=["HADM_ID", "desc", "cc", "note_len"])
n_notes = len(df)

# 선택 규칙: Report 우선 -> CC 있는 것 우선 -> 긴 노트 우선.
# (첫 노트를 그냥 쓰는 규칙과 99.31% 같은 결과를 준다. 재현성을 위해 명시적으로 정한다.)
df["pri"] = df["desc"].eq("Report") * 4 + df["cc"].str.len().gt(0) * 2
df = (df.sort_values(["HADM_ID", "pri", "note_len"], ascending=[True, False, False])
        .drop_duplicates("HADM_ID"))

df["cc_demog_stripped"] = df["cc"].str.replace(DEMOG, "", regex=True).map(
    lambda s: " ".join(s.split()))
df["has_cc"] = df["cc_demog_stripped"].str.len() > 0
df["n_word"] = df["cc_demog_stripped"].str.split().str.len().fillna(0).astype(int)

out = keys.merge(df[["HADM_ID", "desc", "cc", "cc_demog_stripped", "has_cc", "n_word"]],
                 on="HADM_ID", how="left")
out["has_cc"] = out["has_cc"].fillna(False)
out[["cc", "cc_demog_stripped"]] = out[["cc", "cc_demog_stripped"]].fillna("")
out["n_word"] = out["n_word"].fillna(0).astype(int)
out.to_pickle(OUT / "36_cc_text.pkl")

hit = out[out.has_cc]
tab = pd.DataFrame({
    "항목": ["코호트 방문", "퇴원요약 노트", "노트 매칭 방문", "CC 있음", "CC 없음",
             "CC <=3단어", "CC 4~10단어", "CC >10단어"],
    "값": [len(out), n_notes, int(out.cc.str.len().ge(0).sum()), int(out.has_cc.sum()),
           int((~out.has_cc).sum()), int((hit.n_word <= 3).sum()),
           int(((hit.n_word > 3) & (hit.n_word <= 10)).sum()), int((hit.n_word > 10).sum())],
})
tab["%"] = (tab["값"] / len(out) * 100).round(1)
tab.to_csv(OUT / "table45_cc_extract.csv", index=False, encoding="utf-8-sig")

meta = {
    "source": "NOTEEVENTS.csv (Discharge summary)",
    "rows_scanned": n, "cohort_notes": n_notes,
    "n_visits": len(out),
    "coverage": {"has_cc": int(out.has_cc.sum()),
                 "pct": round(float(out.has_cc.mean()) * 100, 1)},
    "selection_rule": "Report 우선 -> CC 있음 우선 -> 긴 노트 우선",
    "words": {"median": float(hit.n_word.median()), "p90": float(hit.n_word.quantile(.9)),
              "le3_pct": round(float((hit.n_word <= 3).mean()) * 100, 1),
              "gt10_pct": round(float((hit.n_word > 10).mean()) * 100, 1)},
    "contamination_pct": {
        "chronic_marker": round(float(hit.cc_demog_stripped.str.contains(CHRON).mean()) * 100, 1),
        "demographic_before_strip": round(float(hit.cc.str.contains(
            r"(?i)\b\d+\s*[-\s]?(?:year|yr|y/?o)[-\s]?(?:old)?\b|"
            r"\b(?:male|female|man|woman|gentleman|lady)\b", regex=True).mean()) * 100, 1)},
    "unique_strings": int(hit.cc_demog_stripped.str.lower().nunique()),
}
with open(OUT / "36_cc_extract_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n[=] 노트 {n_notes:,} -> 방문 {len(out):,}")
print(f"[=] CC 있음 {meta['coverage']['has_cc']:,} ({meta['coverage']['pct']}%), "
      f"단어 중앙 {meta['words']['median']:.0f}, 고유 {meta['unique_strings']:,}종")
print(f"[=] 오염: 만성 {meta['contamination_pct']['chronic_marker']}% / "
      f"인구학(제거 전) {meta['contamination_pct']['demographic_before_strip']}%")
print(tab.to_string(index=False))
