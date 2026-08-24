"""§ICD-9 → 텍스트 — 방문별 진단 목록을 한 문자열로 조립한다.

선생님이 보내주신 D_ICD_DIAGNOSES.csv(Downloads 판)만 쓴다. 루트의 동명 파일은 공식
4컬럼판이라 CONCISE_TITLE 이 없다.

미매핑 코드(진단행의 2.34%)는 버린다. ICD-9-CM 개정 때 하위분류된 상위코드라 사전에
이름이 없다 — REPORT_ICD 에서 이미 확인한 현상이고, 버려도 14,541 방문 전부가 최소
1개 제목을 유지한다.

순서는 SEQ_NUM 오름차순 그대로 둔다. 1번이 청구서 주 진단 자리라 순서 자체가 정보다.
결합은 SEP("; ") 로만 한다. 선생님 지시가 "진단들의 목록을 통째로" 이므로 프롬프트
문구를 얹지 않는다.

2026-08-23: 다른 세션이 여기에 "Principal Diagnosis: … Comorbidities: …" 스캐폴딩을
넣어 두었던 것을 되돌렸다. 방문당 문자 중앙이 +37 늘어(short 282->319) 임베딩과
군집이 전부 바뀌었고, 이미 나간 전달물(long k=25)의 근거 파일이
말없이 갈렸다. 그 판의 산출물은 out/archive_prompt_cls/ 에 스크립트째 남겨 두었다.

소아 방문(AGE<=18, 97건)은 제외한다. 원본 data4LLM_with_note.csv 에 신생아 88건이
섞여 있는데, 진단 어휘가 성인과 겹치지 않는 0.6% 짜리 섬이라 두 가지를 망가뜨렸다:
  (1) k-means 가 센트로이드 하나를 이 섬에 쓸지 말지가 시드마다 갈려서, 규칙B의
      "최소군집>=100" 문턱이 시드 운으로 뒤집혔다 (inertia 차이는 0.05~0.11%).
  (2) 신생아 코드의 전체 유병률이 0.6% 라 군집에 12%만 있어도 lift 가 21.8배로 튀어,
      long k=20 군집13(n=667, 실제 신생아 12%)이 "신생아·미숙아" 로 오라벨되었다.
성인 ICU 약물 추천이 주제이므로 제외가 맞다. 제외 전 산출물은 out/archive_withpeds/.

산출: out/40_dxtext.pkl, out/table61_dxtext_profile.csv, out/40_dxtext_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MAP_CSV = Path.home() / "Downloads" / "D_ICD_DIAGNOSES.csv"
VARIANTS = ["SHORT_TITLE", "LONG_TITLE", "CONCISE_TITLE"]
MIN_AGE = 18          # 초과만 남긴다 — 아래 주석 참조
SEP = "; "

m = pd.read_csv(MAP_CSV, dtype={"ICD9_CODE": str})
assert set(VARIANTS) <= set(m.columns), f"맵핑 파일에 {VARIANTS} 가 없다: {list(m.columns)}"
assert not m["ICD9_CODE"].duplicated().any()
m["ICD9_CODE"] = m["ICD9_CODE"].str.strip()

coh = pd.read_csv(ROOT / "data4LLM_with_note.csv",
                  usecols=["SUBJECT_ID", "HADM_ID", "AGE"])
n_all = len(coh)
peds = coh[coh["AGE"] <= MIN_AGE]
coh = coh[coh["AGE"] > MIN_AGE].drop(columns=["AGE"]).reset_index(drop=True)
print(f"[.] 소아 제외: 방문 {n_all} -> {len(coh)} (제외 {len(peds)}건, "
      f"환자 {peds['SUBJECT_ID'].nunique()}명, 나이 {sorted(peds['AGE'].unique())})",
      flush=True)
d = pd.read_csv(ROOT / "DIAGNOSES_ICD.csv", dtype={"ICD9_CODE": str})
n_raw = len(d)
d = d[d["HADM_ID"].isin(set(coh["HADM_ID"]))].copy()
d["ICD9_CODE"] = d["ICD9_CODE"].str.strip()
n_null = int(d["ICD9_CODE"].isna().sum())
d = d.dropna(subset=["ICD9_CODE"])
n_cohort = len(d)

known = set(m["ICD9_CODE"])
unmapped = d[~d["ICD9_CODE"].isin(known)]
d = d[d["ICD9_CODE"].isin(known)]
print(f"[.] 진단행 {n_raw} -> 코호트 {n_cohort} -> 매핑 {len(d)} "
      f"(미매핑 {len(unmapped)}, {len(unmapped) / n_cohort * 100:.2f}%)", flush=True)

d = d.sort_values(["HADM_ID", "SEQ_NUM"], kind="mergesort")
d = d.merge(m[["ICD9_CODE"] + VARIANTS], on="ICD9_CODE", how="left")

g = d.groupby("HADM_ID", sort=True)
df = pd.DataFrame({"HADM_ID": sorted(d["HADM_ID"].unique())}).set_index("HADM_ID")
df["icd_l"] = g["ICD9_CODE"].apply(list)
df["n_dx"] = df["icd_l"].str.len()
seq1 = d[d["SEQ_NUM"] == 1].set_index("HADM_ID")["ICD9_CODE"]
df["seq1_code"] = seq1.reindex(df.index)
for v in VARIANTS:
    df[v.split("_")[0].lower()] = g[v].apply(lambda s: SEP.join(s.astype(str)))

df = df.reset_index().merge(coh, on="HADM_ID", how="right").sort_values("HADM_ID")
assert df["n_dx"].notna().all(), "제목이 하나도 안 남은 방문이 있다"
df["n_dx"] = df["n_dx"].astype(int)
df = df.reset_index(drop=True)
print(f"[.] 방문 {len(df)} | 진단 수 중앙 {df.n_dx.median():.0f} "
      f"(p10 {df.n_dx.quantile(.1):.0f}, p90 {df.n_dx.quantile(.9):.0f}, 최대 {df.n_dx.max()})",
      flush=True)

cols = ["short", "long", "concise"]
prof = pd.DataFrame([{
    "변형": c,
    "방문": len(df),
    "문자 중앙": int(df[c].str.len().median()),
    "문자 p90": int(df[c].str.len().quantile(.9)),
    "단어 중앙": int(df[c].str.split().str.len().median()),
    "단어 p90": int(df[c].str.split().str.len().quantile(.9)),
    "고유 제목 수": int(m[VARIANTS[i]].nunique()),
} for i, c in enumerate(cols)])
prof.to_csv(OUT / "table61_dxtext_profile.csv", index=False, encoding="utf-8-sig")
print(prof.to_string(index=False), flush=True)

df.to_pickle(OUT / "40_dxtext.pkl")
meta = {
    "map_file": str(MAP_CSV),
    "map_rows": int(len(m)),
    "sep": SEP,
    "order": "SEQ_NUM asc",
    "rows_raw": n_raw, "rows_cohort": n_cohort, "rows_null_code": n_null,
    "rows_mapped": int(len(d)),
    "unmapped_rows": int(len(unmapped)),
    "unmapped_pct": round(len(unmapped) / n_cohort * 100, 3),
    "unmapped_top10": unmapped["ICD9_CODE"].value_counts().head(10).to_dict(),
    "visits": int(len(df)),
    "visits_with_seq1": int(df["seq1_code"].notna().sum()),
    "n_dx": {"mean": round(float(df.n_dx.mean()), 3), "median": int(df.n_dx.median()),
             "max": int(df.n_dx.max())},
}
(OUT / "40_dxtext_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
print("[+] out/40_dxtext.pkl")
