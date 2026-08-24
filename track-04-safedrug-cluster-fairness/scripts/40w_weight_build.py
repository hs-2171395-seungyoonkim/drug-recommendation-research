"""§ICD-9 가중치 텍스트 빌드 — 주진단과 부진단을 분리해서 텍스트로 만든다.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MAP_CSV = Path.home() / "Downloads" / "D_ICD_DIAGNOSES.csv"
VARIANTS = ["SHORT_TITLE", "LONG_TITLE", "CONCISE_TITLE"]
MIN_AGE = 18
SEP = "; "

m = pd.read_csv(MAP_CSV, dtype={"ICD9_CODE": str})
m["ICD9_CODE"] = m["ICD9_CODE"].str.strip()

coh = pd.read_csv(ROOT / "data4LLM_with_note.csv", usecols=["SUBJECT_ID", "HADM_ID", "AGE"])
coh = coh[coh["AGE"] > MIN_AGE].drop(columns=["AGE"]).reset_index(drop=True)

d = pd.read_csv(ROOT / "DIAGNOSES_ICD.csv", dtype={"ICD9_CODE": str})
d = d[d["HADM_ID"].isin(set(coh["HADM_ID"]))].dropna(subset=["ICD9_CODE"])
d["ICD9_CODE"] = d["ICD9_CODE"].str.strip()
d = d[d["ICD9_CODE"].isin(set(m["ICD9_CODE"]))]

d = d.sort_values(["HADM_ID", "SEQ_NUM"], kind="mergesort")
d = d.merge(m[["ICD9_CODE"] + VARIANTS], on="ICD9_CODE", how="left")

g = d.groupby("HADM_ID", sort=True)
df = pd.DataFrame({"HADM_ID": sorted(d["HADM_ID"].unique())}).set_index("HADM_ID")
df["n_dx"] = g["ICD9_CODE"].apply(list).str.len()
seq1 = d[d["SEQ_NUM"] == 1].set_index("HADM_ID")["ICD9_CODE"]
df["seq1_code"] = seq1.reindex(df.index)

def get_main(s):
    l = list(s.astype(str))
    return l[0] if len(l) > 0 else ""

def get_sub(s):
    l = list(s.astype(str))
    return SEP.join(l[1:]) if len(l) > 1 else ""

for v in VARIANTS:
    col = v.split("_")[0].lower()
    df[col + "_main"] = g[v].apply(get_main)
    df[col + "_sub"] = g[v].apply(get_sub)

df = df.reset_index().merge(coh, on="HADM_ID", how="right").sort_values("HADM_ID").reset_index(drop=True)

print(f"[.] 방문 {len(df)} | 진단 수 중앙 {df.n_dx.median():.0f}")
df.to_pickle(OUT / "40w_weight.pkl")
print("[+] out/40w_weight.pkl 생성 완료")
