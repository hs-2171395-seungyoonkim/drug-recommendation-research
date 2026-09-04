"""§CCS 트랙 — 주진단을 AHRQ CCS 범주로 묶는다.

"가장 의학적으로 근거가 탄탄하고 확실하게 주진단별로 환자를 나누고 싶다" 의 원래 안은
CCS 매핑이었는데, 실제로 구현된 것은 42c_chapter_cluster.py 의 **ICD-9 챕터**(17개
대분류)였다. 챕터는 CCS 가 아니다. 심혈관 4,181명이 한 덩어리가 되고 임신 4명이
따로 남는 것은 그래서다. 이 스크립트가 원안을 그대로 구현한다.

매핑 원본: AHRQ HCUP Single-Level CCS for ICD-9-CM, 2015판 (`$dxref 2015.csv`).
ref/ccs_dxref_2015.csv 로 넣어 두었다. 진단 범주 284개.

**미리 고정한 규칙** (결과를 보고 정하지 않는다):
  R1 주진단 = SEQ_NUM=1. 기존 트랙과 같은 정의를 쓴다. SEQ_NUM=1 이 없는 방문은
     "미지정" 으로 따로 세고 군집 분석에서 뺀다.
  R2 CCS 에 매핑되지 않는 코드는 "매핑불가" 로 따로 세고 뺀다.
  R3 소규모 범주 문턱 = 200명. viz_clusters.py 가 챕터에 이미 쓰고 있던 문턱과
     같은 값을 쓴다(그래야 두 트랙을 나란히 볼 수 있다). 주 분석에서는 200명 미만
     범주를 **'기타(소규모)' 한 덩어리로 합친다** — 버리지 않는다.
     민감도로 (a) 문턱 100, (b) 소규모 완전 제외 두 가지를 같이 낸다.

같은 특징공간에서 여러 분할의 실루엣을 나란히 재는 것도 여기서 한다. 실루엣은
"이 라벨이 이 좌표계에서 얼마나 뭉쳐 있나" 이지 "의학적으로 옳은가" 가 아니다.
가중치 공간에서는 ICD 사전을 그대로 베낀 분할이 가장 높은 점수를 받는다.

산출: out/ccs_assignments.csv, out/table80_ccs_groups.csv, out/57_ccs_track.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
REF = ROOT / "ref" / "ccs_dxref_2015.csv"
SMALL, SMALL_ALT = 200, 100
PCA_DIM, SIL_SAMPLE = 50, 5000

# ── CCS 사전 파싱 ───────────────────────────────────────────────────────────
raw = pd.read_csv(REF, skiprows=1, dtype=str, quotechar="'")
raw.columns = [c.strip().strip("'") for c in raw.columns]
raw = raw.rename(columns={"ICD-9-CM CODE": "code", "CCS CATEGORY": "ccs",
                          "CCS CATEGORY DESCRIPTION": "ccs_desc"})
for c in ("code", "ccs", "ccs_desc"):
    raw[c] = raw[c].astype(str).str.strip().str.strip('"').str.strip()
raw = raw[(raw["code"] != "") & (raw["ccs"] != "0")]
CCS = dict(zip(raw["code"], raw["ccs"]))
CDESC = dict(zip(raw["ccs"], raw["ccs_desc"]))
print(f"[.] CCS 사전: 코드 {len(CCS)} -> 범주 {raw['ccs'].nunique()}", flush=True)

# ── 코호트 주진단 ───────────────────────────────────────────────────────────
df = pd.read_pickle(OUT / "40_dxtext.pkl")
N_ALL = len(df)
mapdf = pd.read_csv(ROOT / "D_ICD_DIAGNOSES.csv", dtype={"ICD9_CODE": str})
mapdf["ICD9_CODE"] = mapdf["ICD9_CODE"].str.strip()
NAME = dict(zip(mapdf["ICD9_CODE"], mapdf["SHORT_TITLE"]))

a = df[["SUBJECT_ID", "HADM_ID", "n_dx", "seq1_code"]].copy()
a["seq1_name"] = a["seq1_code"].map(NAME)
a["ccs"] = a["seq1_code"].map(CCS)
n_noseq1 = int(a["seq1_code"].isna().sum())
n_unmapped = int(a["ccs"].isna().sum() - n_noseq1)
a["ccs_desc"] = a["ccs"].map(CDESC)
print(f"[.] 방문 {N_ALL} | SEQ_NUM=1 없음 {n_noseq1} | CCS 매핑불가 {n_unmapped} "
      f"| 사용 {N_ALL - n_noseq1 - n_unmapped}", flush=True)

use = a["ccs"].notna()
sizes = a.loc[use, "ccs"].value_counts()
print(f"[.] 등장한 CCS 범주 {len(sizes)}개 | >={SMALL}명 {int((sizes >= SMALL).sum())}개 "
      f"| >={SMALL_ALT}명 {int((sizes >= SMALL_ALT).sum())}개", flush=True)

big = set(sizes[sizes >= SMALL].index)
big_alt = set(sizes[sizes >= SMALL_ALT].index)
a["group"] = np.where(~use, "미지정/매핑불가",
                      np.where(a["ccs"].isin(big),
                               a["ccs"].map(CDESC), "기타(소규모)"))
a["group_alt100"] = np.where(~use, "미지정/매핑불가",
                             np.where(a["ccs"].isin(big_alt),
                                      a["ccs"].map(CDESC), "기타(소규모)"))
a.to_csv(OUT / "ccs_assignments.csv", index=False, encoding="utf-8-sig")

# ── 챕터·가중치 트랙 라벨 붙이기 ────────────────────────────────────────────
chap = pd.read_csv(OUT / "dxtext_chapter_assignments.csv", usecols=["HADM_ID", "chapter"])
chap = chap.set_index("HADM_ID").reindex(df["HADM_ID"]).reset_index()
wt = pd.read_csv(OUT / "dxtext_weight_cluster_assignments.csv",
                 usecols=["HADM_ID", "primary_cluster"])
wt = wt.set_index("HADM_ID").reindex(df["HADM_ID"]).reset_index()
base = pd.read_csv(OUT / "dxtext_cluster_assignments.csv",
                   usecols=["HADM_ID", "primary_cluster"])
base = base.set_index("HADM_ID").reindex(df["HADM_ID"]).reset_index()

lab_ccs = pd.factorize(a["group"])[0]
lab_chap = pd.factorize(chap["chapter"])[0]
lab_wt = wt["primary_cluster"].to_numpy()
lab_base = base["primary_cluster"].to_numpy()

# ── 범주별 표 ───────────────────────────────────────────────────────────────
rows = []
for gname, g in a.groupby("group"):
    vc = g["seq1_code"].value_counts()
    ch = chap.loc[g.index, "chapter"].value_counts()
    rows.append({
        "CCS범주": gname, "n": len(g), "n%": round(len(g) / N_ALL * 100, 2),
        "고유_SEQ1코드수": int(g["seq1_code"].nunique()),
        "top1_코드": (vc.index[0] if len(vc) else ""),
        "top1_명": NAME.get(vc.index[0], "?") if len(vc) else "",
        "top1_비중%": round(float(vc.iloc[0]) / len(g) * 100, 1) if len(vc) else 0.0,
        "진단수_중앙": int(g["n_dx"].median()),
        "chapter최빈": (ch.index[0] if len(ch) else ""),
        "chapter순도%": round(float(ch.iloc[0]) / len(g) * 100, 1) if len(ch) else 0.0,
    })
gt = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
gt.to_csv(OUT / "table80_ccs_groups.csv", index=False, encoding="utf-8-sig")
print(gt.to_string(index=False), flush=True)

# ── 같은 공간에서 분할들의 실루엣 ───────────────────────────────────────────
def pca50(path, block=None):
    z = np.load(OUT / path)
    E = z["E"]
    return PCA(n_components=PCA_DIM, random_state=0).fit_transform(E).astype(np.float32)


spaces = {"기본_dxtext_long(출하판)": "emb_dxtext_long.npz",
          "가중치_concise(w=5)": "emb_dxtext_weight_concise.npz"}
parts = {"CCS(주분석)": lab_ccs, "ICD9_chapter": lab_chap,
         "가중치_kmeans_k30": lab_wt, "기본_kmeans_k25": lab_base}
sil = {}
for sname, f in spaces.items():
    P = pca50(f)
    sil[sname] = {pn: round(float(silhouette_score(P, pl, sample_size=SIL_SAMPLE,
                                                   random_state=0)), 4)
                  for pn, pl in parts.items()}
    print(f"[=] {sname}: " + "  ".join(f"{k}={v:.4f}" for k, v in sil[sname].items()),
          flush=True)

# ── 분할 간 일치도 ──────────────────────────────────────────────────────────
agree = {}
names = list(parts)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        agree[f"{names[i]} vs {names[j]}"] = {
            "AMI": round(float(adjusted_mutual_info_score(parts[names[i]], parts[names[j]])), 4),
            "ARI": round(float(adjusted_rand_score(parts[names[i]], parts[names[j]])), 4)}
for k, v in agree.items():
    print(f"    {k:44s} AMI={v['AMI']:.4f} ARI={v['ARI']:.4f}", flush=True)

pur = float((gt.loc[gt["CCS범주"] != "미지정/매핑불가", "top1_비중%"]
             * gt.loc[gt["CCS범주"] != "미지정/매핑불가", "n"]).sum()
            / gt.loc[gt["CCS범주"] != "미지정/매핑불가", "n"].sum())
meta = {
    "매핑원본": "AHRQ HCUP Single-Level CCS for ICD-9-CM 2015 ($dxref 2015.csv)",
    "사전": {"코드": len(CCS), "범주": int(raw["ccs"].nunique())},
    "코호트": {"방문": N_ALL, "SEQ1없음": n_noseq1, "CCS매핑불가": n_unmapped,
             "사용": int(N_ALL - n_noseq1 - n_unmapped)},
    "규칙": {"주진단": "SEQ_NUM=1", "소규모문턱": SMALL,
           "소규모처리": "기타(소규모) 로 병합, 제외 아님"},
    "범주수": {"등장": int(len(sizes)),
             f">={SMALL}": int((sizes >= SMALL).sum()),
             f">={SMALL_ALT}": int((sizes >= SMALL_ALT).sum()),
             "주분석_그룹수": int(gt["CCS범주"].nunique())},
    "top1_비중_가중평균%": round(pur, 1),
    "실루엣_공간별": sil,
    "분할간_일치도": agree,
    "_note": "사후 구현이다. 원안(CCS)이 실제로는 챕터로 구현돼 있던 것을 바로잡은 것이고, "
             "새 결론을 만들지 않는다.",
}
(OUT / "57_ccs_track.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
print(f"[=] CCS 주분석 그룹 {gt['CCS범주'].nunique()}개 | "
      f"top1 코드 비중 가중평균 {pur:.1f}%", flush=True)
print("[+] out/ccs_assignments.csv, table80_ccs_groups.csv, 57_ccs_track.json")
