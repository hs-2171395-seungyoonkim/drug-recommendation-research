"""§1 분할 정의 + §2 검정력 표. D1 / D2 / D2b 라벨을 만들고 라벨별 표본 수만 보고한다.

여기서 멈춘다 — §3 평가는 검정력 표를 보고 판단한 뒤에.

산출: out/icd_partition_assignments.csv, out/table13_icd_power.csv, out/23_icd_partition_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MIN_TEST = 30

d = pd.read_pickle(OUT / "20_icd_features.pkl")

# ---------------------------------------------------------------- chapter 매핑 (기존 프롬프트 §3 표 그대로)
CHAPTERS = [
    (1, 139, "감염성·기생충성"), (140, 239, "신생물"), (240, 279, "내분비·대사·면역"),
    (280, 289, "혈액"), (290, 319, "정신"), (320, 389, "신경"), (390, 459, "순환기"),
    (460, 519, "호흡기"), (520, 579, "소화기"), (580, 629, "비뇨생식기"),
    (630, 679, "임신·출산"), (680, 709, "피부"), (710, 739, "근골격"), (740, 759, "선천기형"),
    (760, 779, "주산기"), (780, 799, "증상·불명확"), (800, 999, "손상·중독"),
]
# D2 순환기 세분 — ICD-9 공식 블록 경계
CIRC_BLOCKS = [
    (390, 398, "순환기: 류마티스심질환"), (401, 405, "순환기: 고혈압성"),
    (410, 414, "순환기: 허혈성 심질환"), (415, 417, "순환기: 폐순환"),
    (420, 429, "순환기: 기타 심질환"), (430, 438, "순환기: 뇌혈관"),
    (440, 449, "순환기: 동맥·세동맥"), (451, 459, "순환기: 정맥·림프"),
]


def head3(code):
    """'.' 제거 형식 대비 앞 3자리. V·E 는 별도."""
    c = str(code).strip().replace(".", "")
    if c.startswith("V"):
        return ("V", c[:3])
    if c.startswith("E"):
        return ("E", c[:4])
    return ("N", c[:3])


def chapter_of(code):
    kind, h = head3(code)
    if kind == "V":
        return "V코드 (보조분류)"
    if kind == "E":
        return "E코드 (외인)"
    try:
        v = int(h)
    except ValueError:
        return None
    for lo, hi, name in CHAPTERS:
        if lo <= v <= hi:
            return name
    return None


def d2_of(code):
    ch = chapter_of(code)
    if ch != "순환기":
        return ch
    v = int(head3(code)[1])
    for lo, hi, name in CIRC_BLOCKS:
        if lo <= v <= hi:
            return name
    return "순환기: 블록 미지정"


def d2b_of(code):
    lab = d2_of(code)
    if lab != "순환기: 기타 심질환":
        return lab
    return f"순환기: {head3(code)[1]}"


d = d.copy()
d["icd9_seq1"] = d["icd_seq1"]
d["chapter"] = d["icd9_seq1"].map(chapter_of)
d["d1_label"] = d["chapter"]
d["d2_label"] = d["icd9_seq1"].map(d2_of)
d["d2b_label"] = d["icd9_seq1"].map(d2b_of)

n_unmapped = int(d["chapter"].isna().sum())

# ---------------------------------------------------------------- 3자리 코드 명칭 (D_ICD_DIAGNOSES)
dic_path = ROOT / "D_ICD_DIAGNOSES.csv"
name3 = {}
if dic_path.exists():
    dic = pd.read_csv(dic_path, dtype={"ICD9_CODE": str})
    dic["ICD9_CODE"] = dic["ICD9_CODE"].str.strip()
    short = dict(zip(dic["ICD9_CODE"], dic["SHORT_TITLE"].astype(str)))
    # 각 3자리 그룹의 대표 명칭 = 그 그룹에서 가장 흔한 전체 코드의 SHORT_TITLE
    tmp = d[d["d2_label"] == "순환기: 기타 심질환"]
    for g, sub in tmp.groupby(tmp["icd9_seq1"].map(lambda c: head3(c)[1])):
        # 빈도순으로 훑어 사전에 명칭이 있는 첫 코드를 대표로 쓴다(일부 코드는 사전에 없다)
        titled = [short[c] for c in sub["icd9_seq1"].value_counts().index if c in short]
        name3[g] = titled[0] if titled else "?"
d["d2b_label"] = [
    f"순환기: {l.split(': ')[1]} {name3.get(l.split(': ')[1], '')}".strip()
    if l.startswith("순환기: ") and l.split(": ")[1] in name3 else l
    for l in d["d2b_label"]
]

# ---------------------------------------------------------------- §2 검정력
is_test = d["split"] == "test"
rows, summary = [], []
for part in ["d1_label", "d2_label", "d2b_label"]:
    g = d.groupby(part, dropna=False).agg(
        전체방문=("HADM_ID", "size"), test방문=("split", lambda s: int((s == "test").sum())),
        환자수=("SUBJECT_ID", "nunique"),
    ).sort_values("test방문", ascending=False)
    g["test환자수"] = d[is_test].groupby(part, dropna=False)["SUBJECT_ID"].nunique().reindex(g.index).fillna(0).astype(int)
    g["ge30"] = g["test방문"] >= MIN_TEST
    g = g.reset_index().rename(columns={part: "라벨"})
    g.insert(0, "분할", {"d1_label": "D1", "d2_label": "D2", "d2b_label": "D2b"}[part])
    rows.append(g)
    n_pass = int(g["ge30"].sum())
    test_in_pass = int(g.loc[g["ge30"], "test방문"].sum())
    summary.append({
        "분할": g["분할"].iloc[0], "라벨 수": int(len(g)),
        "test>=30 통과 라벨 수 (실질 k)": n_pass,
        "통과 라벨의 test 방문": test_in_pass,
        "탈락 라벨의 test 방문": int(is_test.sum()) - test_in_pass,
        "탈락 test 비율%": round((int(is_test.sum()) - test_in_pass) / int(is_test.sum()) * 100, 2),
        "매핑 실패(기타)%": round(n_unmapped / len(d) * 100, 2),
        "최소 test 크기(통과 라벨 중)": int(g.loc[g["ge30"], "test방문"].min()) if n_pass else 0,
        "중앙 test 크기(통과 라벨 중)": float(g.loc[g["ge30"], "test방문"].median()) if n_pass else 0,
    })

power = pd.concat(rows, ignore_index=True)
power.to_csv(OUT / "table13_icd_power.csv", index=False)
summ = pd.DataFrame(summary)

d[["SUBJECT_ID", "HADM_ID", "icd9_seq1", "chapter", "d1_label", "d2_label", "d2b_label", "split"]].to_csv(
    OUT / "icd_partition_assignments.csv", index=False)

meta = {
    "n_visits": int(len(d)), "n_test": int(is_test.sum()),
    "chapter_mapping_failures": n_unmapped,
    "circ_share_pct": round(float((d["chapter"] == "순환기").mean()) * 100, 2),
    "circ_block_unassigned": int((d["d2_label"] == "순환기: 블록 미지정").sum()),
    "min_test_visits": MIN_TEST,
    "summary": summary,
    "labels": {k: v for k, v in
               [(s["분할"], power[power["분할"] == s["분할"]][["라벨", "전체방문", "test방문", "ge30"]]
                 .to_dict("records")) for s in summary]},
}
with open(OUT / "23_icd_partition_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

pd.set_option("display.width", 220)
print(f"방문 {len(d):,} / test {int(is_test.sum()):,} · chapter 매핑 실패 {n_unmapped}건 · "
      f"순환기 비중 {meta['circ_share_pct']}% · 순환기 블록 미지정 {meta['circ_block_unassigned']}건\n")
for part, tag in [("d1_label", "D1"), ("d2_label", "D2"), ("d2b_label", "D2b")]:
    sub = power[power["분할"] == tag]
    print(f"===== {tag} — 라벨 {len(sub)}개, test>=30 통과 {int(sub['ge30'].sum())}개 =====")
    print(sub[["라벨", "전체방문", "test방문", "test환자수", "ge30"]].to_string(index=False))
    print()
print("===== 요약 =====")
print(summ.to_string(index=False))
