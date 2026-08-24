"""§1·§2 게이트: DIAGNOSES_ICD 조인 가능성 + SEQ_NUM=1 이 diag_id[0] 과 같은가.

게이트 통과 못 하면 이후 chapter 분할로 진행하지 않는다.
산출: out/19_icd_gate.json
"""
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
rep = {}

# ---------------------------------------------------------------- 게이트 1: 스키마
dcols = pd.read_csv(ROOT / "DIAGNOSES_ICD.csv", nrows=5).columns.tolist()
schema = {
    "columns": dcols,
    "has_ICD9_CODE": "ICD9_CODE" in dcols,
    "has_icd_code_and_version": ("icd_code" in dcols and "icd_version" in dcols),
}
schema["verdict"] = "MIMIC-III (ICD9_CODE)" if schema["has_ICD9_CODE"] else "MIMIC-IV 의심 — 중단"
rep["gate1_schema"] = schema
if not schema["has_ICD9_CODE"]:
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    raise SystemExit("GATE 1 FAIL: ICD9_CODE 없음 — MIMIC-IV 로 보이므로 중단")

d = pd.read_csv(ROOT / "DIAGNOSES_ICD.csv", dtype={"ICD9_CODE": str})
d["ICD9_CODE"] = d["ICD9_CODE"].str.strip()
p = pd.read_pickle(OUT / "parsed.pkl").reset_index(drop=True)

# ---------------------------------------------------------------- 게이트 2: 조인율
ours = set(p["HADM_ID"])
theirs = set(d["HADM_ID"])
hit = ours & theirs
join = {
    "our_visits": len(ours),
    "icd_visits_total": len(theirs),
    "matched": len(hit),
    "join_rate_pct": round(len(hit) / len(ours) * 100, 3),
    "missing_hadm_examples": sorted(list(ours - theirs))[:10],
    "pass_ge95": len(hit) / len(ours) >= 0.95,
}
rep["gate2_join"] = join

# ---------------------------------------------------------------- 게이트 3: 진단 개수 일치
d = d[d["HADM_ID"].isin(hit)].copy()
n_null_seq = int(d["SEQ_NUM"].isna().sum())
d = d.dropna(subset=["SEQ_NUM", "ICD9_CODE"])
d["SEQ_NUM"] = d["SEQ_NUM"].astype(int)
d = d.sort_values(["HADM_ID", "SEQ_NUM"])

icd_lists = d.groupby("HADM_ID")["ICD9_CODE"].apply(list)
pp = p[p["HADM_ID"].isin(hit)].copy()
pp["n_ours"] = pp["diag_id_l"].map(len)
pp["icd_l"] = pp["HADM_ID"].map(icd_lists)
pp["n_icd"] = pp["icd_l"].map(lambda x: len(x) if isinstance(x, list) else 0)
diff = (pp["n_icd"] - pp["n_ours"]).astype(int)
cnt = {
    "rows_with_null_seqnum": n_null_seq,
    "visits_compared": int(len(pp)),
    "exact_count_match": int((diff == 0).sum()),
    "count_match_pct": round(float((diff == 0).mean()) * 100, 3),
    "diff_distribution_icd_minus_ours": {str(k): int(v) for k, v in sorted(Counter(diff).items())[:15]},
    "mean_abs_diff": round(float(diff.abs().mean()), 4),
}
rep["gate3_count"] = cnt

# ---------------------------------------------------------------- 게이트 4 (§2): SEQ_NUM=1 == diag_id[0] ?
# (A) 구조 검정 — 위치 1 이상에서 diag_id->ICD 사전을 학습한 뒤, 위치 0 을 held-out 으로 예측
ok = pp[diff == 0]
pairs_tail = defaultdict(Counter)   # 위치>=1 학습용
head_obs = []                       # 위치 0 검정용
for ids, icds in zip(ok["diag_id_l"], ok["icd_l"]):
    for pos, (i, c) in enumerate(zip(ids, icds)):
        if pos == 0:
            head_obs.append((i, c))
        else:
            pairs_tail[i][c] += 1

tail_map = {i: c.most_common(1)[0][0] for i, c in pairs_tail.items()}
tail_purity = np.mean([c.most_common(1)[0][1] / sum(c.values()) for c in pairs_tail.values()])
covered = [(i, c) for i, c in head_obs if i in tail_map]
head_hit = sum(1 for i, c in covered if tail_map[i] == c)

# 참고: 무작위 순서였다면 맞을 확률 — 방문 내 다른 위치의 코드로 예측했을 때
rng = np.random.default_rng(0)
shuf_hit = 0
shuf_tot = 0
for ids, icds in zip(ok["diag_id_l"], ok["icd_l"]):
    if len(ids) < 2 or ids[0] not in tail_map:
        continue
    j = int(rng.integers(1, len(icds)))
    shuf_tot += 1
    shuf_hit += (tail_map[ids[0]] == icds[j])

structural = {
    "count_match_visits_used": int(len(ok)),
    "distinct_diag_id_in_tail_map": len(tail_map),
    "tail_map_purity_mean": round(float(tail_purity), 4),
    "head_obs_total": len(head_obs),
    "head_obs_covered_by_map": len(covered),
    "head_position0_agreement_pct": round(head_hit / len(covered) * 100, 3) if covered else None,
    "null_baseline_random_other_position_pct": round(shuf_hit / shuf_tot * 100, 3) if shuf_tot else None,
}
rep["gate4_structural_seq1_vs_diagid0"] = structural

# (B) 의미 검정 — D_ICD_DIAGNOSES 명칭 vs diagnose[0] (LLM 재작명이라 문자열 유사도)
dic_path = ROOT / "D_ICD_DIAGNOSES.csv"
sem = {"dictionary_available": dic_path.exists()}
if dic_path.exists():
    dic = pd.read_csv(dic_path, dtype={"ICD9_CODE": str})
    dic["ICD9_CODE"] = dic["ICD9_CODE"].str.strip()
    short = dict(zip(dic["ICD9_CODE"], dic["SHORT_TITLE"].astype(str)))
    long = dict(zip(dic["ICD9_CODE"], dic["LONG_TITLE"].astype(str)))

    seq1 = d[d["SEQ_NUM"] == 1].set_index("HADM_ID")["ICD9_CODE"]
    pp2 = pp[pp["HADM_ID"].isin(seq1.index)].copy()
    pp2["icd_seq1"] = pp2["HADM_ID"].map(seq1)

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())

    def toks(s):
        STOP = {"unspecified", "nos", "with", "without", "other", "and", "of", "the",
                "in", "due", "to", "not", "elsewhere", "classified", "acute", "chronic"}
        return {w for w in norm(s).split() if len(w) > 2 and w not in STOP}

    sims, jacs = [], []
    for name, code in zip(pp2["diagnose_l"].map(lambda l: l[0] if len(l) else ""), pp2["icd_seq1"]):
        t = long.get(code, "") + " " + short.get(code, "")
        sims.append(SequenceMatcher(None, norm(name), norm(t)).ratio())
        a, b = toks(name), toks(t)
        jacs.append(len(a & b) / len(a | b) if (a | b) else 0.0)
    sims, jacs = np.array(sims), np.array(jacs)
    sem.update({
        "visits_with_seq1": int(len(pp2)),
        "seqmatcher_ratio_mean": round(float(sims.mean()), 4),
        "seqmatcher_ratio_median": round(float(np.median(sims)), 4),
        "token_jaccard_mean": round(float(jacs.mean()), 4),
        "token_jaccard_median": round(float(np.median(jacs)), 4),
        "pct_token_overlap_gt0": round(float((jacs > 0).mean()) * 100, 2),
        "pct_token_jaccard_ge_0.34": round(float((jacs >= 1 / 3).mean()) * 100, 2),
    })
    # 대조군: diagnose[0] vs 같은 방문의 SEQ_NUM=2 코드 명칭 (있는 경우)
    seq2 = d[d["SEQ_NUM"] == 2].set_index("HADM_ID")["ICD9_CODE"]
    pp3 = pp2[pp2["HADM_ID"].isin(seq2.index)].copy()
    pp3["icd_seq2"] = pp3["HADM_ID"].map(seq2)
    j2 = []
    for name, code in zip(pp3["diagnose_l"].map(lambda l: l[0] if len(l) else ""), pp3["icd_seq2"]):
        t = long.get(code, "") + " " + short.get(code, "")
        a, b = toks(name), toks(t)
        j2.append(len(a & b) / len(a | b) if (a | b) else 0.0)
    sem["control_token_jaccard_vs_seq2_mean"] = round(float(np.mean(j2)), 4)
    sem["control_n"] = len(j2)
    sem["examples"] = [
        {"diagnose_0": n, "icd_seq1": c, "icd_title": short.get(c, "?")}
        for n, c in list(zip(pp2["diagnose_l"].map(lambda l: l[0] if len(l) else ""), pp2["icd_seq1"]))[:8]
    ]
rep["gate4_semantic_seq1_vs_diagnose0"] = sem

# ---------------------------------------------------------------- 불일치 원인 규명
def is_subseq(a, b):
    it = iter(b)
    return all(x in it for x in a)


bad = pp[diff != 0]
sub_ok = pre_ok = first_ok = tot_bad = 0
for ids, icds in zip(bad["diag_id_l"], bad["icd_l"]):
    mapped = [tail_map.get(i) for i in ids]
    if any(m is None for m in mapped):
        continue
    tot_bad += 1
    sub_ok += is_subseq(mapped, icds)
    pre_ok += (mapped == icds[:len(mapped)])
    first_ok += (mapped[0] == icds[0])

dropped = Counter()
for ids, icds in zip(bad["diag_id_l"], bad["icd_l"]):
    mp = {tail_map.get(i) for i in ids}
    for c in icds:
        if c not in mp:
            dropped["V" if c.startswith("V") else "E" if c.startswith("E") else "numeric"] += 1

# 전체 방문 기준 SEQ_NUM=1 일치율 + SEQ_NUM=1 이 아예 소실된 비율
hit = tot_all = unmapped = lost = tot_lost = 0
for ids, icds in zip(pp["diag_id_l"], pp["icd_l"]):
    if not len(icds):
        continue
    tot_lost += 1
    lost += (icds[0] not in {tail_map.get(i) for i in ids})
    if not len(ids) or ids[0] not in tail_map:
        unmapped += 1
        continue
    tot_all += 1
    hit += (tail_map[ids[0]] == icds[0])

rep["gate5_mismatch_cause"] = {
    "our_diag_vocab_size_ids": len({i for l in pp["diag_id_l"] for i in l}),
    "our_diag_vocab_size_names": len({n for l in pp["diagnose_l"] for n in l}),
    "distinct_icd9_codes_same_visits": len({c for l in pp["icd_l"] for c in l}),
    "mismatch_visits_mappable": tot_bad,
    "our_list_is_subsequence_of_icd_pct": round(sub_ok / tot_bad * 100, 2) if tot_bad else None,
    "our_list_is_prefix_of_icd_pct": round(pre_ok / tot_bad * 100, 2) if tot_bad else None,
    "first_elem_equals_seq1_in_mismatch_pct": round(first_ok / tot_bad * 100, 2) if tot_bad else None,
    "dropped_code_types": dict(dropped),
    "note": "차이 분포가 전부 >=0 이고 부분수열 비율이 높다 = 순서는 보존되고 항목만 누락된다.",
}
rep["gate4_overall"] = {
    "visits_mappable": tot_all,
    "visits_unmappable": unmapped,
    "seq1_equals_diagid0_pct_ALL_VISITS": round(hit / tot_all * 100, 2) if tot_all else None,
    "seq1_code_absent_from_our_list_pct": round(lost / tot_lost * 100, 2) if tot_lost else None,
}

with open(OUT / "19_icd_gate.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(json.dumps(rep, ensure_ascii=False, indent=2))
