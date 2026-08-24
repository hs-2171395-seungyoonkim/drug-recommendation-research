"""
§1 게이트: 리스트 컬럼 파싱을 엄격하게 검증하고 기초 수치를 다시 센다.
실패를 조용히 삼키지 않는다 — 실패 행은 인덱스와 원문을 모은다.
"""
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

LIST_COLS = ["diagnose", "procedure", "drug_name", "diag_id", "pro_id", "drug_id"]

df = pd.read_csv(ROOT / "data4LLM_with_note.csv", dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"})

rep = {"rows": int(len(df))}

# ---------- 엄격 파싱 ----------
failures = {c: [] for c in LIST_COLS}
parsed = {}
for c in LIST_COLS:
    vals = []
    for i, x in enumerate(df[c].tolist()):
        if not isinstance(x, str):
            failures[c].append((i, repr(x)[:200]))
            vals.append(None)
            continue
        try:
            v = ast.literal_eval(x)
        except (ValueError, SyntaxError) as e:
            failures[c].append((i, f"{type(e).__name__}: {x[:200]}"))
            vals.append(None)
            continue
        if not isinstance(v, list):
            failures[c].append((i, f"not a list: {type(v).__name__}: {x[:200]}"))
            vals.append(None)
            continue
        vals.append(v)
    parsed[c] = vals

rep["parse_failures"] = {
    c: {"n": len(f), "examples": f[:5]} for c, f in failures.items()
}
n_fail_total = sum(len(f) for f in failures.values())
rep["parse_failures_total"] = n_fail_total

if n_fail_total:
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    raise SystemExit("파싱 실패 존재 — 멈춤")

for c in LIST_COLS:
    df[c + "_l"] = parsed[c]

# ---------- split(',') 과 비교: 이전 수치가 오염됐는지 직접 증명 ----------
naive = Counter()
for x in df["diagnose"]:
    for t in x.strip("[]").split(","):
        t = t.strip().strip("'\"")
        if t:
            naive[t] += 1
strict = Counter(d for l in df["diagnose_l"] for d in l)
rep["split_vs_literal_eval"] = {
    "n_unique_if_split_comma": len(naive),
    "n_unique_literal_eval": len(strict),
    "comma_containing_names": sorted([d for d in strict if "," in d])[:20],
    "n_comma_containing_names": sum(1 for d in strict if "," in d),
}

# ---------- 길이 일치 ----------
mm = {}
for name, idc in [("diagnose", "diag_id"), ("procedure", "pro_id"), ("drug_name", "drug_id")]:
    bad = df.index[df[name + "_l"].map(len) != df[idc + "_l"].map(len)].tolist()
    mm[f"{name}_vs_{idc}"] = {"n_mismatch": len(bad), "example_rows": bad[:5]}
rep["length_mismatch"] = mm

# ---------- id ↔ name 대응 ----------
from collections import defaultdict

n2i, i2n = defaultdict(set), defaultdict(set)
for names, ids in zip(df["diagnose_l"], df["diag_id_l"]):
    for n, i in zip(names, ids):
        n2i[n].add(i)
        i2n[i].add(n)
rep["diag_vocab"] = {
    "n_unique_diag_id": len(i2n),
    "n_unique_diag_name": len(n2i),
    "names_with_multiple_ids": sum(1 for v in n2i.values() if len(v) > 1),
    "ids_with_multiple_names": sum(1 for v in i2n.values() if len(v) > 1),
    "example_names_multi_id": [
        {"name": n, "ids": sorted(v)} for n, v in list(n2i.items()) if len(v) > 1
    ][:10],
    "diag_id_min": int(min(i2n)),
    "diag_id_max": int(max(i2n)),
}

dn2i, di2n = defaultdict(set), defaultdict(set)
for names, ids in zip(df["drug_name_l"], df["drug_id_l"]):
    for n, i in zip(names, ids):
        dn2i[n].add(i)
        di2n[i].add(n)
rep["drug_vocab"] = {
    "n_unique_drug_id": len(di2n),
    "n_unique_drug_name": len(dn2i),
    "names_with_multiple_ids": sum(1 for v in dn2i.values() if len(v) > 1),
    "ids_with_multiple_names": sum(1 for v in di2n.values() if len(v) > 1),
}

# ---------- 방문당 진단 개수 ----------
nd = df["diagnose_l"].map(len)
rep["diag_per_visit"] = {
    "min": int(nd.min()),
    "q1": float(nd.quantile(0.25)),
    "median": float(nd.median()),
    "q3": float(nd.quantile(0.75)),
    "p95": float(nd.quantile(0.95)),
    "max": int(nd.max()),
    "mean": round(float(nd.mean()), 2),
}
ndr = df["drug_name_l"].map(len)
rep["drug_per_visit"] = {
    "min": int(ndr.min()),
    "q1": float(ndr.quantile(0.25)),
    "median": float(ndr.median()),
    "q3": float(ndr.quantile(0.75)),
    "p95": float(ndr.quantile(0.95)),
    "max": int(ndr.max()),
    "mean": round(float(ndr.mean()), 2),
}

# 진단 리스트 내 중복(같은 방문에 같은 id 두 번)
dup_ids = int(sum(1 for l in df["diag_id_l"] if len(l) != len(set(l))))
rep["visits_with_duplicate_diag_id"] = dup_ids

# ---------- 첫 원소 라벨 ----------
first_id = df["diag_id_l"].map(lambda l: l[0] if l else None)
first_nm = df["diagnose_l"].map(lambda l: l[0] if l else None)
id2name_first = {i: sorted(i2n[i])[0] for i in i2n}

vc_id = first_id.value_counts()
vc_nm = first_nm.value_counts()
rep["first_element_label"] = {
    "n_distinct_by_diag_id": int(vc_id.size),
    "n_distinct_by_name": int(vc_nm.size),
    "top30_by_diag_id": [
        {"diag_id": int(i), "name": id2name_first[i], "visits": int(c), "pct": round(c / len(df) * 100, 2)}
        for i, c in vc_id.head(30).items()
    ],
    "labels_with_ge_100_visits": int((vc_id >= 100).sum()),
    "labels_with_ge_50_visits": int((vc_id >= 50).sum()),
    "labels_with_ge_30_visits": int((vc_id >= 30).sum()),
    "coverage_by_topN_pct": {
        str(n): round(float(vc_id.head(n).sum() / len(df) * 100), 1)
        for n in [10, 20, 50, 100, 200, 300]
        if n <= vc_id.size
    },
    "visits_in_ge100_labels_pct": round(float(vc_id[vc_id >= 100].sum() / len(df) * 100), 1),
}

# 다방문 환자에서 첫 진단 안정성
tmp = df.assign(f=first_id).groupby("SUBJECT_ID")["f"].agg(["nunique", "count"])
multi = tmp[tmp["count"] > 1]
rep["principal_dx_stability"] = {
    "multi_visit_subjects": int(len(multi)),
    "pct_same_first_dx_all_visits": round(float((multi["nunique"] == 1).mean()) * 100, 1),
}

with open(OUT / "10_gate.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2, default=str)

df.to_pickle(OUT / "parsed.pkl")
print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
