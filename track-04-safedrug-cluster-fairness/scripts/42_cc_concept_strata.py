"""§주증상 개념 층 — UMLS 개념을 그대로 층으로 쓰고, 꼬리만 군집화한다.

왜 군집화가 아니라 층인가:
  41 정규화 후 방문의 88.4%가 개념 0~1개다(정확히 1개가 10,420건). 개념 1개짜리
  방문 위에서 k-means 를 돌리면 개념을 다시 발견하는 일이 되고, 대신 층 이름을
  잃는다. 37 의 24개 군집 중 하나의 주소견으로 이름 붙일 수 있었던 것은 10개
  (전체 방문의 27.3%)뿐이었다 — 나머지는 CC 서술 형식으로 뭉친 자리였다.
  개념을 층으로 쓰면 이름이 CUI 로 감사 가능해진다.

층 구성 (사전 고정):
  대표 개념   CC 에 먼저 적힌 개념(주소견을 앞에 쓰는 기록 관행). 대안 규칙(코퍼스
              최빈 개념)과의 일치율을 민감도로 낸다.
  머리        대표 개념 방문수 >= MIN_STRATUM(100) 인 CUI 를 각각 층으로. 100 은 37 과
              같은 기준 — 군집별 SafeDrug 성능을 재려면 필요한 최소 규모다.
  꼬리        나머지(희귀 개념 + 개념 미매칭 905건)를 ClinicalBERT 로 군집화. k 규칙은
              37 과 같다: 최소 군집 >= 100 을 지키는 k 중 가장 큰 것, 시드는 inertia 최소.
  층외        CC 없는 1,737건. 37 과 같이 HPI 트랙 라벨만 붙여 둔다.

검증 (이 스크립트가 같이 내는 것):
  - 층별 test 표본 >= 30 인지 (37 과 같은 기준)
  - eta2(단어수). 37 B 트랙은 0.1411 이었다. 개념 층이면 표기 길이 축이 빠져야 한다.
  - 층 안 CC 원문 top5 — 동의어만 모였는지 눈으로 확인할 수 있게 표로 낸다
  - 대표 개념 규칙 민감도(먼저 적힌 것 vs 최빈)
  - 이름이 비슷한 층 쌍(같은 어간) — UMLS 입도 때문에 갈린 자리를 사람이 판단하도록 낸다
  - 분할이 버리는 정보: 2개 이상 개념 방문에서 부개념도 머리층인 비율

산출: out/42_concept_strata.csv, out/table54_strata.csv, out/table55_strata_checks.csv,
      out/table56_strata_sensitivity.csv, out/table57_strata_collisions.csv,
      out/42_strata_meta.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score as ari
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN, BATCH = 64, 256
SVD_DIM = 50
MIN_STRATUM = 100
K_GRID = [4, 6, 8, 10, 12, 15, 20]
SEEDS = [0, 1, 2, 3, 4]

norm = pd.read_pickle(OUT / "41_cc_umls.pkl")
asg = pd.read_csv(OUT / "cc_cluster_assignments.csv")
feat = pd.read_pickle(OUT / "20_icd_features.pkl")[["HADM_ID", "split", "n_icd"]]

df = norm.merge(asg[["HADM_ID", "stratum", "clusterA", "clusterCC_bert", "n_word"]],
                on="HADM_ID", how="left").merge(feat, on="HADM_ID", how="left")

# ---------------------------------------------------------------- 대표 개념
corpus_ct = Counter(c for v in df.cuis for c in v)
df["primary_first"] = df.cuis.map(lambda v: v[0] if len(v) else None)
df["primary_freq"] = df.cuis.map(
    lambda v: max(v, key=lambda c: corpus_ct[c]) if len(v) else None)
multi = df[df.n_cui >= 2]
agree = float((multi.primary_first == multi.primary_freq).mean()) if len(multi) else 1.0
print(f"[.] 개념 2개 이상 {len(multi):,}건 · 두 규칙 일치 {agree*100:.1f}%")

df["primary"] = df["primary_first"]
prim_ct = df.primary.value_counts()
head_cuis = [c for c, n in prim_ct.items() if n >= MIN_STRATUM]
print(f"[.] 머리 층(대표개념 >= {MIN_STRATUM}) {len(head_cuis)}개")

name_of, group_of = {}, {}
for cu, nm, gs in zip(df.cuis, df.cui_names, df.type_groups):
    for c, n, g in zip(cu, nm, gs):
        name_of.setdefault(c, n)
        group_of.setdefault(c, g)

df["layer"] = np.where(df.primary.isin(head_cuis), "concept", "tail")

# ---------------------------------------------------------------- 꼬리 군집화
tail = df[df.layer == "tail"].copy()
print(f"[.] 꼬리 {len(tail):,}건 (희귀개념 {int((tail.n_cui > 0).sum()):,} / 무개념 "
      f"{int((tail.n_cui == 0).sum()):,}) 군집화", flush=True)

txt = tail.cc.fillna("").tolist()
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"
mdl = mdl.to(dev).eval()
embs = []
with torch.no_grad():
    for i in range(0, len(txt), BATCH):
        b = tok(txt[i:i + BATCH], padding=True, truncation=True, max_length=MAXLEN,
                return_tensors="pt").to(dev)
        o = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        embs.append(((o * m).sum(1) / m.sum(1)).cpu().numpy())
E = np.vstack(embs)
Z = TruncatedSVD(SVD_DIM, random_state=0).fit_transform(E)

sweep = []
for k in K_GRID:
    if k >= len(tail):
        continue
    runs = []
    for s in SEEDS:
        km = KMeans(k, random_state=s, n_init=10).fit(Z)
        sz = np.bincount(km.labels_, minlength=k)
        runs.append((km.inertia_, s, km.labels_, int(sz.min()), float(sz.max() / len(tail) * 100)))
    ine, s, lab, mn, mx = min(runs, key=lambda r: r[0])
    sweep.append({"k": k, "seed": s, "inertia": round(ine, 2), "min_size": mn,
                  "largest_pct": round(mx, 1)})
    print(f"    k={k:3d} seed={s} min={mn:5d} largest={mx:.1f}%")
sw = pd.DataFrame(sweep)
ok = sw[sw.min_size >= MIN_STRATUM]
k_pick = int(ok.k.max()) if len(ok) else int(sw.k.min())
row = sw[sw.k == k_pick].iloc[0]
km = KMeans(k_pick, random_state=int(row.seed), n_init=10).fit(Z)
tail["tail_cluster"] = km.labels_
print(f"[=] 꼬리 k={k_pick} (규칙: 최소 군집 >= {MIN_STRATUM} 중 최대 k)")

# ---------------------------------------------------------------- 층 라벨 통합
df = df.merge(tail[["HADM_ID", "tail_cluster"]], on="HADM_ID", how="left")
df["stratum_id"] = np.where(df.layer == "concept", df.primary,
                            "TAIL" + df.tail_cluster.astype("Int64").astype(str))
df["stratum_name"] = np.where(
    df.layer == "concept", df.primary.map(lambda c: name_of.get(c, "?")),
    "tail " + df.tail_cluster.astype("Int64").astype(str))
df["type_group"] = np.where(df.layer == "concept",
                            df.primary.map(lambda c: group_of.get(c, "?")), "TAIL")

full = asg[["SUBJECT_ID", "HADM_ID", "split", "stratum", "clusterA", "clusterCC_bert"]].merge(
    df[["HADM_ID", "layer", "stratum_id", "stratum_name", "type_group", "primary",
        "n_cui", "n_word"]], on="HADM_ID", how="left")
full["layer"] = full.layer.fillna("no_CC")
full["stratum_id"] = full.stratum_id.fillna("NO_CC")
full["stratum_name"] = full.stratum_name.fillna("CC 없음(HPI 라벨)")
full.to_csv(OUT / "42_concept_strata.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 층 표 + 검증
cc_txt = norm.set_index("HADM_ID").cc
rows = []
for sid, g in df.groupby("stratum_id"):
    strs = cc_txt.loc[g.HADM_ID].str.lower().value_counts()
    rows.append({
        "stratum_id": sid, "name": g.stratum_name.iloc[0], "type_group": g.type_group.iloc[0],
        "n": len(g), "pct": round(len(g) / len(df) * 100, 1),
        "test_n": int((g.split == "test").sum()),
        "words_median": float(g.n_word.median()), "n_icd_median": float(g.n_icd.median()),
        "top_str_share": round(strs.iloc[0] / len(g) * 100, 1),
        "top5": "; ".join(f"{s}({n})" for s, n in strs.head(5).items()),
    })
tab = pd.DataFrame(rows).sort_values("n", ascending=False)
tab.to_csv(OUT / "table54_strata.csv", index=False, encoding="utf-8-sig")


def eta2(labels, y):
    y = np.asarray(y, float)
    gm = y.mean()
    ssb = sum(len(y[labels == l]) * (y[labels == l].mean() - gm) ** 2 for l in np.unique(labels))
    return float(ssb / ((y - gm) ** 2).sum())


e2_new = eta2(df.stratum_id.values, df.n_word.values)
old = asg[asg.stratum == "CC"].dropna(subset=["clusterCC_bert"])
e2_old = eta2(old.clusterCC_bert.values, old.n_word.values)

sec_head = float(multi.apply(
    lambda r: any(c in head_cuis for c in r.cuis[1:]), axis=1).mean()) if len(multi) else 0.0
common = df.dropna(subset=["clusterCC_bert"])
checks = pd.DataFrame([
    ("층 수(머리 개념)", len(head_cuis), ""),
    ("층 수(꼬리 군집)", k_pick, f"규칙: 최소>= {MIN_STRATUM} 중 최대 k"),
    ("머리층이 덮는 방문", int((df.layer == 'concept').sum()),
     f"{(df.layer=='concept').mean()*100:.1f}% of CC층"),
    ("꼬리 방문", int((df.layer == 'tail').sum()), f"{(df.layer=='tail').mean()*100:.1f}%"),
    ("test >= 30 인 층", int((tab.test_n >= 30).sum()), f"/ {len(tab)}"),
    ("최소 층 크기", int(tab.n.min()), ""),
    ("최대 층 비율", f"{tab.pct.max():.1f}%", ""),
    ("eta2(단어수) 새 층", round(e2_new, 4), f"37 B트랙 {e2_old:.4f}"),
    ("대표개념 규칙 일치율", f"{agree*100:.1f}%", f"개념 2개 이상 {len(multi):,}건에서"),
    ("부개념도 머리층인 비율", f"{sec_head*100:.1f}%", "분할이 버리는 정보"),
    ("ARI(새 층, 37 B군집)", round(ari(common.stratum_id.astype(str),
                                      common.clusterCC_bert.astype(int)), 4), ""),
], columns=["항목", "값", "비고"])
checks.to_csv(OUT / "table55_strata_checks.csv", index=False, encoding="utf-8-sig")

alt = df[df.n_cui >= 2][["HADM_ID", "cc", "primary_first", "primary_freq"]].copy()
alt = alt[alt.primary_first != alt.primary_freq]
alt["name_first"] = alt.primary_first.map(name_of)
alt["name_freq"] = alt.primary_freq.map(name_of)
alt.head(200).to_csv(OUT / "table56_strata_sensitivity.csv", index=False, encoding="utf-8-sig")

# 이름이 겹치는 층 쌍 — UMLS 입도로 갈린 자리
STOP = {"of", "the", "and", "or", "to", "in", "with", "ctcae", "nos", "disorder", "finding"}
words = {c: {w for w in re.findall(r"[a-z]+", name_of.get(c, "").lower()) if w not in STOP}
         for c in head_cuis}
col = []
for i, a in enumerate(head_cuis):
    for b in head_cuis[i + 1:]:
        sh = words[a] & words[b]
        if sh:
            col.append({"cui1": a, "name1": name_of.get(a), "n1": int(prim_ct[a]),
                        "cui2": b, "name2": name_of.get(b), "n2": int(prim_ct[b]),
                        "shared": " ".join(sorted(sh))})
colls = pd.DataFrame(col)
colls.to_csv(OUT / "table57_strata_collisions.csv", index=False, encoding="utf-8-sig")

meta = {
    "min_stratum": MIN_STRATUM,
    "n_head_strata": len(head_cuis), "n_tail_clusters": k_pick,
    "tail_k_rule": f"최소 군집 >= {MIN_STRATUM} 중 최대 k", "tail_seed_rule": "inertia 최소",
    "tail_seed": int(row.seed),
    "n_cc_visits": len(df), "n_concept_layer": int((df.layer == "concept").sum()),
    "n_tail": int((df.layer == "tail").sum()), "n_no_cc": int((full.layer == "no_CC").sum()),
    "concept_layer_pct_of_cc": round(float((df.layer == "concept").mean()) * 100, 1),
    "concept_layer_pct_of_all": round(len(df[df.layer == "concept"]) / len(full) * 100, 1),
    "strata_test_ge30": int((tab.test_n >= 30).sum()), "n_strata_total": len(tab),
    "min_stratum_size": int(tab.n.min()), "largest_stratum_pct": float(tab.pct.max()),
    "eta2_word_count": {"new": round(e2_new, 4), "old_37B": round(e2_old, 4)},
    "primary_rule_agreement_pct": round(agree * 100, 1),
    "secondary_also_head_pct": round(sec_head * 100, 1),
    "ari_vs_37B": round(ari(common.stratum_id.astype(str),
                            common.clusterCC_bert.astype(int)), 4),
    "type_group_of_head_strata": Counter(group_of.get(c, "?") for c in head_cuis),
    "name_collision_pairs": len(colls),
    "tail_sweep": sweep,
}
with open(OUT / "42_strata_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=int)

print(f"\n[=] 층 {len(tab)}개 = 개념 {len(head_cuis)} + 꼬리 {k_pick}")
print(f"[=] 머리층 {meta['n_concept_layer']:,}건 = CC층의 {meta['concept_layer_pct_of_cc']}% "
      f"(전체 방문의 {meta['concept_layer_pct_of_all']}%)")
print(f"[=] eta2(단어수) {e2_new:.4f}  (37 B트랙 {e2_old:.4f})")
print(f"[=] test>=30 인 층 {meta['strata_test_ge30']}/{len(tab)} · 최소 층 {tab.n.min()}")
print(f"[=] ARI(새 층, 37B) {meta['ari_vs_37B']}")
print("\n" + checks.to_string(index=False))
print("\n[=] 층 표 (상위 30)")
print(tab.head(30).to_string(index=False, max_colwidth=60))
print(f"\n[=] 이름 겹치는 층 쌍 {len(colls)}")
if len(colls):
    print(colls.to_string(index=False, max_colwidth=34))
