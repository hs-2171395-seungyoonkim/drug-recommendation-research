"""§주증상 개념 그룹 — 방문이 아니라 개념 1,467개를 군집화해 꼬리를 없앤다.

왜:
  42 는 UMLS 개념을 그대로 층으로 썼다. 층 내용은 깨끗했지만(Hematochezia 가 brbpr 과
  bright red blood per rectum 을 묶었다) 대표 개념 1,467종 중 746종이 1회성이라
  방문의 57.7%가 이름 없는 꼬리 군집 6개로 남았다. 37 의 "이름 못 붙이는 큰 군집"
  문제가 해결된 게 아니라 자리를 옮긴 것이다.

  임계값을 낮춰 층을 늘리면(100 -> 10 이면 19개 -> 177개) 이름은 늘지만 test 가
  2,880건뿐이라 층별 SafeDrug 성능을 못 잰다. 방법이 아니라 데이터 상한이다.

  그래서 방문이 아니라 개념을 묶는다. 희귀 개념이 상위 그룹으로 접히면 커버리지가
  올라가고, 그룹 이름은 "개념 목록"으로 남아 감사 가능하다.

규칙 (사전 고정):
  단위     대표 개념으로 등장한 CUI (canonical name 만 임베딩한다. 표면형을 넣으면
           다시 표기가 축이 된다 — 37 에서 겪은 실패다)
  가중     sample_weight = 그 개념의 방문 수. 축에 필요한 것은 개념 균형이 아니라
           방문 균형이다. 비가중 결과는 민감도로 같이 낸다.
  k        그룹의 방문 수 최소 100 을 지키는 k 중 최대 (37·42 와 같은 규칙)
  시드     5개 중 inertia 최소
  무개념   CUI 가 하나도 안 붙은 905건은 그룹에 섞지 않고 NO_CONCEPT 층으로 둔다
  빈템플릿 CC 가 "cc" 뿐인 65건은 층외로 뺀다 (REPORT_CC.md §9 에 적어 두고 못 고친 것)

비순환 검정:
  UMLS 가 CUI 단계에서 못 붙인 2쌍 — unresponsive(C0237284)/unresponsiveness(C0241526),
  Abnormal mental state(C0278061)/Mental Status Change(C0856054) — 이 같은 그룹에
  오는가. 이름 임베딩은 이 쌍을 목표로 삼지 않으므로 자기충족이 아니다.

위험 (미리 적어 둔다):
  이름 임베딩은 단어를 공유하는 무관한 개념을 붙일 수 있다 — chest pain 과
  abdominal pain 이 'pain' 으로 붙는 자리다. 그룹별 이질성을 표로 내고, 가장 나쁜
  그룹을 보고서에 그대로 싣는다.

산출: out/43_concept_groups.csv, out/table58_group_table.csv,
      out/table59_group_checks.csv, out/table60_group_synonym_test.csv,
      out/43_groups_meta.json
"""
import json
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
MAXLEN, BATCH = 32, 256
SVD_DIM = 50
MIN_VISITS = 100
K_GRID = [10, 15, 20, 24, 30, 40, 50]
SEEDS = [0, 1, 2, 3, 4]
EMPTY_CC = {"cc", "chief complaint"}

norm = pd.read_pickle(OUT / "41_cc_umls.pkl")
asg = pd.read_csv(OUT / "cc_cluster_assignments.csv")
strata42 = pd.read_csv(OUT / "42_concept_strata.csv")
feat = pd.read_pickle(OUT / "20_icd_features.pkl")[["HADM_ID", "split", "n_icd"]]

df = norm.merge(asg[["HADM_ID", "clusterCC_bert", "n_word"]], on="HADM_ID", how="left") \
         .merge(feat, on="HADM_ID", how="left")
df["primary"] = df.cuis.map(lambda v: v[0] if len(v) else None)
df["empty_template"] = df.cc.str.lower().str.strip().isin(EMPTY_CC)
print(f"[.] CC 방문 {len(df):,} · 빈템플릿 {int(df.empty_template.sum())} 층외로 제외")

name_of = {}
for cu, nm in zip(df.cuis, df.cui_names):
    for c, n in zip(cu, nm):
        name_of.setdefault(c, n)

work = df[~df.empty_template].copy()
have = work[work.primary.notna()]
vis_ct = have.primary.value_counts()
cuis = list(vis_ct.index)
names = [name_of[c] for c in cuis]
w = vis_ct.values.astype(float)
print(f"[.] 개념 {len(cuis):,}개 (방문 {int(w.sum()):,}) 임베딩", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"
mdl = mdl.to(dev).eval()
embs = []
with torch.no_grad():
    for i in range(0, len(names), BATCH):
        b = tok(names[i:i + BATCH], padding=True, truncation=True, max_length=MAXLEN,
                return_tensors="pt").to(dev)
        o = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        embs.append(((o * m).sum(1) / m.sum(1)).cpu().numpy())
E = np.vstack(embs)
Z = TruncatedSVD(SVD_DIM, random_state=0).fit_transform(E)


def fit(k, seed, weight):
    km = KMeans(k, random_state=seed, n_init=10).fit(Z, sample_weight=weight)
    vc = pd.Series(km.labels_).groupby(km.labels_).size()  # 개념 수
    vis = pd.Series(w).groupby(km.labels_).sum()           # 방문 수
    return km, int(vis.min()), float(vis.max() / w.sum() * 100), int(vc.min())


sweep = []
for k in K_GRID:
    runs = [(fit(k, s, w), s) for s in SEEDS]
    (km, mn, mx, cmin), s = min(runs, key=lambda r: r[0][0].inertia_)
    sweep.append({"k": k, "seed": s, "min_group_visits": mn, "largest_pct": round(mx, 1),
                  "min_group_concepts": cmin, "inertia": round(km.inertia_, 2)})
    print(f"    k={k:3d} seed={s} 최소그룹 방문={mn:5d} 최대={mx:5.1f}% 최소개념수={cmin}")
sw = pd.DataFrame(sweep)
ok = sw[sw.min_group_visits >= MIN_VISITS]
k_pick = int(ok.k.max()) if len(ok) else int(sw.k.min())
seed_pick = int(sw[sw.k == k_pick].iloc[0].seed)
km, _, _, _ = fit(k_pick, seed_pick, w)
lab = km.labels_
print(f"[=] k={k_pick} seed={seed_pick} (규칙: 그룹 방문 >= {MIN_VISITS} 중 최대 k)")

# 비가중 민감도
km_u, _, _, _ = fit(k_pick, seed_pick, None)
agree_unw = ari(lab, km_u.labels_)

cui2grp = dict(zip(cuis, lab))
grp_name = {}
for g in range(k_pick):
    mem = [(c, int(vis_ct[c])) for c in cuis if cui2grp[c] == g]
    mem.sort(key=lambda x: -x[1])
    grp_name[g] = name_of[mem[0][0]]

work["group"] = work.primary.map(cui2grp)
work["stratum_id"] = np.where(work.primary.isna(), "NO_CONCEPT",
                              "G" + pd.Series(work.group, index=work.index).astype("Int64").astype(str))
work["stratum_name"] = np.where(work.primary.isna(), "개념 미매칭",
                                pd.Series(work.group, index=work.index).map(grp_name))

full = asg[["SUBJECT_ID", "HADM_ID", "split", "stratum", "clusterA", "clusterCC_bert"]].merge(
    work[["HADM_ID", "primary", "group", "stratum_id", "stratum_name", "n_cui"]],
    on="HADM_ID", how="left").merge(
    strata42[["HADM_ID", "stratum_id"]].rename(columns={"stratum_id": "stratum42"}),
    on="HADM_ID", how="left")
is_empty = full.HADM_ID.isin(df[df.empty_template].HADM_ID)
full["stratum_id"] = full.stratum_id.fillna(
    pd.Series(np.where(is_empty, "EMPTY_CC", "NO_CC"), index=full.index))
full["stratum_name"] = full.stratum_name.fillna(
    pd.Series(np.where(is_empty, "빈 템플릿(층외)", "CC 없음(HPI 라벨)"), index=full.index))
full.to_csv(OUT / "43_concept_groups.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 표
rows = []
for sid, g in work.groupby("stratum_id"):
    strs = g.cc.str.lower().value_counts()
    mem = sorted([(c, int(vis_ct[c])) for c in cuis
                  if "G%d" % cui2grp[c] == sid], key=lambda x: -x[1]) if sid != "NO_CONCEPT" else []
    topc = "; ".join(f"{name_of[c]}({n})" for c, n in mem[:6])
    rows.append({
        "stratum_id": sid, "name": g.stratum_name.iloc[0], "n": len(g),
        "pct": round(len(g) / len(work) * 100, 1),
        "test_n": int((g.split == "test").sum()), "n_concepts": len(mem),
        "top_concept_share": round(mem[0][1] / len(g) * 100, 1) if mem else 0.0,
        "words_median": float(g.n_word.median()), "n_icd_median": float(g.n_icd.median()),
        "top_concepts": topc,
        "top5_strings": "; ".join(f"{s}({n})" for s, n in strs.head(5).items()),
    })
tab = pd.DataFrame(rows).sort_values("n", ascending=False)
tab.to_csv(OUT / "table58_group_table.csv", index=False, encoding="utf-8-sig")


def eta2(labels, y):
    y = np.asarray(y, float)
    gm = y.mean()
    ssb = sum(len(y[labels == l]) * (y[labels == l].mean() - gm) ** 2 for l in np.unique(labels))
    return float(ssb / ((y - gm) ** 2).sum())


PAIRS = [("C0237284", "C0241526", "unresponsive / unresponsiveness"),
         ("C0278061", "C0856054", "altered mental status / mental status changes"),
         ("C0013404", "C0231807", "dyspnea / dyspnea on exertion"),
         ("C0476273", "C1145670", "respiratory distress / respiratory failure")]
srows = []
for a, b, lbl in PAIRS:
    ga, gb = cui2grp.get(a), cui2grp.get(b)
    srows.append({"pair": lbl, "cui1": a, "name1": name_of.get(a), "group1": ga,
                  "cui2": b, "name2": name_of.get(b), "group2": gb,
                  "same_group": ga is not None and ga == gb})
syn = pd.DataFrame(srows)
syn.to_csv(OUT / "table60_group_synonym_test.csv", index=False, encoding="utf-8-sig")

named = work[work.stratum_id != "NO_CONCEPT"]
e2 = eta2(work.stratum_id.values, work.n_word.values)
old = asg[asg.stratum == "CC"].dropna(subset=["clusterCC_bert"])
e2_old = eta2(old.clusterCC_bert.values, old.n_word.values)
cm = work.dropna(subset=["clusterCC_bert"])
checks = pd.DataFrame([
    ("그룹 수", k_pick, f"규칙: 그룹 방문 >= {MIN_VISITS} 중 최대 k"),
    ("이름 붙는 방문", len(named), f"{len(named)/len(work)*100:.1f}% of CC층, "
     f"{len(named)/len(full)*100:.1f}% of 전체 (42: 37.2%, 37: 27.3%)"),
    ("NO_CONCEPT 층", int((work.stratum_id == 'NO_CONCEPT').sum()), "그룹에 안 섞음"),
    ("빈 템플릿 층외", int(df.empty_template.sum()), "42 에서는 TAIL4 안에 있었다"),
    ("test >= 30 인 층", int((tab.test_n >= 30).sum()), f"/ {len(tab)}"),
    ("최소 층 크기", int(tab.n.min()), ""),
    ("최대 층 비율", f"{tab.pct.max():.1f}%", "42: 14.4%, 37: 10.0%"),
    ("eta2(단어수)", round(e2, 4), f"42: 0.0484, 37 B: {e2_old:.4f}"),
    ("비가중 민감도 ARI", round(agree_unw, 4), "가중 vs 비가중 개념 군집"),
    ("ARI(그룹, 42 층)", round(ari(work.stratum_id.astype(str),
                                  strata42.set_index('HADM_ID').loc[work.HADM_ID]
                                  .stratum_id.astype(str)), 4), ""),
    ("ARI(그룹, 37 B군집)", round(ari(cm.stratum_id.astype(str),
                                    cm.clusterCC_bert.astype(int)), 4), ""),
    ("사전등록 2쌍 병합", f"{int(syn.head(2).same_group.sum())}/2", "UMLS 가 CUI 로 못 붙인 쌍"),
], columns=["항목", "값", "비고"])
checks.to_csv(OUT / "table59_group_checks.csv", index=False, encoding="utf-8-sig")

meta = {
    "unit": "concept (CUI) canonical name", "n_concepts": len(cuis),
    "k": k_pick, "seed": seed_pick, "min_visits_rule": MIN_VISITS,
    "weighted_by": "visit count", "unweighted_ari": round(agree_unw, 4),
    "n_cc_visits": len(work), "n_named": len(named),
    "named_pct_of_cc": round(len(named) / len(work) * 100, 1),
    "named_pct_of_all": round(len(named) / len(full) * 100, 1),
    "n_no_concept": int((work.stratum_id == "NO_CONCEPT").sum()),
    "n_empty_template_excluded": int(df.empty_template.sum()),
    "strata_test_ge30": int((tab.test_n >= 30).sum()), "n_strata": len(tab),
    "min_stratum": int(tab.n.min()), "largest_pct": float(tab.pct.max()),
    "eta2_word_count": round(e2, 4),
    "ari_vs_42": round(ari(work.stratum_id.astype(str),
                           strata42.set_index("HADM_ID").loc[work.HADM_ID]
                           .stratum_id.astype(str)), 4),
    "ari_vs_37B": round(ari(cm.stratum_id.astype(str), cm.clusterCC_bert.astype(int)), 4),
    "synonym_pairs": srows, "sweep": sweep,
}
with open(OUT / "43_groups_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=int)

print(f"\n[=] 그룹 {k_pick}개 · 이름 붙는 방문 {len(named):,} "
      f"({meta['named_pct_of_cc']}% of CC층, {meta['named_pct_of_all']}% of 전체)")
print(f"[=] eta2(단어수) {e2:.4f} · test>=30 {meta['strata_test_ge30']}/{len(tab)} "
      f"· 최소층 {tab.n.min()} · 최대 {tab.pct.max():.1f}%")
print("\n" + checks.to_string(index=False))
print("\n[=] 사전등록 쌍 검정")
print(syn[["pair", "group1", "group2", "same_group"]].to_string(index=False))
