"""§주증상 UMLS 정규화 — CC 문자열을 UMLS 개념(CUI)으로 붙인다. 군집화 전 단계다.

왜:
  37 의 군집화는 표면형으로 갈렸다. shortness of breath(C5 735건) / dyspnea(C10 552건) /
  sob(C13 267건) 이 세 군집으로 흩어졌는데 셋은 같은 개념이다. 41 의 검정에서
  sob 층과 shortness of breath 층은 진단 구성이 구별되지 않았다(순열 p=0.5632).
  동의어를 상류에서 붙이고 나서 군집화한다.

사전:
  umls_kb_2022ab.jsonl — UMLS 2022AB 서브셋(scispacy 공개 배포본, cat0129).
  concept_id / canonical_name / aliases / types(TUI) 를 갖는다.
  손사전 scripts/cc_lexicon.json 은 UMLS 가 못 붙이는 병원 약어(brbpr, gib 등)에만
  폴백으로 쓴다. 폴백이 몇 건을 좌우했는지 stage 별로 집계해 표로 남긴다.

매칭 단계 (사전 고정, 위에서부터):
  s1_exact       atom 정규화 후 alias 완전일치
  s2_abbrev      손사전 철자·약어 확장 후 완전일치
  s3_modifier    수식어 제거(acute/chronic/worsening/s/p/r/o ...) 후 완전일치
  s4_contain     atom 안에 들어 있는 가장 긴 alias (2단어 이상 또는 6자 이상)
  s5_hand        손사전 개념 폴백 (UMLS 밖 병원 약어)
  residual       못 붙인 것. 억지로 붙이지 않는다.

동음이의 해소 (사전 고정): 같은 alias 가 여러 CUI 를 가리키면
  type 우선순위(SYMPTOM > INJURY > DISEASE > TESTRESULT > PROCEDURE > 그밖) ->
  canonical_name 이 짧은 것 -> CUI 문자열이 작은 것.

semantic type 으로 증상/질환/시술을 나눈다. REPORT_CC.md §9 의 한계(“CC 에 질환명·
입원경위·검사수치가 섞여 있는데 라벨로 구분 못 했다”)가 여기서 풀린다.

산출: out/41_cc_umls.pkl, out/table50_cc_umls_coverage.csv, out/table51_cc_cui_top.csv,
      out/table52_cc_synonym_umls.csv, out/41_cc_umls_meta.json
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
KB = ROOT / "umls_kb_2022ab.jsonl"
LEX = Path(__file__).resolve().parent / "cc_lexicon.json"

# 유지할 semantic type. 화학물질·유전자·해부구조는 CC 에 나올 일이 없고 오탐만 만든다.
TYPE_GROUP = {
    "T184": "SYMPTOM", "T033": "FINDING",
    "T037": "INJURY",
    "T047": "DISEASE", "T046": "DISEASE", "T048": "DISEASE", "T191": "DISEASE",
    "T019": "DISEASE", "T020": "DISEASE", "T049": "DISEASE", "T050": "DISEASE",
    "T190": "DISEASE",
    "T034": "TESTRESULT", "T201": "TESTRESULT",
    "T060": "PROCEDURE", "T061": "PROCEDURE", "T059": "PROCEDURE", "T058": "PROCEDURE",
}
PRIO = {"SYMPTOM": 0, "FINDING": 1, "INJURY": 2, "DISEASE": 3,
        "TESTRESULT": 4, "PROCEDURE": 5}

# alias 로 쓰면 안 되는 것 — 짧거나 일반어라 오탐을 만든다.
ALIAS_STOP = {"cc", "na", "n/a", "pt", "patient", "code", "none", "no", "other", "yes",
              "unknown", "report", "service", "male", "female", "man", "woman", "s", "p",
              "dr", "hospital", "admission", "transfer", "history", "chief complaint",
              "complaint", "pain", "mass", "change", "changes", "increase", "decrease",
              "left", "right", "acute", "chronic", "severe", "new", "old", "status"}

MODIFIER = re.compile(
    r"^(?:acute(?:ly)?|chronic(?:ally)?|worsening|increasing|increased|decreased|new onset|"
    r"new|sudden(?:ly)?|progressive|persistent|intermittent|recurrent|severe|mild|moderate|"
    r"possible|probable|likely|suspected|r/o|rule out|s/p|status post|witnessed|unwitnessed|"
    r"episode of|episodes of|complaints? of|c/o|generalized|profound|ongoing|continued|"
    r"unresolved|significant|massive|large|small|left|right|bilateral)\s+")

SEP = re.compile(r"\s*(?:,|;|\||\+|&|\band\b|\bwith\b|\bvs\.?\b|\bthen\b)\s*")
PUNCT = re.compile(r"[^\w\s/'-]+")
WS = re.compile(r"\s+")

lex = json.loads(LEX.read_text(encoding="utf-8"))
PROTECT = {t: t.replace("/", "\x01") for t in lex["protect"]}
SPELL = lex["spelling"]
ABBREV = lex["abbrev"]
EMPTY = set(lex["empty_markers"])
HAND = {}
for cname, blk in lex["concepts"].items():
    for t in blk["terms"]:
        HAND.setdefault(t, cname)
NONSYM = re.compile("|".join(re.escape(p) for p in lex["non_symptom_markers"]["patterns"]))


def protect(s):
    for k, v in PROTECT.items():
        s = s.replace(k, v)
    return s


def unprotect(s):
    return s.replace("\x01", "/")


def norm(s):
    s = PUNCT.sub(" ", str(s).lower())
    s = WS.sub(" ", s).strip(" -'")
    return s


def atoms_of(text):
    t = protect(norm(text))
    out = []
    for a in SEP.split(t):
        a = unprotect(a).strip(" -'.")
        a = WS.sub(" ", a)
        if a:
            out.append(a)
    return out


def expand(a):
    """손사전 철자·약어 확장. 토큰 단위와 전체 문자열 둘 다 시도한다."""
    if a in ABBREV:
        return ABBREV[a]
    toks = [SPELL.get(t, t) for t in a.split()]
    toks = [ABBREV.get(t, t) if len(toks) == 1 else t for t in toks]
    b = " ".join(toks)
    return b


# ---------------------------------------------------------------- KB 색인
print("[.] UMLS KB 색인", flush=True)
alias2cui, cui2info = {}, {}
best = {}
n_kb = n_keep = 0
with open(KB, "r", encoding="utf-8") as f:
    for line in f:
        n_kb += 1
        d = json.loads(line)
        groups = [TYPE_GROUP[t] for t in d.get("types", []) if t in TYPE_GROUP]
        if not groups:
            continue
        n_keep += 1
        g = min(groups, key=lambda x: PRIO[x])
        cui = d["concept_id"]
        canon = d.get("canonical_name", "")
        cui2info[cui] = (canon, g, d.get("types", []))
        key = (PRIO[g], len(canon), cui)
        for al in [canon] + list(d.get("aliases", [])):
            a = norm(al)
            if len(a) < 3 or a in ALIAS_STOP or a.isdigit() or len(a) > 60:
                continue
            if a not in best or key < best[a]:
                best[a] = key
                alias2cui[a] = cui
print(f"[=] KB {n_kb:,}개념 -> 대상 type {n_keep:,}개념 / alias {len(alias2cui):,}", flush=True)

# 포함매칭용: alias 의 단어수 상한
MAXW = 6

# ---------------------------------------------------------------- 매칭
cc = pd.read_pickle(OUT / "36_cc_text.pkl")
work = cc[cc.has_cc].copy()
print(f"[.] CC 있는 방문 {len(work):,}", flush=True)

cache = {}


def match_atom(a):
    if a in cache:
        return cache[a]
    if a in EMPTY or len(a) < 3:
        r = (None, "empty")
    elif a in alias2cui:
        r = (alias2cui[a], "s1_exact")
    else:
        b = expand(a)
        if b != a and b in alias2cui:
            r = (alias2cui[b], "s2_abbrev")
        else:
            c, prev = b, None
            while c != prev:
                prev, c = c, MODIFIER.sub("", c).strip()
            if c and c in alias2cui:
                r = (alias2cui[c], "s3_modifier")
            else:
                # 최장 포함 alias
                w = c.split()
                hit = None
                for n in range(min(MAXW, len(w)), 0, -1):
                    for i in range(len(w) - n + 1):
                        g = " ".join(w[i:i + n])
                        if (n >= 2 or len(g) >= 6) and g in alias2cui:
                            hit = g
                            break
                    if hit:
                        break
                if hit:
                    r = (alias2cui[hit], "s4_contain")
                elif a in HAND or b in HAND or c in HAND:
                    r = ("HAND:" + HAND.get(a, HAND.get(b, HAND.get(c))), "s5_hand")
                else:
                    r = (None, "residual")
    cache[a] = r
    return r


rows = []
stage_ct = Counter()
resid = Counter()
for h, txt in zip(work.HADM_ID, work.cc_demog_stripped):
    ats = atoms_of(txt)
    cuis, stages, names, groups, res = [], [], [], [], []
    for a in ats:
        cui, st = match_atom(a)
        stage_ct[st] += 1
        if cui is None:
            if st == "residual":
                res.append(a)
                resid[a] += 1
            continue
        if cui in cuis:
            continue
        cuis.append(cui)
        stages.append(st)
        if cui.startswith("HAND:"):
            names.append(cui[5:])
            groups.append("HAND")
        else:
            nm, g, _ = cui2info[cui]
            names.append(nm)
            groups.append(g)
    rows.append((int(h), txt, ats, cuis, names, groups, stages, res,
                 bool(NONSYM.search(str(txt).lower()))))

norm_df = pd.DataFrame(rows, columns=[
    "HADM_ID", "cc", "atoms", "cuis", "cui_names", "type_groups", "stages",
    "residual_atoms", "non_symptom_marker"])
norm_df["n_cui"] = norm_df.cuis.str.len()
norm_df["has_symptom"] = norm_df.type_groups.map(
    lambda g: any(x in ("SYMPTOM", "FINDING", "HAND") for x in g))
norm_df["norm_text"] = norm_df.cui_names.map(lambda v: "; ".join(v))
norm_df.to_pickle(OUT / "41_cc_umls.pkl")

# ---------------------------------------------------------------- 표
tot_atom = sum(stage_ct.values())
cov = pd.DataFrame({"stage": list(stage_ct), "atoms": [stage_ct[k] for k in stage_ct]})
cov["atom_pct"] = (cov.atoms / tot_atom * 100).round(1)
cov = cov.sort_values("atoms", ascending=False)
cov.to_csv(OUT / "table50_cc_umls_coverage.csv", index=False, encoding="utf-8-sig")

cui_ct = Counter()
for v in norm_df.cuis:
    cui_ct.update(v)
top = pd.DataFrame(
    [(c, n, (c[5:] if c.startswith("HAND:") else cui2info[c][0]),
      ("HAND" if c.startswith("HAND:") else cui2info[c][1]))
     for c, n in cui_ct.most_common(60)],
    columns=["cui", "n_visits", "name", "type_group"])
top["pct"] = (top.n_visits / len(norm_df) * 100).round(1)
top.to_csv(OUT / "table51_cc_cui_top.csv", index=False, encoding="utf-8-sig")

# 사전등록 동의어 8쌍이 같은 CUI 로 붙는가 (37 docstring 의 가설)
PAIRS = [("shortness of breath", "sob"), ("dyspnea", "shortness of breath"),
         ("abdominal pain", "abd pain"), ("fever", "fevers"),
         ("gi bleed", "gib"), ("gi bleed", "gastrointestinal bleed"),
         ("unresponsive", "unresponsiveness"),
         ("altered mental status", "mental status changes")]
srows = []
for a, b in PAIRS:
    ca, sa = match_atom(norm(a))
    cb, sb = match_atom(norm(b))
    srows.append((a, ca, sa, b, cb, sb, ca is not None and ca == cb))
syn = pd.DataFrame(srows, columns=["term1", "cui1", "stage1", "term2", "cui2", "stage2",
                                   "merged"])
syn.to_csv(OUT / "table52_cc_synonym_umls.csv", index=False, encoding="utf-8-sig")

resid_top = pd.DataFrame(resid.most_common(80), columns=["atom", "n"])
resid_top.to_csv(OUT / "table53_cc_residual_top.csv", index=False, encoding="utf-8-sig")

meta = {
    "kb": KB.name,
    "kb_concepts_total": n_kb, "kb_concepts_kept": n_keep, "aliases_indexed": len(alias2cui),
    "type_groups_kept": sorted(set(TYPE_GROUP.values())),
    "n_visits_with_cc": len(norm_df),
    "atoms_total": tot_atom,
    "stage_pct": {k: round(v / tot_atom * 100, 1) for k, v in stage_ct.most_common()},
    "visits_with_ge1_cui": int((norm_df.n_cui >= 1).sum()),
    "visits_with_ge1_cui_pct": round(float((norm_df.n_cui >= 1).mean()) * 100, 1),
    "visits_with_symptom_concept_pct": round(float(norm_df.has_symptom.mean()) * 100, 1),
    "visits_all_residual": int((norm_df.n_cui == 0).sum()),
    "hand_fallback_atoms": stage_ct.get("s5_hand", 0),
    "hand_fallback_pct": round(stage_ct.get("s5_hand", 0) / tot_atom * 100, 1),
    "unique_cui": len(cui_ct),
    "synonym_pairs_merged": int(syn.merged.sum()), "synonym_pairs_total": len(syn),
    "type_group_visit_pct": {
        g: round(float(norm_df.type_groups.map(lambda v: g in v).mean()) * 100, 1)
        for g in ["SYMPTOM", "FINDING", "DISEASE", "PROCEDURE", "TESTRESULT", "INJURY", "HAND"]},
    "non_symptom_marker_pct": round(float(norm_df.non_symptom_marker.mean()) * 100, 1),
}
with open(OUT / "41_cc_umls_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n[=] atom {tot_atom:,} / 방문 {len(norm_df):,}")
print(cov.to_string(index=False))
print(f"\n[=] CUI 1개 이상 붙은 방문 {meta['visits_with_ge1_cui']:,} "
      f"({meta['visits_with_ge1_cui_pct']}%) · 고유 CUI {len(cui_ct):,}")
print(f"[=] 손사전 폴백 atom {meta['hand_fallback_atoms']:,} ({meta['hand_fallback_pct']}%)")
print(f"[=] type group 방문 비율: {meta['type_group_visit_pct']}")
print("\n[=] 사전등록 동의어 8쌍")
print(syn.to_string(index=False))
print("\n[=] 상위 CUI 20")
print(top.head(20).to_string(index=False))
print("\n[=] 미매칭 상위 25")
print(resid_top.head(25).to_string(index=False))
