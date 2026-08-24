"""§3 HPI 입력 텍스트 구성 — 앵커 분할 + PMH 차감.

목적은 "이 환자가 이번에 무엇 때문에 왔는가"만 남기는 것이다. HPI 원문은 절반가량이
만성 질환명 나열이라(측정: 만성 표현 48.9%, 급성 제시 표현 없이 만성만 22.4%) 그대로
임베딩하면 지배 축이 만성 부담으로 간다 — REPORT_CHRONIC.md 가 기록한 실패와 같은 구조다.

3단계. 전 방문이 같은 처리를 거치게 해서 앵커 유무에 따른 길이 편차를 줄인다
(REPORT_BHC.md 에서 η²(길이)=0.2484 로 길이가 지배 축이 됐던 실패를 피하려는 것).

  1. 앵커 분할 — 급성 제시 표현 뒤 절만 취한다. 없으면 HPI 전체를 그대로 넘긴다(route=full)
  2. PMH 차감 — 같은 노트 past medical history 에 나오는 내용어를 제거. 전 방문 공통
  3. 폴백    — 2 이후 내용어 3개 미만이면 no_acute=True 로 표시(군집화에서 빼지는 않는다)

앵커는 2단이다. 강한 제시 표현(TIER1)을 먼저 찾고, 없으면 맨 present 류(TIER2)를 본다.
`status post` / `underwent` 는 앵커로 쓰지 않는다 — 그 뒤에 오는 것은 급성 사건이 아니라
과거 시술이라 오히려 만성 이력을 남기게 된다. 해당 방문은 route=full 로 흘려보내고
PMH 차감에 맡긴다.

PMH 차감의 알려진 동작: PMH 에 COPD 가 있는 환자가 COPD 악화로 오면 "COPD" 가 지워지고
exacerbation 만 남는다. 급성 사건만 남기는 목적에는 맞지만 병명이 사라지므로 라벨링 때는
원문(hpi_raw)을 함께 본다.

산출: out/32_hpi_text.pkl, out/table39_hpi_prep.csv, out/32_hpi_prep_meta.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SRC = ROOT / "data4LLM_with_note.csv"

MIN_CONTENT = 2          # 차감 후 내용어가 이보다 적으면 no_acute
                         # 2 로 둔다 — "respiratory failure" 같은 2단어 주소견이 실제로 흔하다

# ---------------------------------------------------------------- 섹션 파싱 (01_profile.py 와 동일)
CANON = ["history of present illness", "past medical history", "allergies",
         "medications on admission", "brief hospital course"]
SPLIT_RE = re.compile(r"(?mi)^\s*(" + "|".join(re.escape(h) for h in CANON) + r")\s*:",
                      re.MULTILINE)


def split_sections(text):
    parts = SPLIT_RE.split(str(text))
    out = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        body = parts[i + 1].strip().rstrip(",").strip()
        out[key] = (out.get(key, "") + " " + body).strip()
    return out


# ---------------------------------------------------------------- 앵커
TIER1 = re.compile(
    r"(?i)\b(?:"
    r"present(?:s|ed|ing)?\s+(?:to\s+(?:the\s+)?\S+\s+)?with"
    r"|complain(?:s|ing|ed)?\s+of|c/o"
    r"|develop(?:s|ed|ing)"
    r"|was\s+found|found\s+(?:down|to\s+have|unresponsive)"
    r"|brought\s+(?:in|to)"
    r"|admitted\s+(?:with|for|after)"
    r"|came\s+(?:in|to)(?:\s+\S+)?\s+(?:with|for)"
    r"|report(?:s|ed|ing)"
    r"|note[ds]?\s+to\s+have"
    r"|transferred\s+(?:from|to|for)"
    r"|p/w|presenting\s+w/|present(?:s|ed)\s+w/"
    r"|denies"
    r")\b")
TIER2 = re.compile(r"(?i)\bpresent(?:s|ed|ing)\b")

# 만성 이력 절 — 만성 표지에서 시작해 다음 절 경계까지. PMH 차감의 대체가 아니라 보완이다.
# 실측(1차 구현): 앵커 분할 + PMH 차감만으로는 만성 표현이 48.9% -> 21.8% 에서 멈췄고,
# route=full 쪽 길이가 anchor 쪽의 1.8배로 남아 길이가 다시 축이 될 위험이 있었다.
CHRON_CLAUSE = re.compile(
    r"(?i)\b(?:with\s+(?:a\s+)?(?:known\s+)?(?:past\s+)?(?:medical\s+)?histor(?:y|ies)\s+(?:of\s+)?"
    r"|histor(?:y|ies)\s+of|\bh/o\b|\bhx\b(?:\s+of)?|\bs/p\b|status\s+post|known\s+)"
    r"[^.;]{0,120}?"
    r"(?=\bwho\b|\bpresent|\bp/w\b|\bcomplain|\badmit|\bdevelop|\bbrought\b|\bnow\b|\btransferred\b"
    r"|\bdenies\b|\bexcept\b|\breport(?:s|ed|ing)\b|\bfound\b|\bnoted\b|[.;]|$)")

# "3-day history of hallucinations" 는 급성 서술이다. 기간·급성 수식어가 앞에 붙은 history of 는
# 만성 절에서 보호한다 — 보호하지 않으면 no_acute 가 7.6% -> 17.8% 로 튄다(1차 구현에서 확인).
# 파이썬 lookbehind 는 고정 길이만 되므로 자리표시자로 가린 뒤 되돌린다.
ACUTE_HX = re.compile(
    r"(?i)\b((?:\d+\s*[-\s]?\s*(?:day|week|month|hour|hr|yr|year)s?|recent|acute|new|sudden|"
    r"brief|short|episodic|intermittent|worsening|progressive)\s+(?:onset\s+)?)"
    r"(?:histor(?:y|ies)|hx)\s+of\b")
HX_MARK = "\x00HXOF\x00"

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

SW = set("""the of a an with and who was were is are be been to for in on at by or as from
this that these those there here he she his her him they them their it its we our you your
not no but also then than had has have having do does did
patient patients pt pts mr mrs ms dr year years old yo y/o male female man woman gentleman lady
history hx admitted admission admit hospital ed er icu status post s p per h/o who's""".split())


def content_words(text):
    return [w.lower() for w in WORD.findall(text) if len(w) >= 2 and w.lower() not in SW]


def norm(w):
    """가벼운 단복수 정규화. 어간 추출은 하지 않는다 — 과잉 제거를 피하기 위함."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def anchor_split(text):
    """급성 제시 표현 뒤 절만 반환. (텍스트, route)"""
    m = TIER1.search(text)
    if m:
        return text[m.end():].strip(" ,.;:-"), "anchor"
    m = TIER2.search(text)
    if m:
        return text[m.end():].strip(" ,.;:-"), "anchor2"
    return text, "full"


def tidy(text):
    out = re.sub(r"\s+", " ", text)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"(?:[,;:]\s*){2,}", ", ", out)
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
    return out.strip(" ,.;:-")


def strip_chronic(text):
    guarded = ACUTE_HX.sub(lambda m: m.group(1) + HX_MARK, text)
    out = CHRON_CLAUSE.sub(" ", guarded)
    return tidy(out.replace(HX_MARK, "history of"))


def subtract_pmh(text, pmh_norm):
    """PMH 에 나오는 내용어를 제거하고 문장 구두점을 정리한다."""
    def repl(m):
        w = m.group(0)
        lw = w.lower()
        if len(w) < 2 or lw in SW:
            return w
        return "" if norm(lw) in pmh_norm else w

    return tidy(WORD.sub(repl, text))


# ---------------------------------------------------------------- 본체
print(f"[.] 읽는 중: {SRC.name}", flush=True)
df = pd.read_csv(SRC, usecols=["SUBJECT_ID", "HADM_ID", "NOTE"])
sec = df["NOTE"].map(split_sections)
df["hpi_raw"] = sec.map(lambda d: d.get("history of present illness", ""))
df["pmh_raw"] = sec.map(lambda d: d.get("past medical history", ""))
df = df.drop(columns=["NOTE"])
print(f"[.] 방문 {len(df)} | HPI 존재 {(df.hpi_raw.str.len() > 0).mean() * 100:.1f}% | "
      f"PMH 존재 {(df.pmh_raw.str.len() > 0).mean() * 100:.1f}%", flush=True)

# 1단계 — 앵커 분할
split = df["hpi_raw"].map(anchor_split)
df["hpi_anchor"] = split.map(lambda t: t[0])
df["route"] = split.map(lambda t: t[1])

# 2단계 — 만성 절 제거 후 PMH 차감
df["hpi_nochron"] = df["hpi_anchor"].map(strip_chronic)
pmh_sets = df["pmh_raw"].map(lambda t: {norm(w) for w in content_words(t)})
df["hpi_acute"] = [subtract_pmh(t, s) for t, s in zip(df["hpi_nochron"], pmh_sets)]

n_before = df["hpi_nochron"].map(lambda t: len(content_words(t)))
n_after = df["hpi_acute"].map(lambda t: len(content_words(t)))
df["n_content_raw"] = df["hpi_raw"].map(lambda t: len(content_words(t)))
df["n_content_acute"] = n_after
df["pmh_removed_frac"] = np.where(n_before > 0, (n_before - n_after) / n_before.clip(lower=1), 0.0)
df["n_tok_raw"] = df["hpi_raw"].str.split().map(len)

# 3단계 — 폴백
df["no_acute"] = n_after < MIN_CONTENT

# ---------------------------------------------------------------- 점검표
CHRON = r"(?i)history of|\bh/o\b|chronic|\bs/p\b"
w_raw = df["hpi_raw"].str.split().map(len)
w_ac = df["hpi_acute"].str.split().map(len)


def q(s, p):
    return float(np.quantile(s, p))


meta = {
    "n_visits": int(len(df)),
    "route": {k: int(v) for k, v in df["route"].value_counts().items()},
    "route_pct": {k: round(float(v) * 100, 1) for k, v in
                  df["route"].value_counts(normalize=True).items()},
    "words": {
        "hpi_raw": {"median": q(w_raw, .5), "p90": q(w_raw, .9), "mean": round(float(w_raw.mean()), 1)},
        "hpi_acute": {"median": q(w_ac, .5), "p90": q(w_ac, .9), "mean": round(float(w_ac.mean()), 1)},
        "shrink_pct": round(float(1 - w_ac.sum() / w_raw.sum()) * 100, 1),
    },
    "length_gap_between_routes": {
        r: round(float(w_ac[df["route"] == r].median()), 1) for r in df["route"].unique()
    },
    "chronic_marker_pct": {
        "hpi_raw": round(float(df["hpi_raw"].str.contains(CHRON, regex=True).mean()) * 100, 1),
        "after_anchor": round(float(df["hpi_anchor"].str.contains(CHRON, regex=True).mean()) * 100, 1),
        "after_chron_strip": round(float(df["hpi_nochron"].str.contains(CHRON, regex=True).mean()) * 100, 1),
        "after_pmh_subtract": round(float(df["hpi_acute"].str.contains(CHRON, regex=True).mean()) * 100, 1),
    },
    "pmh_removed_frac": {
        "mean": round(float(df["pmh_removed_frac"].mean()), 3),
        "median": round(q(df["pmh_removed_frac"], .5), 3),
        "p90": round(q(df["pmh_removed_frac"], .9), 3),
        "zero_pct": round(float((df["pmh_removed_frac"] == 0).mean()) * 100, 1),
        "zero_pct_by_route": {r: round(float((df.loc[df["route"] == r, "pmh_removed_frac"] == 0).mean()) * 100, 1)
                              for r in df["route"].unique()},
    },
    "no_acute": {"n": int(df["no_acute"].sum()),
                 "pct": round(float(df["no_acute"].mean()) * 100, 1),
                 "min_content": MIN_CONTENT},
}

tab = pd.DataFrame([
    {"단계": "HPI 원문", "방문": len(df), "단어수 중앙": q(w_raw, .5),
     "만성표현 %": meta["chronic_marker_pct"]["hpi_raw"]},
    {"단계": "앵커 분할 후", "방문": len(df),
     "단어수 중앙": float(df["hpi_anchor"].str.split().map(len).median()),
     "만성표현 %": meta["chronic_marker_pct"]["after_anchor"]},
    {"단계": "만성 절 제거 후", "방문": len(df),
     "단어수 중앙": float(df["hpi_nochron"].str.split().map(len).median()),
     "만성표현 %": meta["chronic_marker_pct"]["after_chron_strip"]},
    {"단계": "PMH 차감 후", "방문": len(df), "단어수 중앙": q(w_ac, .5),
     "만성표현 %": meta["chronic_marker_pct"]["after_pmh_subtract"]},
])
tab.to_csv(OUT / "table39_hpi_prep.csv", index=False, encoding="utf-8-sig")

keep = ["SUBJECT_ID", "HADM_ID", "hpi_raw", "hpi_anchor", "hpi_nochron", "hpi_acute", "route",
        "pmh_removed_frac", "no_acute", "n_content_raw", "n_content_acute", "n_tok_raw"]
df[keep].to_pickle(OUT / "32_hpi_text.pkl")
with open(OUT / "32_hpi_prep_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps(meta, ensure_ascii=False, indent=2))
print("\n--- 처리 예시 5건 ---")
for i in df.sample(5, random_state=11).index:
    print(f"  [{df.at[i, 'route']}] 원문 : {df.at[i, 'hpi_raw'][:120]}")
    print(f"        -> 급성: {df.at[i, 'hpi_acute'][:120]}")
