"""§외부 채점 — 군집을 만들 때 안 쓴 정보로 분할을 평가한다.

실루엣은 "이 라벨이 이 좌표계에서 뭉쳐 있나" 다. 좌표계를 바꾸면 승자가 바뀌고
(기본 공간 승자 = 기본 k-means, 가중치 공간 승자 = 가중치 k-means), 이산적인
라벨일수록 무조건 높아진다. 그래서 실루엣을 최대화하는 최적해는 "SEQ1 코드마다
군집 하나" 즉 ICD 사전 베끼기다. 그 게임은 답이 정해져 있으니 채점을 밖으로 뺀다.

**진단 텍스트에 한 글자도 안 들어간 것들**로 채점한다:
  O1 처방 약물 프로파일  (drug_id, multi-hot)      — "같은 군집이면 같은 약을 받았나"
  O2 시술 프로파일        (pro_id, multi-hot)
  O3 30일 재입원          (SUBJECT_ID + ADMITTIME 로 유도, binary)
  O4 나이                 (AGE, 연속)

척도는 eta² = 1 - SS_within/SS_total (분할이 설명하는 분산 비율).
군집 수 k 가 크면 자동으로 올라가므로, **같은 k·같은 군집 크기로 라벨만 섞은
무작위 분할**을 20회 돌려 eta²_rand 를 구하고 그 차이 delta 를 같이 낸다.
delta 가 0 이면 "이 분할은 크기 배분 말고는 아무것도 설명하지 않는다" 는 뜻이다.

무작위 기준선을 안 깔면 k=30 이 k=18 을 이기는 게 당연해서 비교가 성립하지 않는다.

산출: out/table81_external_check.csv, out/59_external_check.json
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
N_PERM, SEED = 20, 0

# ── 코호트 순서 고정 ────────────────────────────────────────────────────────
df = pd.read_pickle(OUT / "40_dxtext.pkl")
HADM = df["HADM_ID"].to_numpy()
N = len(HADM)

src = pd.read_csv(ROOT / "data4LLM_with_note.csv",
                  usecols=["SUBJECT_ID", "HADM_ID", "AGE", "ADMITTIME",
                           "drug_id", "pro_id"])
src = src.drop_duplicates("HADM_ID").set_index("HADM_ID").reindex(HADM).reset_index()
assert src["HADM_ID"].notna().all(), "코호트 방문이 data4LLM 에 없다"
print(f"[.] 방문 {N} | 외부 정보 결합 완료", flush=True)


def multihot(col):
    """'[1, 2, 3]' 문자열 리스트 -> multi-hot 행렬."""
    lists = [ast.literal_eval(s) if isinstance(s, str) else [] for s in src[col]]
    vocab = sorted({i for l in lists for i in l})
    idx = {v: j for j, v in enumerate(vocab)}
    M = np.zeros((N, len(vocab)), dtype=np.float32)
    for r, l in enumerate(lists):
        for i in l:
            M[r, idx[i]] = 1.0
    return M, len(vocab), float(np.mean([len(l) for l in lists]))


DRUG, n_drug, avg_drug = multihot("drug_id")
PROC, n_proc, avg_proc = multihot("pro_id")
print(f"[.] 약물 어휘 {n_drug} (방문당 평균 {avg_drug:.1f}) | "
      f"시술 어휘 {n_proc} (평균 {avg_proc:.1f})", flush=True)

# 30일 재입원: 같은 환자의 다음 입원이 30일 안에 있으면 1
t = pd.DataFrame({"s": src["SUBJECT_ID"], "t": pd.to_datetime(src["ADMITTIME"])})
t["order"] = np.arange(N)
t = t.sort_values(["s", "t"])
nxt = t.groupby("s")["t"].shift(-1)
readm = ((nxt - t["t"]).dt.days <= 30).fillna(False).to_numpy()
READM = np.zeros(N, dtype=np.float32)
READM[t["order"].to_numpy()] = readm.astype(np.float32)
AGE = src["AGE"].to_numpy(dtype=np.float32)
print(f"[.] 30일 재입원 {int(READM.sum())}건 ({READM.mean() * 100:.1f}%) | "
      f"나이 중앙 {np.median(AGE):.0f}", flush=True)

# L2 정규화 — 진단 개수가 많은 방문이 거리를 지배하지 않게
def l2(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-9)


OUTCOMES = {
    "O1_처방약물": l2(DRUG),
    "O2_시술": l2(PROC),
    "O3_30일재입원": READM.reshape(-1, 1),
    "O4_나이": ((AGE - AGE.mean()) / AGE.std()).reshape(-1, 1),
}

# ── 분할들 ──────────────────────────────────────────────────────────────────
def load(path, col):
    z = pd.read_csv(OUT / path, usecols=["HADM_ID", col])
    return z.set_index("HADM_ID").reindex(HADM)[col].to_numpy()


seq1 = df["seq1_code"].fillna("NA")
top24 = set(seq1.value_counts().index[:24])          # k 를 25 로 맞춘 순수 조회표
lookup25 = np.where(seq1.isin(top24), seq1, "__기타__")

PARTS = {
    "CCS(주분석)": pd.factorize(load("ccs_assignments.csv", "group"))[0],
    "ICD9_chapter": pd.factorize(load("dxtext_chapter_assignments.csv", "chapter"))[0],
    "기본_kmeans_k25(출하판)": load("dxtext_cluster_assignments.csv", "primary_cluster"),
    "가중치_kmeans_k30": load("dxtext_weight_cluster_assignments.csv", "primary_cluster"),
    "SEQ1최빈24+기타(조회표,k=25)": pd.factorize(lookup25)[0],
    "SEQ1코드_그대로(k=1400)": pd.factorize(seq1)[0],
}


def eta2(X, lab):
    """1 - SS_within/SS_total. 군집 중심까지의 제곱거리 합 기준."""
    tot = float(((X - X.mean(0)) ** 2).sum())
    if tot <= 0:
        return 0.0
    within = 0.0
    for g in np.unique(lab):
        m = lab == g
        within += float(((X[m] - X[m].mean(0)) ** 2).sum())
    return 1.0 - within / tot


rng = np.random.default_rng(SEED)
rows = []
for pname, lab in PARTS.items():
    lab = np.asarray(lab)
    k = int(len(np.unique(lab)))
    perms = [rng.permutation(lab) for _ in range(N_PERM)]   # k·크기 보존, 내용만 섞음
    rec = {"분할": pname, "k": k}
    for oname, X in OUTCOMES.items():
        obs = eta2(X, lab)
        rnd = np.array([eta2(X, p) for p in perms])
        rec[f"{oname}_eta2%"] = round(obs * 100, 3)
        rec[f"{oname}_rand%"] = round(float(rnd.mean()) * 100, 3)
        rec[f"{oname}_delta%"] = round((obs - float(rnd.mean())) * 100, 3)
    rows.append(rec)
    print(f"[=] {pname:22s} k={k:4d} | " + "  ".join(
        f"{o.split('_')[0]} d={rec[f'{o}_delta%']:+.2f}%" for o in OUTCOMES), flush=True)

tb = pd.DataFrame(rows)
tb.to_csv(OUT / "table81_external_check.csv", index=False, encoding="utf-8-sig")

meta = {
    "질문": "군집을 만들 때 안 쓴 정보로 채점하면 어느 분할이 이기나",
    "외부정보": {"처방약물_어휘": n_drug, "방문당평균": round(avg_drug, 1),
              "시술_어휘": n_proc, "시술_방문당평균": round(avg_proc, 1),
              "재입원30일_건수": int(READM.sum()),
              "재입원30일_%": round(float(READM.mean()) * 100, 1)},
    "척도": "eta2 = 1 - SS_within/SS_total, 무작위(k·크기 보존) 20회 평균을 뺀 delta",
    "표": rows,
    "_note": "실루엣과 달리 임베딩 공간에 묶이지 않는다. 어느 좌표계에서 나온 분할이든 "
             "같은 자로 잰다. delta=0 이면 크기 배분 말고는 설명하는 게 없다는 뜻이다.",
}
(OUT / "59_external_check.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("[+] out/table81_external_check.csv, 59_external_check.json")
